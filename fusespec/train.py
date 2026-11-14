import os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from .models import build_model, EEGNet
from .data import COV_REG

# ---------------------------------------------------------------- model families
DEEP = ("eeg", "fnirs", "concat", "hybrid")                            # original deep models
VARIANTS = ("hybrid_md", "hybrid_pre", "hybrid_rg", "stack")           # new fusion variants
LDA_KINDS = ("lda_eeg", "lda_fnirs", "lda_both", "lda_fnirs2", "lda_both2")
STD_KINDS = ("csp_lda", "fbcsp_lda", "eegnet")                         # standard EEG baselines (need the raw-EEG cache)
ALL_KINDS = DEEP + LDA_KINDS + ("csp_lda", "fbcsp_lda") + VARIANTS + ("eegnet",)
EEG_ONLY = ("eeg", "lda_eeg", "csp_lda", "fbcsp_lda", "eegnet")
FUSION = ("concat", "hybrid", "hybrid_md", "hybrid_pre", "hybrid_rg", "stack", "lda_both", "lda_both2")
HYBRID_FAMILY = ("hybrid", "hybrid_md", "hybrid_pre", "hybrid_rg")

# per-kind training settings (anything can be overridden through fit_predict keyword arguments)
KIND_DEFAULTS = {
    "hybrid":     dict(gate_init=0.0, gate_lr_mult=1.0),
    "hybrid_rg":  dict(gate_init=0.0, gate_lr_mult=1.0),
    # modality dropout: per training trial, drop the EEG input (25 %) or the fNIRS input (25 %); gate starts half open
    "hybrid_md":  dict(gate_init=0.5, gate_lr_mult=1.0, p_drop_eeg=0.25, p_drop_fnirs=0.25),
    # pretrain both encoders as single-modality models, then fuse (encoders fine-tuned at 0.1x learning rate)
    "hybrid_pre": dict(gate_init=0.5, gate_lr_mult=1.0, enc_lr_mult=0.1),
}
KEYS = ("eeg", "fnirs", "y", "sess", "raw", "cov")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def take(d, idx):
    return {k: d[k][idx] for k in KEYS}


def cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in KEYS}


# ---------------------------------------------------------------- classical features
def _norm_stats(eeg, fnirs):
    """Statistics from the TRAINING split only: per-feature (EEG) / per-channel (fNIRS)."""
    return (eeg.mean((0, 1), keepdims=True), eeg.std((0, 1), keepdims=True) + 1e-6,
            fnirs.mean((0, 2), keepdims=True), fnirs.std((0, 2), keepdims=True) + 1e-6)


def fnirs_features(F, fs=10.0):
    """Richer haemodynamic features per channel from (N, 2*n_ch, T): mean 5-15 s, slope over 0-15 s, peak, time-to-peak.
    HbO channels (first half) use the maximum, HbR channels (second half) the minimum."""
    N, C, T = F.shape
    h = C // 2
    t = np.arange(T) / fs
    tc = t - t.mean()
    mean = F[:, :, int(5 * fs):].mean(-1)
    slope = (F * tc).sum(-1) / (tc ** 2).sum()
    peak = np.concatenate([F[:, :h].max(-1), F[:, h:].min(-1)], 1)
    ttp = np.concatenate([F[:, :h].argmax(-1), F[:, h:].argmin(-1)], 1) / fs
    return np.concatenate([mean, slope, peak, ttp], 1)


def _lda_feats(kind, E, F):
    e = E.mean(1)                                   # EEG band power over the task
    f = F[:, :, 50:].mean(-1)                       # HbO+HbR mean over 5-15 s (original fNIRS feature)
    if kind == "lda_eeg":    return e
    if kind == "lda_fnirs":  return f
    if kind == "lda_both":   return np.concatenate([e, f], 1)
    f2 = fnirs_features(F)
    if kind == "lda_fnirs2": return f2
    if kind == "lda_both2":  return np.concatenate([e, f2], 1)
    raise ValueError(kind)


class FBCSP:
    """(Filter-bank) CSP computed from per-trial band covariances (N, n_bands, ch, ch). Binary problems only.
    Features: log of the CSP-projected variance, n_pairs filters from each end of the spectrum, per band.
    Optional ANOVA-F selection of the k best features (the classic FBCSP uses mutual information)."""
    def __init__(self, bands, n_pairs=2, k=None):
        self.bands, self.n_pairs, self.k = list(bands), n_pairs, k

    def fit(self, C, y):
        from scipy.linalg import eigh
        self.W = {}
        ch = C.shape[-1]
        for b in self.bands:
            C0, C1 = C[y == 0, b].mean(0), C[y == 1, b].mean(0)
            w, V = eigh(C0 + COV_REG * np.eye(ch), C0 + C1 + 2 * COV_REG * np.eye(ch))      # ascending generalised eigenvalues
            p = self.n_pairs
            self.W[b] = V[:, np.r_[0:p, ch - p:ch]]
        X = self._feats(C)
        self.sel = None
        if self.k and self.k < X.shape[1]:
            from sklearn.feature_selection import SelectKBest, f_classif
            self.sel = SelectKBest(f_classif, k=self.k).fit(X, y)
        return self

    def _feats(self, C):
        return np.concatenate([np.log(np.maximum(np.einsum("ck,ncd,dk->nk", self.W[b], C[:, b], self.W[b]), 1e-12))
                               for b in self.bands], 1)

    def transform(self, C):
        X = self._feats(C)
        return self.sel.transform(X) if self.sel is not None else X


def _csp_predict(kind, tr, te):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    m = FBCSP([0], n_pairs=3) if kind == "csp_lda" else FBCSP(range(1, 10), n_pairs=2, k=10)
    m.fit(tr["cov"], tr["y"])
    clf = LDA(solver="lsqr", shrinkage="auto").fit(m.transform(tr["cov"]), tr["y"])
    return clf.predict(m.transform(te["cov"]))


# ---------------------------------------------------------------- deep training
def _fit_loop(model, xs, y, epochs, bs, groups, wd, noise, p_drop=(0.0, 0.0)):
    """Fixed-epoch AdamW + OneCycle loop; light Gaussian input noise; optional modality dropout (2-input models only)."""
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    steps = epochs * int(np.ceil(len(y) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(y), device=y.device).split(bs):
            batch = [x[idx] for x in xs]
            batch = [b + noise * torch.randn_like(b) for b in batch]      # light augmentation
            if len(batch) == 2 and (p_drop[0] > 0 or p_drop[1] > 0):
                r = torch.rand(len(idx), device=y.device)
                drop_e, drop_f = r < p_drop[0], (r >= p_drop[0]) & (r < p_drop[0] + p_drop[1])
                batch = [batch[0] * (~drop_e).float()[:, None, None], batch[1] * (~drop_f).float()[:, None, None]]
            loss = lossf(model(*batch), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return model


@torch.no_grad()
def _proba(model, xs):
    model.eval()
    return torch.softmax(model(*xs), 1).cpu().numpy()


def _groups(model, lr, gate_lr_mult=1.0, enc_lr_mult=1.0):
    gate = [p for n, p in model.named_parameters() if "fusion.gate" in n]
    enc = [p for n, p in model.named_parameters() if n.startswith(("eeg_enc", "fnirs_enc"))] if enc_lr_mult != 1.0 else []
    skip = {id(p) for p in gate + enc}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in skip], lr=lr)]
    if enc:
        groups.append(dict(params=enc, lr=lr * enc_lr_mult))
    if gate:
        groups.append(dict(params=gate, lr=lr * gate_lr_mult))
    return groups


def fit_predict(kind, tr, te, epochs=60, bs=16, lr=1e-3, wd=1e-2, d=64, noise=0.1, device="cpu", seed=0,
                return_model=False, eegnet_epoch_mult=2, pre_epochs=None, ft_epochs=None, **over):
    """Train on `tr`, return accuracy on `te`. Fixed epoch budget + final-epoch model: no test-set peeking.
    tr/te are dicts with keys eeg, fnirs, y, sess, raw, cov.  Per-kind defaults live in KIND_DEFAULTS and can be
    overridden (gate_init, gate_lr_mult, p_drop_eeg, p_drop_fnirs, enc_lr_mult).
    With return_model=True (hybrid family only) returns (acc, model, norm_stats)."""
    set_seed(seed)
    ytr, yte = tr["y"], te["y"]
    if kind in LDA_KINDS:                          # classical references: shrinkage LDA on simple features
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        clf = LDA(solver="lsqr", shrinkage="auto").fit(_lda_feats(kind, tr["eeg"], tr["fnirs"]), ytr)
        return float((clf.predict(_lda_feats(kind, te["eeg"], te["fnirs"])) == yte).mean())
    if kind in ("csp_lda", "fbcsp_lda"):
        return float((_csp_predict(kind, tr, te) == yte).mean())

    cfg = {**KIND_DEFAULTS.get(kind, {}), **over}
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    ytr_t, yte_t = torch.tensor(ytr, device=device), torch.tensor(yte, device=device)
    n_classes = int(max(ytr.max(), yte.max()) + 1)

    if kind == "eegnet":
        Rtr, Rte = tr["raw"].astype(np.float32), te["raw"].astype(np.float32)
        m, s = Rtr.mean((0, 2), keepdims=True), Rtr.std((0, 2), keepdims=True) + 1e-6
        Rtr, Rte = T((Rtr - m) / s), T((Rte - m) / s)
        model = EEGNet(Rtr.shape[1], Rtr.shape[2], n_classes).to(device)
        _fit_loop(model, (Rtr,), ytr_t, epochs * eegnet_epoch_mult, bs, [dict(params=list(model.parameters()), lr=lr)], wd, noise)
        return float((_proba(model, (Rte,)).argmax(1) == yte).mean())

    em, es, fm, fs = _norm_stats(tr["eeg"], tr["fnirs"])
    Etr, Ete = T((tr["eeg"] - em) / es), T((te["eeg"] - em) / es)
    Ftr, Fte = T((tr["fnirs"] - fm) / fs), T((te["fnirs"] - fm) / fs)
    mk = lambda k: build_model(k, Etr.shape[-1], Ftr.shape[1], n_classes=n_classes, d=d, gate_init=cfg.get("gate_init", 0.0)).to(device)
    pd_ = (cfg.get("p_drop_eeg", 0.0), cfg.get("p_drop_fnirs", 0.0))

    if kind == "stack":                            # probability averaging of two independently trained unimodal models
        pe = _proba(_train_uni(mk("eeg"), Etr, Ftr, ytr_t, epochs, bs, lr, wd, noise), (Ete, Fte))
        pf = _proba(_train_uni(mk("fnirs"), Etr, Ftr, ytr_t, epochs, bs, lr, wd, noise), (Ete, Fte))
        return float((((pe + pf) / 2).argmax(1) == yte).mean())

    arch = "hybrid_rg" if kind == "hybrid_rg" else ("hybrid" if kind in ("hybrid", "hybrid_md", "hybrid_pre") else kind)
    model = mk(arch)
    if kind == "hybrid_pre":                       # stage 1: unimodal pre-training of both encoders
        pe_ep = pre_epochs or epochs
        me = _train_uni(mk("eeg"), Etr, Ftr, ytr_t, pe_ep, bs, lr, wd, noise)
        mf = _train_uni(mk("fnirs"), Etr, Ftr, ytr_t, pe_ep, bs, lr, wd, noise)
        model.eeg_enc.load_state_dict(me.eeg_enc.state_dict())
        model.fnirs_enc.load_state_dict(mf.fnirs_enc.state_dict())
        epochs = ft_epochs or max(epochs // 2, 5)  # stage 2: fine-tune the fusion model
    groups = _groups(model, lr, cfg.get("gate_lr_mult", 1.0), cfg.get("enc_lr_mult", 1.0))
    _fit_loop(model, (Etr, Ftr), ytr_t, epochs, bs, groups, wd, noise, p_drop=pd_)
    acc = float((_proba(model, (Ete, Fte)).argmax(1) == yte).mean())
    return (acc, model, (em, es, fm, fs)) if return_model else acc


def _train_uni(model, Etr, Ftr, ytr_t, epochs, bs, lr, wd, noise):
    return _fit_loop(model, (Etr, Ftr), ytr_t, epochs, bs, _groups(model, lr), wd, noise)


# ---------------------------------------------------------------- protocols
def folds(data, protocol, k=5, seed=0, subjects=None):
    """Yield (subject, fold_id, train, test). data: {subj: dict(eeg, fnirs, y, sess, raw, cov)}.
    within : stratified k-fold over the subject's trials (random split inside the same sessions).
    session: leave-one-session-out inside each subject (train on 2 sessions, test on the 3rd; 3 folds).
    loso   : train on all other subjects, test on this one."""
    subs = [s for s in data if subjects is None or s in subjects]
    if protocol == "within":
        for s in subs:
            d = data[s]
            for i, (a, b) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(d["eeg"], d["y"])):
                yield s, i, take(d, a), take(d, b)
    elif protocol == "session":
        for s in subs:
            d = data[s]
            for j in np.unique(d["sess"]):
                yield s, int(j), take(d, np.where(d["sess"] != j)[0]), take(d, np.where(d["sess"] == j)[0])
    elif protocol == "loso":
        for s in subs:
            yield s, 0, cat([data[t] for t in data if t != s]), data[s]
    else:
        raise ValueError(protocol)


def n_folds(data, protocol, k=5):
    return len(data) * {"within": k, "session": 3, "loso": 1}[protocol]


def benchmark(data, kinds=ALL_KINDS, protocol="within", seeds=(0,), out=None, verbose=True, subjects=None, callback=None, **kw):
    """Run every model on every fold. If `out` (a CSV path) is given, each result is appended as soon as it is
    computed and already-finished (model, subject, fold, seed) combinations are skipped -> safe to re-run after a
    disconnect or to add more models / seeds later. One progress line per finished subject."""
    rows, done = [], set()
    if out and os.path.exists(out) and os.path.getsize(out) > 0:
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {(r["model"], int(r["subject"]), int(r["fold"]), int(r["seed"])) for r in rows}
        print(f"resuming: {len(done)} results already in {out}")
    elif out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        open(out, "w").write("model,subject,fold,seed,acc\n")

    t0 = time.time()

    def report(key, accs):
        if callback and accs:
            callback()                                    # e.g. copy the CSVs to Google Drive after every subject
        if verbose and accs:
            line = " ".join(f"{k}={np.mean(v):.2f}" for k, v in accs.items())
            print(f"[{(time.time() - t0) / 60:5.1f} min] seed {key[0]} subj {key[1]:02d} | {line}", flush=True)

    cur_key, cur = None, {}
    for seed in seeds:
        for s, i, tr, te in folds(data, protocol, seed=seed, subjects=subjects):
            if cur_key != (seed, s):
                report(cur_key, cur); cur_key, cur = (seed, s), {}
            for kind in kinds:
                if (kind, int(s), int(i), int(seed)) in done:
                    continue
                acc = fit_predict(kind, tr, te, seed=seed, **kw)
                rows.append(dict(model=kind, subject=s, fold=i, seed=seed, acc=acc))
                cur.setdefault(kind, []).append(acc)
                if out:
                    with open(out, "a") as f:
                        f.write(f"{kind},{s},{i},{seed},{acc}\n")
    report(cur_key, cur)
    return pd.DataFrame(rows)


def time_kinds(data, protocol, kinds, subject=1, **kw):
    """Seconds to train each model on ONE fold (after a warm-up), for projecting the total run time."""
    s, i, tr, te = next(folds(data, protocol, subjects=[subject]))
    fit_predict("eeg", tr, te, **{**kw, "epochs": 1})            # CUDA warm-up
    out = {}
    for kind in kinds:
        t0 = time.time(); fit_predict(kind, tr, te, **kw)
        if kw.get("device", "cpu") != "cpu": torch.cuda.synchronize()
        out[kind] = time.time() - t0
    return out


# ---------------------------------------------------------------- statistics
def _boot_ci(x, n=2000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def _holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.zeros(m); run = 0.0
    for rank, j in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[j])); out[j] = run
    return out


def _paired(diff):
    from scipy.stats import wilcoxon
    diff = diff.dropna()
    p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
    lo, hi = _boot_ci(diff.values)
    return diff.mean(), lo, hi, p, len(diff)


def summarize(df, reference="hybrid", title="", models=None):
    """Mean accuracy per model (averaged over folds/seeds, then over subjects) + paired Wilcoxon tests of `reference`
    against every other model (Holm-corrected) with bootstrap 95 % CIs of the paired difference (over subjects)."""
    per_subj = df.groupby(["model", "subject"]).acc.mean().unstack(0)          # subject x model
    if models: per_subj = per_subj[[m for m in models if m in per_subj]]
    tab = pd.DataFrame({"mean_acc": per_subj.mean(), "std_across_subj": per_subj.std(), "n_subj": per_subj.count()}).round(4)
    print(f"\n=== {title} Accuracy (mean over subjects) ===")
    print(tab.sort_values("mean_acc", ascending=False).to_string())
    if reference in per_subj and len(per_subj) >= 6:
        others = [c for c in per_subj if c != reference]
        res = [(o, *_paired(per_subj[reference] - per_subj[o])) for o in others]
        holm = _holm(np.array([r[4] for r in res]))
        print(f"\n{reference} vs other (paired Wilcoxon over subjects; Holm-corrected over {len(res)} tests)")
        for (o, d, lo, hi, p, n), ph in zip(res, holm):
            print(f"  vs {o:11s} delta={d:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p:.4f}  p_holm={ph:.4f}")
    return tab


def fusion_vs_best_eeg(df, title=""):
    """Headline table: every fusion model against the STRONGEST EEG-only model of the same experiment (chosen by mean
    accuracy, so the comparison is conservative). Positive delta = fusion helps."""
    per_subj = df.groupby(["model", "subject"]).acc.mean().unstack(0)
    eeg_models = [m for m in EEG_ONLY if m in per_subj]
    fus = [m for m in FUSION if m in per_subj]
    if not eeg_models or not fus:
        return None
    best = per_subj[eeg_models].mean().idxmax()
    res = [(m, *_paired(per_subj[m] - per_subj[best])) for m in fus]
    holm = _holm(np.array([r[4] for r in res]))
    print(f"\n=== {title}: fusion vs strongest EEG-only model ({best}, {per_subj[best].mean():.3f}) ===")
    for (m, d, lo, hi, p, n), ph in zip(res, holm):
        print(f"  {m:11s} acc={per_subj[m].mean():.3f}  delta={d:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p:.4f}  p_holm={ph:.4f}")
    return best
