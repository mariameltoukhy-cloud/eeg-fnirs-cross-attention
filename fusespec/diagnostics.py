"""Does a hybrid-family model actually use fNIRS?  (gate value + test-time ablations + attention map)"""
import numpy as np, torch


@torch.no_grad()
def fnirs_dependence(model, stats, te, device="cpu", n_shuffle=20, seed=0):
    """Test-time ablations on a trained hybrid-family model (hybrid, hybrid_md, hybrid_pre, hybrid_rg).
      real        : accuracy with the true paired fNIRS
      shuffled    : fNIRS taken from random other test trials (breaks the EEG-fNIRS pairing)
      mean_fnirs  : fNIRS replaced by the training mean (all zeros after z-scoring)
      gate_closed : fNIRS path forced off (pure EEG path)
      eeg_zero    : EEG replaced by the training mean -> accuracy the model can reach from fNIRS alone
    If real is not clearly above shuffled / mean_fnirs / gate_closed, the model is not relying on fNIRS.
    tanh_gate is the mean gate value on the test trials (an input-dependent gate varies per trial)."""
    em, es, fm, fs = stats
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    E, F, y = T((te["eeg"] - em) / es), T((te["fnirs"] - fm) / fs), te["y"]
    model.eval()
    acc = lambda E_, F_: float((model(E_, F_).argmax(1).cpu().numpy() == y).mean())
    out = dict(real=acc(E, F))
    out["tanh_gate"] = float(model.fusion.last_gate)
    rng = torch.Generator().manual_seed(seed)
    out["shuffled"] = float(np.mean([acc(E, F[torch.randperm(len(F), generator=rng).to(device)]) for _ in range(n_shuffle)]))
    out["mean_fnirs"] = acc(E, torch.zeros_like(F))
    model.fusion.force_closed = True
    try:
        out["gate_closed"] = acc(E, F)
    finally:
        model.fusion.force_closed = False
    out["eeg_zero"] = acc(torch.zeros_like(E), F)
    return out


@torch.no_grad()
def attention_maps(model, stats, te, device="cpu"):
    """Mean EEG->fNIRS attention per class: array (2, T_eeg, T_fnirs_tokens), averaged over heads and trials."""
    em, es, fm, fs = stats
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    E, F, y = T((te["eeg"] - em) / es), T((te["fnirs"] - fm) / fs), np.asarray(te["y"])
    model.eval()
    _, w = model(E, F, return_attn=True)               # (N, H, T_e, T_f')
    w = w.mean(1).cpu().numpy()
    return np.stack([w[y == c].mean(0) for c in (0, 1)])


def summarize_diag(diag, title=""):
    """Mean diagnostics per configuration + paired tests of 'real minus ablated' accuracy over subjects."""
    from .train import _paired
    cols = [c for c in ("tanh_gate", "real", "shuffled", "mean_fnirs", "gate_closed", "eeg_zero") if c in diag]
    print(f"\n=== {title}: mean over subjects (n = {diag.subject.nunique()}) ===")
    print(diag.groupby("config")[cols].mean().round(3).to_string())
    print("\nfNIRS reliance = accuracy with real fNIRS minus accuracy with fNIRS ablated "
          "(paired Wilcoxon over subjects, bootstrap 95 % CI)")
    for cfg, g in diag.groupby("config"):
        g = g.set_index("subject")
        for ab in ("shuffled", "mean_fnirs", "gate_closed"):
            d, lo, hi, p, n = _paired(g["real"] - g[ab])
            print(f"  {cfg:30s} real - {ab:11s} = {d:+.3f} [{lo:+.3f},{hi:+.3f}]  p={p:.4f}")
