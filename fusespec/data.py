"""Loader/preprocessor for Shin et al. 2017 open-access EEG+NIRS dataset (TU Berlin, 29 subjects).

Expected layout (as unzipped from the official download; any parent folder works, files are found by glob):
    <root>/**/subject 01/**/cnt.mat , mrk.mat      (EEG, 200 Hz, 30 EEG + EOG)
    <root>/**/subject 01/**/cnt.mat , mrk.mat      (NIRS, 10 Hz, 36 ch x 2 wavelengths, raw light intensity)
Pass separate eeg_root / nirs_root so the two 'cnt.mat' trees are never mixed up.

Sessions (1-indexed): 1,3,5 = motor imagery (left vs right hand); 2,4,6 = mental arithmetic (MA vs baseline).
Each session has 20 trials -> 60 trials per subject per task. Marker times are in ms.

Real file structure (verified against subject 01):
    EEG  cnt[i]: clab(32), fs=200, x(T,32) [30 EEG + EOG]      mrk[i]: time (ms), y (2,20)
    NIRS cnt[i]: clab(72), fs=10,  x(T,72) RAW LIGHT INTENSITY (36 ch x 2 wavelengths), wavelengths (2,)
         -> converted here to HbO/HbR with the modified Beer-Lambert law.

Output per subject (cached to .npz):
    eeg   (N, T_e, n_ch*n_bands)  log band power, 1 s windows / 0.5 s hop over the 0-10 s task period
    fnirs (N, 2*n_ch, T_f)        HbO then HbR, band-passed, baseline-corrected, 0-15 s after cue
    y     (N,)
Extra cache (raw_<task>_sXX.npz, needed only for CSP/FBCSP and EEGNet):
    raw   (N, 30, 1000)  float16, EEG band-passed 4-40 Hz, 100 Hz, 0-10 s task period (no re-referencing)
    cov   (N, 10, 30, 30) trace-normalised trial covariance in 10 bands: 8-30 Hz, then nine 4 Hz bands 4-8 ... 36-40 Hz
    sess  (N,)           session index (0,1,2) of every trial
"""
import glob, os, re
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
TASK_SESSIONS = {"mi": [0, 2, 4], "ma": [1, 3, 5]}   # 0-indexed into the 1x6 cell arrays
EEG_FS_TARGET = 200
# decadic molar extinction (cm^-1 / M), Prahl table; only the 2x2 ratio matters because channels are z-scored later
_WL = [750, 760, 800, 850]
_E_HBO = [518.0, 586.0, 816.0, 1058.0]
_E_HBR = [1405.24, 1548.52, 761.84, 691.32]
SD_CM, DPF = 3.0, 6.0
WIN_S, HOP_S, TASK_S = 1.0, 0.5, 10.0
NIRS_TMIN, NIRS_TMAX, NIRS_BASE = 0.0, 15.0, (-2.0, 0.0)   # 15 s at 10 Hz -> 150 samples
CSP_BANDS = [(8, 30)] + [(4 + 4 * i, 8 + 4 * i) for i in range(9)]   # band 0 = broad (plain CSP); bands 1-9 = filter bank
RAW_FS = 100                                                          # sampling rate of the stored raw epochs (EEGNet input)
COV_REG = 1e-5                                                        # ridge added before any covariance inversion / CSP


def _find(root, subj, fname):
    hits = sorted(glob.glob(os.path.join(root, "**", f"subject {subj:02d}", "**", fname), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{fname} for subject {subj:02d} not found under {root}")
    pref = [h for h in hits if "without" not in h.lower()]   # keep EOG-contaminated version (we drop EOG ch ourselves)
    return (pref or hits)[0]


def _mat(path, key):
    return np.atleast_1d(sio.loadmat(path, struct_as_record=False, squeeze_me=True)[key])


def _clab(c):
    return [str(s) for s in np.atleast_1d(c.clab)]


def _epoch(x, fs, times_ms, tmin, tmax):
    """x: (ch, T) -> (N, ch, n) epochs for onsets in ms."""
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    out = []
    for t in times_ms:
        s = int(round(t / 1000 * fs))
        out.append(x[:, s + a: s + b])
    return np.stack(out)


def _band_power(epochs, fs):
    """epochs (N, ch, T) -> log band power (N, n_win, ch*n_bands) via Welch in 1 s windows."""
    from mne.time_frequency import psd_array_welch
    N, ch, T = epochs.shape
    w, h = int(WIN_S * fs), int(HOP_S * fs)
    starts = np.arange(0, T - w + 1, h)
    wins = np.stack([epochs[:, :, s:s + w] for s in starts], axis=1)           # (N, nw, ch, w)
    psd, freqs = psd_array_welch(wins.reshape(-1, ch, w), fs, fmin=4, fmax=45, n_fft=w, verbose=False)
    feats = [np.log10(psd[..., (freqs >= lo) & (freqs < hi)].mean(-1) + 1e-12) for lo, hi in BANDS.values()]
    f = np.stack(feats, -1).reshape(N, len(starts), ch * len(BANDS))           # (N, nw, ch*bands)
    return f.astype(np.float32)


def nirs_intensity_to_hb(x, clab, wavelengths, fs, band=(0.01, 0.2)):
    """x (T, 72) raw intensity -> (2*n_ch, T) [HbO ; HbR] in uM (modified Beer-Lambert), band-passed."""
    names = [str(c) for c in clab]
    key = lambda n: re.sub(r"\s*(low|high)\s*wl", "", n, flags=re.I).strip()
    low = {key(n): i for i, n in enumerate(names) if "low" in n.lower()}
    high = {key(n): i for i, n in enumerate(names) if "high" in n.lower()}
    if len(low) == len(high) == len(names) // 2 and set(low) == set(high):
        order = sorted(low)
        lo_idx, hi_idx = [low[k] for k in order], [high[k] for k in order]
    else:
        print("WARNING: could not pair wavelengths by channel name; assuming first half=low, second half=high WL")
        h = len(names) // 2
        lo_idx, hi_idx = list(range(h)), list(range(h, 2 * h))
    wl = np.sort(np.asarray(wavelengths, float))
    E = np.array([[np.interp(w, _WL, _E_HBO), np.interp(w, _WL, _E_HBR)] for w in wl])   # rows: low, high WL
    I = np.clip(np.asarray(x, float), 1e-9, None)
    od = lambda idx: -np.log10(I[:, idx] / I[:, idx].mean(0, keepdims=True))             # (T, n_ch)
    OD = np.stack([od(lo_idx), od(hi_idx)], axis=1)                                      # (T, 2, n_ch)
    conc = np.einsum("ij,tjc->tic", np.linalg.inv(E), OD) / (SD_CM * DPF) * 1e6          # (T, 2, n_ch) -> uM
    hb = np.concatenate([conc[:, 0], conc[:, 1]], axis=1).T                              # (2*n_ch, T)
    sos = butter(3, band, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, hb, axis=-1)


def load_subject(eeg_root, nirs_root, subj, task="mi"):
    import mne
    mne.set_log_level("ERROR")
    sess = TASK_SESSIONS[task]
    e_cnt, e_mrk = _mat(_find(eeg_root, subj, "cnt.mat"), "cnt"), _mat(_find(eeg_root, subj, "mrk.mat"), "mrk")
    n_cnt, n_mrk = _mat(_find(nirs_root, subj, "cnt.mat"), "cnt"), _mat(_find(nirs_root, subj, "mrk.mat"), "mrk")
    E, Fn, Y = [], [], []
    for s in sess:
        c, m = e_cnt[s], e_mrk[s]
        names = _clab(c)
        keep = [i for i, n in enumerate(names) if "EOG" not in n.upper()]
        fs = float(c.fs)
        if len(keep) != 30:
            print(f"WARNING: expected 30 EEG channels after dropping EOG, got {len(keep)}: {names}")
        x = np.asarray(c.x, dtype=np.float64)[:, keep].T                       # (ch, T)
        x = mne.filter.filter_data(x, fs, 1.0, 45.0, verbose=False)
        if abs(fs - EEG_FS_TARGET) > 1e-6:
            x = mne.filter.resample(x, down=fs / EEG_FS_TARGET, verbose=False)
        x -= x.mean(0, keepdims=True)                                          # common average reference
        ep = _epoch(x, EEG_FS_TARGET, np.atleast_1d(m.time), 0.0, TASK_S)
        E.append(_band_power(ep, EEG_FS_TARGET))
        y = np.argmax(np.atleast_2d(m.y), axis=0)

        # ---- fNIRS (session-matched markers: use NIRS markers, same trial order) ----
        nc, nm = n_cnt[s], n_mrk[s]
        nfs = float(nc.fs)
        hb = nirs_intensity_to_hb(nc.x, nc.clab, nc.wavelengths, nfs)
        full = _epoch(hb, nfs, np.atleast_1d(nm.time), NIRS_BASE[0], NIRS_TMAX)
        nb = int((NIRS_BASE[1] - NIRS_BASE[0]) * nfs)
        full = full - full[:, :, :nb].mean(-1, keepdims=True)                  # pre-cue baseline correction
        Fn.append(full[:, :, nb:].astype(np.float32))
        y_n = np.argmax(np.atleast_2d(nm.y), axis=0)
        assert len(y) == len(y_n) and (y == y_n).all(), f"EEG/NIRS labels disagree in session {s}: {list(m.className)} vs {list(nm.className)}"
        Y.append(y)
    return np.concatenate(E), np.concatenate(Fn), np.concatenate(Y).astype(np.int64)


def band_covs_from_epochs(ep, fs):
    """(N, ch, T) epochs -> (N, n_bands, ch, ch) trace-normalised covariances (bands filtered on the epochs themselves;
    used by the synthetic generator. The real loader filters the continuous recording instead, which avoids edge effects)."""
    out = []
    for lo, hi in CSP_BANDS:
        x = sosfiltfilt(butter(4, (lo, hi), btype="band", fs=fs, output="sos"), ep.astype(np.float64), axis=-1)
        C = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
        out.append(C / np.trace(C, axis1=1, axis2=2)[:, None, None])
    return np.stack(out, 1).astype(np.float32)


def load_raw_subject(eeg_root, subj, task="mi"):
    """EEG only. Returns raw (N,30,1000) float16, cov (N,10,30,30) float32, y (N,), sess (N,)."""
    import mne
    mne.set_log_level("ERROR")
    e_cnt, e_mrk = _mat(_find(eeg_root, subj, "cnt.mat"), "cnt"), _mat(_find(eeg_root, subj, "mrk.mat"), "mrk")
    R, C, Y, S = [], [], [], []
    for k, s in enumerate(TASK_SESSIONS[task]):
        c, m = e_cnt[s], e_mrk[s]
        names = _clab(c)
        keep = [i for i, n in enumerate(names) if "EOG" not in n.upper()]
        fs = float(c.fs)
        x = np.asarray(c.x, dtype=np.float64)[:, keep].T                       # (30, T), no re-referencing
        if abs(fs - EEG_FS_TARGET) > 1e-6:
            x = mne.filter.resample(x, down=fs / EEG_FS_TARGET, verbose=False)
        times = np.atleast_1d(m.time)
        xb = sosfiltfilt(butter(4, (4, 40), btype="band", fs=EEG_FS_TARGET, output="sos"), x, axis=-1)
        ep = _epoch(xb, EEG_FS_TARGET, times, 0.0, TASK_S)                      # (N, 30, 2000)
        R.append(ep[:, :, ::EEG_FS_TARGET // RAW_FS].astype(np.float16))        # 4-40 Hz content -> every 2nd sample is safe
        covs = []
        for lo, hi in CSP_BANDS:
            xf = sosfiltfilt(butter(4, (lo, hi), btype="band", fs=EEG_FS_TARGET, output="sos"), x, axis=-1)
            e = _epoch(xf, EEG_FS_TARGET, times, 0.0, TASK_S)
            cv = np.einsum("nct,ndt->ncd", e, e) / e.shape[-1]
            covs.append(cv / np.trace(cv, axis1=1, axis2=2)[:, None, None])
        C.append(np.stack(covs, 1).astype(np.float32))
        n = len(times)
        Y.append(np.argmax(np.atleast_2d(m.y), axis=0)); S.append(np.full(n, k))
    return np.concatenate(R), np.concatenate(C), np.concatenate(Y).astype(np.int64), np.concatenate(S).astype(np.int64)


CACHE_DIR = os.environ.get("FUSESPEC_CACHE", "/content/fusing_spectrum/cache")


def _cached(path, build, keys):
    """Load a .npz cache file; (re)build it with build() if missing or corrupt. Atomic write."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            d = np.load(path)
            return tuple(d[k] for k in keys)
        except Exception as e:                       # truncated file after a disconnect, etc.
            print(f"cache file {path} unreadable ({type(e).__name__}); rebuilding from raw data")
            os.remove(path)
    vals = build()
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **dict(zip(keys, vals)))
    os.replace(tmp, path)                            # atomic: never leaves a half-written cache file
    return vals


def load_cached(eeg_root, nirs_root, subj, task, cache=None):
    """Band-power EEG + HbO/HbR fNIRS (unchanged from the first version of this notebook)."""
    p = os.path.join(cache or CACHE_DIR, f"{task}_s{subj:02d}.npz")
    return _cached(p, lambda: load_subject(eeg_root, nirs_root, subj, task), ("eeg", "fnirs", "y"))


def load_raw_cached(eeg_root, subj, task, cache=None):
    """Raw epochs + band covariances + session ids (needed for CSP/FBCSP/EEGNet and the session-wise protocol)."""
    p = os.path.join(cache or CACHE_DIR, f"raw_{task}_s{subj:02d}.npz")
    return _cached(p, lambda: load_raw_subject(eeg_root, subj, task), ("raw", "cov", "y", "sess"))


def load_all(eeg_root, nirs_root, task, subjects=range(1, 30), verbose=False):
    """{subject: dict(eeg, fnirs, y, sess, raw, cov)}. The two caches are cross-checked (same trial labels)."""
    out = {}
    for s in subjects:
        eeg, fnirs, y = load_cached(eeg_root, nirs_root, s, task)
        raw, cov, y2, sess = load_raw_cached(eeg_root, s, task)
        assert len(y) == len(y2) and (y == y2).all(), f"subject {s}: band-power cache and raw cache disagree on trial labels"
        out[s] = dict(eeg=eeg, fnirs=fnirs, y=y, sess=sess, raw=raw, cov=cov)
        if verbose:
            print(f"  {task} subject {s:02d}: {len(y)} trials, eeg {eeg.shape[1:]}, fnirs {fnirs.shape[1:]}, raw {raw.shape[1:]}", flush=True)
    return out


def _euclid_align(C, reg=COV_REG):
    """Label-free Euclidean alignment of one subject's covariances (N, nb, ch, ch): whiten by the subject's mean covariance."""
    out = np.empty_like(C, dtype=np.float64)
    for b in range(C.shape[1]):
        Rm = C[:, b].astype(np.float64).mean(0) + reg * np.eye(C.shape[-1])
        w, V = np.linalg.eigh(Rm)
        Wm = (V / np.sqrt(w)) @ V.T                                   # Rm^(-1/2)
        out[:, b] = Wm @ C[:, b].astype(np.float64) @ Wm
    return out.astype(np.float32)


def subject_norm(data):
    """Standardise each subject with its OWN statistics (no labels used). EEG: per feature; fNIRS: per channel;
    raw EEG: per channel; covariances: Euclidean alignment. This is the 'calibration-free' cross-subject setup."""
    out = {}
    for s, d in data.items():
        E, F, R = d["eeg"], d["fnirs"], d["raw"].astype(np.float32)
        E = (E - E.mean((0, 1), keepdims=True)) / (E.std((0, 1), keepdims=True) + 1e-6)
        F = (F - F.mean((0, 2), keepdims=True)) / (F.std((0, 2), keepdims=True) + 1e-6)
        R = (R - R.mean((0, 2), keepdims=True)) / (R.std((0, 2), keepdims=True) + 1e-6)
        out[s] = dict(eeg=E.astype(np.float32), fnirs=F.astype(np.float32), y=d["y"], sess=d["sess"],
                      raw=R.astype(np.float16), cov=_euclid_align(d["cov"]))
    return out
