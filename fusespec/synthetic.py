"""Synthetic paired data in the same feature space as data.py - for smoke tests without the dataset.
Class info sits in different places per modality (EEG: band power in a few features AND amplitude of a few raw channels;
fNIRS: a delayed HbO bump / HbR dip on other channels), so complementary fusion can win."""
import numpy as np
from .data import band_covs_from_epochs, RAW_FS


def make_subject(n=60, n_ch=30, n_bands=4, T_e=19, n_fn_ch=36, T_f=150, snr_e=0.35, snr_f=0.35, T_raw=1000, seed=0):
    r = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    r.shuffle(y)
    sgn = (2 * y - 1)[:, None, None]
    eeg = r.normal(size=(n, T_e, n_ch * n_bands)).astype(np.float32)
    eeg[:, :, :8] += snr_e * sgn * np.linspace(0.3, 1, T_e)[None, :, None]
    t = np.arange(T_f) / 10.0
    hrf = (t ** 5 * np.exp(-t) / 120.0)[None, None, :]
    fn = r.normal(size=(n, 2 * n_fn_ch, T_f)).astype(np.float32) * 0.8
    fn[:, :6] += snr_f * 3 * sgn * hrf
    fn[:, n_fn_ch:n_fn_ch + 6] -= snr_f * 1.5 * sgn * hrf
    raw = r.normal(size=(n, n_ch, T_raw))
    raw[:, :3] *= 1 + snr_e * 1.2 * sgn[:, :, :1]                      # class-dependent amplitude on 3 channels (CSP-friendly)
    cov = band_covs_from_epochs(raw, RAW_FS)
    sess = np.repeat(np.arange(3), int(np.ceil(n / 3)))[:n]
    return dict(eeg=eeg, fnirs=fn.astype(np.float32), y=y.astype(np.int64), sess=sess.astype(np.int64),
                raw=raw.astype(np.float16), cov=cov)


def make_dataset(n_subjects=5, **kw):
    return {s: make_subject(seed=s, **kw) for s in range(1, n_subjects + 1)}
