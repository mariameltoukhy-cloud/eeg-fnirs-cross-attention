"""EEG-queries-fNIRS cross-attention decoder + single-modality / late-fusion baselines.

Shapes
------
eeg   : (B, T_e, F_e)  log band-power sequence   (F_e = n_channels * n_bands)
fnirs : (B, C_f, T_f)  HbO/HbR time series       (C_f = 2 * n_channels)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("eeg", "fnirs", "concat", "hybrid", "hybrid_rg")   # architectures; hybrid_md / hybrid_pre reuse "hybrid" (see train.py)


def sinusoidal(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class EEGEncoder(nn.Module):
    """Spectral-power sequence -> (B, T_e, d)."""
    def __init__(self, in_dim, d=64, heads=4, drop=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(drop))
        self.tf = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)

    def forward(self, x):
        h = self.proj(x)
        h = h + sinusoidal(h.size(1), h.size(2), h.device)
        return self.tf(h)


class FNIRSEncoder(nn.Module):
    """HbO/HbR series -> (B, ~T_f/10, d). Average-pool first (haemodynamics are slow), then a small strided conv.
    Channel dropout fights over-fitting at ~50 training trials."""
    def __init__(self, in_ch, d=64, heads=4, drop=0.3, ch_drop=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout1d(ch_drop), nn.AvgPool1d(5),
            nn.Conv1d(in_ch, d, 3, padding=1), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d, d, 3, stride=2, padding=1), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(drop))
        self.tf = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        h = h + sinusoidal(h.size(1), h.size(2), h.device)
        return self.tf(h)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from EEG tokens, keys/values from fNIRS tokens."""
    def __init__(self, d=64, heads=4, drop=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(drop)

    def forward(self, eeg_tok, fnirs_tok):
        B, Tq, d = eeg_tok.shape
        Tk = fnirs_tok.size(1)
        split = lambda t, T: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = split(self.q(eeg_tok), Tq), split(self.k(fnirs_tok), Tk), split(self.v(fnirs_tok), Tk)
        attn = self.drop(F.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), dim=-1))  # (B,H,Tq,Tk)
        out = (attn @ v).transpose(1, 2).reshape(B, Tq, d)
        return self.o(out), attn


class CrossFusionBlock(nn.Module):
    """EEG tokens attend to fNIRS tokens. A gate scales the fNIRS contribution.
    gate_mode="scalar": one learned scalar (tanh-gated); gate_init=0 -> the block starts as EEG-only.
    gate_mode="input" : the gate also depends on the trial's fNIRS tokens (zero-initialised linear layer on the mean fNIRS
                        token), so the model *can* learn to open the fNIRS path only when the fNIRS looks informative."""
    def __init__(self, d=64, heads=4, drop=0.1, gate_init=0.0, gate_mode="scalar"):
        super().__init__()
        self.attn = CrossAttention(d, heads, drop)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.gate_net = None
        if gate_mode == "input":
            self.gate_net = nn.Linear(d, 1)
            nn.init.zeros_(self.gate_net.weight); nn.init.zeros_(self.gate_net.bias)
        self.force_closed = False        # diagnostics: True -> fNIRS path switched off
        self.last_gate = None            # diagnostics: mean tanh(gate) of the last forward pass

    def forward(self, z_eeg, z_fnirs):
        a, w = self.attn(self.n1(z_eeg), z_fnirs)
        g = self.gate
        if self.gate_net is not None:
            g = g + self.gate_net(z_fnirs.mean(1))                 # (B, 1)
        gate = torch.tanh(g).reshape(-1, 1, 1)                     # (1,1,1) for the scalar gate, (B,1,1) for the input gate
        self.last_gate = gate.detach().mean()
        if self.force_closed:
            gate = gate * 0.0
        h = z_eeg + gate * a
        h = h + self.ffn(self.n2(h))
        return h, w


class Decoder(nn.Module):
    def __init__(self, kind, eeg_dim, fnirs_ch, n_classes=2, d=64, heads=4, drop=0.3, gate_init=0.0):
        super().__init__()
        assert kind in KINDS, f"unknown model kind {kind!r}"
        self.kind = kind
        self.eeg_enc = EEGEncoder(eeg_dim, d, heads, drop) if kind != "fnirs" else None
        self.fnirs_enc = FNIRSEncoder(fnirs_ch, d, heads, drop) if kind != "eeg" else None
        self.fusion = (CrossFusionBlock(d, heads, gate_init=gate_init, gate_mode="input" if kind == "hybrid_rg" else "scalar")
                       if kind in ("hybrid", "hybrid_rg") else None)
        w = d * (2 if kind == "concat" else 1)
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Dropout(drop), nn.Linear(w, n_classes))

    def forward(self, eeg, fnirs, return_attn=False):
        w = None
        if self.kind == "eeg":
            pooled = self.eeg_enc(eeg).mean(1)
        elif self.kind == "fnirs":
            pooled = self.fnirs_enc(fnirs).mean(1)
        else:
            ze, zf = self.eeg_enc(eeg), self.fnirs_enc(fnirs)
            if self.kind == "concat":                       # late-fusion control (no interaction)
                pooled = torch.cat([ze.mean(1), zf.mean(1)], dim=-1)
            else:                                           # hybrid / hybrid_rg: EEG queries fNIRS
                h, w = self.fusion(ze, zf)
                pooled = h.mean(1)
        logits = self.head(pooled)
        return (logits, w) if return_attn else logits


def build_model(kind, eeg_dim, fnirs_ch, **kw):
    return Decoder(kind, eeg_dim, fnirs_ch, **kw)


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018) on raw EEG (B, n_ch, n_t) sampled at 100 Hz. Standard layout; the max-norm weight
    constraints of the original are left out (AdamW weight decay is used instead)."""
    def __init__(self, n_ch=30, n_t=1000, n_classes=2, F1=8, D=2, F2=16, k1=50, k3=16, drop=0.5):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding="same", bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k3), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_feat = self.b2(self.b1(torch.zeros(2, 1, n_ch, n_t))).numel() // 2
        self.fc = nn.Linear(n_feat, n_classes)

    def forward(self, x):
        return self.fc(self.b2(self.b1(x.unsqueeze(1))).flatten(1))
