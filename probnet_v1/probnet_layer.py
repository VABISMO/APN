"""
ProbNet Layer - Deterministic replacement for nn.Linear in Transformers
=======================================================================
Based on Probnet 0.1.0.0 by Vicent Nos Ripolles / Cobalt Technologies Panamá

Instead of learned weight matrices (linear regression), ProbNetLayer uses
ratio-based prediction (logarithmic differences between consecutive values),
which is 100% deterministic and requires no gradient-based training.

The core math:
  ratios(x) = x[:-1] / x[1:]           # ratio between consecutive elements
  predict(x) = ratio[nearest_idx] * x[-1]  # apply ratio from nearest match
  error_term = prerr(x)                 # second-order correction
  output = predict(x) + error_term

This mirrors the original Haskell 'probnet' function exactly, extended to tensors.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Core ProbNet math (translated from Haskell to PyTorch)
# ---------------------------------------------------------------------------

def percents(dat: torch.Tensor) -> torch.Tensor:
    """Ratios between consecutive elements (logarithmic differences).
    dat shape: (..., L) — operates on the last dimension
    returns:   (..., L-1)
    """
    num = dat[..., :-1]
    den = dat[..., 1:].clone()
    den[den == 0] = 1e-8
    return num / den


def nearnum_index(val: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Index of element in candidates nearest to val (element-wise).
    val:        (...,)
    candidates: (..., L)
    returns:    (...,)  LongTensor
    """
    diff = (candidates - val.unsqueeze(-1)).abs()
    return diff.argmin(dim=-1)


def predict1(dat: torch.Tensor) -> torch.Tensor:
    """Predict next value for each row using the ProbNet ratio rule.
    dat shape: (..., seq_len)   — at least length 2
    returns:   (...,)
    """
    if dat.shape[-1] < 2:
        return dat[..., -1]

    ratios    = percents(dat)                # (..., L-1)
    last_val  = dat[..., -1]                 # (...,)
    init_dat  = dat[..., :-1]               # (..., L-1)

    # find ratio index corresponding to nearest init value to last value
    idx       = nearnum_index(last_val, init_dat)              # (...,)
    idx       = idx.clamp(0, ratios.shape[-1] - 1)

    # gather the matching ratio
    chosen    = ratios.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # (...,)
    return chosen * last_val


def prerr(dat: torch.Tensor) -> torch.Tensor:
    """Predict the error correction term.
    dat shape: (..., seq_len)
    returns:   (...,)
    """
    seq_len = dat.shape[-1]
    if seq_len < 3:
        return torch.zeros(dat.shape[:-1], device=dat.device, dtype=dat.dtype)

    # compute prediction error at each position
    preds = []
    for i in range(seq_len):
        if i < 2:
            preds.append(torch.zeros(dat.shape[:-1], device=dat.device, dtype=dat.dtype))
        else:
            preds.append(predict1(dat[..., :i]))

    pred_tensor = torch.stack(preds, dim=-1)   # (..., seq_len)
    err         = dat - pred_tensor             # element-wise error

    # predict next error from the trimmed error series
    err_trimmed = err[..., 2:]
    if err_trimmed.shape[-1] < 2:
        return torch.zeros(dat.shape[:-1], device=dat.device, dtype=dat.dtype)

    last_err = err_trimmed[..., -1]
    if last_err.abs().max() < 1e-9:
        return torch.zeros_like(last_err)

    return predict1(err_trimmed)


def probnet_predict(dat: torch.Tensor) -> torch.Tensor:
    """Single-step ProbNet prediction with error correction.
    dat: (..., seq_len)
    returns: (...,)  — next predicted value
    """
    return predict1(dat) + prerr(dat)


# ---------------------------------------------------------------------------
# ProbNetLayer — drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class ProbNetLayer(nn.Module):
    """
    Deterministic ProbNet layer replacing nn.Linear(in_features, out_features).

    Strategy
    --------
    For each output neuron j (0..out_features-1):
      - Select a sliding window of width `window` from the input,
        starting at position (j * stride) % in_features.
      - Apply probnet_predict() on that window to get one scalar.

    This requires NO learned weight matrices and NO matrix multiplication.
    The only (optional) parameters are per-output biases.

    Parameters
    ----------
    in_features  : int
    out_features : int
    window       : int  — how many input elements to use per prediction (≥2)
    bias         : bool — small learned scalar bias per output
    """

    def __init__(self, in_features: int, out_features: int,
                 window: int = 8, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.window       = max(window, 2)
        self.use_bias     = bias

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

        # Precompute which input indices each output neuron reads
        # Shape: (out_features, window)
        stride  = max(1, in_features // out_features)
        indices = []
        for j in range(out_features):
            start = (j * stride) % in_features
            idx   = [(start + k) % in_features for k in range(self.window)]
            indices.append(idx)
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        returns: (..., out_features)
        """
        batch_shape = x.shape[:-1]
        x_flat      = x.reshape(-1, self.in_features)     # (B, in)
        B           = x_flat.shape[0]

        # Gather windows: (B, out_features, window)
        gathered = x_flat[:, self.indices]                 # (B, out, window)

        # Reshape for batched predict: (B*out, window)
        seq = gathered.reshape(B * self.out_features, self.window)

        # ProbNet prediction — deterministic
        out_vals = probnet_predict(seq)                    # (B*out,)
        out      = out_vals.reshape(B, self.out_features)  # (B, out)

        if self.use_bias:
            out = out + self.bias

        return out.reshape(*batch_shape, self.out_features)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"window={self.window}, bias={self.use_bias}")
