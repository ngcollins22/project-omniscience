"""
NaN-safe Normalized Cross-Correlation (NCC) for TRN patch scoring.

NCC is scale-invariant: NCC(A, k·A) = 1.0 for any k ≠ 0.
This means the ×40 vertical exaggeration in the STL does not affect scoring.
"""

from __future__ import annotations
import numpy as np


def ncc(measured: np.ndarray, predicted: np.ndarray,
        min_valid_frac: float = 0.4) -> float:
    """
    Scalar NCC between two same-shape patches, masking NaN pixels.

    Returns a value in [-1, 1], or NaN if fewer than *min_valid_frac* of
    pixels are jointly finite.
    """
    valid = np.isfinite(measured) & np.isfinite(predicted)
    if valid.mean() < min_valid_frac:
        return float("nan")
    a = measured[valid].astype(np.float64)
    b = predicted[valid].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ncc_batch(measured: np.ndarray,
              predicted_batch: np.ndarray,
              min_valid_frac: float = 0.4) -> np.ndarray:
    """
    Vectorised NCC between one measured patch and N predicted patches.

    Parameters
    ----------
    measured        : (ny, nx) float32/float64.
    predicted_batch : (N, ny, nx) float32/float64.
    min_valid_frac  : minimum fraction of jointly valid pixels required.

    Returns
    -------
    scores : (N,) float32 in [-1, 1], NaN where insufficient valid pixels.
    """
    measured = np.asarray(measured, dtype=np.float64)
    predicted_batch = np.asarray(predicted_batch, dtype=np.float64)

    valid_m = np.isfinite(measured)                          # (ny, nx)
    valid_p = np.isfinite(predicted_batch)                   # (N, ny, nx)
    valid   = valid_m[np.newaxis, :, :] & valid_p            # (N, ny, nx)

    n_valid     = valid.sum(axis=(1, 2), keepdims=True).astype(np.float64)   # (N,1,1)
    total_cells = measured.size
    valid_frac  = (n_valid / total_cells).squeeze((1, 2))    # (N,)

    # Zero-fill invalid pixels before computing means
    m_fill = np.where(valid, measured[np.newaxis], 0.0)      # (N, ny, nx)
    p_fill = np.where(valid, predicted_batch,     0.0)       # (N, ny, nx)

    m_mean = m_fill.sum(axis=(1, 2), keepdims=True) / np.maximum(n_valid, 1)
    p_mean = p_fill.sum(axis=(1, 2), keepdims=True) / np.maximum(n_valid, 1)

    a = np.where(valid, m_fill - m_mean, 0.0)               # (N, ny, nx)
    b = np.where(valid, p_fill - p_mean, 0.0)               # (N, ny, nx)

    dot    = (a * b).sum(axis=(1, 2))                        # (N,)
    norm_a = np.sqrt((a * a).sum(axis=(1, 2)))               # (N,)
    norm_b = np.sqrt((b * b).sum(axis=(1, 2)))               # (N,)
    denom  = norm_a * norm_b

    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(denom > 1e-9, dot / denom, 0.0)

    # Mark insufficient coverage as NaN
    scores = np.where(valid_frac >= min_valid_frac, scores, np.nan)
    return scores.astype(np.float32)
