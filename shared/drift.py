"""Distribution-drift helpers — Population Stability Index (PSI) + projections.

PSI is the standard interpretable scalar for "did the live distribution drift
from a baseline distribution?". Operating thresholds (Wieland & Wallace,
*Tracking Ecology with Statistical Process Control*, 2008; widely reused in
ML monitoring):

    PSI < 0.10   no significant change
    0.10–0.25    moderate change, investigate
    PSI > 0.25   significant change, alarm

We compute PSI on a low-dimensional projection of embeddings (top-N PCA via
`numpy.linalg.svd`) so the histogram bins stay sample-efficient. Pure
numpy, no sklearn dep.

Used by Course 3 ch4 (monitoring) and ch5 (end-to-end pipeline).
"""

from __future__ import annotations

import numpy as np


def fit_projection(baseline_emb: np.ndarray, n_components: int) -> np.ndarray:
    """Top-N right-singular-vectors of the centered baseline embedding matrix.

    Returns a `[d, n_components]` projection matrix `W` such that
    `emb @ W` is the low-dim representation. Pure SVD; deterministic.
    """
    if baseline_emb.ndim != 2:
        raise ValueError(f"baseline_emb must be 2D [N, D]; got shape {baseline_emb.shape}")
    n_components = min(int(n_components), baseline_emb.shape[1], baseline_emb.shape[0])
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}")
    centered = baseline_emb - baseline_emb.mean(axis=0, keepdims=True)
    # SVD: centered = U @ diag(s) @ Vh; the top-k right singular vectors
    # (rows of Vh) are the principal components.
    _, _, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    return vh[:n_components].T.astype(np.float32)  # [d, k]


def histogram_along_projection(emb: np.ndarray, projection: np.ndarray,
                               n_bins: int, edges: np.ndarray | None = None
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Project `emb @ projection`, take the FIRST projected coordinate, and
    histogram it into `n_bins` bins.

    Returns `(hist, edges)` where `hist` is the per-bin probability mass
    (sums to 1) and `edges` is the `n_bins+1` bin boundaries. Pass the
    baseline's `edges` back in to compute the live histogram on the same
    binning so PSI is comparable.
    """
    projected = (emb @ projection)[:, 0]
    if edges is None:
        edges = np.histogram_bin_edges(projected, bins=n_bins)
    counts, _ = np.histogram(projected, bins=edges)
    total = counts.sum()
    hist = counts.astype(np.float64) / max(total, 1)
    return hist, edges


def psi(baseline_hist: np.ndarray, live_hist: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index between two probability histograms.

    Both inputs must sum to ~1 and have the same length (same binning).
    `eps` floors zeros so log/division stay defined.
    """
    if baseline_hist.shape != live_hist.shape:
        raise ValueError(
            f"shape mismatch: baseline {baseline_hist.shape} vs live {live_hist.shape}"
        )
    b = np.clip(baseline_hist.astype(np.float64), eps, None)
    l = np.clip(live_hist.astype(np.float64), eps, None)
    return float(((l - b) * np.log(l / b)).sum())
