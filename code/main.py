"""
Structural Geometry of Financial Time Series (Reference Implementation)

This script implements the core pipeline described in the accompanying working paper:
1) download price data
2) compute returns and time-delay embeddings
3) compute rolling covariance matrices
4) measure (a) dominant subspace rotation and (b) entropy-based effective dimensionality
5) visualize smoothed dynamics and extreme events

Notes:
- No prediction; diagnostics only.
- Focus is clarity and reproducibility, not production optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from numpy.linalg import eigh, svd


# =========================
# CONFIG
# =========================
CONFIG = {
    # Examples:
    # "^GSPC" = S&P 500, "EURUSD=X" = EUR/USD
    "ticker": "^GSPC",
    "start": "2000-01-01",
    "end": "2026-01-01",

    # Embedding + rolling window
    "L": 60,          # embedding dimension
    "H": 20,          # rolling window length (embedded vectors)
    "k": 3,           # dominant PCA subspace dimension
    "center": True,   # center each rolling window
    "smooth_w": 20,   # smoothing window for visualization

    # Extremes (quantiles)
    "high_rotation_q": 0.975,  # top 2.5% rotation events
    "low_effdim_q": 0.025,     # bottom 2.5% eff. dimensionality events
}


# =========================
# CORE METHODS
# =========================
def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close (preferred) or close prices from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    price = data[price_col].dropna().rename("price")
    return price


def compute_returns(price: pd.Series) -> pd.Series:
    """Simple returns from a price series."""
    return price.pct_change().dropna().rename("ret")


def embed_returns(returns: np.ndarray, L: int) -> np.ndarray:
    """Time-delay embedding: x_t = (r_{t-L}, ..., r_{t-1}) in R^L."""
    X = [returns[i - L : i] for i in range(L, len(returns) + 1)]
    return np.asarray(X)


def pca_basis_from_cov(cov: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors (principal components) of a symmetric covariance matrix."""
    vals, vecs = eigh(cov)                  # ascending eigenvalues
    idx = np.argsort(vals)[::-1]            # descending
    return vecs[:, idx[:k]]


def subspace_rotation(Q_prev: np.ndarray, Q_curr: np.ndarray) -> float:
    """
    Subspace rotation based on principal angles.

    Let s_i be singular values of Q_prev^T Q_curr. Then:
        R = 1 - mean(s_i),  where R in [0, 1] (approximately).
    """
    s = svd(Q_prev.T @ Q_curr, compute_uv=False)
    return float(1.0 - np.mean(s))


def effective_dimensionality(eigs: np.ndarray) -> float:
    """
    Entropy-based effective dimensionality:
        D_eff = exp( -sum p_i log p_i ),  p_i = eig_i / sum eig_i
    """
    eigs = np.clip(eigs, 1e-12, None)
    p = eigs / np.sum(eigs)
    return float(np.exp(-np.sum(p * np.log(p))))


def rescale_to_price_range(x: pd.Series, price: pd.Series, pad: float = 0.05) -> pd.Series:
    """Rescale x to the price range for shape-only overlay."""
    x = x.replace([np.inf, -np.inf], np.nan).astype(float)
    x_valid = x.dropna()
    if x_valid.empty:
        return pd.Series(np.nan, index=x.index)

    pmin, pmax = float(price.min()), float(price.max())
    span = pmax - pmin
    xmin, xmax = float(x_valid.min()), float(x_valid.max())

    if np.isclose(xmax - xmin, 0.0):
        return pd.Series(pmin + 0.5 * span, index=x.index)

    return pmin + pad * span + (x - xmin) / (xmax - xmin) * (1 - 2 * pad) * span


# =========================
# PIPELINE
# =========================
def compute_structural_metrics(returns: pd.Series, L: int, H: int, k: int, center: bool) -> pd.DataFrame:
    """
    Compute rolling structural metrics on embedded returns:
    - subspace rotation of dominant PCA subspace (dimension k)
    - entropy-based effective dimensionality (using all eigenvalues)
    """
    r = returns.to_numpy()
    X = embed_returns(r, L)                 # shape: (n_samples, L)

    rotation, eff_dim = [], []
    Q_prev = None

    for t in range(H, len(X)):
        window = X[t - H : t].copy()
        if center:
            window -= window.mean(axis=0)

        cov = (window.T @ window) / H
        vals, _ = eigh(cov)                 # ascending eigs
        eff_dim.append(effective_dimensionality(vals))

        Q = pca_basis_from_cov(cov, k)
        rotation.append(np.nan if Q_prev is None else subspace_rotation(Q_prev, Q))
        Q_prev = Q

    # Align dates: X index corresponds to returns index starting at L-1 onward.
    dates_for_X = returns.index[L-1:]       # aligns with X rows
    idx = dates_for_X[H:]                  # aligns with t = H..len(X)-1

    return pd.DataFrame({"rotation": rotation, "eff_dim": eff_dim}, index=idx)


def plot_overlays(price: pd.Series, results: pd.DataFrame, smooth_w: int) -> None:
    """Plot price with rescaled (shape-only) overlays of smoothed metrics."""
    results = results.copy()
    results["avg_rotation"] = results["rotation"].rolling(smooth_w).mean()
    results["avg_eff_dim"] = results["eff_dim"].rolling(smooth_w).mean()

    aligned_price = price.loc[results.index]

    rot_scaled = rescale_to_price_range(results["avg_rotation"], aligned_price)
    dim_scaled = rescale_to_price_range(results["avg_eff_dim"], aligned_price)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(aligned_price.index, aligned_price, linewidth=1.6, label="Price")
    ax.plot(results.index, rot_scaled, linewidth=1.3, alpha=0.85, label="Avg. rotation (shape)")
    ax.plot(results.index, dim_scaled, linewidth=1.3, alpha=0.85, label="Avg. eff. dimensionality (shape)")
    ax.set_title("Price with Structural Dynamics (Shape-only overlays)")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_extremes(price: pd.Series, results: pd.DataFrame, smooth_w: int, high_q: float, low_q: float) -> None:
    """Plot price with extreme-event markers from smoothed structural metrics."""
    results = results.copy()
    results["avg_rotation"] = results["rotation"].rolling(smooth_w).mean()
    results["avg_eff_dim"] = results["eff_dim"].rolling(smooth_w).mean()

    valid = results.dropna(subset=["avg_rotation", "avg_eff_dim"])
    aligned_price = price.loc[valid.index]

    rot_thr = valid["avg_rotation"].quantile(high_q)
    dim_thr = valid["avg_eff_dim"].quantile(low_q)

    hi_rot = valid[valid["avg_rotation"] >= rot_thr]
    lo_dim = valid[valid["avg_eff_dim"] <= dim_thr]

    plt.figure(figsize=(15, 6))
    plt.plot(aligned_price.index, aligned_price, linewidth=1.6, label="Price")

    plt.scatter(hi_rot.index, price.loc[hi_rot.index], s=55, alpha=0.85,
                label=f"Top {(1-high_q)*100:.1f}% rotation", zorder=6)
    plt.scatter(lo_dim.index, price.loc[lo_dim.index], s=55, alpha=0.85, marker="^",
                label=f"Bottom {low_q*100:.1f}% eff. dim.", zorder=6)

    plt.title("Structural Extremes on Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    cfg = CONFIG
    price = download_prices(cfg["ticker"], cfg["start"], cfg["end"])
    rets = compute_returns(price)

    results = compute_structural_metrics(
        returns=rets,
        L=cfg["L"],
        H=cfg["H"],
        k=cfg["k"],
        center=cfg["center"],
    )

    plot_overlays(price, results, smooth_w=cfg["smooth_w"])
    plot_extremes(price, results, smooth_w=cfg["smooth_w"],
                  high_q=cfg["high_rotation_q"], low_q=cfg["low_effdim_q"])


if __name__ == "__main__":
    main()


