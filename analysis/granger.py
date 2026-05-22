"""Conditional Granger causality + rolling window — self-contained."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import statsmodels.api as sm


FRAC_SIG_THRESHOLD = 0.30
FRAC_BORDERLINE_THRESH = 0.15


def normalize(df_in: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each column."""
    std = df_in.std()
    std = std.replace(0, 1.0)
    return (df_in - df_in.mean()) / std


def conditional_granger_test(
    df_segment: pd.DataFrame,
    target: str,
    source: str,
    all_vars: list[str],
    p: int,
) -> tuple[float, float, float]:
    """Test if source Granger-causes target, controlling for all other vars.

    Returns (F_stat, p_value, lag1_coef).
    """
    df_z = normalize(df_segment)
    other_vars = [v for v in all_vars if v != target and v != source]

    reg_r = pd.DataFrame(index=df_z.index)
    reg_r["y_t"] = df_z[target].values
    for k in range(1, p + 1):
        reg_r[f"{target}_t-{k}"] = df_z[target].shift(k).values
    for ov in other_vars:
        for k in range(1, p + 1):
            reg_r[f"{ov}_t-{k}"] = df_z[ov].shift(k).values
    reg_r = reg_r.dropna()

    reg_u = reg_r.copy()
    for k in range(1, p + 1):
        reg_u[f"{source}_t-{k}"] = df_z[source].shift(k).reindex(reg_u.index).values
    reg_u = reg_u.dropna()
    reg_r = reg_r.loc[reg_u.index]

    if len(reg_u) <= len(reg_u.columns):
        return np.nan, np.nan, np.nan
    try:
        res_r = sm.OLS(reg_r["y_t"], sm.add_constant(reg_r.drop(columns=["y_t"]))).fit()
        res_u = sm.OLS(reg_u["y_t"], sm.add_constant(reg_u.drop(columns=["y_t"]))).fit()
    except Exception:
        return np.nan, np.nan, np.nan

    rss_r, rss_u = res_r.ssr, res_u.ssr
    n = int(res_u.nobs)
    k_params = len(res_u.params)
    df_num, df_denom = p, n - k_params
    if df_denom <= 0 or rss_u <= 0:
        return np.nan, np.nan, np.nan
    F = ((rss_r - rss_u) / df_num) / (rss_u / df_denom)
    pval = 1 - scipy_stats.f.cdf(F, dfn=df_num, dfd=df_denom)
    coef = res_u.params.get(f"{source}_t-1", 0)
    return float(F), float(pval), float(coef)


def make_pair_keys(all_vars: list[str]) -> list[tuple[str, str]]:
    """Generate all directional pairs (source, target) for n vars: n*(n-1)."""
    pairs = []
    for src in all_vars:
        for tgt in all_vars:
            if src != tgt:
                pairs.append((src, tgt))
    return pairs


def build_windows(
    df_diff: pd.DataFrame,
    window: int,
    shift: int,
    p_lag: int,
    progress_callback=None,
) -> tuple[np.ndarray, list[dict], list[tuple[str, str]]]:
    """Roll a window of size `window` across df_diff, shift by `shift` days.

    For each window, run conditional Granger on every directional pair.
    Returns (X, meta, pair_keys) where:
      X: ndarray (n_windows, n_pairs) of p-values
      meta: list of dicts with start/end/mid dates and coefficients
      pair_keys: list of (source, target) tuples for each column of X
    """
    all_vars = list(df_diff.columns)
    pair_keys = make_pair_keys(all_vars)
    starts = list(range(0, len(df_diff) - window + 1, shift))
    n_windows = len(starts)

    rows, meta = [], []
    for wi, st in enumerate(starts):
        seg = df_diff.iloc[st : st + window]
        pvals, fstats, coefs = [], [], []
        for s, t in pair_keys:
            F, pv, c = conditional_granger_test(seg, t, s, all_vars, p_lag)
            pvals.append(pv)
            fstats.append(F)
            coefs.append(c)
        rows.append(pvals)
        meta.append(
            {
                "win_idx": wi,
                "start_date": seg.index[0],
                "end_date": seg.index[-1],
                "mid_date": seg.index[len(seg) // 2],
                "fstats": fstats,
                "coefs": coefs,
            }
        )
        if progress_callback is not None:
            progress_callback(wi + 1, n_windows)

    X = np.array(rows, dtype=float)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = np.where(nan_mask, 0.5, X)
    return X, meta, pair_keys


def cluster_fraction_sig(labels: np.ndarray, c: int, X: np.ndarray) -> np.ndarray:
    """Return fraction of windows in cluster c where p < 0.05 for each pair."""
    idx = np.where(labels == c)[0]
    if len(idx) == 0:
        return np.zeros(X.shape[1])
    mask = X[idx] < 0.05
    return mask.mean(axis=0)


def build_cluster_edge_data(
    labels: np.ndarray,
    c: int,
    X: np.ndarray,
    meta: list[dict],
    pair_keys: list[tuple[str, str]],
) -> list[dict]:
    """Build per-edge data for network plot of cluster c."""
    idx = np.where(labels == c)[0]
    frac_sig = cluster_fraction_sig(labels, c, X)
    out = []
    for pi, (s, t) in enumerate(pair_keys):
        cs = [meta[wi]["coefs"][pi] for wi in idx if not np.isnan(meta[wi]["coefs"][pi])]
        out.append(
            {
                "Source": s,
                "Target": t,
                "p_value": float(frac_sig[pi]),
                "frac_sig": float(frac_sig[pi]),
                "coef": float(np.mean(cs)) if cs else 0.0,
                "Significant": frac_sig[pi] >= FRAC_SIG_THRESHOLD,
                "Borderline": frac_sig[pi] >= FRAC_BORDERLINE_THRESH,
            }
        )
    return out
