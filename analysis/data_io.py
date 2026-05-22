"""Data loading + ADF stationarity check."""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


SAMPLE_PATH = "data/sample_us_idx.csv"


def load_sample() -> pd.DataFrame:
    """Load the bundled US→IHSG sample dataset."""
    df = pd.read_csv(SAMPLE_PATH, parse_dates=["date"])
    return df


def parse_uploaded(buffer: io.BytesIO) -> pd.DataFrame:
    """Parse user-uploaded CSV. Returns DataFrame with date columns preserved as strings."""
    df = pd.read_csv(buffer)
    return df


def autodetect_date_col(df: pd.DataFrame) -> Optional[str]:
    """Guess the date column. Prefer columns named date/time/timestamp/ds, else try parsing."""
    name_hints = ["date", "datetime", "time", "timestamp", "ds", "observation_date"]
    for c in df.columns:
        if c.lower() in name_hints:
            return c
    # Try parsing each column
    for c in df.columns:
        sample = df[c].dropna().head(20).astype(str)
        if len(sample) == 0:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().sum() / max(len(sample), 1) > 0.8:
            return c
    return None


def numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    nums = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nums.append(c)
        else:
            # Try coerce
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() / max(len(df), 1) > 0.8:
                nums.append(c)
    return nums


def prepare_panel(
    df: pd.DataFrame, date_col: str, value_cols: list[str]
) -> pd.DataFrame:
    """Return a clean datetime-indexed numeric panel."""
    out = df[[date_col] + list(value_cols)].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    out = out.set_index(date_col).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "date"
    return out


def adf_test(series: pd.Series) -> dict:
    """Run augmented Dickey-Fuller. Returns {stat, pvalue, stationary}."""
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 20:
        return {"stat": float("nan"), "pvalue": float("nan"), "stationary": False, "n": len(s)}
    try:
        stat, pvalue, *_ = adfuller(s, autolag="AIC")
        return {"stat": float(stat), "pvalue": float(pvalue), "stationary": pvalue < 0.05, "n": len(s)}
    except Exception as e:  # noqa: BLE001
        return {"stat": float("nan"), "pvalue": float("nan"), "stationary": False, "error": str(e)}


def adf_panel(panel: pd.DataFrame) -> dict[str, dict]:
    """Run ADF on each column. Returns {col: result}."""
    return {col: adf_test(panel[col]) for col in panel.columns}


def difference(panel: pd.DataFrame) -> pd.DataFrame:
    """First-order difference. Drops the first row."""
    return panel.diff().dropna()


def needs_differencing(adf_results: dict[str, dict], threshold: float = 0.05) -> bool:
    """True if any column has ADF p-value ≥ threshold."""
    return any(r.get("pvalue", 1.0) >= threshold for r in adf_results.values())
