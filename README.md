# Granger Regime Explorer

A no-code web app to discover **recurring transmission regimes** in multivariate
time series using rolling-window **conditional Granger causality** + **clustering**.

Built around the methodology in Wesley Teo's research paper on US → IHSG financial
market transmission (2021–2026).

> ⚠️ **Granger causality is NOT true causation.** It measures predictive precedence
> only — whether past values of X help forecast Y beyond Y's own history. Use
> results as exploratory, not causal claims.

## What it does

1. Upload a daily multivariate time series (or use the bundled sample of US/IHSG data)
2. Pick variables and parameters (window size, lag, K clusters)
3. Auto-runs ADF stationarity check + first-differences if needed
4. Runs **conditional Granger** across rolling windows → vector of p-values per window
5. **K-means + Hierarchical Ward** cluster the windows into regimes
6. Visualises: cluster timeline, leader networks (fraction-significant), PCA + t-SNE
7. Compares K-means vs Ward (ARI + confusion matrix)
8. Download all figures as ZIP

## Methodology choices (preserved from the research paper)

- **Conditional Granger** (controls for all other system variables) — not bivariate
- **Fraction-significant** edge weighting (% windows where p<0.05) — never average p-values
- **Threshold ★** at ≥30% (6× null), **◐** at 15-30% (3× null)
- **First-difference** if ADF p ≥ 0.05 — silent transformations are bad science, the app shows when this happens
- **PCA variance explained** displayed prominently — low variance = non-linear, justifies t-SNE
- **t-SNE axes labelled "no inherent meaning"** — common interpretation pitfall

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at <http://localhost:8501>.

## Public deployment

Deployed via [Streamlit Community Cloud](https://streamlit.io/cloud) on push to `main`.

## Repo layout

```
granger-explorer/
├── app.py                    # Streamlit entry
├── analysis/
│   ├── data_io.py            # CSV parsing + ADF + differencing
│   ├── rolling_window.py     # rolling Granger orchestration (Phase 1 stub)
│   ├── clustering.py         # K-means + Hierarchical Ward
│   └── plotting.py           # matplotlib figures (re-uses paper styling)
├── data/
│   └── sample_us_idx.csv     # 1,152 rows, 6 cols (BAA10Y, DGS10, DTWEXBGS, IHSG, SP500, VIXCLS)
├── requirements.txt
└── README.md
```

## Sample data

Public market data only (FRED + Bursa Efek Indonesia):

- **BAA10Y** — Moody's Baa corporate bond yield minus 10-year Treasury (credit spread)
- **DGS10** — 10-year US Treasury yield
- **DTWEXBGS** — USD broad index
- **IHSG** — Jakarta Composite Index
- **SP500** — S&P 500
- **VIXCLS** — CBOE Volatility Index

Date range: 2021-02-01 → 2026-01-23 (1,152 trading days).

## License

MIT. Methodology adapted from the research paper at <https://github.com/wesleyteo/granger-regime-explorer> — see paper for full theoretical justification.
