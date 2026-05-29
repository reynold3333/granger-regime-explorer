"""
Granger Regime Explorer — Streamlit app.

Personal research dashboard for rolling-window conditional Granger causality
+ K-means / Hierarchical Ward clustering on multivariate time series.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from analysis.data_io import (
    SAMPLE_PATH,
    adf_panel,
    autodetect_date_col,
    difference,
    load_sample,
    needs_differencing,
    numeric_columns,
    parse_uploaded,
    prepare_panel,
)
from analysis.granger import (
    FRAC_BORDERLINE_THRESH,
    FRAC_SIG_THRESHOLD,
    build_cluster_edge_data,
    build_windows,
    cluster_fraction_sig,
)
from analysis.clustering import (
    compute_ari,
    elbow_scan,
    run_hierarchical,
    run_kmeans,
)
from analysis.plotting import (
    CLUSTER_PALETTE,
    get_pca_variance,
    plot_cluster_heatmap,
    plot_cluster_network,
    plot_elbow,
    plot_pca_scatter,
    plot_timeline,
    plot_tsne_scatter,
)
from analysis.periods import (
    add_period,
    load_periods,
    remove_period,
    save_periods,
)
from analysis.interpret import full_interpretation


st.set_page_config(
    page_title="Granger Regime Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================
def init_state() -> None:
    defaults = {
        "df_raw": None,
        "data_source": None,
        "date_col": None,
        "value_cols": [],
        "auto_diff": True,
        "panel": None,
        "panel_subset": None,        # date-range-filtered
        "adf": {},
        "panel_used": None,          # what feeds analysis
        "differencing_applied": False,
        "params": {"window": 60, "shift": 10, "p_lag": 5, "K": 4},
        "date_range": None,          # (start, end) or None
        "results": None,
        "periods": load_periods(),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


def _autoload_sample_if_missing() -> None:
    if st.session_state.df_raw is not None:
        return
    if not Path(SAMPLE_PATH).exists():
        return
    df = load_sample()
    st.session_state.df_raw = df
    st.session_state.data_source = "sample"
    st.session_state.date_col = "date"
    st.session_state.value_cols = [c for c in df.columns if c != "date"]
    _recompute_panel_and_adf()


def _recompute_panel_and_adf() -> None:
    df = st.session_state.df_raw
    dcol = st.session_state.date_col
    vcols = st.session_state.value_cols
    if df is None or dcol is None or not vcols:
        st.session_state.panel = None
        st.session_state.panel_used = None
        st.session_state.adf = {}
        return
    panel = prepare_panel(df, dcol, vcols)
    st.session_state.panel = panel

    # Apply date range subset if set
    dr = st.session_state.date_range
    if dr is not None and len(panel) > 0:
        start, end = dr
        subset = panel.loc[pd.to_datetime(start) : pd.to_datetime(end)]
    else:
        subset = panel
    st.session_state.panel_subset = subset

    if len(subset) == 0:
        st.session_state.adf = {}
        st.session_state.panel_used = None
        return

    st.session_state.adf = adf_panel(subset)
    if st.session_state.auto_diff and needs_differencing(st.session_state.adf):
        st.session_state.panel_used = difference(subset)
        st.session_state.differencing_applied = True
    else:
        st.session_state.panel_used = subset
        st.session_state.differencing_applied = False


_autoload_sample_if_missing()


# ============================================================
# RUN PIPELINE
# ============================================================
def run_analysis() -> None:
    panel_used = st.session_state.panel_used
    if panel_used is None or len(panel_used) < 50:
        st.error("Not enough data to run analysis.")
        return
    p = st.session_state.params
    window, shift, p_lag, K = p["window"], p["shift"], p["p_lag"], p["K"]
    n_obs = len(panel_used)
    if window > n_obs:
        st.error(f"Window ({window}) > observations ({n_obs}). Pick a smaller window or wider date range.")
        return

    progress = st.progress(0.0, text="Computing rolling Granger…")

    def cb(done, total):
        progress.progress(min(done / max(total, 1), 1.0),
                          text=f"Window {done}/{total}")

    try:
        X, meta, pair_keys = build_windows(panel_used, window, shift, p_lag, progress_callback=cb)
    except Exception as e:
        st.error(f"Granger pipeline failed: {e}")
        progress.empty()
        return

    progress.progress(1.0, text="Clustering…")

    elbow = elbow_scan(X, K_max=min(10, len(X) - 1))
    km = run_kmeans(X, K)
    hw = run_hierarchical(X, K)
    ari = compute_ari(km["labels"], hw["labels"])

    pc1, pc2 = get_pca_variance(X)

    st.session_state.results = {
        "X": X,
        "meta": meta,
        "pair_keys": pair_keys,
        "nodes": list(panel_used.columns),
        "km": km,
        "hw": hw,
        "ari": ari,
        "elbow": elbow,
        "pca_var": (pc1, pc2),
        "params": dict(p),
    }
    progress.empty()
    st.toast("✓ Analysis complete", icon="✅")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div style="background:#FFF3CD;border:1px solid #FFEEBA;color:#856404;
                padding:8px 14px;border-radius:6px;font-size:13px;margin-bottom:8px;">
    ⚠️ <strong>Granger causality is NOT true causation.</strong>
    It measures predictive precedence only.
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Granger Regime Explorer")
st.caption(
    "Rolling-window conditional Granger causality + clustering. "
    "Discover recurring transmission regimes in multivariate time series."
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # ---------- Data ----------
    st.header("📂 Data")
    source = st.radio(
        "Source",
        options=["Sample (bundled)", "Upload CSV"],
        index=0 if st.session_state.data_source != "upload" else 1,
        horizontal=True,
    )

    if source == "Sample (bundled)":
        if st.button("🔄 Reload sample", use_container_width=True):
            st.session_state.df_raw = None
            st.session_state.results = None
            st.session_state.date_range = None
            _autoload_sample_if_missing()
            st.rerun()
        if st.session_state.data_source == "sample" and st.session_state.df_raw is not None:
            st.caption(
                f"✓ Sample · {len(st.session_state.df_raw):,} rows · "
                f"{len(st.session_state.df_raw.columns)} cols"
            )
    else:
        uploaded = st.file_uploader("CSV (one date col + numeric cols)", type=["csv"])
        if uploaded is not None:
            try:
                df = parse_uploaded(uploaded)
                st.session_state.df_raw = df
                st.session_state.data_source = "upload"
                st.session_state.date_col = autodetect_date_col(df) or df.columns[0]
                st.session_state.value_cols = list(
                    numeric_columns(df, exclude=[st.session_state.date_col])
                )
                st.session_state.results = None
                st.session_state.date_range = None
                _recompute_panel_and_adf()
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")

    st.divider()

    # ---------- Columns ----------
    st.subheader("📐 Columns")
    df_raw = st.session_state.df_raw
    if df_raw is None:
        st.info("Pick a data source above.")
    else:
        date_options = list(df_raw.columns)
        default_idx = (
            date_options.index(st.session_state.date_col)
            if st.session_state.date_col in date_options
            else 0
        )
        new_date = st.selectbox("Date column", options=date_options, index=default_idx)
        if new_date != st.session_state.date_col:
            st.session_state.date_col = new_date
            _recompute_panel_and_adf()

        candidate_vars = list(numeric_columns(df_raw, exclude=[new_date]))
        new_vars = st.multiselect(
            "Variables (≥ 2 required)",
            options=candidate_vars,
            default=[c for c in st.session_state.value_cols if c in candidate_vars]
            or candidate_vars,
        )
        if new_vars != st.session_state.value_cols:
            st.session_state.value_cols = new_vars
            _recompute_panel_and_adf()

        new_autodiff = st.checkbox(
            "Auto-difference if non-stationary (ADF p ≥ 0.05)",
            value=st.session_state.auto_diff,
        )
        if new_autodiff != st.session_state.auto_diff:
            st.session_state.auto_diff = new_autodiff
            _recompute_panel_and_adf()

        if st.session_state.adf:
            with st.expander("ADF stationarity results", expanded=False):
                for col, r in st.session_state.adf.items():
                    pval = r.get("pvalue", float("nan"))
                    if r.get("stationary"):
                        st.markdown(f"- **{col}** ✅ stationary (p={pval:.3f})")
                    else:
                        st.markdown(f"- **{col}** ⚠️ non-stationary (p={pval:.3f})")
            if st.session_state.differencing_applied:
                st.success("→ Differencing applied.")

    st.divider()

    # ---------- Date Range ----------
    st.subheader("📅 Date Range")
    panel = st.session_state.panel
    if panel is not None and len(panel) > 0:
        dmin = panel.index.min().to_pydatetime()
        dmax = panel.index.max().to_pydatetime()
        if st.session_state.date_range is None:
            st.session_state.date_range = (dmin, dmax)
        cur_start, cur_end = st.session_state.date_range
        try:
            cur_start = max(dmin, pd.to_datetime(cur_start).to_pydatetime())
            cur_end = min(dmax, pd.to_datetime(cur_end).to_pydatetime())
        except Exception:
            cur_start, cur_end = dmin, dmax

        new_range = st.slider(
            "Subset of data to analyze",
            min_value=dmin,
            max_value=dmax,
            value=(cur_start, cur_end),
            format="YYYY-MM-DD",
        )
        if new_range != st.session_state.date_range:
            st.session_state.date_range = new_range
            _recompute_panel_and_adf()
        if st.session_state.panel_used is not None:
            st.caption(
                f"{len(st.session_state.panel_used):,} obs after subset/diff"
            )
    else:
        st.info("Load data first.")

    st.divider()

    # ---------- Custom periods ----------
    st.subheader("🏷 Custom Periods")
    with st.expander("Add / manage periods", expanded=False):
        for p_def in st.session_state.periods:
            cols = st.columns([3, 1])
            cols[0].caption(f"**{p_def['name']}** — {p_def['start']} → {p_def['end']}")
            if cols[1].button("✕", key=f"del_{p_def['name']}"):
                st.session_state.periods = remove_period(st.session_state.periods, p_def["name"])
                save_periods(st.session_state.periods)
                st.rerun()
        st.markdown("**Add new:**")
        new_name = st.text_input("Name", key="new_period_name", placeholder="e.g. COVID Crash")
        new_start = st.text_input("Start (YYYY-MM-DD)", key="new_period_start")
        new_end = st.text_input("End (YYYY-MM-DD)", key="new_period_end")
        if st.button("➕ Add period"):
            updated, err = add_period(
                st.session_state.periods, new_name, new_start, new_end
            )
            if err:
                st.error(err)
            else:
                st.session_state.periods = updated
                save_periods(updated)
                st.success(f"Added '{new_name}'")
                st.rerun()
    if st.session_state.periods:
        st.caption(f"{len(st.session_state.periods)} period(s) defined")

    st.divider()

    # ---------- Parameters ----------
    st.subheader("⚙️ Parameters")
    p = st.session_state.params
    p["window"] = st.slider("Window size (days)", 30, 250, p["window"], step=5)
    p["shift"]  = st.slider("Shift (days)", 1, 60, p["shift"], step=1)
    p["p_lag"]  = st.slider("VAR lag p", 1, 20, p["p_lag"], step=1)
    p["K"]      = st.slider("Clusters K", 2, 10, p["K"], step=1)

    st.divider()

    # ---------- Run ----------
    run_disabled = (
        st.session_state.panel_used is None
        or len(st.session_state.value_cols) < 2
    )
    if st.button("▶ Run Analysis", use_container_width=True, type="primary", disabled=run_disabled):
        run_analysis()


# ============================================================
# STATE STRIP
# ============================================================
panel_used = st.session_state.panel_used
n_obs = len(panel_used) if panel_used is not None else 0
window = st.session_state.params["window"]
shift = st.session_state.params["shift"]
n_windows = max(0, (n_obs - window) // shift + 1) if n_obs >= window else 0
K = st.session_state.params["K"]

results = st.session_state.results

state_cols = st.columns([2, 1, 1, 1])
with state_cols[0]:
    if st.session_state.df_raw is None:
        st.markdown("**State:** ⚪ Waiting for data")
    elif results is None:
        st.markdown("**State:** 🟡 Ready — click **Run Analysis**")
    else:
        st.markdown("**State:** 🟢 Analysis complete")
with state_cols[1]:
    st.markdown(f"**Windows:** {n_windows}")
with state_cols[2]:
    st.markdown("**Methods:** K-means · Ward")
with state_cols[3]:
    if results is not None:
        n_pairs = results["X"].shape[1]
        n_strong = 0
        n_border = 0
        for c in range(results["km"]["labels"].max() + 1):
            f = cluster_fraction_sig(results["km"]["labels"], c, results["X"])
            n_strong += int((f >= FRAC_SIG_THRESHOLD).sum())
            n_border += int(((f >= FRAC_BORDERLINE_THRESH) & (f < FRAC_SIG_THRESHOLD)).sum())
        total = n_pairs * (results["km"]["labels"].max() + 1)
        st.markdown(f"**Edges:** {n_strong}/{total} ★ · {n_border}/{total} ◐")
    else:
        st.markdown("**Edges:** —")


# ============================================================
# TABS
# ============================================================
(tab_explain, tab_overview, tab_timeline, tab_networks, tab_2d,
 tab_heatmaps, tab_compare) = st.tabs(
    ["💬 Explain (plain English)", "📋 Overview", "📅 Timeline", "🕸 Networks",
     "📍 2D (PCA · t-SNE)", "🔥 Heatmaps", "🆚 Compare"]
)


# ---------- Explain (plain English) ----------
with tab_explain:
    st.subheader("What do these results mean?")
    if results is None:
        st.info(
            "Click **▶ Run Analysis** in the sidebar. This tab will then explain "
            "your results in plain English — no statistics jargon."
        )
    else:
        report = full_interpretation(results, st.session_state.periods)
        st.markdown(report)


# ---------- Overview ----------
with tab_overview:
    st.subheader("Overview")
    if st.session_state.panel_used is not None:
        pu = st.session_state.panel_used
        date_min = pu.index.min().date()
        date_max = pu.index.max().date()
        if st.session_state.differencing_applied:
            st.info(
                "ℹ️ Your data was non-stationary — we applied first-order differencing "
                "before analysis. Results describe **changes**, not levels."
            )
        st.markdown(
            f"**Data ready** &nbsp;·&nbsp; {len(pu):,} obs &nbsp;·&nbsp; "
            f"{date_min} → {date_max} &nbsp;·&nbsp; "
            f"{len(pu.columns)} variables: `{', '.join(pu.columns)}`"
        )
        with st.expander("Preview first 5 rows", expanded=False):
            st.dataframe(pu.head(), use_container_width=True)

    if results is not None:
        st.divider()
        st.markdown("### Auto-generated summary")
        meta = results["meta"]
        n_w = len(meta)
        dmin = min(m["mid_date"] for m in meta).date()
        dmax = max(m["mid_date"] for m in meta).date()
        K_ = results["km"]["labels"].max() + 1
        ari = results["ari"]
        pc1, pc2 = results["pca_var"]
        pca_pct = (pc1 + pc2) * 100
        elbow_k = results["elbow"]["elbow"]
        pca_signal = (
            "non-linear (justifies t-SNE)"
            if (pc1 + pc2) < 0.40
            else "primarily linear"
        )

        st.markdown(
            f"Analyzed **{n_w} windows** ({dmin} → {dmax}). "
            f"Clustered into **K={K_}** regimes using K-means and Hierarchical Ward. "
            f"PCA captures **{pca_pct:.1f}%** of variance in 2D — structure is **{pca_signal}**. "
            f"Adjusted Rand Index between K-means and Ward: **{ari:.2f}** "
            f"({'strong agreement' if ari > 0.5 else 'moderate agreement' if ari > 0.3 else 'weak agreement'}). "
            f"Elbow-suggested K: **{elbow_k}**."
        )

        # Per-cluster mini summary
        st.markdown("### Cluster fingerprints (K-means)")
        rows = []
        labels = results["km"]["labels"]
        X = results["X"]
        pair_keys = results["pair_keys"]
        for c in range(K_):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            frac = cluster_fraction_sig(labels, c, X)
            top_i = np.argsort(-frac)[:3]
            top_str = " · ".join(
                f"{pair_keys[i][0]}→{pair_keys[i][1]} {frac[i]*100:.0f}%"
                for i in top_i if frac[i] > 0.05
            )
            rows.append({
                "Cluster": f"C{c}",
                "N": len(idx),
                "% sample": f"{len(idx)/n_w*100:.0f}%",
                "Top edges": top_str or "(none significant)",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Elbow plot")
        e = results["elbow"]
        fig = plot_elbow(e["K_list"], e["wcss"], e["elbow"])
        st.pyplot(fig)
    else:
        st.divider()
        st.caption("Click **Run Analysis** in the sidebar to populate this tab.")


# ---------- Timeline ----------
with tab_timeline:
    st.subheader("Cluster timeline")
    if results is None:
        st.info("Run analysis first.")
    else:
        meta = results["meta"]
        labels = results["km"]["labels"]
        medoids = results["km"]["medoids"]
        K_ = labels.max() + 1
        fig = plot_timeline(
            labels, K_, medoids, meta,
            periods=st.session_state.periods,
            title=f"Cluster Timeline — K-means K={K_} (★ = medoid)",
        )
        st.pyplot(fig)
        st.caption(
            "Colored bars = cluster assignment per window. "
            "★ = medoid (most representative window per cluster). "
            "Background bands = your custom periods (if defined)."
        )


# ---------- Networks ----------
with tab_networks:
    st.subheader("Leader networks per cluster (fraction-significant)")
    st.caption(
        f"★ = edge ≥ {int(FRAC_SIG_THRESHOLD*100)}% of windows (6× null rate)  ·  "
        f"◐ = edge {int(FRAC_BORDERLINE_THRESH*100)}–{int(FRAC_SIG_THRESHOLD*100)}% (3× null rate)"
    )
    if results is None:
        st.info("Run analysis first.")
    else:
        labels = results["km"]["labels"]
        X = results["X"]
        meta = results["meta"]
        pair_keys = results["pair_keys"]
        nodes = results["nodes"]
        K_ = labels.max() + 1

        cols = st.columns(2)
        for c in range(K_):
            idx = np.where(labels == c)[0]
            n_in = len(idx)
            if n_in == 0:
                continue
            edge_data = build_cluster_edge_data(labels, c, X, meta, pair_keys)
            dmin = min(meta[wi]["mid_date"] for wi in idx).date()
            dmax = max(meta[wi]["mid_date"] for wi in idx).date()
            ttl = f"C{c}  ({n_in} win, {n_in/len(labels)*100:.0f}%)\n{dmin} → {dmax}"
            fig = plot_cluster_network(edge_data, nodes, ttl,
                                       target_node=nodes[-1] if nodes else None)
            with cols[c % 2]:
                st.pyplot(fig)


# ---------- 2D ----------
with tab_2d:
    st.subheader("Feature space projection")
    if results is None:
        st.info("Run analysis first.")
    else:
        X = results["X"]
        labels = results["km"]["labels"]
        meta = results["meta"]
        pc1, pc2 = results["pca_var"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PCA (linear)**")
            fig_pca, _ = plot_pca_scatter(X, labels, meta, (pc1, pc2))
            st.pyplot(fig_pca)
            if (pc1 + pc2) < 0.4:
                st.caption(
                    f"⚠️ Only {(pc1+pc2)*100:.1f}% variance captured in 2D — "
                    "data structure is non-linear, t-SNE is more informative."
                )
        with col2:
            st.markdown("**t-SNE (non-linear)**")
            fig_tsne = plot_tsne_scatter(X, labels, perplexity=15)
            st.pyplot(fig_tsne)
            st.caption(
                "Axes carry no inherent meaning — only relative distances matter. "
                "Well-separated clusters here = real structure (not artifact)."
            )


# ---------- Heatmaps ----------
with tab_heatmaps:
    st.subheader("Fraction-significant heatmaps per cluster")
    if results is None:
        st.info("Run analysis first.")
    else:
        labels = results["km"]["labels"]
        X = results["X"]
        nodes = results["nodes"]
        pair_keys = results["pair_keys"]
        K_ = labels.max() + 1
        cols = st.columns(2)
        for c in range(K_):
            fig = plot_cluster_heatmap(X, labels, c, nodes, pair_keys,
                                       title=f"Cluster {c}")
            with cols[c % 2]:
                st.pyplot(fig)


# ---------- Compare ----------
with tab_compare:
    st.subheader("K-means vs Hierarchical Ward")
    if results is None:
        st.info("Run analysis first.")
    else:
        km = results["km"]
        hw = results["hw"]
        ari = results["ari"]
        meta = results["meta"]
        K_ = km["labels"].max() + 1

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("ARI", f"{ari:.3f}",
                     help="Adjusted Rand Index between K-means and Ward")
        mcol2.metric("K-means silhouette", f"{km['silhouette']:.3f}")
        mcol3.metric("Ward silhouette", f"{hw['silhouette']:.3f}")

        st.markdown("**K-means timeline**")
        fig_km = plot_timeline(km["labels"], K_, km["medoids"], meta,
                                periods=st.session_state.periods,
                                title="K-means")
        st.pyplot(fig_km)

        st.markdown("**Hierarchical Ward timeline**")
        fig_hw = plot_timeline(hw["labels"], K_, hw["medoids"], meta,
                                periods=st.session_state.periods,
                                title="Hierarchical Ward")
        st.pyplot(fig_hw)

        # Confusion matrix
        st.markdown("**Confusion matrix (rows: K-means, cols: Ward)**")
        conf = np.zeros((K_, K_), dtype=int)
        for k_lbl, w_lbl in zip(km["labels"], hw["labels"]):
            conf[int(k_lbl), int(w_lbl)] += 1
        conf_df = pd.DataFrame(
            conf,
            index=[f"K-means C{c}" for c in range(K_)],
            columns=[f"Ward C{c}" for c in range(K_)],
        )
        st.dataframe(conf_df, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "⚠️ Granger ≠ causation. Exploratory research methodology. "
    "Try multiple (window, p, K) configurations to test robustness."
)
