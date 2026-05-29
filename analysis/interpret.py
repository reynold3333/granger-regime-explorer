"""Generate plain-English interpretation of analysis results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.granger import (
    FRAC_BORDERLINE_THRESH,
    FRAC_SIG_THRESHOLD,
    cluster_fraction_sig,
)


def _date_in_period(d, start, end) -> bool:
    return pd.to_datetime(start) <= pd.to_datetime(d) <= pd.to_datetime(end)


def _ari_words(ari: float) -> str:
    if ari > 0.7:
        return "very strong agreement"
    if ari > 0.5:
        return "strong agreement"
    if ari > 0.3:
        return "moderate agreement"
    if ari > 0.1:
        return "weak agreement"
    return "little to no agreement"


def _silhouette_words(sil: float) -> str:
    if sil > 0.5:
        return "well-separated, tight clusters"
    if sil > 0.25:
        return "reasonably separated clusters"
    if sil > 0.1:
        return "loosely separated clusters (gradual transitions)"
    return "heavily overlapping clusters (very gradual transitions)"


def cluster_top_edges(labels, c, X, pair_keys, top_n=3):
    """Return list of (source, target, fraction) for top edges in cluster c."""
    frac = cluster_fraction_sig(labels, c, X)
    order = np.argsort(-frac)[:top_n]
    return [(pair_keys[i][0], pair_keys[i][1], float(frac[i])) for i in order]


def describe_cluster(labels, c, X, meta, pair_keys, n_total) -> dict:
    """Build a plain-English description of one cluster."""
    idx = np.where(labels == c)[0]
    n = len(idx)
    if n == 0:
        return {"empty": True, "c": c}
    frac = cluster_fraction_sig(labels, c, X)
    top = cluster_top_edges(labels, c, X, pair_keys, top_n=3)
    strong = [(pair_keys[i][0], pair_keys[i][1], float(frac[i]))
              for i in range(len(frac)) if frac[i] >= FRAC_SIG_THRESHOLD]
    border = [(pair_keys[i][0], pair_keys[i][1], float(frac[i]))
              for i in range(len(frac))
              if FRAC_BORDERLINE_THRESH <= frac[i] < FRAC_SIG_THRESHOLD]
    dmin = min(meta[wi]["mid_date"] for wi in idx).date()
    dmax = max(meta[wi]["mid_date"] for wi in idx).date()

    # Character label
    if strong:
        s, t, f = strong[0]
        character = f"Active-transmission regime — strongest link **{s} → {t}** ({f*100:.0f}% of windows)"
    elif border:
        s, t, f = border[0]
        character = f"Mild-transmission regime — leading link {s} → {t} ({f*100:.0f}% of windows)"
    else:
        character = "Calm / low-transmission regime — no consistent links above noise"

    return {
        "empty": False,
        "c": c,
        "n": n,
        "pct": n / n_total * 100,
        "dmin": dmin,
        "dmax": dmax,
        "top": top,
        "strong": strong,
        "border": border,
        "character": character,
    }


def detect_recurrence(labels, meta, periods, min_presence=0.20) -> dict:
    """Detect which clusters recur across non-adjacent custom periods."""
    if not periods:
        return {"has_periods": False}

    K = int(labels.max()) + 1
    # For each cluster, find which periods it has >= min_presence in
    period_names = [p["name"] for p in periods]
    cluster_periods = {c: [] for c in range(K)}
    for pi, p in enumerate(periods):
        idx_p = [wi for wi in range(len(labels))
                 if _date_in_period(meta[wi]["mid_date"], p["start"], p["end"])]
        if not idx_p:
            continue
        for c in range(K):
            n_c = sum(1 for wi in idx_p if labels[wi] == c)
            if n_c / len(idx_p) >= min_presence:
                cluster_periods[c].append(pi)

    recurring = {}
    for c, plist in cluster_periods.items():
        if len(plist) >= 2:
            non_adjacent = any(
                abs(plist[i] - plist[j]) >= 2
                for i in range(len(plist))
                for j in range(i + 1, len(plist))
            )
            recurring[c] = {
                "periods": [period_names[i] for i in plist],
                "non_adjacent": non_adjacent,
            }

    n_recurring_nonadj = sum(1 for v in recurring.values() if v["non_adjacent"])
    return {
        "has_periods": True,
        "n_periods": len(periods),
        "recurring": recurring,
        "n_recurring_nonadj": n_recurring_nonadj,
        "K": K,
    }


def full_interpretation(results: dict, periods: list[dict]) -> str:
    """Generate a complete plain-English markdown report of results."""
    X = results["X"]
    meta = results["meta"]
    pair_keys = results["pair_keys"]
    nodes = results["nodes"]
    labels = results["km"]["labels"]
    ari = results["ari"]
    pc1, pc2 = results["pca_var"]
    km_sil = results["km"]["silhouette"]
    K = int(labels.max()) + 1
    n_total = len(labels)
    dmin = min(m["mid_date"] for m in meta).date()
    dmax = max(m["mid_date"] for m in meta).date()
    p = results["params"]

    lines = []

    # --- 1. What we did ---
    lines.append("### 🧭 What this analysis did, in plain English\n")
    lines.append(
        f"We took your **{len(nodes)} variables** ({', '.join(nodes)}) "
        f"over the period **{dmin} → {dmax}**, and slid a "
        f"**{p['window']}-day window** across the data, moving it "
        f"**{p['shift']} days** at a time. That gave us **{n_total} snapshots** "
        f"of how the markets were behaving at different times.\n"
    )
    lines.append(
        f"For each snapshot, we asked: *\"which variable's past movements help "
        f"predict which other variable?\"* (that's Granger causality). Then we "
        f"grouped the {n_total} snapshots into **{K} clusters** — each cluster is "
        f"a distinct \"regime\" or way the markets were behaving.\n"
    )

    # --- 2. The regimes ---
    lines.append("### 🎭 The regimes we found\n")
    descs = [describe_cluster(labels, c, X, meta, pair_keys, n_total) for c in range(K)]
    # Sort: most active first
    descs_sorted = sorted(
        [d for d in descs if not d["empty"]],
        key=lambda d: -(len(d["strong"]) * 100 + len(d["border"])),
    )
    for d in descs_sorted:
        lines.append(
            f"**Cluster {d['c']}** — {d['n']} snapshots ({d['pct']:.0f}% of the time), "
            f"spanning {d['dmin']} → {d['dmax']}.\n"
        )
        lines.append(f"  - {d['character']}\n")
        if d["top"]:
            top_str = ", ".join(
                f"{s}→{t} ({f*100:.0f}%)" for s, t, f in d["top"] if f > 0.05
            )
            if top_str:
                lines.append(f"  - Most frequent links: {top_str}\n")

    # --- 3. Headline ---
    headline = next((d for d in descs_sorted if d["strong"]), None)
    lines.append("### ⭐ Headline finding\n")
    if headline:
        s, t, f = headline["strong"][0]
        lines.append(
            f"The most distinctive regime is **Cluster {headline['c']}**. In this "
            f"regime, **{s} → {t}** is a statistically significant predictor in "
            f"**{f*100:.0f}% of windows** — that's **{f/0.05:.0f}× more often** than "
            f"you'd expect by random chance (the 5% baseline). In plain terms: when "
            f"the markets are in this regime, movements in **{s}** reliably foreshadow "
            f"movements in **{t}**.\n"
        )
    else:
        lines.append(
            "No single regime had a transmission link that crossed the \"strong\" "
            f"threshold ({int(FRAC_SIG_THRESHOLD*100)}% of windows). This means the "
            "Granger signals are relatively sparse — the markets in this dataset don't "
            "show one dominant, persistent transmission channel. The structure is "
            "more about *distributional* differences between regimes than one loud "
            "signal.\n"
        )

    # --- 4. Recurrence (if periods defined) ---
    rec = detect_recurrence(labels, meta, periods)
    lines.append("### 🔁 Do regimes repeat?\n")
    if not rec["has_periods"]:
        lines.append(
            "_You haven't defined any custom periods yet._ To test whether regimes "
            "**recur** (appear in separate, non-adjacent time periods), add periods "
            "in the sidebar (e.g. 'COVID', 'Recovery', 'Rate Hikes') with their date "
            "ranges. Then re-run — this section will tell you which regimes come back.\n"
        )
    else:
        if rec["n_recurring_nonadj"] > 0:
            lines.append(
                f"**Yes — {rec['n_recurring_nonadj']} of {rec['K']} regimes recur in "
                f"non-adjacent periods.** This is meaningful: it suggests the regimes "
                f"are **state-driven, not time-driven** — the same market behavior "
                f"comes back whenever similar conditions occur, rather than each "
                f"period having its own unique, one-time pattern.\n"
            )
            for c, info in rec["recurring"].items():
                if info["non_adjacent"]:
                    lines.append(
                        f"  - **Cluster {c}** appears in: {', '.join(info['periods'])} "
                        f"(non-adjacent → genuine recurrence)\n"
                    )
        else:
            lines.append(
                "The clusters mostly stay within their own periods (or only appear in "
                "adjacent ones). This is more consistent with a **time-driven** story "
                "— each period has its own distinct regime — rather than recurring "
                "regimes.\n"
            )

    # --- 5. How much to trust this ---
    lines.append("### 🔬 How much should you trust this?\n")
    lines.append(
        f"- **Two different algorithms show {_ari_words(ari)}** "
        f"(Adjusted Rand Index = {ari:.2f}). K-means and Hierarchical Ward are built "
        f"on completely different math, so when they find similar groupings, that's "
        f"good evidence the clusters are real, not an artifact of one method.\n"
    )
    lines.append(
        f"- **Cluster shape:** the silhouette score ({km_sil:.2f}) indicates "
        f"{_silhouette_words(km_sil)}. "
        + (
            "Low scores here aren't bad — they usually mean market regimes blend into "
            "each other gradually rather than switching abruptly.\n"
            if km_sil < 0.25
            else "\n"
        )
    )
    total_var = (pc1 + pc2) * 100
    lines.append(
        f"- **Data complexity:** when we squash all the data down to 2 dimensions "
        f"for the scatter plot, only **{total_var:.0f}%** of the information survives. "
        + (
            "That's low, which tells us the real structure is **non-linear and lives "
            "in many dimensions** — so trust the t-SNE plot (which handles non-linear "
            "structure) over the PCA plot for judging whether clusters separate.\n"
            if total_var < 40
            else "That's a fair amount, so the PCA plot is a reasonable view of the "
            "structure.\n"
        )
    )

    # --- 6. Caveats ---
    lines.append("### ⚠️ Important caveats\n")
    lines.append(
        "- **Granger causality is not real causation.** It only means \"past values "
        "of X help predict Y\". A hidden third factor could drive both.\n"
        "- **Results depend on your settings.** Try different window sizes, lags, and "
        "K values — if the story holds across settings, it's robust. If it flips "
        "easily, be cautious.\n"
        "- **This is exploratory, not a forecast.** It describes what *has* happened, "
        "not what *will* happen.\n"
    )

    return "\n".join(lines)
