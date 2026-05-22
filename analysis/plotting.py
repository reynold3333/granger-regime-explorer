"""Plotting helpers — generic, work with any variable set."""
from __future__ import annotations

from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

from analysis.granger import FRAC_BORDERLINE_THRESH, FRAC_SIG_THRESHOLD


CLUSTER_PALETTE = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#FBC02D", "#5D4037",
]


def _circle_layout(nodes: list[str]) -> dict[str, tuple[float, float]]:
    n = len(nodes)
    pos = {}
    for i, v in enumerate(nodes):
        angle = 2 * np.pi * i / n - np.pi / 2
        pos[v] = (np.cos(angle), np.sin(angle))
    return pos


def plot_timeline(
    labels: np.ndarray,
    K: int,
    medoids: dict[int, int],
    meta: list[dict],
    periods: list[dict] | None = None,
    title: str = "Cluster Timeline",
):
    """Render cluster timeline. periods = [{name,start,end}] optional."""
    fig, ax = plt.subplots(figsize=(14, 5))

    if periods:
        period_colors = ["#FFE0B2", "#C8E6C9", "#BBDEFB", "#F8BBD0", "#D1C4E9", "#FFCCBC"]
        for i, p in enumerate(periods):
            try:
                s = pd.to_datetime(p["start"])
                e = pd.to_datetime(p["end"])
            except Exception:
                continue
            color = period_colors[i % len(period_colors)]
            ax.axvspan(s, e, alpha=0.35, color=color, zorder=0)
            mid = s + (e - s) / 2
            ax.text(mid, K - 0.3, p.get("name", f"P{i+1}"),
                    ha="center", va="top", fontsize=9, fontweight="bold",
                    color="#37474F", alpha=0.85)

    for wi in range(len(labels)):
        c = int(labels[wi])
        m = meta[wi]
        ax.barh(c, (m["end_date"] - m["start_date"]).days,
                left=m["start_date"], height=0.7,
                color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                alpha=0.85, edgecolor="white", linewidth=0.3)

    for c, mi in medoids.items():
        m = meta[mi]
        ax.plot(m["mid_date"], c, "k*", markersize=18,
                markeredgecolor="white", markeredgewidth=1.5, zorder=10)

    ax.set_yticks(range(K))
    ax.set_yticklabels([f"C{c}" for c in range(K)])
    ax.set_xlabel("Date")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3, axis="x")
    ax.set_ylim(-0.7, K - 0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_cluster_network(
    edge_data: list[dict],
    nodes: list[str],
    title: str,
    target_node: str | None = None,
):
    """Network graph for one cluster. Edges colored by fraction-significant."""
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = _circle_layout(nodes)

    G = nx.DiGraph()
    for v in nodes:
        G.add_node(v)
    for e in edge_data:
        G.add_edge(e["Source"], e["Target"])

    bg_edges = [(e["Source"], e["Target"]) for e in edge_data
                if not e["Borderline"] and not e["Significant"]]
    border_edges = [(e["Source"], e["Target"]) for e in edge_data
                    if e["Borderline"] and not e["Significant"]]
    sig_edges = [(e["Source"], e["Target"]) for e in edge_data
                 if e["Significant"]]
    sig_widths = [max(2.5, e["frac_sig"] * 8) for e in edge_data if e["Significant"]]

    nx.draw_networkx_edges(G, pos, edgelist=bg_edges, width=0.4,
                           edge_color="#CFD8DC", alpha=0.3, arrows=True,
                           arrowsize=6, connectionstyle="arc3,rad=0.12", ax=ax,
                           min_source_margin=20, min_target_margin=22)
    if border_edges:
        nx.draw_networkx_edges(G, pos, edgelist=border_edges, width=1.4,
                               edge_color="#FFB74D", alpha=0.65, arrows=True,
                               arrowsize=12, connectionstyle="arc3,rad=0.15", ax=ax,
                               min_source_margin=20, min_target_margin=22)
    if sig_edges:
        nx.draw_networkx_edges(G, pos, edgelist=sig_edges, width=sig_widths,
                               edge_color="#C62828", alpha=0.95, arrows=True,
                               arrowsize=22, connectionstyle="arc3,rad=0.15", ax=ax,
                               min_source_margin=20, min_target_margin=22)

    for v in nodes:
        size = 2400 if v == target_node else 1800
        nx.draw_networkx_nodes(G, pos, nodelist=[v], node_color="#37474F",
                               node_size=size, alpha=0.92, ax=ax,
                               edgecolors="white", linewidths=2.5)
    for v in nodes:
        x, y = pos[v]
        ax.text(x, y, v, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

    edge_labels = {(e["Source"], e["Target"]): f"{e['frac_sig']*100:.0f}%"
                   for e in edge_data if e["Significant"] or e["Borderline"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax,
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="white", edgecolor="none",
                                           alpha=0.85))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_cluster_heatmap(
    X: np.ndarray,
    labels: np.ndarray,
    c: int,
    nodes: list[str],
    pair_keys: list[tuple[str, str]],
    title: str,
):
    """Heatmap of fraction-significant per directional pair for cluster c."""
    idx = np.where(labels == c)[0]
    if len(idx) == 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "Empty cluster", ha="center", va="center")
        ax.axis("off")
        return fig
    frac = (X[idx] < 0.05).mean(axis=0)
    n = len(nodes)
    M = np.full((n, n), np.nan)
    for (s, t), fv in zip(pair_keys, frac):
        if s in nodes and t in nodes:
            M[nodes.index(s), nodes.index(t)] = fv

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(M, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=nodes, yticklabels=nodes,
                cbar_kws={"label": "% windows p<0.05"},
                linewidths=0.5, linecolor="white", ax=ax,
                mask=np.isnan(M))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Target →")
    ax.set_ylabel("Source ↓")
    fig.tight_layout()
    return fig


def plot_elbow(K_list: list[int], wcss: list[float], elbow_k: int):
    """Elbow plot showing WCSS vs K."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_list, wcss, "o-", color="#1E88E5", linewidth=2, markersize=8)
    ax.axvline(elbow_k, color="#C62828", linestyle="--",
               label=f"Elbow K = {elbow_k}")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("WCSS (within-cluster sum of squares)")
    ax.set_title("Elbow Method", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pca_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    meta: list[dict],
    var_explained: tuple[float, float],
):
    """2D PCA scatter colored by cluster."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=CLUSTER_PALETTE[int(c) % len(CLUSTER_PALETTE)],
                   s=70, alpha=0.75, edgecolors="white", linewidths=1.2,
                   label=f"C{int(c)} (n={int(mask.sum())})", zorder=3)
    ax.plot(X2[:, 0], X2[:, 1], "-", color="#9E9E9E", alpha=0.15,
            linewidth=0.5, zorder=1)
    pc1, pc2 = var_explained
    ax.set_xlabel(f"PC1 ({pc1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2*100:.1f}% var)")
    total = (pc1 + pc2) * 100
    ax.set_title(f"PCA — total variance explained: {total:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, (pc1, pc2)


def plot_tsne_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 15,
):
    """2D t-SNE scatter colored by cluster."""
    from sklearn.manifold import TSNE

    perplexity = min(perplexity, max(2, len(X) // 4))
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    init="pca", learning_rate="auto")
    except TypeError:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X2 = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=CLUSTER_PALETTE[int(c) % len(CLUSTER_PALETTE)],
                   s=70, alpha=0.75, edgecolors="white", linewidths=1.2,
                   label=f"C{int(c)} (n={int(mask.sum())})", zorder=3)
    ax.set_xlabel("t-SNE 1 (axis has no inherent meaning)")
    ax.set_ylabel("t-SNE 2 (axis has no inherent meaning)")
    ax.set_title(f"t-SNE — perplexity = {perplexity}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def get_pca_variance(X: np.ndarray) -> tuple[float, float]:
    """Return (PC1, PC2) variance ratios."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X)
    ev = pca.explained_variance_ratio_
    return float(ev[0]), float(ev[1])
