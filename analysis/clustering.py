"""K-means + Hierarchical Ward clustering with elbow detection."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def find_elbow_k(K_list: list[int], inerts: list[float]) -> int:
    """Find the elbow K using the maximum-distance-from-line method."""
    pts = np.array(list(zip(K_list, inerts)))
    p1, p2 = pts[0], pts[-1]
    line_n = (p2 - p1) / np.linalg.norm(p2 - p1)
    dists = []
    for p in pts:
        v = p - p1
        proj = np.dot(v, line_n) * line_n
        dists.append(np.linalg.norm(v - proj))
    return K_list[int(np.argmax(dists))]


def elbow_scan(X: np.ndarray, K_max: int = 10, seed: int = 42) -> dict:
    """Run K-means for K=2..K_max, return WCSS list + suggested elbow K."""
    K_list = list(range(2, K_max + 1))
    inerts = []
    for k in K_list:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(X)
        inerts.append(float(km.inertia_))
    elbow = find_elbow_k(K_list, inerts)
    return {"K_list": K_list, "wcss": inerts, "elbow": elbow}


def run_kmeans(X: np.ndarray, K: int, seed: int = 42) -> dict:
    """Run K-means, return labels + centroids + medoid indices + silhouette."""
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    medoids = {}
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
        medoids[c] = int(idx[int(np.argmin(d))])
    sil = float(silhouette_score(X, labels)) if K >= 2 and K < len(X) else float("nan")
    return {
        "labels": labels,
        "centroids": km.cluster_centers_,
        "medoids": medoids,
        "silhouette": sil,
        "wcss": float(km.inertia_),
    }


def run_hierarchical(X: np.ndarray, K: int, method: str = "ward") -> dict:
    """Hierarchical (default Ward) clustering, returns labels + linkage matrix."""
    Z = linkage(X, method=method)
    labels = fcluster(Z, t=K, criterion="maxclust") - 1
    medoids = {}
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        center = X[idx].mean(axis=0)
        d = np.linalg.norm(X[idx] - center, axis=1)
        medoids[c] = int(idx[int(np.argmin(d))])
    sil = float(silhouette_score(X, labels)) if K >= 2 and K < len(X) else float("nan")
    return {
        "labels": labels,
        "linkage": Z,
        "medoids": medoids,
        "silhouette": sil,
    }


def compute_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index between two label assignments."""
    return float(adjusted_rand_score(labels_a, labels_b))
