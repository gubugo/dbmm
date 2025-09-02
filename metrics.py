#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import suppress
from functools import wraps
from typing import Optional

import numpy as np
from matplotlib.collections import LineCollection
from scipy import spatial, stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def nan_when_raises(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        with suppress(ValueError):
            return func(*args, **kwargs)
        return np.nan

    return new_func


@nan_when_raises
def compute_distance_list(X):
    return spatial.distance.pdist(X, "euclidean")


def metric_distance_to_nearest_neighbor(inverted_grid, nearest_neighbor):
    dist, _ = nearest_neighbor.kneighbors(
        inverted_grid, n_neighbors=1, return_distance=True
    )
    return dist.reshape(np.shape(dist)[0])

def metric_distance_to_nearest_same_class_neighbor(inverted_grid, n_classes, classes, grid_points, per_class_neighbor_finder):
    # this retains the data type, which might be handy.
    distances = np.zeros((grid_points,), dtype=np.float32)

    #inverted_grid = self.inverter(self.grid)
    # inverted_grid_classes = self.classifier.classify(inverted_grid)
    # inverted_grid = inverted_grid.cpu().numpy()
    # inverted_grid_classes = inverted_grid_classes.cpu().numpy()
    for cl in range(n_classes):
        mask = classes == cl
        if not np.any(mask):
            continue
        dist, _ = per_class_neighbor_finder[cl].kneighbors(
            inverted_grid[classes == cl],
            n_neighbors=1,
            return_distance=True,
        )
        dist = dist.squeeze()
        distances[mask] = dist

    return distances#.reshape((self.grid_width, self.grid_width)).copy()

def metric_difference_dbms(base_colors, colors, grid_res, ax):
    subbed = np.subtract(base_colors[:, 0:3], colors[:, 0:3])
    if np.size(np.nonzero(subbed.flatten())) == 0:
        ax.imshow(base_colors.reshape((grid_res, grid_res, 4)), interpolation='none', resample=False, origin='lower')
    else:
        ax.imshow(np.abs(subbed).reshape((grid_res, grid_res, 3)), interpolation='none', resample=False, origin='lower')
    

# TODO handle precomputed distances. Euclidean doesn't make sense for spherical.
@nan_when_raises
def metric_neighborhood_hit(X, y, k=7, precomputed=False):
    knn = KNeighborsClassifier(n_neighbors=k)
    if precomputed:
        # tell `knn` to assume X is a distance matrix
        knn.set_params(metric="precomputed")

    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(
        np.mean(
            (y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype("uint8"), axis=1
        )
    )


@nan_when_raises
def metric_trustworthiness(X_high, X_low, D_high_m, D_low_m, k=7):
    D_high = spatial.distance.squareform(D_high_m)
    D_low = spatial.distance.squareform(D_low_m)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, : k + 1][:, 1:]
    knn_proj = nn_proj[:, : k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


@nan_when_raises
def metric_continuity(X_high, X_low, D_high_l, D_low_l, k=7):
    D_high = spatial.distance.squareform(D_high_l)
    D_low = spatial.distance.squareform(D_low_l)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, : k + 1][:, 1:]
    knn_proj = nn_proj[:, : k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


@nan_when_raises
def metric_shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high, D_low)[0]


@nan_when_raises
def metric_normalized_stress(D_high, D_low, ndim_high=None):
    return np.sqrt(np.sum((D_high - D_low) ** 2) / np.sum(D_high**2))


@nan_when_raises
def metric_mse(X, X_hat):
    return np.mean(np.square(X - X_hat))


def class_variances(X, y, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(y))
    variances = np.zeros(n_classes, dtype=np.float32)
    for cl in range(n_classes):
        class_data = X[y == cl]
        variances[cl] = np.var(class_data, axis=0, ddof=1).sum()
        # variances[cl] = np.trace(np.cov(class_data, rowvar=False))
    return variances


@nan_when_raises
def cluster_size_consistency_r(
    X_high,
    y,
    X_proj,
    outlier_detector_maker=IsolationForest,
    n_classes=None,
    return_hulls=False,
):
    interm = cluster_size_consistency(
        X_high, y, X_proj, outlier_detector_maker, n_classes, return_hulls
    )

    if return_hulls:
        return interm[0].correlation, interm[1]
    return interm.correlation


def cluster_size_consistency(
    X_high,
    y,
    X_proj,
    outlier_detector_maker=IsolationForest,
    n_classes=None,
    return_hulls=False,
):
    if n_classes is None:
        n_classes = len(np.unique(y))
    areas = np.zeros(n_classes, dtype=np.float32)
    hulls = [None for _ in range(n_classes)]
    variances = class_variances(X_high, y)
    for cl in range(n_classes):
        proj_class_data = X_proj[y == cl]
        main_cluster_points = proj_class_data[
            outlier_detector_maker().fit_predict(proj_class_data) == 1
        ]
        hull = spatial.ConvexHull(main_cluster_points, incremental=False)
        hulls[cl] = hull
        areas[cl] = hull.area
    if return_hulls:
        return stats.spearmanr(variances, areas), hulls
    return stats.spearmanr(variances, areas)


def _get_plot_lims(X_proj):
    x_min, x_max = X_proj[:, 0].min(), X_proj[:, 0].max()
    y_min, y_max = X_proj[:, 1].min(), X_proj[:, 1].max()
    w, h = (x_max - x_min), (y_max - y_min)
    x_min, x_max = (x_min - 0.05 * w), (x_max + 0.05 * w)
    y_min, y_max = (y_min - 0.05 * h), (y_max + 0.05 * h)

    return x_min, x_max, y_min, y_max


def plot_data_and_hulls(
    X_proj,
    y,
    hulls: list[spatial.ConvexHull],
    n_classes=None,
    fname=None,
    save_without_hulls=False,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if n_classes is None:
        n_classes = len(np.unique(y))
    fig, ax = plt.subplots(figsize=(10, 10))

    x_min, x_max, y_min, y_max = _get_plot_lims(X_proj)

    cmap = ListedColormap(
        [
            "#aee39a",
            "#9e37d0",
            "#7cee4d",
            "#713d83",
            "#1be19f",
            "#fb2076",
            "#458612",
            "#e89ff0",
            "#115d52",
            "#f79302",
        ]
    )
    for cl in range(n_classes):
        class_data = X_proj[y == cl, :]
        ax.scatter(
            class_data[:, 0],
            class_data[:, 1],
            c=[cmap(cl) for _ in range(class_data.shape[0])],
            label=cl,
            s=25,
            alpha=0.5,
        )
        ax.axis("off")
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    # plt.legend()
    if save_without_hulls:
        import os

        base, ext = os.path.splitext(fname)
        plt.savefig(f"{base}_nohull{ext}")
    for h in hulls:
        line_segments = [h.points[simplex] for simplex in h.simplices]
        ax.add_collection(LineCollection(line_segments, colors="k", linestyle="solid"))
    ax.axis("off")
    plt.show() if fname is None else plt.savefig(fname)

    plt.close(fig)


def distance_consistency(
    X_proj: np.ndarray, y: np.ndarray, n_classes: Optional[int] = None
):
    label_enc = LabelEncoder().fit(y)
    if n_classes is None:
        n_classes = len(label_enc.classes_)
    # Protect ourselves in case class labels are not in [0, n_classes-1]
    y_ = label_enc.transform(y)
    X_proj = np.copy(X_proj)

    sort_indices = y_.argsort()
    X_proj = X_proj[sort_indices]
    y_ = y_[sort_indices]
    per_class = np.split(X_proj, np.unique(y_, return_index=True)[1][1:])
    centroids = np.array(
        [
            np.mean(data, axis=0) if len(data) > 0 else np.array([np.nan, np.nan])
            for data in per_class
        ]
    )
    closest_centroid = np.linalg.norm(
        np.expand_dims(X_proj, axis=1) - centroids, axis=2
    ).argmin(axis=1)

    return np.mean(closest_centroid == y_)
