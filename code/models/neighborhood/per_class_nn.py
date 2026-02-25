
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_per_class_nn_model(X, y, n_neighbors):
    n_classes = len(np.unique(y))
    per_class_neighbor_finder = {}
    for cl in range(n_classes):
        neighbor_finder2 = NearestNeighbors(n_neighbors)
        neighbor_finder2.fit(X[y == cl])
        per_class_neighbor_finder[cl] = neighbor_finder2

    return per_class_neighbor_finder