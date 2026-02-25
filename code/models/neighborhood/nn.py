

from sklearn.neighbors import NearestNeighbors


def get_nn_model(X, n_neighbors):
    neighbor_finder_model = NearestNeighbors(n_neighbors=n_neighbors) 
    neighbor_finder_model.fit(X)

    return neighbor_finder_model