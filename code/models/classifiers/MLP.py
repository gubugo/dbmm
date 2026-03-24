from sklearn.neural_network import MLPClassifier

def make_and_fit_mlp(X, y) -> MLPClassifier:
    return MLPClassifier(
        verbose=True,
        hidden_layer_sizes=(512, 128, 64, 32),
        activation="relu",
        max_iter=100,
        random_state=420,
    ).fit(X, y)