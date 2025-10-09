import os

from joblib import dump, load
from sklearn.neural_network import MLPClassifier

def make_and_fit_mlp(X, y) -> MLPClassifier:
    return MLPClassifier(
        verbose=True,
        hidden_layer_sizes=(512, 128, 32),
        activation="relu",
        max_iter=100,
        random_state=420,
    ).fit(X, y)

def load_or_fit_mlp_classifier(X, y, output_dir):
    if os.path.exists(f'{output_dir}/class.joblib'):
        clf = load(f'{output_dir}/class.joblib')
    else:
        clf = make_and_fit_mlp(X, y)
        dump(clf, f'{output_dir}/class.joblib')

    return clf