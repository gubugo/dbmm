import os
from joblib import dump, load

from code.models.classifiers.MLP import make_and_fit_mlp
from code.models.classifiers.SVC import make_and_fit_svc
from code.models.classifiers.naive_bayes import make_and_fit_nbg
from code.models.classifiers.random_forest import make_and_fit_rf

def create_and_fit_classifier(X, y, name):
    if name == "mlp":
        return make_and_fit_mlp(X, y)
    elif name == "random_forest":
        return make_and_fit_rf(X, y)
    elif name == "svc":
        return make_and_fit_svc(X, y)
    elif name == "nb_gaussian":
        return make_and_fit_nbg(X, y)
    

def load_or_fit_classifier(X, y, name, output_dir):
    if os.path.exists(f'{output_dir}/{name}.joblib'):
        clf = load(f'{output_dir}/{name}.joblib')
    else:
        clf = create_and_fit_classifier(X, y, name)
        dump(clf, f'{output_dir}/{name}.joblib')

    return clf