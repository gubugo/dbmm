import os

from joblib import dump, load

from code.models.projection.tsne import load_or_project_tsne


def load_or_fit_model_inv_proj(X_train, y_train, aug, X_test, _inv_model, output_dir, dataset_name, model_name, method, epochs):

    dir = f'{output_dir}/{dataset_name}'

    proj_train, proj_test = load_or_project_tsne(X_train, X_test, dir, method)

    if os.path.exists(os.path.join(dir, model_name, method)):
        _inv_model.load_weights(os.path.join(dir, model_name, method))
    else:
        _inv_model.fit_random(proj_train, X_train, aug, epochs=epochs)
        _inv_model.save_weights(os.path.join(dir, model_name, method))

    return proj_test, _inv_model
