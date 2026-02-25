import os

from joblib import dump, load


def load_or_fit_model_inv_proj(X, y, aug, X_train, output_dir, _model, _inv_model, dataset_name, model_name, method, epochs):

    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib'):
        tsne_proj = load(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    else:
        tsne_proj = _model.fit_transform(X)
        dump(tsne_proj, f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    
    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib'):
        projected_data = load(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')
    else:
        projected_data = _model.fit_transform(X_train)
        dump(projected_data, f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')

    if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
        _inv_model.load_weights(os.path.join(output_dir, dataset_name, model_name, method))
    else:
        _inv_model.fit_random(tsne_proj, X, aug, epochs=epochs)
        _inv_model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))

    return projected_data, _inv_model, [-1.0, 1.0, -1.0, 1.0]
