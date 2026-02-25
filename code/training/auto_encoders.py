import os


def load_or_fit_model_ae(X, y, aug, X_test, output_dir, _model, dataset_name, model_name, method, epochs):

    if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
        _model.load_weights(export_path=os.path.join(output_dir, dataset_name, model_name, method))
    else:
        _model.fit(X, y, aug, epochs=epochs)
        _model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
    projected_data = _model.transform(X_test)
    
    return projected_data, _model
