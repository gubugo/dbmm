import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_or_fit_model_ae(X, y, aug, X_train, output_dir, _model, dataset_name, model_name, method, epochs):

    if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
        _model.load_weights(export_path=os.path.join(output_dir, dataset_name, model_name, method))
    else:
        _model.fit(X, y, aug, epochs=epochs)
        _model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
    projected_data = _model.transform(X_train)
    
    return projected_data, _model, [-1.0, 1.0, -1.0, 1.0]
