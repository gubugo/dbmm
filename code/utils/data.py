
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import tensorflow as tf

from code.models.neighborhood.nn import get_nn_model
from code.models.classifiers.classifiers import load_or_fit_classifier
from code.training.auto_encoders import load_or_fit_model_ae
from code.training.inv_proj import load_or_fit_model_inv_proj

def include_classes(X_data, y_data, classes):
    if len(classes) == 0:
        return X_data, y_data
    
    X_train2 = []
    y_train2 = []
    
    for i in range(np.shape(X_data)[0]):
        if y_data[i] in classes:
            X_train2.append(X_data[i,:])
            y_train2.append(y_data[i])

    return np.array(X_train2), np.array(y_train2)

def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def get_dimensions_and_class(dataset_name):
    dims_classes = {}
    dims_classes["fashionmnist"] = [784, 10]
    dims_classes["mnist"] = [784, 10]
    dims_classes["har"]  = [561, 6]
    dims_classes["reuters"] = [5000, 6]
    dims_classes["hate_speech"] = [100, 3]

    dims = dims_classes[dataset_name][0]
    classes = dims_classes[dataset_name][1]

    return dims, classes

def train_test_split_augmented(X, y, method, train_size=6500, test_size=2000, random_state=420):
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=train_size, test_size=500, random_state=random_state, stratify=y
    )
    # X_train, y_train = include_classes(X_train, y_train, [0,1])

    if method == "noise":
        # noise = tf.random.stateless_uniform(seed=(420,420), minval=-1, maxval=1, shape=(X_train.shape[0],2))
        pca = PCA(n_components=2)
        noise = pca.fit_transform(X_train)
        noise = minmax_scale(noise, feature_range=(-1,1))
        
    else:
        noise = tf.zeros((X_train.shape[0],0))

    X_train= np.concatenate((X_train,noise), axis=1)

    _, X_test, _, y_test = train_test_split(
        X_train, y_train, train_size=100, test_size=test_size, random_state=random_state, stratify=y_train
    )     
    noise_train = X_train[:,-2:]
    X_train = X_train[:,:-2]
    noise_test = X_test[:,-2:]
    X_test = X_test[:,:-2]

    return X_train, y_train, noise_train, X_test, y_test, noise_test

def get_inv_proj_data_nninv(output_dir, _inv_model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    X, y = Load_data(data_dir, dataset_name)

    X_train, y_train, noise_train, X_test, y_test, noise_test = train_test_split_augmented(X, y, method, train_size=6500, test_size=2000, random_state=420)

    neighbor_finder = get_nn_model(X, 5)

    X_proj, _inv_model = load_or_fit_model_inv_proj(X_train, y_train, noise_train, X_test, _inv_model, output_dir, dataset_name, model_name, method, epochs)
    clf = load_or_fit_classifier(X_train, y_train, "mlp", f'{output_dir}/{dataset_name}')

    return X_test, y_test, noise_test, X_proj, clf, _inv_model, neighbor_finder


def get_inv_proj_data_ae(output_dir, _model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    X, y = Load_data(data_dir, dataset_name)

    X_train, y_train, noise_train, X_test, y_test, noise_test = train_test_split_augmented(X, y, method, train_size=6500, test_size=2000, random_state=420)

    neighbor_finder = get_nn_model(X, 5)

    X_proj, _model = load_or_fit_model_ae(X_train, y_train, noise_train, X_test, output_dir, _model, dataset_name, model_name, method, epochs)
    clf = load_or_fit_classifier(X_train, y_train, "mlp", f'{output_dir}/{dataset_name}')

    # plot(_model.transform(X_train), y_train, f"sharp_{dataset_name}.png") # DEBUG

    return X_test, y_test, noise_test, X_proj, clf, _model, neighbor_finder