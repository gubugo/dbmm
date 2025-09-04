#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import warnings

from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from sklearn.decomposition import PCA 
from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import nninv

tf.random.set_seed(420)


cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("viridis")

def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )
    
    return np.c_[xx.ravel(), yy.ravel()]

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def make_and_fit_mlp(X, y) -> MLPClassifier:
    return MLPClassifier(
        verbose=True,
        hidden_layer_sizes=(512, 128, 32),
        activation="relu",
        max_iter=100,
        random_state=420,
    ).fit(X, y)

def plot_matrix(classifier, inverter, x_data, y_data, grid_res, figname=None):
    fig_main, ax_main = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))

    bounding_box = get_bounding_box(x_data)

    grid = make_grid(*bounding_box, grid_res)
    inverted_grid = inverter.inverse_transform(grid)

    classes = classifier.predict(inverted_grid).astype(np.uint8)

    cmapped = cmap(classes)
    
    ax_main.imshow(
        cmapped.reshape((grid_res, grid_res, 4)),
        origin="lower",
        interpolation="none",
        resample=False,
    )
    
    # ax_main.axis("off") 
    ax_main.set_title("0,0", fontsize=grid_res/(2*400), x=0.5, y=1-5/grid_res) 
 
    fig_main.savefig(f"{figname}.png")
    plt.close("all")


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

# @st.cache_resource
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def get_inv_proj_data_i(output_dir, _model, _inv_model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    X, y = Load_data(data_dir, d)

    n_samples = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=50000, test_size=1250, random_state=420, stratify=y
    )

    X_train, y_train = include_classes(X_train, y_train, [1,2])
    X_test, y_test = include_classes(X_test, y_test, [1,2])

    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib'):
        tsne_proj = load(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    else:
        tsne_proj = _model.fit_transform(X_train)
        dump(tsne_proj, f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    
    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib'):
        X_model_res = load(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')
    else:
        X_model_res = _model.fit_transform(X_test)
        dump(X_model_res, f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')

    if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
        _inv_model.load_weights(os.path.join(output_dir, dataset_name, model_name, method))
    else:
        _inv_model.fit_random(tsne_proj, y=X_train, epochs=epochs)
        _inv_model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
    
    X_model_2d = X_model_res

    if os.path.exists(f'{output_dir}/{dataset_name}/class.joblib'):
        clf = load(f'{output_dir}/{dataset_name}/class.joblib')
    else:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/class.joblib')

    return X_model_2d, y_test, clf, _inv_model

if __name__ == "__main__":

    output_dir = "weights"
    model_name = "nninv"
    # dataset = "mnist"
    dataset_ops = ["mnist", "fashionmnist"] # , "har", "reuters"
    dataset = dataset_ops[0]

    method = "noise" # , "har", "reuters"
    
    grid_res = 200

    if model_name == "nninv":
        results_2d, y_values, clf, inv_model = get_inv_proj_data_i(
            output_dir, 
            TSNE(
                n_jobs=4, 
                random_state=420, 
                n_components=(2 if method=="noise" else 4)
            ),
            nninv.NNInv(
                latent_dims=(2 if method=="noise" else 4)
            ),
            dataset,
            model_name,
            method,
            300
        )

    fig = plot_matrix(clf, inv_model, results_2d, y_values, grid_res, figname=f"./matrices/matrices_{model_name}_{method}_{grid_res}_1")
  

