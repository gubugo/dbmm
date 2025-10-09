#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import warnings

from sklearn.decomposition import PCA 
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import minmax_scale, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import keras
import tensorflow as tf
from keras import backend as K
from keras.initializers import Constant
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from tensorflow.keras import datasets as kdatasets

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
# from MulticoreTSNE import MulticoreTSNE as TSNE

tf.random.set_seed(420)

class NNInv:
    def __init__(
        self,
        init=PCA(n_components=2),
        # size="medium",
        # style="bottleneck",
        loss="mean_squared_error",
        opt=keras.optimizers.Adam(learning_rate=0.001),
        l1=0.0,
        l2=0.01,
        dropout=False,
        latent_dims=2,
        verbose=1,
        **kwargs,
    ):
        self.verbose = verbose
        self.init = init
        self.dropout = dropout
        self.opt = opt
        # self.epochs = epochs
        self.loss = loss
        self.l1 = l1
        self.l2 = l2
        self.latent_dims = latent_dims

        self.inv = None
        self.is_fitted = False
        K.clear_session()

    def fit(self, X, y=None, epochs=300, **kwargs):
        main_input = Input(shape=(self.latent_dims,), name="main_input")
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l1",
        )(main_input)
        
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l2",
        )(x)
        
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l3",
        )(x)
        
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l4",
        )(x)
        
        x = Dense(
            y.shape[1],
            activation="sigmoid",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="output",
        )(x)

        self.model = Model(inputs=main_input, outputs=x)

        self.model.summary()

        self.model.compile(loss=self.loss, optimizer=self.opt)

        self.model.fit(
            X,
            y,
            batch_size=128,
            epochs=epochs,
            verbose=self.verbose,
            validation_split=0.05,
            # callbacks=self.callbacks,
            **kwargs,
        )

        encoded_input = Input(shape=(self.latent_dims,))
        l = self.model.get_layer("l1")(encoded_input)
        l = self.model.get_layer("l2")(l)
        l = self.model.get_layer("l3")(l)
        l = self.model.get_layer("l4")(l)
        decoder_layer = self.model.get_layer("output")(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.is_fitted = True

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception("Model not trained. Call fit() before calling transform()")

    def inverse_transform(self, X):
        
        if self._is_fit():
            return self.inv.predict(X)
        
    def save_weights(self, export_path: str):
        # Route `save_weights` to specific models.
        self.inv.save(os.path.join(export_path, "inv"))


    def load_weights(self, export_path: str):
        # Same for `load_weights`
        self.is_fitted = True
        self.inv = load_model(os.path.join(export_path, "inv"))

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


# @st.cache_resource
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def get_inv_proj_data_i(output_dir, _model, _inv_model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    (X, y), (_, _) = kdatasets.mnist.load_data()
    X = MinMaxScaler().fit_transform(X.reshape((-1, 28 * 28)).astype("float32"))
    y = LabelEncoder().fit_transform(y)
    # X, y = fetch_openml("mnist_784", as_frame=False, return_X_y=True, parser='pandas')
    # X = minmax_scale(X.astype(np.float32))
    # y = LabelEncoder().fit_transform(y)
    # X, y = Load_data(data_dir, d)

    n_samples = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, test_size=1250, random_state=420, stratify=y
    )

    tsne_proj = _model.fit_transform(X_train)

    X_model_res = tsne_proj

    _inv_model.fit(tsne_proj, y=X_train, epochs=epochs)

    X_model_2d = X_model_res

    clf = make_and_fit_mlp(X_train, y_train)

    return X_model_2d, X_train, y_train, clf, _inv_model

if __name__ == "__main__":

    output_dir = "weights"
    model_name = "nninv"
    # dataset = "mnist"
    dataset_ops = ["mnist", "fashionmnist"] # , "har", "reuters"
    dataset = dataset_ops[0]

    method = "noise" # , "har", "reuters"
    
    grid_res = 200

    if model_name == "nninv":
        results_2d, X_values, y_values, clf, inv_model = get_inv_proj_data_i(
            output_dir, 
            TSNE(
                n_jobs=4, 
                random_state=420, 
                n_components=2
            ),
            NNInv(
                latent_dims=2
            ),
            dataset,
            model_name,
            method,
            300
        )

    fig = plot_matrix(clf, inv_model, results_2d, y_values, grid_res, figname=f"./matrices/matrices_{model_name}_{method}_{grid_res}_1")
  
    fig, axes = plt.subplots(2, 5, subplot_kw={'box_aspect': 1})

    for i, (proj_i, train_i) in enumerate(zip(results_2d[:5], X_values[:5])):
        axes[0,i].imshow(train_i.reshape((28, 28)))
        axes[1,i].imshow(inv_model.inverse_transform(proj_i[None,:]).reshape((28,28)))

    fig.savefig(f"matrices/examples.png")
