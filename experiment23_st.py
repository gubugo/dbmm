#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import sleep
from typing import Union
import warnings

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
from ipycanvas import canvas

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly' 

# from MulticoreTSNE import MulticoreTSNE as TSNE
# from umap import UMAP
import sharp
import ssnp
import nnproj

cmap = plt.get_cmap("tab10")

def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, v1: float, v2: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )

    return np.array([[i[0], i[1], v1, v2] for i in np.c_[xx.ravel(), yy.ravel()]])

def dbm_for_estimator(
    model: MLPClassifier,
    inverter: Union[sharp.ShaRP, ssnp.SSNP],
    bounding_box: tuple[float, float, float, float],
    grid_res: int,
    # ax: Axes,
    v1: float,
    v2: float, 
    cmap=cmap,
):
    grid = make_grid(*bounding_box, v1, v2, grid_res)
    aux = inverter.inverse_transform(grid)
    classes = model.predict(aux).astype(np.uint8)

    cmapped = cmap(classes)*255

    fig = px.imshow(np.reshape(cmapped,(grid_res, grid_res, 4)))
    fig.update_layout(
      hovermode=False,
      xaxis=dict(visible=False),  # Hide x-axis
      yaxis=dict(visible=False),  # Hide y-axis
      margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
    )
    return fig

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def gen_and_save_dbm(
    X_2d: np.ndarray,
    classifier: ClassifierMixin,
    inverter: Union[sharp.ShaRP, ssnp.SSNP],
    output_dir: str,
    grid_res: int,
    dataset_name: str,
    alg_name: str,
    v1: float,
    v2: float,
):
    # fig, ax = plt.subplots(figsize=(20, 20))
    fig = dbm_for_estimator(
        classifier,
        inverter,
        get_bounding_box(X_2d),
        grid_res=grid_res,
        # ax=ax,
        v1=v1,
        v2=v2,
        cmap=cmap if len(classifier.classes_) <= 10 else plt.get_cmap("tab20"),
    )
    # print(fig)
    return fig

def make_and_fit_mlp(X, y) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(512, 128, 32),
        activation="relu",
        max_iter=100,
        random_state=420,
    ).fit(X, y)

def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(20, 20))

    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=[cmap(cl)], label=cl, s=375)
        ax.axis("off")

    if figname is not None:
        fig.savefig(figname)

    plt.close("all")
    del fig
    del ax

@st.cache_resource
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

@st.cache_resource
def get_inv_proj_data_sharp(output_dir):
    verbose = False

    data_dir = "../data/"

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 10
    epochs_dataset["har"] = 10
    epochs_dataset["hatespeech"] = 20
    epochs_dataset["reuters"] = 10

    dataset_name = "mnist"
    d = dataset_name

    X, y = Load_data(data_dir, d)

    n_samples = X.shape[0]

    train_size = min(int(n_samples * 0.9), 5000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=360, stratify=y
    )

    epochs = epochs_dataset[dataset_name]

    print(X.shape[1], len(np.unique(y)))
    sharp_model = sharp.ShaRP(
        X.shape[1],
        len(np.unique(y)),
        "diagonal_normal",
        latent_dim=4,
        variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
        var_leaky_relu_alpha=-0.0001,
        bottleneck_activation="linear",
        bottleneck_l1=0.0,
        bottleneck_l2=0.1,
    )


    sharp_model.fit(X_train, y_train, epochs=15)
    sharp_model.save_weights(os.path.join(output_dir, dataset_name, "sharp"))


    clf = make_and_fit_mlp(X_train, y_train)
    dump(clf, f'{output_dir}/{dataset_name}/sharp.joblib')


    X_sharp_gt = sharp_model.transform(X_test)

    #Outros metodos (alem do sharp e outras distribuições possiveis), em vez de usar 4 dimensoes latentes, adicionar 2 valores aleatorios, outros datasets
    #Gera os valores aleatorios com o tamanho dos dados de treino, e mandar como parametro a para gerar o modelo, para assim nao gerar valores aleatorios a cada epoch
    #Dropdown para o tipo de projecao e para a resolusao, e um canvas para alterar a posicao do ponto no espaco latente, ao inves de 2 sliders
    #Fazer um git
    # pca = PCA()
    # plot(pca.fit_transform(X=X_sharp_gt, y=y_test), y_test, figname="pca.png")
    # plot(np.array([[i[0], i[1]] for i in X_sharp_gt]), y_test, figname="01.png")
    # plot(np.array([[i[0], i[2]] for i in X_sharp_gt]), y_test, figname="02.png")
    # plot(np.array([[i[0], i[3]] for i in X_sharp_gt]), y_test, figname="03.png")
    # plot(np.array([[i[1], i[2]] for i in X_sharp_gt]), y_test, figname="12.png")
    # plot(np.array([[i[1], i[3]] for i in X_sharp_gt]), y_test, figname="13.png")
    # plot(np.array([[i[2], i[3]] for i in X_sharp_gt]), y_test, figname="23.png")

    X_sharp_res = np.array([[i[0], i[1]] for i in X_sharp_gt])

    _, _, z_min, w_min = np.round(X_sharp_gt.min(axis=0),2)
    _, _, z_max, w_max = np.round(X_sharp_gt.max(axis=0),2)

    return X_sharp_res, clf, sharp_model, [float(z_min), float(z_max), float(w_min), float(w_max)]

@st.cache_resource
def get_inv_proj_data(output_dir, _model, dataset_name, model_name):
    verbose = False

    data_dir = "../data/"
    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 10
    epochs_dataset["har"] = 10
    epochs_dataset["hatespeech"] = 20
    epochs_dataset["reuters"] = 10

    # dataset_name = "mnist"
    d = dataset_name
    epochs = epochs_dataset[dataset_name]

    X, y = Load_data(data_dir, d)

    n_samples = X.shape[0]

    train_size = min(int(n_samples * 0.9), 5000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=360, stratify=y
    )

    try:
        _model.load_weights(os.path.join(output_dir, dataset_name, model_name))
    except:
        _model.fit(X_train, y_train, epochs=epochs)
        _model.save_weights(os.path.join(output_dir, dataset_name, model_name))
    
    try:
        clf = load(f'{output_dir}/{dataset_name}/{model_name}.joblib')
    except:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/{model_name}.joblib')

    X_model_res = _model.transform(X_test)
    X_model_2d = np.array([[i[0], i[1]] for i in X_model_res])

    _, _, z_min, w_min = np.round(X_model_res.min(axis=0),2)
    _, _, z_max, w_max = np.round(X_model_res.max(axis=0),2)

    return X_model_2d, clf, _model, [float(z_min), float(z_max), float(w_min), float(w_max)]

def run_sharp_dbm():
    output_dir = "results_inverse"
    sharp_dims_classes = {}
    sharp_dims_classes["fashionmnist"] = [784, 10]# errado
    sharp_dims_classes["mnist"] = [784, 10]
    sharp_dims_classes["har"]  = [784, 10]# errado
    sharp_dims_classes["hatespeech"] = [784, 10]# errado
    sharp_dims_classes["reuters"] = [784, 10] # errado
    dims = sharp_dims_classes["mnist"][0]
    classes = sharp_dims_classes["mnist"][1]

    X_sharp_2d, clf, sharp_model, limits = get_inv_proj_data( #get_inv_proj_data_sharp(output_dir)
        output_dir, 
        sharp.ShaRP(
            dims,
            classes,
            "laplace",
            latent_dim=4,
            variational_layer_kwargs=dict(prior_scale=0.1),
            var_leaky_relu_alpha=-0.0001,
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.5,
        ),
        "mnist",
        "sharp"
    )

    x = st.sidebar.slider('x', limits[0], limits[1], 0.02)
    y = st.sidebar.slider('y', limits[2], limits[3], 0.02)

    grid_res = st.sidebar.selectbox(
    "Image Resolution",
    (50, 75, 100, 150, 200),
    )

    fig = gen_and_save_dbm(X_sharp_2d, clf, sharp_model, output_dir, grid_res, "mnist", "sharp", x, y)
    st.plotly_chart(fig, use_container_width=True)

def run_ssnp_dbm():
    grid_res = 75
    output_dir = "results_inverse"

    X_ssnp_2d, clf, ssnp_model, limits = get_inv_proj_data(
        output_dir, 
        ssnp.SSNP(
            verbose=False,
            latent_dims=4,
            patience=0,
            opt="adam",
            bottleneck_activation="linear",
        ),
        "mnist",
        "ssnp"
    )

    x = st.sidebar.slider('x', limits[0], limits[1], 0.02)
    y = st.sidebar.slider('y', limits[2], limits[3], 0.02)

    canvas = st.canvas(size=(200, 200))
    # bg = load_image('test.png')
    canvas.draw_image([1,1,1,1], 50, 50)

    canvas.fill_rect(0, 0, 50, 50)

    def handle_mouse_down(x, y):
        print("im here")
        pass
    canvas.on_mouse_down(handle_mouse_down)

    fig = gen_and_save_dbm(X_ssnp_2d, clf, ssnp_model, output_dir, grid_res, "mnist", "ssnp", x, y)
    st.plotly_chart(fig, use_container_width=True)

    

# def run_nnp_tsne_dbm():
#     grid_res = 75
#     output_dir = "results_inverse"
#     X_ssnp_gt, X_ssnp_2d, clf, ssnp_model, limits = get_inv_proj_data_nnp_tsne(output_dir)

#     x = st.sidebar.slider('x', limits[0], limits[1], 0.02)
#     y = st.sidebar.slider('y', limits[2], limits[3], 0.02)

#     fig = gen_and_save_dbm(X_ssnp_2d, clf, ssnp_model, output_dir, grid_res, "mnist", "ssnp", x, y)
#     # fig.show()
#     st.plotly_chart(fig, use_container_width=True)

# run_ssnp_dbm()
run_sharp_dbm()