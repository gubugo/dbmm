#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import subprocess



warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# from ipycanvas import canvas

import streamlit as st
import plotly.io as pio
pio.templates.default = 'plotly' 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# from MulticoreTSNE import MulticoreTSNE as TSNE
# from umap import UMAP
from utils.augmentations import get_augmentation_pca
from utils.dbm import gen_and_save_dbm
import models.sharp as sharp
import models.ssnp as ssnp
import models.nninv as nninv

from training.classifier import load_or_fit_mlp_classifier
from training.auto_encoders import load_or_fit_model_ae
from training.inv_proj import load_or_fit_model_inv_proj

cmap = plt.get_cmap("tab10")

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

# @st.cache_resource
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

@st.cache_resource
def get_inv_proj_data_ae(output_dir, _model, dataset_name, model_name, method, epochs, random_state):
    data_dir = "./data/"
    X, y = Load_data(data_dir, dataset_name)

    n_samples = X.shape[0]
    train_size = min(int(n_samples * 0.9), 5000)

    X, _, y, _ = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    _, X_test, _, y_test = train_test_split(
        X, y, train_size=int(train_size*0.9), random_state=random_state, stratify=y
    )

    augmentation = get_augmentation_pca(X)

    X_proj, _model, limits = load_or_fit_model_ae(X, y, augmentation, X_test, output_dir, _model, dataset_name, model_name, method, epochs)
    classifier = load_or_fit_mlp_classifier(X, y, f'{output_dir}/{dataset_name}')
    return X_proj, _model, classifier, limits

@st.cache_resource
def get_inv_proj_data_mlp(output_dir, _model, _inv_model, dataset_name, model_name, method, epochs, random_state):
    data_dir = "./data/"
    X, y = Load_data(data_dir, dataset_name)

    n_samples = X.shape[0]
    train_size = min(int(n_samples * 0.9), 30000)

    X, _, y, _ = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    _, X_test, _, y_test = train_test_split(
        X, y, train_size=int(train_size*0.9), random_state=random_state, stratify=y
    )

    augmentation = get_augmentation_pca(X)

    X_proj, _inv_model, limits = load_or_fit_model_inv_proj(X, y, augmentation, X_test, output_dir, _model, _inv_model, dataset_name, model_name, method, epochs, random_state)
    classifier = load_or_fit_mlp_classifier(X, y, f'{output_dir}/{dataset_name}')
    return X_proj, _inv_model, classifier, limits

if __name__ == "__main__":

    python_executable = sys.executable
    if not os.path.exists("data"):
        subprocess.run([python_executable, "get_data.py"])

    output_dir = "weights"
    # model_name = st.sidebar.selectbox("Inverse Projection Method", ("ssnp", "sharp", "nninv"))
    # dataset = "mnist"
    dataset = st.sidebar.selectbox("Dataset Used", ("mnist", "fashionmnist")) # , "har", "reuters"

    # method = st.sidebar.selectbox("Training Method Used", ("latent_space", "noise")) # , "har", "reuters"
    
    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 10
    epochs_dataset["har"] = 10
    epochs_dataset["hatespeech"] = 20
    epochs_dataset["reuters"] = 30
    
    epochs = epochs_dataset[dataset]

    sharp_dims_classes = {}
    sharp_dims_classes["fashionmnist"] = [784, 10]
    sharp_dims_classes["mnist"] = [784, 10]
    sharp_dims_classes["har"]  = [561, 6]
    sharp_dims_classes["reuters"] = [5000, 6]

    dims = sharp_dims_classes[dataset][0]
    classes = sharp_dims_classes[dataset][1]

    results_2d, inv_model, clf, limits = get_inv_proj_data_ae( #get_inv_proj_data_sharp(output_dir)
        output_dir, 
        sharp.ShaRP(
            # dims,
            # classes,
            # "laplace",
            # latent_dim=4,
            # variational_layer_kwargs=dict(prior_scale=0.1),
            # var_leaky_relu_alpha=-0.0001,
            # bottleneck_activation="linear",
            # bottleneck_l1=0.0,
            # bottleneck_l2=0.5,
            dims,
            classes,
            "diagonal_normal",
            latent_dim=2,
            variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
            var_leaky_relu_alpha=-0.0001,
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.1,
        ),
        dataset,
        "sharp",
        "pca",
        epochs,
        420
    )

        # model.fit(X=)
    sliderx_step = np.floor(np.log10(limits[1]-limits[0]))-2
    slidery_step = np.floor(np.log10(limits[3]-limits[2]))-2

    # print("here", sliderx_step, slidery_step)
    x = st.sidebar.slider('x', limits[0], limits[1], 0.0, step=10**(sliderx_step), format=f"%0.{np.array([np.abs(sliderx_step)], dtype=np.int8)[0]}f")
    y = st.sidebar.slider('y', limits[2], limits[3], 0.0, step=10**(slidery_step), format=f"%0.{np.array([np.abs(slidery_step)], dtype=np.int8)[0]}f")

    grid_res = st.sidebar.selectbox("DBM Resolution", (50, 75, 100, 150, 200))

    fig = gen_and_save_dbm(results_2d, clf, inv_model, output_dir, grid_res, "reuters", "pca", x, y)
    st.plotly_chart(fig, use_container_width=True)




        
        # canvas = st.canvas(size=(200, 200))
        # # bg = load_image('test.png')
        # canvas.draw_image([1,1,1,1], 50, 50)

        # canvas.fill_rect(0, 0, 50, 50)

        # def handle_mouse_down(x, y):
        #     print("im here")
        #     pass
        # canvas.on_mouse_down(handle_mouse_down)
