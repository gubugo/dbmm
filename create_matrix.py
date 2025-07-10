#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from time import sleep
from typing import Union
import warnings

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
from MulticoreTSNE import MulticoreTSNE as TSNE


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# from MulticoreTSNE import MulticoreTSNE as TSNE
# from umap import UMAP
import sharp
import ssnp
import nninv

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
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    bounding_box: tuple[float, float, float, float],
    grid_res: int,
    ax: Axes,
    v1: float,
    v2: float, 
    cmap=cmap,
):
    grid = make_grid(*bounding_box, v1, v2, grid_res)
    aux = inverter.inverse_transform(grid)

    classes = model.predict(aux).astype(np.uint8)

    cmapped = cmap(classes)
    ax.imshow(
        cmapped.reshape((grid_res, grid_res, 4)),
        origin="lower",
        interpolation="none",
        resample=False,
    )
    ax.axis("off")

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def gen_and_save_dbm(
    X_2d: np.ndarray,
    classifier: ClassifierMixin,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    output_dir: str,
    grid_res: int,
    dataset_name: str,
    alg_name: str,
    v1: float,
    v2: float,
):
    fig, ax = plt.subplots(5,5,figsize=(20, 20))
    fig = dbm_for_estimator(
        classifier,
        inverter,
        get_bounding_box(X_2d),
        grid_res=grid_res,
        ax=ax,
        v1=v1,
        v2=v2,
        cmap=cmap if len(classifier.classes_) <= 10 else plt.get_cmap("tab20"),
    )
    # print(fig)
    return fig

def make_and_fit_mlp(X, y) -> MLPClassifier:
    return MLPClassifier(
        verbose=True,
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

def plot_matrix(classifier, inverter, bounding_box, grid_res, matrix_side_size, matrix_origin, format_or, step, format_step, figname=None):
    fig, ax = plt.subplots(matrix_side_size,matrix_side_size,figsize=(grid_res/10, grid_res/10))
    # print(figname)

    xx, yy = np.meshgrid(
        np.linspace(limits[0], limits[1], matrix_size), np.linspace(limits[2], limits[3], matrix_size)
    )
    values = np.c_[xx.ravel(), yy.ravel()]

    for i in range(matrix_side_size):
        for j in range(matrix_side_size):
            grid = make_grid(*bounding_box, matrix_origin[0]+i*step[0], matrix_origin[1]+j*step[1], grid_res)
            aux = inverter.inverse_transform(grid)

            classes = classifier.predict(aux).astype(np.uint8)

            cmapped = cmap(classes)

            # fig.subplot(matrix_side_size, matrix_side_size, (j+1)+(i)*matrix_side_size)
            ax[i,j].imshow(
                cmapped.reshape((grid_res, grid_res, 4)),
                origin="lower",
                interpolation="none",
                resample=False,
            )

            
            coords = f"({np.round(matrix_origin[0]+i*step[0],np.uint8(format_step[0]))},{np.round(matrix_origin[1]+j*step[1],np.uint8(format_step[1]))})"
            ax[i,j].axis("off") 
            ax[i,j].set_title(coords, fontsize=grid_res/(2*matrix_side_size), x=0.5, y=1-5/grid_res) 
            plt.subplots()
            plt.imshow(
                cmapped.reshape((grid_res, grid_res, 4)),
                origin="lower",
                interpolation="none",
                resample=False,
            )
            plt.axis("off")
            plt.title(coords, fontsize="medium", x=0.5, y=1)
            coords = coords.replace(",","_")
            plt.savefig(os.path.join(figname,f"{coords}.png"))
    #plt.show()
    
    fig.savefig(f"{figname}.png")

    plt.close("all")
    """
    for i, index in enumerate(values):
        grid = make_grid(*bounding_box, i[0], i[1], grid_res)
        aux = inverter.inverse_transform(grid)

        classes = classifier.predict(aux).astype(np.uint8)

        cmapped = cmap(classes)

        # fig.subplot(matrix_side_size, matrix_side_size, (j+1)+(i)*matrix_side_size)
        ax[index//matrix_side_size,index%matrix_side_size].imshow(
            cmapped.reshape((grid_res, grid_res, 4)),
            origin="lower",
            interpolation="none",
            resample=False,
        )
        
        coords = f"({np.round(i[0],np.uint8(format_step[0]))},{np.round(i[1],np.uint8(format_step[1]))})"
        ax[index//matrix_side_size,index%matrix_side_size].axis("off") 
        ax[index//matrix_side_size,index%matrix_side_size].set_title(coords, fontsize=grid_res/10, x=0.5, y=1-5/grid_res) 
        plt.subplots()
        plt.imshow(
            cmapped.reshape((grid_res, grid_res, 4)),
            origin="lower",
            interpolation="none",
            resample=False,
        )
        plt.axis("off")
        plt.title(coords, fontsize="medium", x=0.5, y=1)
        coords = coords.replace(",","_")
        plt.savefig(os.path.join(figname,f"{coords}.png"))
    #plt.show()
    
    fig.savefig(f"{figname}.png")

    plt.close("all")
    """

# @st.cache_resource
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def get_inv_proj_data(output_dir, _model, _inv_model, dataset_name, model_name, method, epochs):
    verbose = False

    data_dir = "./data/"
    # dataset_name = "mnist"
    d = dataset_name

    X, y = Load_data(data_dir, d)

    # print(X.shape)
    # print(y.shape)

    n_samples = X.shape[0]

    train_size = min(int(n_samples * 0.9), 5000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=10, stratify=y
    )

    # print(X_train.shape, X_test.shape, y_train.shape)
    # if method == "noise":
    
    print(os.path.abspath(os.path.join(output_dir, dataset_name, model_name, method)))
    
    # ugh refactor this ugly shit
    if method == "noise":
        noise = tf.random.Generator.from_seed(360).normal(stddev=1, shape=(X_train.shape[0],2))
    else:
        noise = tf.zeros((X_train.shape[0],0))
    # print(noise)
    if model_name != "nninv":
        ## SSNP and ShaRP
        if method == "noise":
            try:
                _model.load_weights(export_path=os.path.join(output_dir, dataset_name, model_name, method))
            except:
                _model.fit(X_train, y_train, noise, epochs=epochs)
                _model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
            X_model_res = _model.transform(X_test)
            X_model_2d = X_model_res
            z_min, w_min = -2, -2
            z_max, w_max =  2,  2
        else:
            try:
                _model.load_weights(export_path=os.path.abspath(os.path.join(output_dir, dataset_name, model_name, method)))
            except:
                _model.fit(X_train, y_train, noise, epochs=epochs)
                _model.save_weights(export_path=os.path.abspath(os.path.join(output_dir, dataset_name, model_name, method)))

            X_model_res = _model.transform(X_test)
            X_model_2d = np.array([[i[2], i[3]] for i in X_model_res])
            z_min, w_min, _, _ = np.round(X_model_res.min(axis=0),2)
            z_max, w_max, _, _ = np.round(X_model_res.max(axis=0),2)
    else:
        ## NNInv
        if method == "noise":
            try:
                X_model_res = load(f'{output_dir}/{dataset_name}/tsneData2d.joblib')
            except:
                X_model_res = _model.fit_transform(X_test)
                dump(X_model_res, f'{output_dir}/{dataset_name}/tsneData2d.joblib')
            try:
                _inv_model.load_weights(os.path.join(output_dir, dataset_name, model_name, method))
            except:
                print(X_model_res.shape)
                _inv_model.fit_random(X_model_res, X_test, epochs=epochs)
                _inv_model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
            
            X_model_2d = X_model_res
            z_min, w_min = -2, -2
            z_max, w_max =  2,  2

        else:
            try:
                X_model_res = load(f'{output_dir}/{dataset_name}/tsneData4d.joblib')
            except:
                X_model_res = _model.fit_transform(X_test)
                dump(X_model_res, f'{output_dir}/{dataset_name}/tsneData4d.joblib')
            try:
                _inv_model.load_weights(os.path.join(output_dir, dataset_name, model_name, method))
            except:
                _inv_model.fit(X_model_res, X_test, epochs=epochs)
                _inv_model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
            
            X_model_2d = np.array([[i[2], i[3]] for i in X_model_res])
            z_min, w_min, _, _ = np.round(X_model_res.min(axis=0),2)
            z_max, w_max, _, _ = np.round(X_model_res.max(axis=0),2)

    try:
        clf = load(f'{output_dir}/{dataset_name}/class.joblib')
    except:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/class.joblib')

    return X_model_2d, clf, _model, _inv_model, [float(z_min), float(z_max), float(w_min), float(w_max)]

if __name__ == "__main__":

    output_dir = "weights"
    model_name_ops = ["ssnp", "sharp", "nninv"]
    model_name = model_name_ops[0]
    # dataset = "mnist"
    dataset_ops = ["mnist", "fashionmnist"] # , "har", "reuters"
    dataset = dataset_ops[0]

    method_ops = ["latent_space", "noise"] # , "har", "reuters"
    method = method_ops[0] # , "har", "reuters"
    
    grid_res_ops = [100, 150, 200, 300, 500]
    grid_res = grid_res_ops[2]

    matrix_size = 9

    matrix_origin = (-2.0,-2.0)

    matrix_step = (0.5,0.5)

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 10
    epochs_dataset["har"] = 10
    epochs_dataset["hatespeech"] = 20
    epochs_dataset["reuters"] = 30
    
    epochs = epochs_dataset[dataset]

    if model_name == "sharp":
        sharp_dims_classes = {}
        sharp_dims_classes["fashionmnist"] = [784, 10]
        sharp_dims_classes["mnist"] = [784, 10]
        sharp_dims_classes["har"]  = [561, 6]
        sharp_dims_classes["reuters"] = [5000, 6]

        dims = sharp_dims_classes[dataset][0]
        classes = sharp_dims_classes[dataset][1]

        results_2d, clf, inv_model, _, limits = get_inv_proj_data( #get_inv_proj_data_sharp(output_dir)
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
                latent_dim= (2 if method=="noise" else 4),
                variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
                var_leaky_relu_alpha=-0.0001,
                bottleneck_activation="linear",
                bottleneck_l1=0.0,
                bottleneck_l2=0.1,
            ),
            {},
            dataset,
            model_name,
            method,
            epochs
        )

    if model_name == "ssnp":
        results_2d, clf, inv_model, _, limits = get_inv_proj_data(
            output_dir, 
            ssnp.SSNP(
                verbose=True,
                latent_dims=(2 if method=="noise" else 4),
                patience=0,
                opt="adam",
                bottleneck_activation="linear",
            ),
            {},
            dataset,
            model_name,
            method,
            epochs
        )

    if model_name == "nninv":
        results_2d, clf, _, inv_model, limits = get_inv_proj_data(
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

        # model.fit(X=)

    
    # print(np.size(np.c_[xx.ravel(), yy.ravel()]))
    format_step =  (0 if np.floor(np.log10(matrix_step[0])) >= 0 else np.abs(np.floor(np.log10(matrix_step[0]))), 
                   0 if np.floor(np.log10(matrix_step[1])) >= 0 else np.abs(np.floor(np.log10(matrix_step[1]))))
    format_or =  (0 if np.floor(np.log10(matrix_origin[0])) >= 0 else np.abs(np.floor(np.log10(matrix_origin[0]))), 
                   0 if np.floor(np.log10(matrix_origin[1])) >= 0 else np.abs(np.floor(np.log10(matrix_origin[1]))))
    txt = f"({np.round(matrix_origin[0],np.uint8(format_or[0]))}_{np.round(matrix_origin[1],np.uint8(format_or[1]))})_({np.round(matrix_step[0],np.uint8(format_step[0]))}_{np.round(matrix_step[1],np.uint8(format_step[1]))})"
    # print("here", sliderx_step, slidery_step)
    if not os.path.exists(f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}"):
        os.makedirs(f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}")
    fig = plot_matrix(clf, inv_model, get_bounding_box(results_2d), grid_res, matrix_size, matrix_origin, format_or, matrix_step, format_step, figname=f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}")
  



"""

1. define projection+inverse projection methods (SSNP, ShaRP, t-SNE+NNInv)
2. define augmentation method (4D proj or noise addition)
3. define classifier, we're using only a MLP one
4. define grid resolution (n x n, with n being an integer), normally something like 100x100, 200x200, 500x500
5. define matrix size, odd squared number preferably (using 5x5)
6. define matrices origin, and x and y steps
example: start in position (0,0), with steps 0.1 and 0.2. With a 5x5 matrix, we would create 25 dbms, with augmented dimesions domain: [0, 0.1, 0.2, 0.3, 0.4]x[0, 0.2, 0.4, 0.6, 0.8]

"""

#ponto d treino mais próximo 

#projetar os pontos em cima do DBM