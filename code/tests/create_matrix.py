#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from time import sleep
from typing import Union
import warnings

from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from sklearn.decomposition import PCA 
from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale


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


# from MulticoreTSNE import MulticoreTSNE as TSNE
# from umap import UMAP
import interface.models.sharp as sharp
import interface.models.ssnp as ssnp
import interface.models.nninv as nninv
import interface.utils.metrics as metrics
import interface.utils.scatterplot as scatterplot
from interface.utils.expand_augmentations import expand_projection, repel_particles_all1, repel_particles_all2


tf.random.set_seed(420)


cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("viridis")
x_test_global = None
y_test_global = None

noise = []

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
        # ax.axis("off")

    if figname is not None:
        fig.savefig(figname)

    plt.close("all")
    del fig
    del ax
    
def plot_matrix(classifier, inverter, neighbor_finder, x_data, y_data, noise, nd_data, grid_res, matrix_side_size, matrix_origin, step, format_step, figname=None):
    fig_main, ax_main = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))
    # fig_conf, ax_conf = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_mainconf, ax_mainconf = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_metric, ax_metric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_mainmetric, ax_mainmetric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_glob_metric, ax_glob_metric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_glob_mainmetric, ax_glob_mainmetric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_scatter, ax_scatter = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_scatter3, ax_scatter3 = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))

    # center_coord = matrix_origin + step*np.array(matrix_side_size//2)
    bounding_box = get_bounding_box(x_data)

    # metric_matrix = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    # metric_matrix2 = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    cmapped = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res,4))
    # conf_dbm = np.zeros((grid_res*grid_res,4))
    # ntp_dbm = np.zeros((grid_res*grid_res,4))
    # alpha = np.zeros((np.shape(nd_data)[0],))

    # scatterplot.plot_decision_map_with_points(x_data, cmap(y_data), grid_res, matrix_side_size, fig=ax_scatter)
    # fig_scatter.savefig(f"sharp_scatterplot.png", bbox_inches="tight", pad_inches=0.0)
    for i in range(matrix_side_size):
        for j in range(matrix_side_size):
            # fig_scatter3, ax_scatter3 = plt.subplots(1,1,figsize=(50, 50))
            grid = make_grid(*bounding_box, matrix_origin[0]+i*step[0], matrix_origin[1]+j*step[1], grid_res)
            inverted_grid = inverter.inverse_transform(grid)

            # invp_grid_neighbor_finder = NearestNeighbors(
            #     n_neighbors=5
            # ) 
            # invp_grid_neighbor_finder.fit(inverted_grid)

            classes = classifier.predict(inverted_grid).astype(np.uint8)

            coords = f"({np.round(matrix_origin[0]+i*step[0],np.uint8(format_step[0]))},{np.round(matrix_origin[1]+j*step[1],np.uint8(format_step[1]))})"
            # print(y_data)
            
            n_classes = 10 # im lazy asf
            # metric_matrix[matrix_side_size*i+j] = metrics.metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)
            # values = metrics.metric_distance_to_nearest_neighbor(nd_data, invp_grid_neighbor_finder)

            # for index, value in enumerate(values):
            #     # v = get_normal_dist(value[0],map_extra_coords[0],inv_sqrt_2pi)*get_normal_dist(value[1],map_extra_coords[1],inv_sqrt_2pi)
            #     # # print(v)
            #     # # labels[i,0:3] = v*labels[i,0:3]
            #     # labels[i,3] = (np.exp(v-1)-1*np.exp(-1))*labels[i,3]
            #     alpha[index] = (1/(1+np.exp(3*value-17)))**4#np.exp(-((value-1.1)**2)/12)

            # np.save(f"sharp/dbm_scatter_local/{coords}.npy", alpha)
            
            # scatterplot.plot_decision_map_with_points(classes.reshape((grid_res, grid_res, 1)), x_data, cmap(y_data), grid_res, matrix_side_size, fig=ax_scatter)
            # scatterplot.plot_decision_map_with_points_relative(classes.reshape((grid_res, grid_res, 1)), x_data, cmap(y_data), alpha, grid_res, matrix_side_size, fig=ax_scatter3)

            # ntp_values = metrics.metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)

            cmapped[matrix_side_size*i+j] = cmap(classes)

            # res = classifier.predict_proba(inverted_grid)
     
            # confidence = np.zeros(np.shape(res)[0])

            # for k,lis in enumerate(res):
            #     confidence[k] = np.max(lis)

           
            # fig_scatter.savefig(f"sharp/dbm_scatter/{coords}.png", bbox_inches="tight", pad_inches=0.0)
            # fig_scatter3.savefig(f"sharp/dbm_scatter_local/{coords}.png", bbox_inches="tight", pad_inches=0.0)
            # del invp_grid_neighbor_finder
            ax_main[j,i].imshow(
                cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4)),
                origin="lower",
                interpolation="none",
                resample=False,
            )
            ax_main[j,i].axis("off") 

            print(f"finish {i} {j}")

    plt.subplots_adjust(wspace=0, hspace=0) 
    fig_main.savefig(f"noise/dbm/mat.png", bbox_inches="tight", pad_inches=0.0)
            
            # ax_conf.imshow(
            #     cmap2(confidence).reshape((grid_res, grid_res, 4)),
            #     origin="lower",
            #     interpolation="none",
            #     resample=False,
            # )
            # ax_conf.axis("off") 
            # fig_conf.savefig(f"sharp/conf/{coords}.png", bbox_inches="tight", pad_inches=0.0)
            
            # conf_dbm[:,0] = cmapped[matrix_side_size*i+j,:,0]*confidence
            # conf_dbm[:,1] = cmapped[matrix_side_size*i+j,:,1]*confidence
            # conf_dbm[:,2] = cmapped[matrix_side_size*i+j,:,2]*confidence
            # conf_dbm[:,3] = cmapped[matrix_side_size*i+j,:,3]

            # ax_mainconf.imshow(
            #     conf_dbm.reshape((grid_res, grid_res, 4)),
            #     origin="lower",
            #     interpolation="none",
            #     resample=False,
            # )
            # ax_mainconf.axis("off") 
            # fig_mainconf.savefig(f"sharp/dbm_conf/{coords}.png", bbox_inches="tight", pad_inches=0.0)

            # ax_metric.imshow(
            #     cmap2(1.0-minmax_scale(metric_matrix[matrix_side_size*i+j])).reshape((grid_res, grid_res, 4)),
            #     origin="lower",
            #     interpolation="none",
            #     resample=False,
            # )
            # ax_metric.axis("off") 
            # fig_metric.savefig(f"sharp/ntp/{coords}.png", bbox_inches="tight", pad_inches=0.0)

            # ntp_dbm[:,0] = cmapped[matrix_side_size*i+j,:,0]*(1.0-minmax_scale(metric_matrix[matrix_side_size*i+j]))
            # ntp_dbm[:,1] = cmapped[matrix_side_size*i+j,:,1]*(1.0-minmax_scale(metric_matrix[matrix_side_size*i+j]))
            # ntp_dbm[:,2] = cmapped[matrix_side_size*i+j,:,2]*(1.0-minmax_scale(metric_matrix[matrix_side_size*i+j]))
            # ntp_dbm[:,3] = cmapped[matrix_side_size*i+j,:,3]

            # ax_mainmetric.imshow(
            #     ntp_dbm.reshape((grid_res, grid_res, 4)),
            #     origin="lower",
            #     interpolation="none",
            #     resample=False,
            # )
            # ax_mainmetric.axis("off") 
            # fig_mainmetric.savefig(f"sharp/dbm_ntp/{coords}.png", bbox_inches="tight", pad_inches=0.0)


    # metric_matrix_scaled  = 1.0-minmax_scale(metric_matrix)#-np.min(metric_matrix))/(np.max(metric_matrix)-np.min(metric_matrix))
    # cmapped2  = cmap2(1.0-minmax_scale(metric_matrix))

    # for i in range(matrix_side_size):
    #     for j in range(matrix_side_size):
    #         coords = f"({np.round(matrix_origin[0]+i*step[0],np.uint8(format_step[0]))},{np.round(matrix_origin[1]+j*step[1],np.uint8(format_step[1]))})"

    #         cmapped[matrix_side_size*i+j,:,0] = cmapped[matrix_side_size*i+j,:,0]*metric_matrix_scaled[matrix_side_size*i+j]
    #         cmapped[matrix_side_size*i+j,:,1] = cmapped[matrix_side_size*i+j,:,1]*metric_matrix_scaled[matrix_side_size*i+j]
    #         cmapped[matrix_side_size*i+j,:,2] = cmapped[matrix_side_size*i+j,:,2]*metric_matrix_scaled[matrix_side_size*i+j]
    #         cmapped[matrix_side_size*i+j,:,3] = cmapped[matrix_side_size*i+j,:,3]

    #         ax_glob_metric.imshow(
    #             cmapped2[matrix_side_size*i+j].reshape((grid_res, grid_res, 4)),
    #             origin="lower",
    #             interpolation="none",
    #             resample=False,
    #         )
    #         ax_glob_metric.axis("off") 
    #         fig_glob_metric.savefig(f"sharp/ntp_glob/{coords}.png", bbox_inches="tight", pad_inches=0.0)

    #         ax_glob_mainmetric.imshow(
    #             cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4)),
    #             origin="lower",
    #             interpolation="none",
    #             resample=False,
    #         )
    #         ax_glob_mainmetric.axis("off") 
    #         fig_glob_mainmetric.savefig(f"sharp/dbm_ntp_glob//{coords}.png", bbox_inches="tight", pad_inches=0.0)


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

    train_size = min(int(n_samples * 0.9), 10000)

    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=10000, test_size=10, random_state=420, stratify=y
    )

    if method == "noise":
        # noise = tf.random.stateless_uniform(seed=(420,420), minval=-1, maxval=1, shape=(X_train.shape[0],2))
        pca = PCA(n_components=2)
        noise = pca.fit_transform(X_train)
        noise = minmax_scale(noise, feature_range=(-1,1))
    else:
        noise = tf.zeros((X_train.shape[0],0))

    X_train= np.concatenate((X_train,noise), axis=1)

    _, X_test, _, y_test = train_test_split(
        X_train, y_train, train_size=10, test_size=5000, random_state=420, stratify=y_train
    )
    noise = X_train[:,-2:]
    X_train = X_train[:,:-2]
    noise_test = X_test[:,-2:]
    X_test = X_test[:,:-2]

    neighbor_finder = NearestNeighbors(
        n_neighbors=5
    ) 
    neighbor_finder.fit(X_train)
    
    # n_classes = len(np.unique(y_train))
    # per_class_neighbor_finder = {}
    # for cl in range(n_classes):
    #     neighbor_finder2 = NearestNeighbors(n_neighbors=5)
    #     neighbor_finder2.fit(X_train[y_train == cl])
    #     per_class_neighbor_finder[cl] = neighbor_finder2

    if method == "noise":
        if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib'):
            tsne_proj = load(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
        else:
            tsne_proj = _model.fit_transform(X_train)
            tsne_proj = minmax_scale(tsne_proj, feature_range=(-1,1))
            dump(tsne_proj, f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
        
        if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib'):
            X_model_res = load(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')
        else:
            # X_model_res = tsne_proj
            X_model_res = _model.fit_transform(X_test)
            X_model_res = minmax_scale(X_model_res, feature_range=(-1,1))
            dump(X_model_res, f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')

        if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
            _inv_model.load_weights(os.path.join(output_dir, dataset_name, model_name, method))
        else:
            # noise = X
            _inv_model.fit_random(tsne_proj, X_train, noise, epochs=epochs)
            _inv_model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))

        X_model_2d = X_model_res

    if os.path.exists(f'{output_dir}/{dataset_name}/class.joblib'):
        clf = load(f'{output_dir}/{dataset_name}/class.joblib')
    else:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/class.joblib')

    return X_test, X_model_2d, y_test, noise_test, clf, _inv_model, neighbor_finder

def get_inv_proj_data_sharp(output_dir, _model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    X, y = Load_data(data_dir, d)

    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=6500, test_size=500, random_state=420, stratify=y
    )
    X_train, y_train = include_classes(X_train, y_train, [0,1])
    # print(np.shape(X))
    if method == "noise":
        # noise = tf.random.stateless_uniform(seed=(420,420), minval=-1, maxval=1, shape=(X_train.shape[0],2))
        pca = PCA(n_components=2)
        noise = pca.fit_transform(X_train)
        noise = minmax_scale(noise, feature_range=(-1,1))
        plot(noise, y_train, f"pca_{dataset_name}.png")

    else:
        noise = tf.zeros((X_train.shape[0],0))

    X_train= np.concatenate((X_train,noise), axis=1)

    _, X_test, _, y_test = train_test_split(
        X_train, y_train, train_size=100, test_size=2000, random_state=360, stratify=y_train
    )     
    noise = X_train[:,-2:]
    X_train = X_train[:,:-2]
    noise_test = X_test[:,-2:]
    X_test = X_test[:,:-2]

    neighbor_finder = NearestNeighbors(
        n_neighbors=5
    ) 
    neighbor_finder.fit(X_train)
    
    n_classes = len(np.unique(y_train))
    per_class_neighbor_finder = {}
    for cl in range(n_classes):
        neighbor_finder2 = NearestNeighbors(n_neighbors=5)
        neighbor_finder2.fit(X_train[y_train == cl])
        per_class_neighbor_finder[cl] = neighbor_finder2

    if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
        _model.load_weights(export_path=os.path.join(output_dir, dataset_name, model_name, method))
    else:
        _model.fit(X_train, y_train, noise, epochs=epochs)
        _model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
    X_model_res = _model.transform(X_test)
    X_model_2d = X_model_res
    plot(_model.transform(X_train), y_train, f"sharp_{dataset_name}.png")
    
    if os.path.exists(f'{output_dir}/{dataset_name}/class.joblib'):
        clf = load(f'{output_dir}/{dataset_name}/class.joblib')
    else:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/class.joblib')

    return X_test, X_model_2d, y_test, noise_test, clf, _model, neighbor_finder


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    output_dir = "weights"
    model_name_ops = ["ssnp", "sharp", "nninv"]
    model_name = model_name_ops[1]
    # dataset = "mnist"
    dataset_ops = ["mnist", "fashionmnist", "har", "reuters", "hate_speech"] # 
    dataset = dataset_ops[4]

    method_ops = ["latent_space", "noise"] # , "har", "reuters"
    method = method_ops[1] # , "har", "reuters"
    
    grid_res_ops = [100, 150, 200, 300, 500]
    grid_res = 300#grid_res_ops[3]

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 20
    epochs_dataset["mnist"] = 20
    epochs_dataset["har"] = 20
    epochs_dataset["hate_speech"] = 20
    epochs_dataset["reuters"] = 30
    
    epochs = epochs_dataset[dataset]

    if model_name == "sharp":
        sharp_dims_classes = {}
        sharp_dims_classes["fashionmnist"] = [784, 10]
        sharp_dims_classes["mnist"] = [784, 10]
        sharp_dims_classes["har"]  = [561, 6]
        sharp_dims_classes["reuters"] = [5000, 6]
        sharp_dims_classes["hate_speech"] = [100, 3]

        dims = sharp_dims_classes[dataset][0]
        classes = sharp_dims_classes[dataset][1]

        results_nd, results_2d, y_values, noise, clf, inv_model, neighbor_finder = get_inv_proj_data_sharp( #get_inv_proj_data_sharp(output_dir)
            output_dir, 
            sharp.ShaRP(
                dims,
                classes,
                "diagonal_normal",
                latent_dim= 2,
                variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
                var_leaky_relu_alpha=-0.0001,
                bottleneck_activation="linear",
                bottleneck_l1=0.0,
                bottleneck_l2=0.1,
            ),
            dataset,
            model_name,
            "noise",
            epochs
        )

    if model_name == "ssnp":
        results_nd, results_2d, y_values, noise, clf, inv_model, neighbor_finder = get_inv_proj_data_sharp(
            output_dir, 
            ssnp.SSNP(
                verbose=True,
                latent_dims=(2 if method=="noise" else 4),
                patience=0,
                opt="adam",
                bottleneck_activation="linear",
            ),
            dataset,
            model_name,
            method,
            epochs
        )

    if model_name == "nninv":
        results_nd, results_2d, y_values, noise, clf, inv_model, neighbor_finder = get_inv_proj_data_i(
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

    matrix_size = 9
    if method == "noise":
        matrix_origin = (-1.0,-1.0)

        matrix_step = (0.25,0.25)#(1.0,1.0)#
        # matrix_step = (0.125,0.125) if model_name == "nninv" else (1.0,1.0)

    # print(np.size(np.c_[xx.ravel(), yy.ravel()]))
    format_step =  ((0 if np.floor(np.log10(matrix_step[0])) >= 0 else np.abs(np.floor(np.log10(matrix_step[0])))) +1, 
                   (0 if np.floor(np.log10(matrix_step[1])) >= 0 else np.abs(np.floor(np.log10(matrix_step[1])))) +1) 
    # format_or =  (0 if np.floor(np.log10(matrix_origin[0])) >= 0 else np.abs(np.floor(np.log10(matrix_origin[0]))), 
    #                0 if np.floor(np.log10(matrix_origin[1])) >= 0 else np.abs(np.floor(np.log10(matrix_origin[1]))))
    txt = f"({np.round(matrix_origin[0],np.uint8(format_step[0]))}_{np.round(matrix_origin[1],np.uint8(format_step[1]))})_({np.round(matrix_step[0],np.uint8(format_step[0]))}_{np.round(matrix_step[1],np.uint8(format_step[1]))})"
    # print("here", sliderx_step, slidery_step)
    # if not os.path.exists(f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}"):
    #     os.makedirs(f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}")
    
    fig = plot_matrix(clf, inv_model, neighbor_finder, results_2d, y_values, noise, results_nd, grid_res, matrix_size, matrix_origin, matrix_step, format_step, figname=f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}")
  



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


#DBM_with_images_200_9_(0.0,0.0)_(0.125,0.125)
#DBM_differences_200_9_(0.0,0.0)_(0.125,0.125)
#DBM_nearest_training_point_200_9_(0.0,0.0)_(0.125,0.125)
#DBM_scatterplot_200_9_(0.0,0.0)_(0.125,0.125)
#DBM_200_9_(0.0,0.0)_(0.125,0.125)

#DBM_with_images_200_9_(-1.0,-1.0)_(0.25,0.25)
#DBM_differences_200_9_(-1.0,-1.0)_(0.25,0.25)
#DBM_nearest_training_point_200_9_(-1.0,-1.0)_(0.25,0.25)
#DBM_scatterplot_200_9_(-1.0,-1.0)_(0.25,0.25)
#DBM_200_9_(-1.0,-1.0)_(0.25,0.25)


#nninv 30k training data
#ssnp 5k training data
#sharp 5k training data