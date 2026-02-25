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

from code.tests.create_matrix.save_dbm import save_single_dbm, show_single_dbm
from code.training.auto_encoders import load_or_fit_model_ae
from code.training.inv_proj import load_or_fit_model_inv_proj
from code.models.neighborhood.nn import get_nn_model
from code.models.classifiers.MLP import load_or_fit_mlp_classifier
from code.utils.data import get_inv_proj_data_ae, get_inv_proj_data_nninv, train_test_split_augmented

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
import code.models.tensorflow.sharp as sharp
import code.models.tensorflow.ssnp as ssnp
import code.models.tensorflow.nninv as nninv
import code.utils.metrics as metrics
import code.utils.scatterplot as scatterplot
from code.utils.expand_augmentations import expand_projection, repel_particles_all1, repel_particles_all2


tf.random.set_seed(420)


cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("viridis")

def alpha_function(x):
    return (1/(1+np.exp(3*x-17)))**4

def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, v1: float, v2: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )
    
    return np.array([[i[0], i[1], v1, v2] for i in np.c_[xx.ravel(), yy.ravel()]])

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def plot_matrix(classifier, inverter, neighbor_finder, x_data, y_data, noise, nd_data, grid_res, matrix_side_size, matrix_origin, step, format_step, figname=None):
    fig_main, ax_main = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))
    fig_conf, ax_conf = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))
    fig_confdbm, ax_confdbm = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))
    fig_dntp, ax_dntp = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))
    fig_dntpdbm, ax_dntpdbm = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))
    # fig_glob_metric, ax_glob_metric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_glob_mainmetric, ax_glob_mainmetric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    fig_scatter, ax_scatter = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))
    fig_scatteraplha, ax_scatteralpha = plt.subplots(9,9,figsize=(grid_res/10, grid_res/10))

    # center_coord = matrix_origin + step*np.array(matrix_side_size//2)
    bounding_box = get_bounding_box(x_data)

    metric_matrix = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    # metric_matrix2 = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    cmapped = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res,4))
    conf_dbm = np.zeros((grid_res*grid_res,4))
    ntp_dbm = np.zeros((grid_res*grid_res,4))
    alpha = np.zeros((np.shape(nd_data)[0],))

    # scatterplot.plot_decision_map_with_points(x_data, cmap(y_data), grid_res, matrix_side_size, fig=ax_scatter)
    # fig_scatter.savefig(f"sharp_scatterplot.png", bbox_inches="tight", pad_inches=0.0)
    for i in range(matrix_side_size):
        for j in range(matrix_side_size):
            # fig_scatter3, ax_scatter3 = plt.subplots(1,1,figsize=(50, 50))
            grid = make_grid(*bounding_box, matrix_origin[0]+i*step[0], matrix_origin[1]+j*step[1], grid_res)
            inverted_grid = inverter.inverse_transform(grid)

            classes = classifier.predict(inverted_grid).astype(np.uint8)

            coords = f"({np.round(matrix_origin[0]+i*step[0],np.uint8(format_step[0]))},{np.round(matrix_origin[1]+j*step[1],np.uint8(format_step[1]))})"

            cmapped[matrix_side_size*i+j] = cmap(classes)

            # NORMAL MATRIX_DBM
            show_single_dbm(cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4)), i, j, ax_main)


            # SCATTERPLOTS
            show_single_dbm(cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4)), i, j, ax_main)
            scatterplot.plot_decision_map_with_points(classes.reshape((grid_res, grid_res, 1)), x_data, cmap(y_data), grid_res, matrix_side_size, ax=ax_scatter)


            # SCATTERPLOTS ALPHA
            show_single_dbm(cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4)), i, j, ax_scatteralpha)
            invp_grid_neighbor_finder = NearestNeighbors(
                n_neighbors=5
            ) 
            invp_grid_neighbor_finder.fit(inverted_grid)

            values = metrics.metric_distance_to_nearest_neighbor(nd_data, invp_grid_neighbor_finder)
            for index, value in enumerate(values):
                # v = get_normal_dist(value[0],map_extra_coords[0],inv_sqrt_2pi)*get_normal_dist(value[1],map_extra_coords[1],inv_sqrt_2pi)
                # # print(v)
                # # labels[i,0:3] = v*labels[i,0:3]
                # labels[i,3] = (np.exp(v-1)-1*np.exp(-1))*labels[i,3]
                alpha[index] = alpha_function(value)#np.exp(-((value-1.1)**2)/12)

            scatterplot.plot_decision_map_with_points_relative(classes.reshape((grid_res, grid_res, 1)), x_data, cmap(y_data), alpha, grid_res, matrix_side_size, ax=ax_scatteralpha)


            # CLASS CONFIDENCE
            res = classifier.predict_proba(inverted_grid)
     
            confidence = np.zeros(np.shape(res)[0])

            for k,lis in enumerate(res):
                confidence[k] = np.max(lis)

            show_single_dbm(cmap2(confidence).reshape((grid_res, grid_res, 4)), i, j, ax_conf)
            

            # CC ONTOP OF DBM
            conf_dbm[:,0] = cmapped[matrix_side_size*i+j,:,0]*confidence
            conf_dbm[:,1] = cmapped[matrix_side_size*i+j,:,1]*confidence
            conf_dbm[:,2] = cmapped[matrix_side_size*i+j,:,2]*confidence
            conf_dbm[:,3] = cmapped[matrix_side_size*i+j,:,3]

            show_single_dbm(conf_dbm.reshape((grid_res, grid_res, 4)), i, j, ax_confdbm)

            
            # DISTANCE TO NEAREST TRAINING POINT
            metric_matrix[matrix_side_size*i+j] = metrics.metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)

            minmaxed_dntp = 1.0-minmax_scale(metric_matrix[matrix_side_size*i+j])

            show_single_dbm(cmap2(minmaxed_dntp).reshape((grid_res, grid_res, 4)), i, j, ax_dntp)
            
            
            # DNTP ONTOP OF DBM
            ntp_dbm[:,0] = cmapped[matrix_side_size*i+j,:,0]*(minmaxed_dntp)
            ntp_dbm[:,1] = cmapped[matrix_side_size*i+j,:,1]*(minmaxed_dntp)
            ntp_dbm[:,2] = cmapped[matrix_side_size*i+j,:,2]*(minmaxed_dntp)
            ntp_dbm[:,3] = cmapped[matrix_side_size*i+j,:,3]

            show_single_dbm(ntp_dbm.reshape((grid_res, grid_res, 4)), i, j, ax_dntpdbm)



            # np.save(f"sharp/dbm_scatter_local/{coords}.npy", alpha)

            # ntp_values = metrics.metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)

            

            # res = classifier.predict_proba(inverted_grid)
     
            # confidence = np.zeros(np.shape(res)[0])

            # for k,lis in enumerate(res):
            #     confidence[k] = np.max(lis)

           
            # fig_scatter.savefig(f"sharp/dbm_scatter/{coords}.png", bbox_inches="tight", pad_inches=0.0)
            # fig_scatter3.savefig(f"sharp/dbm_scatter_local/{coords}.png", bbox_inches="tight", pad_inches=0.0)
            # del invp_grid_neighbor_finder

            

            print(f"finish {i} {j}")

    plt.subplots_adjust(wspace=0, hspace=0) 
    fig_main.savefig(f"mat.png", bbox_inches="tight", pad_inches=0.0)
            
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
    
    dataset_ops = ["mnist", "fashionmnist", "har", "reuters", "hate_speech"]
    dataset = dataset_ops[0]

    method_ops = ["latent_space", "noise"]
    method = method_ops[1] 
    
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

        results_nd, y_values, noise, results_2d, clf, inv_model, neighbor_finder = get_inv_proj_data_ae( #get_inv_proj_data_sharp(output_dir)
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
        results_nd, y_values, noise, results_2d, clf, inv_model, neighbor_finder = get_inv_proj_data_ae(
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
        results_nd, y_values, noise, results_2d, clf, inv_model, neighbor_finder = get_inv_proj_data_nninv(
            output_dir, 
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

        matrix_step = (0.25,0.25)

    format_step =  ((0 if np.floor(np.log10(matrix_step[0])) >= 0 else np.abs(np.floor(np.log10(matrix_step[0])))) +1, 
                   (0 if np.floor(np.log10(matrix_step[1])) >= 0 else np.abs(np.floor(np.log10(matrix_step[1])))) +1) 

    txt = f"({np.round(matrix_origin[0],np.uint8(format_step[0]))}_{np.round(matrix_origin[1],np.uint8(format_step[1]))})_({np.round(matrix_step[0],np.uint8(format_step[0]))}_{np.round(matrix_step[1],np.uint8(format_step[1]))})"
    
    fig = plot_matrix(clf, inv_model, neighbor_finder, results_2d, y_values, noise, results_nd, grid_res, matrix_size, matrix_origin, matrix_step, format_step, figname=f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}")
  
