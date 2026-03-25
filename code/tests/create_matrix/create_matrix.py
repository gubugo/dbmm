#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings

import tensorflow as tf

from code.tests.create_matrix import save_dbm_matrix
from code.utils.data import get_inv_proj_data_ae, get_inv_proj_data_nninv
from code.utils.utils import plot

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from MulticoreTSNE import MulticoreTSNE as TSNE
# from umap import UMAP
# import code.models.tensorflow.sharp as sharp
import code.models.pytorch.sharp_4d as sharp
import code.models.pytorch.ssnp as ssnp
import code.models.pytorch.nninv as nninv

tf.random.set_seed(420)

cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("viridis")

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

def plot_matrix_separated(classifier, inverter, neighbor_finder, x_data, y_data, noise, nd_data, grid_res, matrix_side_size, matrix_origin, step, format_step, figname=None):
    fig, ax = plt.subplots()
    # center_coord = matrix_origin + step*np.array(matrix_side_size//2)
    bounding_box = get_bounding_box(x_data)

    dntp_matrix = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    cmapped = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res,4))

    dbm_saver = save_dbm_matrix.dbm_saver()

    for i in range(matrix_side_size):
        for j in range(matrix_side_size):
            coords = f"{np.round(matrix_origin[0]+i*step[0],np.uint8(format_step[0]))},{np.round(matrix_origin[1]+j*step[1],np.uint8(format_step[1]))}"
            
            grid = make_grid(*bounding_box, matrix_origin[0]+i*step[0], matrix_origin[1]+j*step[1], grid_res)
            inverted_grid = inverter.inverse_transform(grid)
            classes = classifier.predict(inverted_grid).astype(np.uint8)

            cmapped[matrix_side_size*i+j] = cmap(classes)
            dbm = cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4))

            # NORMAL MATRIX_DBM
            dbm_saver.show_dbm(dbm, ax)
            fig.savefig(f"results/dbm/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()

            # SCATTERPLOTS
            dbm_saver.show_dbm_scatterplot(dbm, x_data, cmap(y_data), grid_res, ax)
            fig.savefig(f"results/scatter/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()

            # SCATTERPLOTS ALPHA
            dbm_saver.show_dbm_local_scatterplot(dbm, inverted_grid, nd_data, x_data, cmap(y_data), grid_res, ax)
            fig.savefig(f"results/scatterlocal/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()
            
            # CLASS CONFIDENCE
            dbm_saver.show_class_confidence_map(classifier, inverted_grid, grid_res, ax)
            fig.savefig(f"results/cc/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()
            
            # CC ONTOP OF DBM
            dbm_saver.show_class_confidence_dbm(dbm, classifier, inverted_grid, grid_res, ax)
            fig.savefig(f"results/cc_dbm/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()

            # DISTANCE TO NEAREST TRAINING POINT
            dbm_saver.show_distance_to_nearest_training_point_map(neighbor_finder, inverted_grid, grid_res, ax)
            fig.savefig(f"results/dntp/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()

            # DNTP ONTOP OF DBM
            dbm_saver.show_distance_to_nearest_training_point_dbm(dbm, neighbor_finder, inverted_grid, grid_res, ax)
            fig.savefig(f"results/dntp_dbm/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()

            dntp_matrix[matrix_side_size*i+j] = dbm_saver.dntp_map

            dbm_saver.flush_variables()
            print(f"finish {i} {j}")

    for i in range(matrix_side_size):
        for j in range(matrix_side_size):
            coords = f"{np.round(matrix_origin[0]+i*step[0],np.uint8(format_step[0]))},{np.round(matrix_origin[1]+j*step[1],np.uint8(format_step[1]))}"
            
            # GLOBAL DNTP
            dbm_saver.show_global_distance_to_nearest_training_point_map(dntp_matrix, matrix_side_size, grid_res, ax)
            fig.savefig(f"results/g_dntp/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()

            # GLOBAL DNTP DBM
            dbm_saver.show_global_distance_to_nearest_training_point_dbm(cmapped, dntp_matrix, matrix_side_size, grid_res, ax)
            fig.savefig(f"results/g_dntp_dbm/({coords}).png", bbox_inches="tight", pad_inches=0.0)
            ax.clear()

    plt.close("all")


def plot_matrix_whole(classifier, inverter, neighbor_finder, x_data, y_data, noise, nd_data, grid_res, matrix_side_size, matrix_origin, step, format_step, figname=None):
    fig_main, ax_main = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_conf, ax_conf = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_confdbm, ax_confdbm = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_dntp, ax_dntp = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_dntpdbm, ax_dntpdbm = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_glob_dntp, ax_glob_dntp = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_glob_dntpdbm, ax_glob_dntpdbm = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_scatter, ax_scatter = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))
    fig_scatteraplha, ax_scatteralpha = plt.subplots(matrix_side_size, matrix_side_size,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))

    # fig, ax = plt.subplots(1, 1,gridspec_kw={'wspace': 0.0, 'hspace': 0.0},figsize=(10, 10))

    # center_coord = matrix_origin + step*np.array(matrix_side_size//2)
    bounding_box = get_bounding_box(x_data)

    dntp_matrix = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    cmapped = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res,4))

    dbm_saver = save_dbm_matrix.dbm_saver()

    for i in range(matrix_side_size):
        for j in range(matrix_side_size):
            grid = make_grid(*bounding_box, matrix_origin[0]+i*step[0], matrix_origin[1]+j*step[1], grid_res)
            inverted_grid = inverter.inverse_transform(grid)

            # ax.imshow(inverted_grid[0].reshape((28,28)), cmap="gray")
            # fig.savefig(f"Test{i}{j}.png")
            classes = classifier.predict(inverted_grid).astype(np.uint8)

            cmapped[matrix_side_size*i+j] = cmap(classes)
            dbm = cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4))

            # NORMAL MATRIX_DBM
            dbm_saver.show_dbm(dbm, ax_main[j,i])

            # SCATTERPLOTS
            dbm_saver.show_dbm_scatterplot(dbm, x_data, cmap(y_data), grid_res, ax_scatter[j,i])

            # SCATTERPLOTS ALPHA
            # dbm_saver.show_dbm_local_scatterplot(dbm, inverted_grid, nd_data, x_data, cmap(y_data), grid_res, ax_scatteralpha[j,i])

            # CLASS CONFIDENCE
            # dbm_saver.show_class_confidence_map(classifier, inverted_grid, grid_res, ax_conf[j,i])

            # CC ONTOP OF DBM
            # dbm_saver.show_class_confidence_dbm(dbm, classifier, inverted_grid, grid_res, ax_confdbm[j,i])
            
            # DISTANCE TO NEAREST TRAINING POINT
            # dbm_saver.show_distance_to_nearest_training_point_map(neighbor_finder, inverted_grid, grid_res, ax_dntp[j,i])
            
            # DNTP ONTOP OF DBM
            # dbm_saver.show_distance_to_nearest_training_point_dbm(dbm, neighbor_finder, inverted_grid, grid_res, ax_dntpdbm[j,i])
            
            # dntp_matrix[matrix_side_size*i+j] = dbm_saver.dntp_map

            dbm_saver.flush_variables()
            print(f"finish {i} {j}")

    # for i in range(matrix_side_size):
    #     for j in range(matrix_side_size):
    #         # GLOBAL DNTP
    #         dbm_saver.show_global_distance_to_nearest_training_point_map(dntp_matrix, matrix_side_size*i+j, grid_res, ax_glob_dntp[j,i])

    #         # GLOBAL DNTP DBM
    #         dbm_saver.show_global_distance_to_nearest_training_point_dbm(cmapped, dntp_matrix, matrix_side_size*i+j, grid_res, ax_glob_dntpdbm[j,i])

    plt.subplots_adjust(wspace=0, hspace=0)
    fig_main.savefig(f"results/dbm/matrix.png", bbox_inches="tight", pad_inches=0.0)
    # fig_conf.savefig(f"results/cc/matrix.png", bbox_inches="tight", pad_inches=0.0)
    # fig_confdbm.savefig(f"results/cc_dbm/matrix.png", bbox_inches="tight", pad_inches=0.0)
    # fig_dntp.savefig(f"results/dntp/matrix.png", bbox_inches="tight", pad_inches=0.0)
    # fig_dntpdbm.savefig(f"results/dntp_dbm/matrix.png", bbox_inches="tight", pad_inches=0.0)
    fig_scatter.savefig(f"results/scatter/matrix.png", bbox_inches="tight", pad_inches=0.0)
    # fig_scatteraplha.savefig(f"results/scatterlocal/matrix.png", bbox_inches="tight", pad_inches=0.0)
    # fig_glob_dntpdbm.savefig(f"results/g_dntp_dbm/matrix.png", bbox_inches="tight", pad_inches=0.0)
    # fig_glob_dntp.savefig(f"results/g_dntp/matrix.png", bbox_inches="tight", pad_inches=0.0)
    
    plt.close("all")

if __name__ == "__main__":

    # gpus = tf.config.list_physical_devices('GPU')
    # print(gpus)
    # if gpus:
    # # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             gpus[0],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)

    output_dir = "weights/pytorch"
    model_name_ops = ["ssnp", "sharp", "nninv"]
    model_name = model_name_ops[1]
    
    dataset_ops = ["mnist", "fashionmnist", "har", "reuters", "hate_speech"]
    dataset = dataset_ops[0]

    method_ops = ["latent_space", "noise"]
    method = method_ops[1] 

    classifier_names = ["mlp", "svc, random_forest", "nb_gaussian"]
    classifier_name = "mlp"
    
    grid_res_ops = [100, 150, 200, 300, 500]
    grid_res = 100#grid_res_ops[3]

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
            # sharp.ShaRP(
            #     dims,
            #     classes,
            #     "diagonal_normal",
            #     latent_dim= 2,
            #     variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
            #     var_leaky_relu_alpha=-0.0001,
            #     bottleneck_activation="linear",
            #     bottleneck_l1=0.0,
            #     bottleneck_l2=0.1,
            # ),
            sharp.ShaRP(
                dims,
                2,
                classes,
                "diagonal_normal",
                bottleneck_activation="linear",
                variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
            ),
            dataset,
            model_name,
            classifier_name,
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
    
    fig = plot_matrix_whole(clf, inv_model, neighbor_finder, results_2d, y_values, noise, results_nd, grid_res, matrix_size, matrix_origin, matrix_step, format_step, figname=f"./matrices/matrices_{model_name}_{method}_{grid_res}_{matrix_size}_{txt}")
  
