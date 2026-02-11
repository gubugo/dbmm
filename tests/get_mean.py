import os
from typing import Union
from joblib import dump, load
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
import tensorflow as tf

from interface.models import nninv, sharp, ssnp
from interface.utils import metrics, scatterplot

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

def plot_matrix(classifier, inverter, neighbor_finder, x_data, y_data, noise, nd_data, grid_res, matrix_side_size, matrix_origin, step, format_step, figname=None):
    fig_main, ax_main = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    fig_conf, ax_conf = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    fig_mainconf, ax_mainconf = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    fig_metric, ax_metric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    fig_mainmetric, ax_mainmetric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    fig_glob_metric, ax_glob_metric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    fig_glob_mainmetric, ax_glob_mainmetric = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_metric2, ax_metric2 = plt.subplots(matrix_side_size,matrix_side_size,figsize=(grid_res/10, grid_res/10))
    fig_scatter, ax_scatter = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_scatter2, ax_scatter2 = plt.subplots(matrix_side_size,matrix_side_size,figsize=(grid_res/10, grid_res/10))
    fig_scatter3, ax_scatter3 = plt.subplots(1,1,figsize=(grid_res/10, grid_res/10))
    # fig_diff, ax_diff = plt.subplots(matrix_side_size,matrix_side_size,figsize=(grid_res/10, grid_res/10))
    # fig_combined, ax_combined = plt.subplots(matrix_side_size,matrix_side_size,figsize=(grid_res/10, grid_res/10))
    
    # print(figname)

    center_coord = matrix_origin + step*np.array(matrix_side_size//2)
    bounding_box = get_bounding_box(x_data)

    metric_matrix = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    metric_matrix2 = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res))
    cmapped = np.zeros((matrix_side_size*matrix_side_size,grid_res*grid_res,4))
    conf_dbm = np.zeros((grid_res*grid_res,4))
    ntp_dbm = np.zeros((grid_res*grid_res,4))
    alpha = np.zeros((np.shape(nd_data)[0],))
    # print(metric_matrix)
    # print(np.size(metric_matrix[0]))

    # scatterplot.plot_decision_map_with_points(x_data, cmap(y_data), grid_res, matrix_side_size, fig=ax_scatter)
    # fig_scatter.savefig(f"sharp_scatterplot.png", bbox_inches="tight", pad_inches=0.0)
    for i in range(matrix_side_size):
        for j in range(matrix_side_size):
            fig_scatter3, ax_scatter3 = plt.subplots(1,1,figsize=(50, 50))
            grid = make_grid(*bounding_box, matrix_origin[0]+i*step[0], matrix_origin[1]+j*step[1], grid_res)
            inverted_grid = inverter.inverse_transform(grid)

            classes = classifier.predict(inverted_grid).astype(np.uint8)

            coords = f"({np.round(matrix_origin[0]+i*step[0],np.uint8(format_step[0]))},{np.round(matrix_origin[1]+j*step[1],np.uint8(format_step[1]))})"
            # print(y_data)
            
            n_classes = 10 # im lazy asf
            metric_matrix[matrix_side_size*i+j] = metrics.metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)

            alpha = np.load(f"sharp/dbm_scatter_local/{coords}.npy")
            
            # scatterplot.plot_decision_map_with_points(classes.reshape((grid_res, grid_res, 1)), x_data, cmap(y_data), grid_res, matrix_side_size, fig=ax_scatter)
            scatterplot.plot_decision_map_with_points_relative(classes.reshape((grid_res, grid_res, 1)), x_data, cmap(y_data), alpha, grid_res, matrix_side_size, fig=ax_scatter3)

            # ntp_values = metrics.metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)

            cmapped[matrix_side_size*i+j] = cmap(classes)

            # res = classifier.predict_proba(inverted_grid)
     
            # confidence = np.zeros(np.shape(res)[0])

            # for k,lis in enumerate(res):
            #     confidence[k] = np.max(lis)

           
            # fig_scatter.savefig(f"sharp/dbm_scatter/{coords}.png", bbox_inches="tight", pad_inches=0.0)
            fig_scatter3.savefig(f"sharp/dbm_scatter_local/{coords}.png", bbox_inches="tight", pad_inches=0.0)

            # ax_main.imshow(
            #     cmapped[matrix_side_size*i+j].reshape((grid_res, grid_res, 4)),
            #     origin="lower",
            #     interpolation="none",
            #     resample=False,
            # )
            # ax_main.axis("off") 
            # fig_main.savefig(f"nninv/dbm/{coords}.png", bbox_inches="tight", pad_inches=0.0)
            
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

            print(f"finish {i} {j}")

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

# @st.cache_resource
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def get_inv_proj_data_sharp(output_dir, _model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    X, y = Load_data(data_dir, d)

    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=10000, test_size=500, random_state=420, stratify=y
    )
    # print(np.shape(X))
    if method == "noise":
        # noise = tf.random.stateless_uniform(seed=(420,420), minval=-1, maxval=1, shape=(X_train.shape[0],2))
        pca = PCA(n_components=2)
        noise = pca.fit_transform(X_train)
        noise = minmax_scale(noise, feature_range=(-1,1))

    else:
        noise = tf.zeros((X_train.shape[0],0))

    X_train= np.concatenate((X_train,noise), axis=1)

    _, X_test, _, y_test = train_test_split(
        X_train, y_train, train_size=100, test_size=5000, random_state=420, stratify=y_train
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

    if os.path.exists(f'{output_dir}/{dataset_name}/class.joblib'):
        clf = load(f'{output_dir}/{dataset_name}/class.joblib')
    else:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/class.joblib')

    return X_test, X_model_2d, y_test, noise_test, clf, _model, neighbor_finder


if __name__ == "__main__":

    output_dir = "weights"
    model_name_ops = ["ssnp", "sharp", "nninv"]
    model_name = model_name_ops[1]
    # dataset = "mnist"
    dataset_ops = ["mnist", "fashionmnist"] # , "har", "reuters"
    dataset = dataset_ops[0]

    method_ops = ["latent_space", "noise"] # , "har", "reuters"
    method = method_ops[1] # , "har", "reuters"
    
    grid_res_ops = [100, 150, 200, 300, 500]
    grid_res = 500#grid_res_ops[3]

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 20
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
            method,
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
  