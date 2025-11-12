import os
import sys
import time
from joblib import dump, load
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from numba import njit
import skdim
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
from joblib import Parallel, delayed
import torch
torch.backends.cuda.preferred_linalg_library("magma")

import interface.models.sharp as sharp
import interface.models.sharp_og as sharp_og
import interface.models.nninv as nninv

def compute_knn_indices(X, k):
    n_samples = X.shape[0]
    if k >= n_samples:
        k = n_samples - 1
        print(f"[warning] k ajustado para {k} pois era >= n_samples")

    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    knn_indices = indices[:, 1:] 
    return knn_indices


# CPU THREADING
def local_id_from_covariances(X, knn_indices, theta=0.95):
    # n_samples = covariances.shape[0]
    n_samples = X.shape[0]
    di_list = np.zeros(n_samples)

    # pega o rage, faz uma lista, separa a lista e passa como argumento pras threads...?
    di_list = Parallel(n_jobs=-3)(delayed(get_intrinsic_dimension_pca)(i, X, knn_indices, theta) for i in range(n_samples))

    return di_list

def get_intrinsic_dimension_pca(i, X, knn_indices, theta):
    S = X[knn_indices[i]]
    N_samples, N_features = S.shape
    pca = PCA(n_components=min(N_samples, N_features))
    pca.fit(S)
    eigenvalues = pca.explained_variance_
    eigenvals = np.sort(eigenvalues)[::-1]
    total = np.sum(eigenvals)
    if total == 0:
        return 0
    eigenvals = eigenvals / total
    cumulative = np.cumsum(eigenvals)
    return np.searchsorted(cumulative, theta) + 1

# BASE IMPLEMENTATION
# def local_id_from_covariances(X, knn_indices, theta):
#     n_samples = knn_indices.shape[0]
#     di_list = np.empty(n_samples, dtype=np.int32)
#     for i in range(n_samples):
#         S = X[knn_indices[i]]
#         S_mean = np.mean(S)
#         S_centered = S - S_mean
#         cov = (S_centered.T @ S_centered) / S.shape[0]  # same as np.cov(rowvar=False, ddof=0)
#         eigenvals = np.linalg.eigvalsh(cov)
#         eigenvals = eigenvals[::-1]  # descending order
#         total = np.sum(eigenvals)
#         if total == 0:
#             di_list[i] = 0
#             continue
#         eigenvals /= total
#         cumulative = np.cumsum(eigenvals)
#         d_i = np.searchsorted(cumulative, theta) + 1
#         di_list[i] = d_i
#     d_avg = np.mean(di_list)
#     return di_list, d_avg

def get_intrinsic_dimension(X, k, theta):
    knn_idx = compute_knn_indices(X, k)
    # covariancias = neighborhood_covariances()
    di_list = local_id_from_covariances(X, knn_idx, theta)
    d_avg = np.mean(di_list)
    return d_avg

def get_intrinsic_dimension_sv(index, X, k, theta):
    knn_idx = compute_knn_indices(X, k)
    # covariancias = neighborhood_covariances()
    di_list = get_intrinsic_dimension_pca(index, X, knn_idx, theta)
    d_avg = np.mean(di_list)
    return d_avg

cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("viridis")

def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, v1: float, v2: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )
    
    return np.array([[i[0], i[1], v1, v2] for i in np.c_[xx.ravel(), yy.ravel()]])

def make_grid_normal(
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

def plot_matrix(classifier, inverter, nd_data, x_data, y_data, noise, grid_res, matrix_side_size, matrix_origin, step, format_step, figname=None):

    grid_res = 50
    
    X_size = np.shape(x_data)[0]

    bounding_box = get_bounding_box(x_data)
    
    print("computing...")
    
    # # id_nd_data = get_intrinsic_dimension(nd_data, 120, 0.95)
    # # print(f"id of nd data: {id_nd_data}")
    
    # # id_2d_data = get_intrinsic_dimension(x_data, 120, 0.95)
    # # print(f"id of 2d data: {id_2d_data}") 
    
    # # concat = np.concatenate((x_data,noise), axis=1)
    # # id_4d_data = get_intrinsic_dimension(concat, 120, 0.95)
    # # print(f"id of 4d data: {id_4d_data}") 

    # # ip_data = inverter.inverse_transform(concat) # concat
    # # id_ndd_data = get_intrinsic_dimension(ip_data, 120, 0.95)
    # # print(f"id of nd' data: {id_ndd_data}") 

    # grid = make_grid(*bounding_box, 0.0, 0.0, grid_res)
    # dbm00_data = inverter.inverse_transform(grid)
    # id_dbm00_data = get_intrinsic_dimension(dbm00_data, 120, 0.95)
    # print(f"id of dbm00: {id_dbm00_data}")
    
    k = 120
    pixel_width = 51#np.ceil(np.sqrt(X_size))
    # pixel_width = np.uint16(pixel_width+1 if pixel_width//2 == 0 else pixel_width)
    
    half_pixel_width = pixel_width//2
    theta = 0.95
    index_coord = 0#pixel_width**2//2

    dist = 12 #6 #22 
    coords_grid = make_grid_normal(*bounding_box, dist)
    c0 = coords_grid[65]#14#130##
    c1 = coords_grid[78]#21#153##

    normal_grid = make_grid_normal(c0[0], c1[0], c0[1], c1[1], grid_res)
    normal_grid_ = make_grid(c0[0], c1[0], c0[1], c1[1], -1.0, -1.0, grid_res)
    # normal_grid = make_grid_normal(*bounding_box, grid_res)
    # normal_grid_ = make_grid(*bounding_box, -1.0, -1.0, grid_res)
    invp_grid_base = inverter.inverse_transform(normal_grid_)
    #(1/21)*
    rngex = (1/11)*(np.array(list(range(pixel_width))*pixel_width)-half_pixel_width)/half_pixel_width
    rngey = (1/11)*(np.array(sorted(list(range(pixel_width))*pixel_width))-half_pixel_width)/half_pixel_width
    matrix_values = np.c_[rngex, rngey]
    di_list = np.ones(np.shape(normal_grid)[0])
    print(np.shape(di_list))

    for index,i in enumerate(normal_grid):
        start_time = time.perf_counter()
        grid = [[i[0], i[1], j[0], j[1]] for j in matrix_values]
        invp_grid = np.concatenate((inverter.inverse_transform(grid),invp_grid_base))
        # print(np.shape(invp_grid))
        id_pixelv = get_intrinsic_dimension_sv(index_coord, invp_grid, k, theta)
        di_list[index] = id_pixelv
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        if not bool(index % 10):
            print(f"done pixel: {index}")


    # for index,i in enumerate(normal_grid):
    #     pixelv = [[i[0], i[1], j[0], j[1]] for j in matrix_values]
    #     invp_pixelv = inverter.inverse_transform(pixelv)
    #     start_time = time.perf_counter()
    #     id_pixelv = get_intrinsic_dimension(invp_pixelv, k, theta)
    #     di_list[index] = id_pixelv
    #     end_time = time.perf_counter()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time: {elapsed_time:.4f} seconds")
    #     if not bool(index % 10):
    #         print(f"done pixel: {index}")
    


    fig_id, ax_id = plt.subplots(1,1,figsize=(200,200))
    np.save(f'my_array({pixel_width})_{k}_{index_coord}_central_11.npy', di_list)
    max_v = np.max(di_list)
    min_v = np.min(di_list)
    # for index, i in enumerate(di_list):
    #     if i != 6.0 and i != 7.0 and i != 8.0:
    #         print(f"{index}: {i}")
    # di_list = (di_list - min_v)/(max_v-min_v)
    
    ax_id.imshow(
        di_list.reshape((grid_res, grid_res,1)),
        cmap="viridis",
        interpolation="none",
        resample=False,
    )

    # coords_grid = minmax_scale(coords_grid)
    # ax_id.scatter(
    #     100*coords_grid[:,0],100*coords_grid[:,1], 1600, "black"
    # )
    print(max_v)
    print(min_v)
    fig_id.savefig(f"DBM_PIXEL({pixel_width})_{k}_{index_coord}_central_11.png")

def plot_matrix_true(clf, inverter, nd_data, x_data, y_data, noise, grid_res, matrix_side_size, matrix_origin, step, format_step, figname=None):

    grid_res = 40
    
    X_size = np.shape(x_data)[0]

    bounding_box = get_bounding_box(x_data)
    
    print("computing...")
    k = 120
    pixel_width = 41
    half_pixel_width = pixel_width//2
    theta = 0.95
    matrix_side = 5
    half_matrix_side = matrix_side//2

    res_id =  np.zeros((matrix_side,matrix_side,grid_res,grid_res))
    res_dbm = np.zeros((matrix_side,matrix_side,grid_res*grid_res,4))

    step_x =             (pixel_width-1)//(matrix_side-1)
    step_y = pixel_width*(pixel_width-1)//(matrix_side-1)

    cmap_dbm = plt.get_cmap('tab10')
    
    for x in range(matrix_side):
        for y in range(matrix_side):
            index_coord = x*step_x+y*step_y#pixel_width**2//2

            normal_grid = make_grid_normal(*bounding_box, grid_res)
            normal_grid_ = make_grid(*bounding_box, (x-half_matrix_side)/half_matrix_side, (y-half_matrix_side)/half_matrix_side, grid_res)
            invp_grid_base = inverter.inverse_transform(normal_grid_)
            
            #(1/21)*
            rngex = (np.array(list(range(pixel_width))*pixel_width)-half_pixel_width)/half_pixel_width
            rngey = (np.array(sorted(list(range(pixel_width))*pixel_width))-half_pixel_width)/half_pixel_width
            matrix_values = np.c_[rngex, rngey]
            di_list = np.ones(np.shape(normal_grid)[0])

            # for index,i in enumerate(normal_grid):
            #     start_time = time.perf_counter()
            #     grid = [[i[0], i[1], j[0], j[1]] for j in matrix_values]
            #     invp_grid = np.concatenate((inverter.inverse_transform(grid),invp_grid_base))
            #     # print(np.shape(invp_grid))
            #     id_pixelv = get_intrinsic_dimension_sv(index_coord, invp_grid, k, theta)
            #     di_list[index] = id_pixelv
            #     end_time = time.perf_counter()
            #     elapsed_time = end_time - start_time
            #     print(f"Elapsed time: {elapsed_time:.4f} seconds")
            #     if not bool(index % 10):
            #         print(f"done pixel: {index}")
            res_dbm[x,y] = cmap_dbm(clf.predict(invp_grid_base).astype(np.uint8))
            res_id[x,y] = di_list.reshape((grid_res, grid_res))
            print(f"ready {x},{y}")

    # fig_id, ax_id = plt.subplots(matrix_side,matrix_side,figsize=(100,100))
    fig_dbm, ax_dbm = plt.subplots(matrix_side,matrix_side,figsize=(100,100))
    
    # np.save(f'matrix({pixel_width})_{k}_{matrix_side}.npy', res_id)
    # max_v = np.max(res_id)
    # min_v = np.min(res_id)
    # cmap = plt.get_cmap('jet', int(max_v-min_v+1))
    
    # images = []
    for x in range(matrix_side):
        for y in range(matrix_side):
            # images.append(
            #     ax_id[x,y].imshow(
            #         res_id[x,y].reshape((grid_res, grid_res,1)),
            #         interpolation="none",
            #         resample=False,
            #         cmap=cmap,
            #         vmin=min_v,
            #         vmax=max_v,
            #     )
            # )
            # ax_id[x,y].set_title(f"{(x-half_matrix_side)/half_matrix_side}, {(y-half_matrix_side)/half_matrix_side}", fontsize=200)
            # ax_id[x,y].axis("off")
            ax_dbm[x,y].imshow(
                res_dbm[x,y].reshape((grid_res, grid_res,4)),
                interpolation="none",
                resample=False,
            )
            ax_dbm[x,y].set_title(f"{(x-half_matrix_side)/half_matrix_side}, {(y-half_matrix_side)/half_matrix_side}", fontsize=100)
            ax_dbm[x,y].axis("off")
             

    # bounds = np.arange(int(max_v-min_v+2)) + min_v - 0.5 # Boundaries for each color segment
    # norm = BoundaryNorm(bounds, cmap.N)
    # cbar = fig_id.colorbar(images[0], ax=ax_id, orientation='horizontal', fraction=0.05, norm=norm, boundaries=bounds, ticks=list(range(int(min_v), int(max_v+1))))
    # cbar.ax.tick_params(labelsize=200)
    
    # print(max_v)
    # print(min_v)
    fig_dbm.savefig(f"matrix_dbm({pixel_width})_{k}_{matrix_side}.png")
    # fig_id.savefig(f"matrix_id({pixel_width})_{k}_{matrix_side}.png")


def make_and_fit_mlp(X, y) -> MLPClassifier:
    return MLPClassifier(
        verbose=True,
        hidden_layer_sizes=(512, 128, 32),
        activation="relu",
        max_iter=100,
        random_state=420,
    ).fit(X, y)

def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def get_inv_proj_data_i_wo_augmentation(output_dir, _model, _inv_model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    X, y = Load_data(data_dir, d)

    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=30000, test_size=10, random_state=420, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_train, y_train, train_size=10, test_size=5000, random_state=420, stratify=y_train
    )

    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib'):
        tsne_proj = load(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    else:
        tsne_proj = _model.fit_transform(X_train)
        tsne_proj = minmax_scale(tsne_proj)
        dump(tsne_proj, f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    
    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib'):
        X_model_res = load(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')
    else:
        # X_model_res = tsne_proj
        X_model_res = _model.fit_transform(X_test)
        X_model_res = minmax_scale(X_model_res)
        dump(X_model_res, f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')

    if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
        _inv_model.load_weights(os.path.join(output_dir, dataset_name, model_name, method))
    else:
        # noise = X
        _inv_model.fit(tsne_proj, X_train, epochs=epochs)
        _inv_model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
        
    X_model_2d = X_model_res

    if os.path.exists(f'{output_dir}/{dataset_name}/class.joblib'):
        clf = load(f'{output_dir}/{dataset_name}/class.joblib')
    else:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/class.joblib')

    return X_test, X_model_2d, y_test, [], clf, _inv_model

def get_inv_proj_data_i(output_dir, _model, _inv_model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    X, y = Load_data(data_dir, d)

    n_samples = X.shape[0]

    train_size = min(int(n_samples * 0.9), 10000)

    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=30000, test_size=10, random_state=420, stratify=y
    )

    pca = PCA(n_components=2)
    noise = pca.fit_transform(X_train)
    noise = minmax_scale(noise, feature_range=(-1,1))

    X_train= np.concatenate((X_train,noise), axis=1)

    _, X_test, _, y_test = train_test_split(
        X_train, y_train, train_size=10, test_size=5000, random_state=420, stratify=y_train
    )
    noise = X_train[:,-2:]
    X_train = X_train[:,:-2]
    noise_test = X_test[:,-2:]
    X_test = X_test[:,:-2]

    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib'):
        tsne_proj = load(f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    else:
        tsne_proj = _model.fit_transform(X_train)
        tsne_proj = minmax_scale(tsne_proj)
        dump(tsne_proj, f'{output_dir}/{dataset_name}/tsneData2d_train.joblib')
    
    if os.path.exists(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib'):
        X_model_res = load(f'{output_dir}/{dataset_name}/tsneData2d_test.joblib')
    else:
        # X_model_res = tsne_proj
        X_model_res = _model.fit_transform(X_test)
        X_model_res = minmax_scale(X_model_res)
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

    return X_test, X_model_2d, y_test, noise_test, clf, _inv_model

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

    return X_test, X_model_2d, y_test, noise_test, clf, _model


if __name__ == "__main__":

    output_dir = "weights"
    model_name = "sharp"
    dataset_ops = ["mnist", "fashionmnist"] 
    dataset = dataset_ops[0]
    method = "noise"
    
    grid_res = 100

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 20
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
    noise = []
    if model_name == "sharp":
        results_nd, results_2d, y_values, noise, clf, inv_model = get_inv_proj_data_sharp(
            output_dir, 
            sharp.ShaRP(
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
            model_name,
            method,
            epochs
        )
    if model_name == "nninv":#noise,
        results_nd, results_2d, y_values, noise, clf, inv_model = get_inv_proj_data_i_wo_augmentation(
            output_dir, 
            TSNE(
                n_jobs=4, 
                random_state=420, 
                n_components=2
            ),
            nninv.NNInv(
                latent_dims=2
            ),
            dataset,
            model_name,
            method,
            300
        )

    matrix_size = 5
    matrix_origin = (-1.0,-1.0)
    matrix_step = (0.5,0.5)

    format_step =  ((0 if np.floor(np.log10(matrix_step[0])) >= 0 else np.abs(np.floor(np.log10(matrix_step[0])))) +1, 
                   (0 if np.floor(np.log10(matrix_step[1])) >= 0 else np.abs(np.floor(np.log10(matrix_step[1])))) +1) 
    
    txt = f"({np.round(matrix_origin[0],np.uint8(format_step[0]))}_{np.round(matrix_origin[1],np.uint8(format_step[1]))})_({np.round(matrix_step[0],np.uint8(format_step[0]))}_{np.round(matrix_step[1],np.uint8(format_step[1]))})"
    
    fig = plot_matrix_true(clf, inv_model, results_nd, results_2d, y_values, noise, grid_res, matrix_size, matrix_origin, matrix_step, format_step, figname=f"./matrices/matrices/{model_name}/{method}/{grid_res}_{matrix_size}_{txt}_ID_EXPERIMENT")
  

import numpy as np
from sklearn.decomposition import PCA


