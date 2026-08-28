import os
import sys
import time
from joblib import dump, load
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from numba import njit
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
from joblib import Parallel, delayed

import code.models.tensorflow.sharp as sharp
import code.models.tensorflow.sharp_og as sharp_og
import code.models.tensorflow.nninv as nninv
from code.utils.data import get_inv_proj_data_nninv, get_inv_proj_data_ae
import faiss 
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# faiss.omp_set_num_threads(8)  
gpu_res = faiss.StandardGpuResources() 
gpu_res.setTempMemory(2*61*61*784)

def compute_knn_indices(X, k):
    n_samples = X.shape[0]
    if k >= n_samples:
        k = n_samples - 1

    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    knn_indices = indices[:, 1:] 
    return knn_indices

def compute_knn_indices_single(X, k, index):
    n_samples = X.shape[0]
    n_dims = X.shape[1]
    if k >= n_samples:
        k = n_samples - 1
    
    search = faiss.IndexFlatL2(X.shape[1])   # build the inde

    search = faiss.index_cpu_to_gpu(gpu_res, 0, search)  # Move index to GPU (GPU 0)
    search.add(X)
    _, I = search.search(np.array([X[index]]), k)
    knn_indices = I[:, 1:] 
    return knn_indices
    # nn = copy.copy(nnf)
    # nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    # nn.fit(X)
    # distances, indices = nn.kneighbors([X[index]], return_distance=True)
    # knn_indices = indices[:, 1:] 
    # return knn_indices


# CPU THREADING
def local_id_from_covariances(X, knn_indices, theta=0.95):
    # n_samples = covariances.shape[0]
    n_samples = X.shape[0]
    di_list = np.zeros(n_samples)

    di_list = Parallel(n_jobs=-3)(delayed(get_intrinsic_dimension_pca)(i, X, knn_indices, theta) for i in range(n_samples))

    return di_list

def get_intrinsic_dimension_pca(i, X, knn_indices, theta):
    S = X[knn_indices[i]]
    N_samples, N_features = S.shape
    pca = PCA(n_components=N_samples)#min(N_samples, N_features))
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

    return di_list

def get_intrinsic_dimension_sv(index, X, k, theta):
    knn_idx = compute_knn_indices_single(X, k, index)
    # covariancias = neighborhood_covariances()
    di_list = get_intrinsic_dimension_pca(0, X, knn_idx, theta)
    return di_list
    

cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("viridis")

def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, v1: float, v2: float, side_length: int
) -> np.ndarray:
    # Create 1D arrays of evenly spaced values for each dimension
    x_values = np.linspace(x_min, x_max, side_length)
    y_values = np.linspace(y_min, y_max, side_length)
    
    xx, yy = np.meshgrid(x_values, y_values) # Create the 2D grid using meshgrid
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    extra_coords_collumn = np.column_stack([np.full(grid_points.shape[0], v1), 
                                    np.full(grid_points.shape[0], v2)])
    grid_points = np.hstack([grid_points, extra_coords_collumn])
    return grid_points

def make_grid_normal(
    x_min: float, x_max: float, y_min: float, y_max: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )
    
    return np.c_[xx.ravel(), yy.ravel()]

def weave_grid(values, grid_points):
    return np.hstack([np.column_stack([np.full(grid_points.shape[0], values[0]), 
                                            np.full(grid_points.shape[0], values[1])]), grid_points])


def make_grid_reverse(
    base: np.ndarray, vl: np.ndarray, side_length: int
) -> np.ndarray:

    grid_points_res = np.hstack(
                                [np.column_stack(
                                    [np.full(base.shape[0], vl[0,0]), np.full(base.shape[0], vl[0,1])]), 
                                    base
                                ])
    for i in range(1, np.shape(vl)[0]):
        grid_points_res = np.concatenate((grid_points_res, np.hstack(
                                                            [np.column_stack(
                                                                [np.full(base.shape[0], vl[i,0]), np.full(base.shape[0], vl[i,1])]), 
                                                                base
                                                            ]
                                                          )
                                                        ), axis=0)
    return np.array(grid_points_res)

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def pct_bb(x_min, x_max, y_min, y_max, mult_x_min, mult_x_max, mult_y_min, mult_y_max) -> tuple[float, float, float, float]:
    n_x_min = mult_x_min*(x_max - x_min)+x_min 
    n_x_max = mult_x_max*(x_max - x_min)+x_min 
    n_y_min = mult_y_min*(y_max - y_min)+y_min 
    n_y_max = mult_y_max*(y_max - y_min)+y_min 

    return n_x_min, n_x_max, n_y_min, n_y_max

def numerical_id(inverter, nd_data, x_data, y_data, noise):
    base_id = np.mean(get_intrinsic_dimension(nd_data, 120, 0.95))
    print(f"nd id: {base_id}")
    proj_id = np.mean(get_intrinsic_dimension(x_data, 2, 0.95))
    print(f"proj id: {proj_id}")
    print(np.shape(np.concatenate((x_data, noise), axis=-1)))
    inv_proj_id = np.mean(get_intrinsic_dimension(inverter.inverse_transform(np.concatenate((x_data, noise), axis=-1)), 120, 0.95))
    print(f"inv proj id: {inv_proj_id}")

def plot_matrix(inverter, nd_data, x_data, noise, grid_res):

    n_dimensions = nd_data.shape[1]
    grid_res = 60
    n_pieces = 5 # true grid res = n_pieces * grid_res
    padding = 0.1 # 10% of padding on all sides
    padded_grid_space = np.int16(np.round(padding*grid_res))
    k = 120
    pixel_width = 61#np.ceil(np.sqrt(X_size))
    pixel_width_sqr = pixel_width**2
    half_pixel_width = pixel_width//2
    theta = 0.95
    index_coord = pixel_width**2-1#//2##0  #CHANGE HERE WHEN RUNNING DIFFERENT DBM COORD
    
    X_size = np.shape(x_data)[0]
    sum_all = 0.0

    bb = get_bounding_box(x_data)
    for x_c in range(n_pieces):
        for y_c in range(n_pieces):
            start_time = time.perf_counter()
            bounding_box_dbm = pct_bb(*bb, (x_c-padding)/n_pieces, (x_c+1+padding)/n_pieces, (y_c-padding)/n_pieces, (y_c+1+padding)/n_pieces)
            bounding_box_dpt = pct_bb(*bb, x_c/n_pieces, (x_c+1)/n_pieces, y_c/n_pieces, (y_c+1)/n_pieces)

            print("computing...")

            normal_grid = make_grid_normal(*bounding_box_dpt, grid_res)
            normal_grid_ = make_grid(*bounding_box_dbm, 1.0, 1.0, grid_res + 2*padded_grid_space) #CHANGE HERE WHEN RUNNING DIFFERENT DBM COORD
            invp_grid_base = inverter.inverse_transform(normal_grid_)

            di_list = np.ones(np.shape(normal_grid)[0])

            grid_base = make_grid_normal(0.8-1.0/n_pieces,0.8+1.0/n_pieces,0.8-1.0/n_pieces,0.8+1.0/n_pieces, pixel_width)#CHANGE HERE WHEN RUNNING DIFFERENT DBM COORD

            batch_size = 8
            for batch_index, start in enumerate(range(0, np.shape(normal_grid)[0], batch_size)):
                batch = normal_grid[start:start + batch_size]
                batch_coord = batch_index*batch_size
                # print(batch_index, batch.shape)
                grid = make_grid_reverse(grid_base,batch,pixel_width)
                # grid = grid.reshape(-1, grid.shape[2])

                inv_p_batch_grid = (inverter.inverse_transform(grid, verbose=False)).reshape(batch_size, pixel_width_sqr, n_dimensions)
                for index, i in enumerate(inv_p_batch_grid):
                    # print(np.shape(i))
                    invp_grid = np.concatenate((i,invp_grid_base))
                    id_pixelv = get_intrinsic_dimension_sv(index_coord, invp_grid, k, theta)
                    di_list[index+batch_coord] = id_pixelv

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds, done batch: {n_pieces*x_c+ y_c}")

            fig_id, ax_id = plt.subplots(1,1,figsize=(50,50))
            np.save(f'_ID_res3/dbm_4d_({pixel_width})_{k}_{x_c}_{y_c}.npy', di_list)
            max_v = np.max(di_list)
            min_v = np.min(di_list)
            sum_all += np.sum(di_list)

            cmap = plt.get_cmap('jet', int(5-2+1))
            
            ax_id.imshow(
                di_list.reshape((grid_res, grid_res,1)),
                cmap=cmap,
                interpolation="none",
                resample=False,
                vmin=2,
                vmax=5,
            )

            ax_id.axis("off")  
            print(max_v)
            print(min_v)
            fig_id.savefig(f"_ID_res3/img_4d_({pixel_width})_{k}_{x_c}_{y_c}.png", bbox_inches="tight", pad_inches=0.0)
    sum_all /= (grid_res*n_pieces)**2
    print(sum_all)

def plot_matrix_pixel_plane(inverter, nd_data, x_data, noise, grid_res):

    n_dimensions = nd_data.shape[1]
    grid_res = 60
    n_pieces = 5 # true grid res = n_pieces * grid_res
    k = 120
    pixel_width = 61
    pixel_width_sqr = pixel_width**2
    half_pixel_width = pixel_width//2
    theta = 0.95
    matrix_side = 5
    half_matrix_side = matrix_side//2
    index_coord = 0#pixel_width**2//2

    bb = get_bounding_box(x_data)
    for x_c in range(n_pieces):
        for y_c in range(n_pieces):
            start_time = time.perf_counter()
            bounding_box = pct_bb(*bb, x_c/n_pieces, (x_c+1)/n_pieces, y_c/n_pieces, (y_c+1)/n_pieces)

            normal_grid = make_grid_normal(*bounding_box, grid_res)

            di_list = np.ones(np.shape(normal_grid)[0])

            grid_base = make_grid_normal(-1.0/n_pieces,1.0/n_pieces,-1.0/n_pieces,1.0/n_pieces, pixel_width)#CHANGE HERE WHEN RUNNING DIFFERENT DBM COORD

            batch_size = 4
            for batch_index, start in enumerate(range(0, np.shape(normal_grid)[0], batch_size)):
                batch = normal_grid[start:start + batch_size]
                batch_coord = batch_index*batch_size
                # print(batch_index, batch.shape)
                grid = make_grid_reverse(grid_base,batch,pixel_width)
                # grid = grid.reshape(-1, grid.shape[2])

                inv_p_batch_grid = (inverter.inverse_transform(grid, verbose=False)).reshape((batch_size, pixel_width_sqr, n_dimensions))        
                for index, i in enumerate(inv_p_batch_grid):
                    # print(np.shape(i))
                    invp_grid = i
                    id_pixelv = get_intrinsic_dimension_sv(index_coord, invp_grid, k, theta)
                    di_list[index+batch_coord] = id_pixelv

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds, done batch: {n_pieces*x_c+ y_c}")

            fig_id, ax_id = plt.subplots(1,1,figsize=(50,50))
            
            res_id = di_list
            np.save(f'_ID_res_reuters/dbm_({pixel_width})_{k}_{x_c}_{y_c}.npy', res_id)
            max_v = np.max(res_id)
            min_v = np.min(res_id)
            cmap = plt.get_cmap('jet', int(5-2+1))
            
            ax_id.imshow(
                    res_id.reshape((grid_res, grid_res,1)),
                    interpolation="none",
                    resample=False,
                    cmap=cmap,
                    vmin=2,
                    vmax=5,
            )

            print(max_v)
            print(min_v)
    
            ax_id.axis("off")     
            fig_id.savefig(f"_ID_res_reuters/img_({pixel_width})_{k}_{x_c}_{y_c}.png", bbox_inches="tight", pad_inches=0.0)
    
def plot_matrix_dbm_plane(inverter, nd_data, x_data, noise, grid_res):

    grid_res = 500
    X_size = np.shape(x_data)[0]
    bounding_box = get_bounding_box(x_data)
    k = 120
    pixel_width = 501
    half_pixel_width = pixel_width//2
    theta = 0.95
    matrix_side = 5
    half_matrix_side = matrix_side//2

    res_id =  np.zeros((matrix_side,matrix_side,grid_res,grid_res))

    print("computing...")

    normal_grid = make_grid_normal(*bounding_box, grid_res)
    normal_grid_ = make_grid(*bounding_box, -1.0, -1.0, grid_res)#(x-half_matrix_side)/half_matrix_side, (y-half_matrix_side)/half_matrix_side
    invp_grid_base = inverter.inverse_transform(normal_grid_)
    
    rngex = (np.array(list(range(pixel_width))*pixel_width)-half_pixel_width)/half_pixel_width
    rngey = (np.array(sorted(list(range(pixel_width))*pixel_width))-half_pixel_width)/half_pixel_width
    matrix_values = np.c_[rngex, rngey]
    di_list = np.ones(np.shape(normal_grid)[0])

    # for index,i in enumerate(invp_grid_base):
    start_time = time.perf_counter()
    # grid = [[i[0], i[1], j[0], j[1]] for j in matrix_values]
    invp_grid = invp_grid_base
    # print(np.shape(invp_grid))
    di_list = get_intrinsic_dimension(invp_grid, k, theta)#index_coord
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    res_id = np.array(di_list)
    
    np.save(f'dbm({grid_res})_{k}_{matrix_side}.npy', res_id)
    max_v = np.max(res_id)
    min_v = np.min(res_id)
    
    print(max_v)
    print(min_v)

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
    model_name = "sharp"
    for i in range(4):
        dataset_ops = ["mnist", "fashionmnist", "reuters", "har"] 
        dataset = dataset_ops[i]
        method = "noise"
        
        grid_res = 100

        epochs_dataset = {}
        epochs_dataset["fashionmnist"] = 20
        epochs_dataset["mnist"] = 20
        epochs_dataset["har"] = 20
        epochs_dataset["hate_speech"] = 20
        epochs_dataset["reuters"] = 30
        
        epochs = epochs_dataset[dataset]

        sharp_dims_classes = {}
        sharp_dims_classes["fashionmnist"] = [784, 10]
        sharp_dims_classes["mnist"] = [784, 2]
        sharp_dims_classes["har"]  = [561, 6]
        sharp_dims_classes["reuters"] = [5000, 6]
        sharp_dims_classes["hate_speech"] = [100, 3]
        dims = sharp_dims_classes[dataset][0]
        classes = sharp_dims_classes[dataset][1]
        noise = []

        if model_name == "sharp":
            results_nd, y_values, noise, results_2d, clf, inv_model, nn = get_inv_proj_data_ae(
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
                "mlp",
                method,
                epochs
            )
        if model_name == "nninv":#noise,
            results_nd, results_2d, y_values, noise, clf, inv_model = get_inv_proj_data_nninv(
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
        
        fig = plot_matrix(inv_model, results_nd, results_2d, noise, grid_res)
        print(dataset)
    # numerical_id(inv_model, results_nd, results_2d, y_values, noise)