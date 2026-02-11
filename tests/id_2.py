import os
from joblib import Parallel, delayed, dump, load
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
import umap

from interface.models import ae, nninv, ssnp
import interface.models.sharp_og as sharp
from interface.utils import metrics

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
    print(X.shape)

    # pega o rage, faz uma lista, separa a lista e passa como argumento pras threads...?
    di_list = Parallel(n_jobs=-3)(delayed(local_id_from_covariances_mt)(i, X, knn_indices, theta) for i in range(n_samples))

    return di_list

def local_id_from_covariances_mt(i, X, knn_indices, theta):
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

def get_intrinsic_dimension(X, k, theta):
    print("here2")
    knn_idx = compute_knn_indices(X, k)
    # covariancias = neighborhood_covariances()
    print("here3")
    di_list = local_id_from_covariances(X, knn_idx, theta)
    return di_list

def intrinsic_dimension_map(ssnp_model, data, k=120, theta=0.95, grid_res=50):
    bb = get_bounding_box(data)
    coords = make_grid_normal(*bb, grid_res)

    generated_points = ssnp_model.inverse_transform(coords)

    id_map = np.zeros(grid_res*grid_res)
    print("heres")
    d_avg = get_intrinsic_dimension(generated_points, k=k, theta=theta)
    id_map = d_avg

    return id_map

def average_intrinsic_dimension(id_map):
    valid = np.isfinite(id_map)
    if np.any(valid):
        return np.mean(id_map[valid])
    else:
        return np.nan
    
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def make_and_fit_mlp(X, y) -> MLPClassifier:
    return MLPClassifier(
        verbose=True,
        hidden_layer_sizes=(512, 128, 32),
        activation="relu",
        max_iter=100,
        random_state=420,
    ).fit(X, y)

def get_inv_proj_data_sharp(output_dir, _model, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    X, y = Load_data(data_dir, d)

    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=10000, test_size=500, random_state=420, stratify=y
    )

    _, X_test, _, y_test = train_test_split(
        X_train, y_train, train_size=100, test_size=5000, random_state=420, stratify=y_train
    )     

    neighbor_finder = NearestNeighbors(
        n_neighbors=5
    ) 
    neighbor_finder.fit(X_train)

    if os.path.exists(os.path.join(output_dir, dataset_name, model_name, method)):
        _model.load_weights(export_path=os.path.join(output_dir, dataset_name, model_name, method))
    else:
        _model.fit(X_train, y_train, epochs=epochs)
        _model.save_weights(os.path.join(output_dir, dataset_name, model_name, method))
    X_model_res = _model.transform(X_test)
    X_model_2d = X_model_res

    if os.path.exists(f'{output_dir}/{dataset_name}/class.joblib'):
        clf = load(f'{output_dir}/{dataset_name}/class.joblib')
    else:
        clf = make_and_fit_mlp(X_train, y_train)
        dump(clf, f'{output_dir}/{dataset_name}/class.joblib')

    return X_test, X_model_2d, y_test, _model, clf, neighbor_finder


if __name__ == "__main__":
    output_dir = "weights"
    method = "none"

    grid_res = 500

    epochs = 20

    sharp_dims_classes = {}
    sharp_dims_classes["fashionmnist"] = [784, 10]
    sharp_dims_classes["mnist"] = [784, 10]
    sharp_dims_classes["har"]  = [561, 6]
    sharp_dims_classes["reuters"] = [5000, 6]

    noise = []

    results_nd, results_2d, y_values, inv_model, classifier, neighbor_finder = get_inv_proj_data_sharp(
        output_dir, 
        sharp.ShaRP(
            784,
            10,
            "diagonal_normal",
            latent_dim=2,
            variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
            var_leaky_relu_alpha=-0.0001,
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.1,
        ),
        "mnist",
        "sharp",
        "none",
        epochs
    )

    fig_dbm, ax_dbm = plt.subplots(1,1,figsize=(50, 50))
    fig_scatter, ax_scatter = plt.subplots(3,2,figsize=(50, 50))

    cmap_tab = plt.get_cmap('tab10')
    point_colors = cmap_tab(y_values)

    bounding_box = get_bounding_box(results_2d)
    grid = make_grid_normal(*bounding_box, grid_res)
    inverted_grid = inv_model.inverse_transform(grid)

    classes = classifier.predict(inverted_grid).astype(np.uint8)

    res = classifier.predict_proba(inverted_grid)     
    confidence = np.zeros(np.shape(res)[0])

    for k,lis in enumerate(res):
        confidence[k] = np.max(lis)

    metric_matrix = metrics.metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)

    # ax_dbm.grid(False)
    # ax_dbm.axis("off") 
    ax_scatter[0,0].grid(False)
    ax_scatter[0,0].axis("off") 
    # ax_scatter[0,0].set_aspect(1)
    ax_scatter[0,1].grid(False)
    ax_scatter[0,1].axis("off") 
    # ax_scatter[0,1].set_aspect(1)
    ax_scatter[1,0].grid(False)
    ax_scatter[1,0].axis("off") 
    # ax_scatter[1,0].set_aspect(1)
    ax_scatter[1,1].grid(False)
    ax_scatter[1,1].axis("off") 
    # ax_scatter[1,1].set_aspect(1)
    ax_scatter[2,0].grid(False)
    ax_scatter[2,0].axis("off") 
    # ax_scatter[2,0].set_aspect(1)
    ax_scatter[2,1].grid(False)
    ax_scatter[2,1].axis("off") 
    # ax_scatter[2,1].set_aspect(1)
    

    # putting the dbm in the background
    ax_scatter[0,0].imshow(cmap_tab(classes).reshape((grid_res,grid_res,4)), interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    # fig_dbm.savefig(f"sharp_2d_dbm.png", bbox_inches="tight", pad_inches=0.0)

    ax_scatter[0,1].imshow(cmap_tab(classes).reshape((grid_res,grid_res,4)), interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')
    ax_scatter[0,1].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
                      (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
                      c=point_colors, s=100, edgecolor='k', linewidth=0.7*point_colors[:,3])
    
    # fig_dbm.savefig(f"sharp_2d_dbm_scatter.png", bbox_inches="tight", pad_inches=0.0)

    conf_dbm = np.zeros((grid_res*grid_res,4))
    conf_dbm[:,0] = cmap_tab(classes)[:,0]*confidence
    conf_dbm[:,1] = cmap_tab(classes)[:,1]*confidence
    conf_dbm[:,2] = cmap_tab(classes)[:,2]*confidence
    conf_dbm[:,3] = cmap_tab(classes)[:,3]

    ax_scatter[1,0].imshow(conf_dbm.reshape((grid_res,grid_res,4)), interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    ax_scatter[1,1].imshow(conf_dbm.reshape((grid_res,grid_res,4)), interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')
    ax_scatter[1,1].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
                       (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
                       c=point_colors, s=100, edgecolor='k', linewidth=0.7*point_colors[:,3])
    
    ntp_dbm = np.zeros((grid_res*grid_res,4))
    ntp_dbm[:,0] = cmap_tab(classes)[:,0]*(minmax_scale(metric_matrix))
    ntp_dbm[:,1] = cmap_tab(classes)[:,1]*(minmax_scale(metric_matrix))
    ntp_dbm[:,2] = cmap_tab(classes)[:,2]*(minmax_scale(metric_matrix))
    ntp_dbm[:,3] = cmap_tab(classes)[:,3]

    ax_scatter[2,0].imshow(ntp_dbm.reshape((grid_res,grid_res,4)), interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    ax_scatter[2,1].imshow(ntp_dbm.reshape((grid_res,grid_res,4)), interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')
    ax_scatter[2,1].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
                       (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
                       c=point_colors, s=100, edgecolor='k', linewidth=0.7*point_colors[:,3])
    
    # fig_scatter[0,0].savefig(f"sharp_2d_scatter.png", bbox_inches="tight", pad_inches=0.0)

    # id_map = intrinsic_dimension_map(ssnp_model=inv_model, data=results_2d, k=120, theta=0.95, grid_res=grid_res)

    # print(np.mean(id_map))

    # fig_main, ax_main = plt.subplots(1,1,figsize=(50,50))
    # np.save(f'sharp_2d_id_{grid_res}.npy', id_map)
    # max_v = np.max(id_map)
    # min_v = np.min(id_map)
    # print(np.shape(id_map))

    # cmap = plt.get_cmap('jet', int(5-2+1))

    # ax_main.imshow(
    #     id_map.reshape((grid_res, grid_res,1)),
    #     cmap=cmap,
    #     interpolation="none",
    #     resample=False,
    #     vmin=2,
    #     vmax=5
    # )
    # ax_main.grid(False)
    # ax_main.axis("off") 
    # print(max_v)
    # print(min_v)
    # fig_main.savefig(f'sharp_2d_id_{grid_res}.png', bbox_inches="tight", pad_inches=0.0)

##

#     results_nd, results_2d, y_values, inv_model, classifier = get_inv_proj_data_sharp(
#         output_dir, 
#         ssnp.SSNP(
#             verbose=True,
#             latent_dims=2,
#             patience=0,
#             opt="adam",
#             bottleneck_activation="linear",
#         ),
#         "mnist",
#         "ssnp",
#         "none",
#         epochs
#     )

#     # fig_scatter, ax_scatter = plt.subplots(1,1,figsize=(50, 50))

#     cmap_tab = plt.get_cmap('tab10')
#     point_colors = cmap_tab(y_values)

#     ax_scatter[0,1].grid(False)
#     ax_scatter[0,1].axis("off") 
#     ax_scatter[0,1].set_aspect(1)

#     ax_scatter[0,1].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
#                        (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
#                        c=point_colors, s=300, edgecolor='k', linewidth=0.7*point_colors[:,3])
    
#     # fig_scatter.savefig(f"ssnp_2d_scatter.png", bbox_inches="tight", pad_inches=0.0)

# ##


#     results_nd, results_2d, y_values, inv_model, classifier = get_inv_proj_data_sharp(
#         output_dir, 
#         ae.AutoencoderProjection(epochs),
#         "mnist",
#         "ae",
#         "none",
#         epochs
#     )

#     # fig_scatter, ax_scatter = plt.subplots(1,1,figsize=(50, 50))

#     cmap_tab = plt.get_cmap('tab10')
#     point_colors = cmap_tab(y_values)

#     ax_scatter[0,0].grid(False)
#     ax_scatter[0,0].axis("off") 
#     ax_scatter[0,0].set_aspect(1)

#     ax_scatter[0,0].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
#                        (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
#                        c=point_colors, s=300, edgecolor='k', linewidth=0.7*point_colors[:,3])
    
#     # fig_scatter.savefig(f"ae_2d_scatter.png", bbox_inches="tight", pad_inches=0.0)

# ##

#     tsne = TSNE(
#         n_jobs=4, 
#         random_state=420, 
#         n_components=2
#     )

#     results_2d = tsne.fit_transform(results_nd)

#     # fig_scatter, ax_scatter = plt.subplots(1,1,figsize=(50, 50))

#     ax_scatter[1,1].grid(False)
#     ax_scatter[1,1].axis("off") 
#     ax_scatter[1,1].set_aspect(1)

#     ax_scatter[1,1].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
#                        (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
#                        c=point_colors, s=300, edgecolor='k', linewidth=0.7*point_colors[:,3])
    
#     # fig_scatter.savefig(f"tsne_2d_scatter.png", bbox_inches="tight", pad_inches=0.0)

# ##
#     ump = umap.UMAP(n_components=2)

#     results_2d = ump.fit_transform(results_nd)

#     # fig_scatter, ax_scatter = plt.subplots(1,1,figsize=(50, 50))

#     ax_scatter[1,2].grid(False)
#     ax_scatter[1,2].axis("off") 
#     ax_scatter[1,2].set_aspect(1)

#     ax_scatter[1,2].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
#                        (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
#                        c=point_colors, s=300, edgecolor='k', linewidth=0.7*point_colors[:,3])
    
#     fig_scatter.savefig(f"umap_2d_scatter.png", bbox_inches="tight", pad_inches=0.0)

# ##
#     pca = PCA(n_components=2)

#     results_2d = pca.fit_transform(results_nd)

#     # fig_scatter, ax_scatter = plt.subplots(1,1,figsize=(50, 50))

#     ax_scatter[1,0].grid(False)
#     ax_scatter[1,0].axis("off")
#     ax_scatter[1,0].set_aspect(1) 

#     ax_scatter[1,0].scatter((grid_res-1)*(results_2d[:, 0]-np.min(results_2d[:, 0]))/(np.max(results_2d[:, 0])-np.min(results_2d[:, 0])), 
#                        (grid_res-1)*(results_2d[:, 1]-np.min(results_2d[:, 1]))/(np.max(results_2d[:, 1])-np.min(results_2d[:, 1])), 
#                        c=point_colors, s=300, edgecolor='k', linewidth=0.7*point_colors[:,3])
    plt.subplots_adjust(wspace=0, hspace=0) 
    fig_scatter.savefig(f"dbms_3x22.png", bbox_inches="tight", pad_inches=0.0)