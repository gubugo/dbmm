import os
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np

def get_normal_dist(x, mu, sig):
    return np.exp(-1*((x-mu)/sig)**2)#(1/sig*np.sqrt(2*np.pi))*

# def sigmoid(x):
#     return 1/(1+np.exp(-x))

# def sigmoid_der(x):
#     return sigmoid(x)*(1-sigmoid(x))

def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, v1: float, v2: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )
    
    return np.array([[i[0], i[1], v1, v2] for i in np.c_[xx.ravel(), yy.ravel()]])

def plot_decision_map_with_accuracy(decision_map, coordinates, true_labels, predictions,
                                    nninv_model, classifier, map_size, matrix_side, v1, v2, fig=None, batch_size=128,
                                    cmap='tab10', figsize=(10, 8), save_path=None):

    ### For simplicity, inv transform of data points will be done only with 0,0 augmented pos (before calling func)
    # predict raw data positions
    # preds = []
    # for i in range(0, len(coordinates), batch_size):
    #     batch_coords = coordinates[i:i + batch_size]
    #     gen_imgs = nninv_model.inverse_transform(np.array([[i[0], i[1], v1, v2] for i in batch_coords]))
    #     # gen_imgs = gen_imgs.reshape(-1, 28, 28, 1).astype('float32')
    #     batch_preds = classifier.predict(gen_imgs)
    #     # print(batch_preds)
    #     preds.append(batch_preds)
    # # print(preds)
    # predictions = np.concatenate(preds)

    # verify if it is equal
    correct_mask = predictions == true_labels

    # print(predictions)
    # print(true_labels)
    # print(np.shape(predictions))
    # print(np.shape(true_labels))

    # plot figure
    # fig.figure(figsize=figsize)

    # DBM de fundo
    fig.imshow(decision_map, interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    # hits
    # print(coordinates[correct_mask, 0])
    # print(np.shape(coordinates[correct_mask, 0]))
    points1 = coordinates[correct_mask, 0]
    points2 = coordinates[correct_mask, 1]
    fig.scatter(map_size*(points1-np.min(points1))/(np.max(points1)-np.min(points1)), 
                map_size*(points2-np.min(points2))/(np.max(points2)-np.min(points2)),
                c='lime', s=36/(matrix_side), label='Acerto', alpha=0.7, edgecolor='k', linewidth=0.2)

    #misses
    points1 = coordinates[~correct_mask, 0]
    points2 = coordinates[~correct_mask, 1]
    fig.scatter(map_size*(points1-np.min(points1))/(np.max(points1)-np.min(points1)), 
                map_size*(points2-np.min(points2))/(np.max(points2)-np.min(points2)),
                c='red', s=36/(matrix_side), label='Erro', alpha=0.7, edgecolor='k', linewidth=0.2)

    fig.set_title(f'DBM accuracy: {np.mean(correct_mask):.2%}', fontsize=map_size/(2*matrix_side), x=0.5, y=1-5/map_size)
    fig.grid(False)
    fig.axis("off") 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    # return np.mean(correct_mask)


def plot_decision_map_with_points(points, labels, map_size, ax=None):
    # putting the dbm in the background
    # fig.imshow(decision_map, interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    # scatter points above the dbm
    # print(labels)
    # print(points)
    scatter = ax.scatter((map_size-1)*(points[:, 0]-np.min(points[:, 0]))/(np.max(points[:, 0])-np.min(points[:, 0])), 
                          (map_size-1)*(points[:, 1]-np.min(points[:, 1]))/(np.max(points[:, 1])-np.min(points[:, 1])), 
                          c=labels, s=12, edgecolor='k', linewidth=0.2*labels[:,3], alpha=1.0)

    # cbar = fig.colorbar(img, ticks=range(10))
    ax.grid(False)
    ax.axis("off") 

def plot_points_on_decision_map(points, labels, grid_res, locally, map_extra_coords, augmentation, ax=None):

    if locally:
        labels = apply_local_points_to_alpha(labels, augmentation, map_extra_coords)
    # else:
    #     cmap_colors = np.ones((np.shape(points)[0],4))
    map_size = grid_res-1
    scatter = ax.scatter((map_size-1)*(points[:, 0]-np.min(points[:, 0]))/(np.max(points[:, 0])-np.min(points[:, 0])), 
                          (map_size-1)*(points[:, 1]-np.min(points[:, 1]))/(np.max(points[:, 1])-np.min(points[:, 1])), 
                          c=labels, s=36, edgecolor='k', linewidth=0.2*labels[:,3])
    #/(matrix_side)
    # fig.grid(False)
    # fig.axis("off") 


def plot_generated_images_grid_with_dbm(model_nninv, classifier, grid_res, bb, matrix_origin, step, image_shape=(28,28),
                                    latent_range=(0.0,1.0), img_size=0.05, proximity=1.0,
                                    projection_technique_name="t-SNE", dataset_name="MNIST",
                                    figsize=(10,8), cmap_images='gray', cmap_dbm='tab10', save_path="generated_images_grid_with_dbm.png"):
    fig_main, ax_main = plt.subplots(3,3,figsize=(grid_res/10, grid_res/10))
    
    matrix_end = [1.0,1.0]
    steps_x = (matrix_origin[0],(matrix_origin[0]+matrix_end[0])/2,matrix_end[0])
    steps_y = (matrix_origin[1],(matrix_origin[1]+matrix_end[1])/2,matrix_end[1])
    for a in range(3):
        for b in range(3):
            grid = make_grid(*bb, steps_x[a], steps_y[b], grid_res)
            inverted_grid = model_nninv.inverse_transform(grid)
            decision_map = classifier.predict(inverted_grid).astype(np.uint8).reshape(grid_res,grid_res)
            
            ax = ax_main[a,b]#.subplot(111)

            # DBM in the background
            # plot_dbm_background(ax, decision_map, cmap=cmap_dbm, vmin=0, vmax=int(decision_map.max()))
            ax.imshow(
                decision_map,
                cmap=cmap_dbm,
                interpolation='nearest',
                extent=[0, 1, 0, 1],
                vmin=0,
                vmax=int(decision_map.max()),
                origin='lower'
            )


            # generate coordinates and images
            # coords = generate_grid_coords(latent_range, img_size, proximity)
            
            num_cols = int(proximity*(latent_range[1]-latent_range[0]) / img_size)
            num_rows = int(proximity*(latent_range[1]-latent_range[0]) / img_size)
            xs = np.linspace(bb[0], bb[1], num_cols)
            ys = np.linspace(bb[2], bb[3], num_rows)
            xx, yy = np.meshgrid(xs, ys)
            coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
            coordsInv = np.array([[i[0],i[1], steps_x[a], steps_y[b]] for i in coords])

            xs = np.linspace(latent_range[0], latent_range[1], num_cols)
            ys = np.linspace(latent_range[0], latent_range[1], num_rows)
            xx, yy = np.meshgrid(xs, ys)
            coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

            # images = generate_images_from_coords(model_nninv, coords, image_shape)
            images = model_nninv.inverse_transform(coordsInv)
            images = images.reshape((-1, *image_shape))
            images = np.clip(images, 0, 1)   
            

            # plot images on grid above DBM
            # plot_images_on_ax(ax, coords, images, img_size=img_size*10, cmap=cmap_images)
            images = np.clip(images, 0, 1)
            for (x, y), img in zip(coords, images):
                img_obj = OffsetImage(img, cmap="gray", zoom=img_size*10)
                ab = AnnotationBbox(img_obj, (x, y), frameon=False)
                ax.add_artist(ab)

            ax.set_xlim(latent_range)
            ax.set_ylim(latent_range)
            ax.axis('off')
            ax.set_title(f"({steps_x[a]},{steps_y[b]})", fontsize=grid_res/(2*3), x=0.5, y=1.05-5/grid_res) 

    # save_fig(fig, projection_technique_name=projection_technique_name, dataset_name=dataset_name, filename=save_path)
    folder = os.path.join("results", projection_technique_name, dataset_name)
    os.makedirs(folder, exist_ok=True)  

    filepath = os.path.join(folder, save_path)

    # saves the figure
    fig_main.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {filepath}")

def plot_decision_map_with_points_relative(points, labels, values, map_size, ax=None):
    # putting the dbm in the background
    # fig.imshow(decision_map, interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    # inv_sqrt_2pi = 1/np.sqrt(2*np.pi)

    # for i, value in enumerate(values):
    #     # v = get_normal_dist(value[0],map_extra_coords[0],inv_sqrt_2pi)*get_normal_dist(value[1],map_extra_coords[1],inv_sqrt_2pi)
    #     # # print(v)
    #     # # labels[i,0:3] = v*labels[i,0:3]
    #     # labels[i,3] = (np.exp(v-1)-1*np.exp(-1))*labels[i,3]
    #     labels[i,3] = (1/(1+np.exp(3*value-17)))**4#np.exp(-((value-1.1)**2)/12)

    labels[:,3] = values

    line_width = [(1.6*i if 1.6*i < 1.0 else 1.0) for i in values]

    # for i,j in zip(values,labels):
    #     print(i,j[3])

    # scatter points above the dbm
    # print(labels)
    # print(points)
    scatter = ax.scatter((map_size-1)*(points[:, 0]-np.min(points[:, 0]))/(np.max(points[:, 0])-np.min(points[:, 0])), 
                          (map_size-1)*(points[:, 1]-np.min(points[:, 1]))/(np.max(points[:, 1])-np.min(points[:, 1])), 
                          c=labels, s=12, edgecolor='k', linewidth=0.2*labels[:,3])

    # cbar = fig.colorbar(img, ticks=range(10))
    ax.grid(False)
    ax.axis("off") 


def apply_local_points_to_alpha(colors, extra_dims, map_extra_coords):
    inv_sqrt_2pi = 1/np.sqrt(2*np.pi)
    
    for i, value in enumerate(extra_dims):
        v = get_normal_dist(value[0],map_extra_coords[0],inv_sqrt_2pi)*get_normal_dist(value[1],map_extra_coords[1],inv_sqrt_2pi)
        # print(v)
        # labels[i,0:3] = v*labels[i,0:3]
        colors[i,3] = (np.exp(3*v-3)-np.exp(-3))*colors[i,3]

    return colors
    