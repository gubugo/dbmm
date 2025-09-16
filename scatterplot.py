import os
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np

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


def plot_decision_map_with_points(decision_map, points, labels, map_size, matrix_side, cmap='tab10', fig=None, save_path=None):
    # putting the dbm in the background
    fig.imshow(decision_map, interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    # scatter points above the dbm
    # print(labels)
    # print(points)
    scatter = fig.scatter(map_size*(points[:, 0]-np.min(points[:, 0]))/(np.max(points[:, 0])-np.min(points[:, 0])), 
                          map_size*(points[:, 1]-np.min(points[:, 1]))/(np.max(points[:, 1])-np.min(points[:, 1])), 
                          c=labels, cmap=cmap, s=36/(matrix_side), edgecolor='k', linewidth=0.2, alpha=0.7)

    # cbar = fig.colorbar(img, ticks=range(10))
    fig.grid(False)
    fig.axis("off") 

    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_generated_images_grid_with_dbm(decision_map, model_nninv, image_shape=(28,28),
                                    latent_range=(0.0,1.0), img_size=0.05, proximity=1.4,
                                    projection_technique_name="t-SNE", dataset_name="MNIST",
                                    figsize=(10,8), cmap_images='gray', cmap_dbm='tab10', save_path="generated_images_grid_with_dbm.png"):
    fig_main, ax_main = plt.subplots(9,9,figsize=(200/10, 200/10))
    
    vals = [0, 1, 2]

    for (x,y) in zip(vals,vals):
        fig = ax_main[x,y].figure(figsize=figsize)
        ax = ax_main[x,y].subplot(111)

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
        xs = np.linspace(latent_range[0], latent_range[1], num_cols)
        ys = np.linspace(latent_range[0], latent_range[1], num_rows)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
        coordsInv = np.array([[i[0],i[1], 0, 0] for i in coords])

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

    # save_fig(fig, projection_technique_name=projection_technique_name, dataset_name=dataset_name, filename=save_path)
    folder = os.path.join("results", projection_technique_name, dataset_name)
    os.makedirs(folder, exist_ok=True)  

    filepath = os.path.join(folder, save_path)

    # saves the figure
    fig_main.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {filepath}")