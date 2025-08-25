import matplotlib.pyplot as plt
import numpy as np

def plot_decision_map_with_accuracy(decision_map, coordinates, true_labels, 
                                    nninv_model, classifier, map_size, v1, v2, fig=None, batch_size=128,
                                    cmap='tab10', figsize=(10, 8), save_path=None):

    # predict raw data positions
    preds = []
    for i in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[i:i + batch_size]
        gen_imgs = nninv_model.inverse_transform(np.array([[i[0], i[1], v1, v2] for i in batch_coords]))
        # gen_imgs = gen_imgs.reshape(-1, 28, 28, 1).astype('float32')
        batch_preds = classifier.predict(gen_imgs)
        # print(batch_preds)
        preds.append(batch_preds)
    # print(preds)
    predictions = np.concatenate(preds)

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
                c='lime', s=12, label='Acerto', alpha=0.7, edgecolor='k', linewidth=0.2)

    #misses
    points1 = coordinates[~correct_mask, 0]
    points2 = coordinates[~correct_mask, 1]
    fig.scatter(map_size*(points1-np.min(points1))/(np.max(points1)-np.min(points1)), 
                map_size*(points2-np.min(points2))/(np.max(points2)-np.min(points2)),
                c='red', s=12, label='Erro', alpha=0.7, edgecolor='k', linewidth=0.2)

    fig.set_title(f'DBM accuracy: {np.mean(correct_mask):.2%}')
    fig.grid(False)
    fig.axis("off") 
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    # return np.mean(correct_mask)


def plot_decision_map_with_points(decision_map, points, labels, map_size, cmap='tab10', fig=None, save_path=None):
    # putting the dbm in the background
    fig.imshow(decision_map, interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    # scatter points above the dbm
    print(labels)
    print(points)
    scatter = fig.scatter(map_size*(points[:, 0]-np.min(points[:, 0]))/(np.max(points[:, 0])-np.min(points[:, 0])), 
                          map_size*(points[:, 1]-np.min(points[:, 1]))/(np.max(points[:, 1])-np.min(points[:, 1])), 
                          c=labels, cmap=cmap, s=15, edgecolor='k', linewidth=0.2, alpha=0.7)

    # cbar = fig.colorbar(img, ticks=range(10))
    fig.grid(False)
    fig.axis("off") 

    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()