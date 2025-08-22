import matplotlib.pyplot as plt
import numpy as np

def plot_decision_map_with_accuracy(decision_map, coordinates, true_labels, 
                                    nninv_model, classifier, batch_size=128, 
                                    cmap='tab10', figsize=(10, 8), save_path=None):

    # predict raw data positions
    preds = []
    for i in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[i:i + batch_size]
        gen_imgs = nninv_model.inverse_transform(batch_coords)
        gen_imgs = gen_imgs.reshape(-1, 28, 28, 1).astype('float32')
        batch_preds = classifier.predict(gen_imgs, verbose=0)
        preds.append(np.argmax(batch_preds, axis=1))
    predictions = np.concatenate(preds)

    # verify if it is equal
    correct_mask = predictions == true_labels

    # plot figure
    plt.figure(figsize=figsize)

    # DBM de fundo
    plt.imshow(decision_map, cmap=cmap, interpolation='nearest',
               extent=[0, 1, 0, 1], vmin=0, vmax=9, origin='lower')

    # hits
    plt.scatter(coordinates[correct_mask, 0], coordinates[correct_mask, 1],
                c='lime', s=12, label='Acerto', alpha=0.7, edgecolor='k', linewidth=0.2)

    #misses
    plt.scatter(coordinates[~correct_mask, 0], coordinates[~correct_mask, 1],
                c='red', s=12, label='Erro', alpha=0.7, edgecolor='k', linewidth=0.2)

    plt.title(f'DBM accuracy: {np.mean(correct_mask):.2%}')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return np.mean(correct_mask)


def plot_decision_map_with_points(decision_map, points, labels, cmap='tab10', fig=None, save_path=None):
    # putting the dbm in the background
    fig.imshow(decision_map, interpolation='none', cmap='tab10',  vmin=0, vmax=9, origin='lower')

    # scatter points above the dbm
    print(labels)
    print(points)
    scatter = fig.scatter(200*(points[:, 0]-np.min(points[:, 0]))/(np.max(points[:, 0])-np.min(points[:, 0])), 
                          200*(points[:, 1]-np.min(points[:, 1]))/(np.max(points[:, 1])-np.min(points[:, 1])), 
                          c=labels, cmap=cmap, s=10, edgecolor='k', linewidth=0.2, alpha=0.7)

    # cbar = fig.colorbar(img, ticks=range(10))
    fig.grid(False)
    fig.axis("off") 

    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()