# @st.cache_resource
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import tensorflow as tf


def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

def include_classes(X_data, y_data, classes):
    if len(classes) == 0:
        return X_data, y_data
    
    X_train2 = []
    y_train2 = []
    
    for i in range(np.shape(X_data)[0]):
        if y_data[i] in classes:
            X_train2.append(X_data[i,:])
            y_train2.append(y_data[i])

    return np.array(X_train2), np.array(y_train2)

def get_inv_proj_data_sharp(output_dir, dataset_name, model_name, method, epochs):
    data_dir = "./data/"

    d = dataset_name

    X, y = Load_data(data_dir, d)

    img_ = [[],[],[],[],[],[],[],[],[],[]]

    img_[0], _ = include_classes(X, y, [0])
    img_[1], _ = include_classes(X, y, [1])
    img_[2], _ = include_classes(X, y, [2])
    img_[3], _ = include_classes(X, y, [3])
    img_[4], _ = include_classes(X, y, [4])
    img_[5], _ = include_classes(X, y, [5])
    img_[6], _ = include_classes(X, y, [6])
    img_[7], _ = include_classes(X, y, [7])
    img_[8], _ = include_classes(X, y, [8])
    img_[9], _ = include_classes(X, y, [9])

    fig_main, ax_main = plt.subplots(3,10,figsize=(50, 15))

    for i in range(3):
        for j in range(10):
            ax_main[i,j].imshow(
                img_[j][i].reshape((28, 28, 1)),
                interpolation="none",
                resample=False,
                cmap="gray"
            )
            ax_main[i,j].axis("off") 

            print(f"finish {i} {j}")

    fig_main.subplots_adjust(wspace=0.01, hspace=0.01) 
    # fig_main.tight_layout(pad=0)
    fig_main.savefig(f"noise/dbm/mnist.png", bbox_inches="tight", pad_inches=0.0)


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
            dataset,
            model_name,
            method,
            epochs
        )
