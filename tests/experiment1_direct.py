#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from PIL import Image, ImageFont
from skimage import io
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from umap import UMAP
from sklearn.manifold import Isomap

from code.models.tensorflow import sharp_og as sharp
# from code.models.pytorch import sharp
from code.utils import metrics

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None, X_test=None, X_inv_test=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low, ndim_high=X.shape[1])
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)
    DSC = metrics.distance_consistency(X_2d, y)
    CC = metrics.cluster_size_consistency_r(X, y, X_2d)

    if X_inv is not None:
        MSE_train = metrics.metric_mse(X, X_inv)
    else:
        MSE_train = -99.0

    if X_inv_test is not None:
        assert X_test is not None, "if X_inv_test is provided, X_test must be too"
        MSE_test = metrics.metric_mse(X_test, X_inv_test)
    else:
        MSE_test = -99.0

    return T, C, R, S, N, DSC, CC, MSE_train, MSE_test


def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap("tab10")
    else:
        cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(20, 20))

    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=[cmap(cl)], label=cl, s=375)
        ax.axis("off")

    if figname is not None:
        fig.savefig(figname)

    plt.close("all")
    del fig
    del ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment_1")

    parser.add_argument("--output-dir", type=str, default="results_direct")
    parser.add_argument(
        "--datasets", nargs="*", default=["mnist", "fashionmnist"]#, "har", "reuters", "usps"]
    )
    parser.add_argument("--append-metrics", action="store_true", default=False)
    args = parser.parse_args()

    patience = 5
    epochs = 200

    min_delta = 0.05

    verbose = False
    if args.append_metrics:
        assert os.path.exists(
            args.output_dir
        ), "when using --append-metrics, output_dir must already exist"
        df = pd.read_csv(os.path.join(args.output_dir, "metrics.csv"))
        results = df.to_records(index=False).tolist()
    else:
        results = []

    output_dir = args.output_dir
    print(f"Outputting data to {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = "./data"
    data_dirs = args.datasets  # ["mnist", "fashionmnist", "har", "reuters", "usps"]

    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10 * 2
    epochs_dataset["mnist"] = 10 * 5
    epochs_dataset["har"] = 10 * 2
    epochs_dataset["reuters"] = 10 * 2
    epochs_dataset["usps"] = 10 * 2
    epochs_dataset["quickdraw2"] = 50

    classes_mult = {}
    classes_mult["fashionmnist"] = 2
    classes_mult["mnist"] = 1
    classes_mult["har"] = 2
    classes_mult["reuters"] = 1
    classes_mult["usps"] = 2
    classes_mult["quickdraw2"] = 1

    for d in data_dirs:
        dataset_name = d

        X = np.load(os.path.join(data_dir, d, "X.npy"))
        y = np.load(os.path.join(data_dir, d, "y.npy"))

        print("------------------------------------------------------")
        print("Dataset: {0}".format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        n_clusters = len(np.unique(y)) * classes_mult.get(dataset_name, 1)
        n_samples = X.shape[0]

        train_size = min(int(n_samples * 0.9), 10000)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=420, stratify=y
        )
        label_bin = LabelBinarizer()
        label_bin.fit(y_train)
        D_high = metrics.compute_distance_list(X_train)

        epochs = epochs_dataset.get(dataset_name, 50)

        # sharp_gt = sharp.ShaRP(
        #     X.shape[1],
        #     2,
        #     len(np.unique(y_train)),
        #     "diagonal_normal",
        # )
        sharp_gt = sharp.ShaRP(
            X.shape[1],
            n_clusters,
            "diagonal_normal",
            latent_dim=2,
            variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
            var_leaky_relu_alpha=-0.0001,
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.1,
        )

        sharp_gt.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=256,
        )
        X_sharp_gt = sharp_gt.transform(X_train)
        X_inv_sharp_gt = sharp_gt.inverse_transform(X_sharp_gt)
        X_inv_sharp_gt_test = sharp_gt.inverse_transform(sharp_gt.transform(X_test))
        print("CORRECT")


        D_sharp_gt = metrics.compute_distance_list(X_sharp_gt)

        # results.append(
        #     (dataset_name, "ShaRP-GT")
        #     + compute_all_metrics(
        #         X_train,
        #         X_sharp_gt,
        #         D_high,
        #         D_sharp_gt,
        #         y_train,
        #         X_inv_sharp_gt,
        #         X_test,
        #         X_inv_sharp_gt_test,
        #     )
        # )


        for X_, label in zip(
            [
                X_sharp_gt,
            ],
            [
                "ShaRP-GT",
            ],
        ):
            fname = os.path.join(output_dir, "{0}_{1}.png".format(dataset_name, label))
            print(fname)
            plot(X_, y_train, fname)

        # df = pd.DataFrame(
        #     results,
        #     columns=[
        #         "dataset_name",
        #         "test_name",
        #         "T_train",
        #         "C_train",
        #         "R_train",
        #         "S_train",
        #         "N_train",
        #         "DSC_train",
        #         "CC_train",
        #         "MSE_train",
        #         "MSE_test",
        #     ],
        # )

        # df.to_csv(os.path.join(output_dir, "metrics.csv"), header=True, index=None)

    # don't plot NNP
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    pri_images = [
        "ShaRP-GT",
        "ShaRP-KMeans",
        "ShaRP-AG",
    ]

    images = glob(output_dir + "/*.png")
    base = 2000

    for d in data_dirs:
        dataset_name = d
        to_paste = []

        for i, label in enumerate(pri_images):
            to_paste += [
                f
                for f in images
                if os.path.basename(f) == "{0}_{1}.png".format(dataset_name, label)
            ]

        img = np.zeros((base, base * len(pri_images), 3)).astype("uint8")

        for i, im in enumerate(to_paste):
            tmp = io.imread(im)
            img[:, i * base : (i + 1) * base, :] = tmp[:, :, :3]

        pimg = Image.fromarray(img)
        pimg.save(output_dir + "/composite_full_{0}.png".format(dataset_name))

        for i, label in enumerate(pri_images):
            print(
                "/composite_full_{0}.png".format(dataset_name),
                "{0} {1}".format(dataset_name, label),
            )

    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    # pri_images = ["SSNP-KMeans", "SSNP-AG", "AE"]

    # images = glob(output_dir + "/*.png")
    # base = 2000

    # for d in data_dirs:
    #     dataset_name = d
    #     to_paste = []

    #     for i, label in enumerate(pri_images):
    #         to_paste += [
    #             f
    #             for f in images
    #             if os.path.basename(f) == "{0}_{1}.png".format(dataset_name, label)
    #         ]

    #     img = np.zeros((base, base * 3, 3)).astype("uint8")

    #     for i, im in enumerate(to_paste):
    #         tmp = io.imread(im)
    #         img[:, i * base : (i + 1) * base, :] = tmp[:, :, :3]

    #     pimg = Image.fromarray(img)
    #     pimg.save(output_dir + "/composite_{0}.png".format(dataset_name))

    #     for i, label in enumerate(pri_images):
    #         print(
    #             "/composite_{0}.png".format(dataset_name),
    #             "{0} {1}".format(dataset_name, label),
    #         )
