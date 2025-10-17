#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import subprocess

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
# from ipycanvas import canvas

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = 'plotly' 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from MulticoreTSNE import MulticoreTSNE as TSNE
# from umap import UMAP
from utils.augmentations import get_augmentation_pca
from utils.dbm import gen_and_save_dbm
from utils.dbm_matrix import gen_and_save_dbm_matrix
from utils.metrics import metric_distance_to_nearest_neighbor
from utils.maps import gen_and_save_ccm, gen_and_save_nnm
from utils.utils import get_bounding_box, make_grid, make_titles

import models.sharp as sharp
import models.ssnp as ssnp
import models.nninv as nninv

from training.classifier import load_or_fit_mlp_classifier
from training.auto_encoders import load_or_fit_model_ae
from training.inv_proj import load_or_fit_model_inv_proj

cmap_main = plt.get_cmap("tab10")
cmap_nn   = plt.get_cmap("viridis")

@st.cache_resource
def Load_data(path, dataset):
    X = np.load(os.path.join(path, dataset, "X.npy"))
    y = np.load(os.path.join(path, dataset, "y.npy"))
    return X, y

@st.cache_resource
def get_inv_proj_data_ae(output_dir, _model, dataset_name, model_name, method, epochs, random_state):
    data_dir = "./data/"
    X, y = Load_data(data_dir, dataset_name)

    n_samples = X.shape[0]
    train_size = min(int(n_samples * 0.9), 5000)

    X, _, y, _ = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    augmentation = get_augmentation_pca(X)

    X = np.concatenate((X,augmentation), axis=1)

    _, X_test, _, y_test = train_test_split(
        X, y, train_size=int(train_size*0.9), random_state=random_state, stratify=y
    )

    augmentation = X[:,-2:]
    X = X[:,:-2]
    augmentation_test = X_test[:,-2:]
    X_test = X_test[:,:-2]

    X_proj, _model, limits = load_or_fit_model_ae(X, y, augmentation, X_test, output_dir, _model, dataset_name, model_name, method, epochs)
    classifier = load_or_fit_mlp_classifier(X, y, f'{output_dir}/{dataset_name}')

    neighbor_finder_model = NearestNeighbors(n_neighbors=5) 
    neighbor_finder_model.fit(X)

    return X_proj, y_test, augmentation_test, _model, classifier, neighbor_finder_model, limits

@st.cache_resource
def get_inv_proj_data_mlp(output_dir, _model, _inv_model, dataset_name, model_name, method, epochs, random_state):
    data_dir = "./data/"
    X, y = Load_data(data_dir, dataset_name)

    n_samples = X.shape[0]
    train_size = min(int(n_samples * 0.9), 30000)

    X, _, y, _ = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    _, X_test, _, y_test = train_test_split(
        X, y, train_size=int(train_size*0.9), random_state=random_state, stratify=y
    )

    augmentation = get_augmentation_pca(X)

    X_proj, _inv_model, limits = load_or_fit_model_inv_proj(X, y, augmentation, X_test, output_dir, _model, _inv_model, dataset_name, model_name, method, epochs, random_state)
    classifier = load_or_fit_mlp_classifier(X, y, f'{output_dir}/{dataset_name}')
    return X_proj, _inv_model, classifier, limits

@st.cache_resource
def get_nn_matrix(results_2d, _nn_model, _inv_model, grid_res, start, step, size):
    metric_matrix = np.zeros((size*size,grid_res*grid_res))
    bounding_box = get_bounding_box(results_2d)

    for i in range(size):
        for j in range(size):
            grid = make_grid(*bounding_box, start[0]+i*step[0], start[1]+j*step[1], grid_res)
            inverted_grid = inv_model.inverse_transform(grid)

            metric_matrix[size*i+j] = metric_distance_to_nearest_neighbor(inverted_grid, _nn_model)
    
    max_v = np.max(metric_matrix)
    min_v = np.min(metric_matrix)

    return minmax_scale(metric_matrix), max_v, min_v

@st.cache_resource
def get_matrix_fig(results_2d, labels, _clf, _inv_model, grid_res, start, step, size):
    fig = make_subplots(rows=size, cols=size, horizontal_spacing=0.01, vertical_spacing=0.01, subplot_titles=titles)
    for i in range(size):
        for j in range(size):
            fig = gen_and_save_dbm_matrix(results_2d, labels, _clf, _inv_model, grid_res, i, j, fig, start, step)
    
    fig.update_layout(
        hovermode='closest',
        width=1000,  # Set the width in pixels
        height=800,  # Set the height in pixels
        xaxis=dict(visible=False),  # Hide x-axis
        yaxis=dict(visible=False),  # Hide y-axis
        margin=dict(l=0, r=0, t=15, b=0), # Remove margins
    )
    fig.update_annotations(font_size=10, yshift=-5) # New coordinates
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def update_radio1_options():
    """Callback function to update radio2 options based on radio1 selection."""
    if st.session_state.radio2_key == st.session_state.radio1_key:
        st.session_state.radio1_value = "Off" # Reset selected value
        st.session_state.radio1_key = "Off" # Reset selected value

def update_radio2_options():
    """Callback function to update radio2 options based on radio1 selection."""
    if st.session_state.radio2_key == st.session_state.radio1_key:
        st.session_state.radio2_value = "Off" # Reset selected value
        st.session_state.radio2_key = "Off" # Reset selected value

if __name__ == "__main__":

    python_executable = sys.executable
    if not os.path.exists("data"):
        subprocess.run([python_executable, "get_data.py"])

    st.set_page_config(layout="wide")


    output_dir = "weights"
    # model_name = st.sidebar.selectbox("Inverse Projection Method", ("ssnp", "sharp", "nninv"))
    # dataset = "mnist"
    dataset = st.sidebar.selectbox("Dataset Used", ("mnist", "fashionmnist")) # , "har", "reuters"

    # method = st.sidebar.selectbox("Training Method Used", ("latent_space", "noise")) # , "har", "reuters"
    
    epochs_dataset = {}
    epochs_dataset["fashionmnist"] = 10
    epochs_dataset["mnist"] = 10
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

    results_2d, labels, augmentation_values, inv_model, clf, nn_model, limits = get_inv_proj_data_ae( #get_inv_proj_data_sharp(output_dir)
        output_dir, 
        sharp.ShaRP(
            # dims,
            # classes,
            # "laplace",
            # latent_dim=4,
            # variational_layer_kwargs=dict(prior_scale=0.1),
            # var_leaky_relu_alpha=-0.0001,
            # bottleneck_activation="linear",
            # bottleneck_l1=0.0,
            # bottleneck_l2=0.5,
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
        "sharp",
        "pca",
        epochs,
        420
    )

    sliderx_step = np.floor(np.log10(limits[1]-limits[0]))-2
    slidery_step = np.floor(np.log10(limits[3]-limits[2]))-2

    # print("here", sliderx_step, slidery_step)
    x = st.sidebar.slider('x', limits[0], limits[1], 0.0, step=10**(sliderx_step), format=f"%0.{np.array([np.abs(sliderx_step)], dtype=np.int8)[0]}f")
    y = st.sidebar.slider('y', limits[2], limits[3], 0.0, step=10**(slidery_step), format=f"%0.{np.array([np.abs(slidery_step)], dtype=np.int8)[0]}f")

    scatter = st.sidebar.radio(
        "Scatterplots",
        ["Off", "On", "Locally"],
        key="radio0_key"
    )

    closest_tp = st.sidebar.radio(
        "Nearest Training Point",
        ["Off", "On", "Exclusive"],
        key="radio1_key", # Unique key for this widget
        on_change=update_radio2_options,
    )

    class_confidence = st.sidebar.radio(
        "Classifier Confidence",
        ["Off", "On", "Exclusive"],
        key="radio2_key", # Unique key for this widget
        on_change=update_radio1_options,
    )

    grid_res = st.sidebar.selectbox("DBM Resolution", (50, 75, 100, 150, 200))
    start = (-1.0,-1.0)
    step  = (0.25,0.25)
    size = 9

    nn_matrix, nn_max_distance, nn_min_distance = get_nn_matrix(results_2d, nn_model, inv_model, grid_res, start, step, size)

    col1, col2 = st.columns(2)

    with col1:
        titles = make_titles(start,step,size)

        fig = get_matrix_fig(results_2d, labels, clf, inv_model, grid_res, start, step, size)
        # fig.show() # debug
        
        st.plotly_chart(fig, use_container_width=True)#, 

    with col2:
        fig2 = go.Figure()
        if closest_tp == "Exclusive":
            fig2 = gen_and_save_nnm(results_2d, labels, augmentation_values, clf, nn_model, inv_model, grid_res, x, y, fig2, scatter, class_confidence, cmap_nn)
        elif class_confidence == "Exclusive":
            fig2 = gen_and_save_ccm(results_2d, labels, augmentation_values, clf, nn_model, inv_model, grid_res, x, y, fig2, scatter, closest_tp, cmap_nn)
        else:
            fig2 = gen_and_save_dbm(results_2d, labels, augmentation_values, clf, nn_model, inv_model, grid_res, x, y, fig2, scatter, closest_tp, class_confidence, cmap_main)
        fig2.update_layout(
            hovermode='closest',
            width=1000,  # Set the width in pixels
            height=650,  # Set the height in pixels
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            margin=dict(l=100, r=0, t=0, b=0), # Remove margins
        )
       
        selected_points = st.plotly_chart(fig2, use_container_width=True, on_select="rerun", selection_mode="points", key="my_chart")#, 
        # st.write(selected_points)

        img = []
        if len(selected_points.selection.points) == 0:
            img = np.ones((28, 28, 3))
        else:
            point = selected_points.selection.points[0]
            img_1d = inv_model.inverse_transform(np.reshape(np.array([(point["x"]-25)/25,(point["y"]-25)/25,x,y]),(1,4)))
            img = np.reshape(img_1d,(28, 28))
            # print(img)
            img = 255*np.stack([img, img, img, np.ones(np.shape(img))], axis=-1)
        # print(np.shape(img))

        col21, col22 = st.columns([1, 2])

        with col21:
            fi = go.Figure()
            fi.add_trace(
                go.Image(z=img)
            )
            fi.update_layout(
                hovermode=False,
                width=400,  # Set the width in pixels
                height=100,  # Set the height in pixels
                xaxis=dict(visible=False),  # Hide x-axis
                yaxis=dict(visible=False),  # Hide y-axis
                margin=dict(l=0, r=0, t=5, b=0), # Remove margins
            )
            st.plotly_chart(fi)#, 
        with col22:
            st.markdown("""
                <style>
                .font {
                    font-size:14px !important; 
                    color: green;
                    margin: 0px;
                }
                </style>
                """, unsafe_allow_html=True
            )
            st.markdown('<p class="font">This text is big and blue!</p>', unsafe_allow_html=True)
            st.markdown('<p class="font">This text is small and green.</p>', unsafe_allow_html=True)



        
        # canvas = st.canvas(size=(200, 200))
        # # bg = load_image('test.png')
        # canvas.draw_image([1,1,1,1], 50, 50)

        # canvas.fill_rect(0, 0, 50, 50)

        # def handle_mouse_down(x, y):
        #     print("im here")
        #     pass
        # canvas.on_mouse_down(handle_mouse_down)
