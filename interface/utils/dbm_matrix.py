

from typing import Union, Any
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import models.sharp as sharp
import models.ssnp as ssnp
import models.nninv as nninv

cmap = plt.get_cmap("tab10")

def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, v1: float, v2: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )
    
    return np.array([[i[0], i[1], v1, v2] for i in np.c_[xx.ravel(), yy.ravel()]])

def generate_dbm(
    model: MLPClassifier,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    data: np.ndarray,
    labels: np.ndarray,
    grid_res: int,
    # ax: Axes,
    v1: float,
    v2: float, 
    start_step: tuple,
    step: tuple,
    fig: Any,
    cmap=cmap,
):
    pos1 = start_step[0]+step[0]*v1
    pos2 = start_step[1]+step[1]*v2

    bounding_box = get_bounding_box(data)
    grid = make_grid(*bounding_box, pos1, pos2, grid_res)
    aux = inverter.inverse_transform(grid)

    classes = model.predict(aux).astype(np.uint8)

    cmapped = cmap(classes)*255

    fig.add_trace(
        go.Image(z=np.reshape(cmapped,(grid_res, grid_res, 4))), row=v1+1, col=v2+1
    )
    return fig

def generate_dbm_w_scatterplots(
    data: np.ndarray,
    labels: np.ndarray,
    grid_res: int,
    v1: float,
    v2: float, 
    fig: Any,
    cmap=cmap,
):

    groundtruth_colors = [colors.to_hex(i) for i in cmap(labels)]

    fig.add_trace(
        go.Scatter(
            x=minmax_scale(data[:,0], feature_range=(0,grid_res)), 
            y=minmax_scale(data[:,1], feature_range=(0,grid_res)), 
            marker=dict(
                size=5,
                opacity=0.5,
                color=groundtruth_colors  # Assign the NumPy array of colors here
            ),
            marker_line_width=1,
            mode='markers', 
            visible=True,
            hoverinfo='none',
            showlegend=False
        ), row=v1+1, col=v2+1
        
    )

    return fig

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def gen_and_save_dbm_matrix(
    X_2d: np.ndarray,
    y: np.ndarray,
    classifier: ClassifierMixin,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    grid_res: int,
    v1: int,
    v2: int,
    fig: Any,
    scatter: bool,
    closest_tp:bool,
    start: tuple,
    step: tuple
):
    fig = generate_dbm(
        classifier,
        inverter,
        X_2d,
        y,
        grid_res=grid_res,
        # ax=ax,
        v1=v1,
        v2=v2,
        start_step=start,
        step=step,
        fig=fig,
        cmap=cmap if len(classifier.classes_) <= 10 else plt.get_cmap("tab20"),  
    )
    if scatter:    
        fig = generate_dbm_w_scatterplots(
            X_2d,
            y,
            grid_res=grid_res,
            v1=v1,
            v2=v2,
            fig=fig,
            cmap=cmap if len(classifier.classes_) <= 10 else plt.get_cmap("tab20"),        
        )
    # elif closest_tp:

              
    # print(fig)
    return fig
