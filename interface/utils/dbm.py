

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
    fig: Any,
    cmap=cmap,
):
    bounding_box = get_bounding_box(data)
    grid = make_grid(*bounding_box, v1, v2, grid_res)
    aux = inverter.inverse_transform(grid)

    classes = model.predict(aux).astype(np.uint8)

    cmapped = cmap(classes)*255

    fig.add_trace(
        go.Image(z=np.reshape(cmapped,(grid_res, grid_res, 4)))
    )
    fig.update_layout(
      hovermode=False,
      xaxis=dict(visible=False),  # Hide x-axis
      yaxis=dict(visible=False),  # Hide y-axis
      margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
    )
    return fig

def generate_dbm_w_scatterplots(
    model: MLPClassifier,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    data: np.ndarray,
    labels: np.ndarray,
    grid_res: int,
    # ax: Axes,
    v1: float,
    v2: float,
    fig: Any,
    cmap=cmap,
):
    bounding_box = get_bounding_box(data)
    grid = make_grid(*bounding_box, v1, v2, grid_res)
    aux = inverter.inverse_transform(grid)

    classes = model.predict(aux).astype(np.uint8)

    print(np.shape(cmap(labels)))
    gd_v = [colors.to_hex(i) for i in cmap(labels)]

    cmapped = cmap(classes)*255

    # fig.add_trace(
    #     go.Image(z=np.reshape(cmapped,(grid_res, grid_res, 4)))
    # )
    
    fig = px.imshow(np.reshape(cmapped,(grid_res, grid_res, 4)))
    fig.add_trace(
        go.Scatter(
            x=data[:100,0], 
            y=minmax_scale(data[:100,1], feature_range=(0,grid_res)), 
            marker=dict(
                size=10,
                color=gd_v  # Assign the NumPy array of colors here
            ),
            marker_line_width=1,
            mode='markers', 
            name='Initial Trace',
            visible=True
        )
        
    )
    # fig.update_layout(
    #   hovermode=False,
    #   xaxis=dict(visible=False),  # Hide x-axis
    #   yaxis=dict(visible=False),  # Hide y-axis
    #   margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
    # )

    return fig

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def gen_and_save_dbm(
    X_2d: np.ndarray,
    y: np.ndarray,
    classifier: ClassifierMixin,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    output_dir: str,
    grid_res: int,
    dataset_name: str,
    alg_name: str,
    v1: float,
    v2: float,
    fig: Any,
):
    # fig, ax = plt.subplots(figsize=(20, 20))
    fig = generate_dbm_w_scatterplots(
        classifier,
        inverter,
        X_2d,
        y,
        grid_res=grid_res,
        # ax=ax,
        v1=v1,
        v2=v2,
        fig=fig,
        cmap=cmap if len(classifier.classes_) <= 10 else plt.get_cmap("tab20"),        
    )
    # print(fig)
    return fig
