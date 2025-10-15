

import string
from typing import Union, Any
from matplotlib import colors, pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

import plotly.graph_objects as go

from utils.metrics import metric_distance_to_nearest_neighbor
from utils.utils import generate_map_w_scatterplots, make_grid, get_bounding_box
import models.sharp as sharp
import models.ssnp as ssnp
import models.nninv as nninv

def generate_nnm(
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    data: np.ndarray,
    grid_res: int,
    # ax: Axes,
    pos1: float,
    pos2: float, 
    fig: Any,
    cmap: Any,
):
    bounding_box = get_bounding_box(data)
    grid = make_grid(*bounding_box, pos1, pos2, grid_res)
    inverted_grid = inverter.inverse_transform(grid)
    metric_matrix = metric_distance_to_nearest_neighbor(inverted_grid, nn_model)

    # res = model.predict_proba(inverted_grid)

    cmapped = cmap(minmax_scale(metric_matrix))
    # print(np.shape(cmapped))
    v1 = np.max(metric_matrix)
    v2 = np.min(metric_matrix)
    
    cmapped = cmapped*255

    fig.add_trace(
        go.Image(z=np.reshape(cmapped,(grid_res, grid_res, 4)))
    )
    return fig, v1, v2

def gen_and_save_nnm(
    X_2d: np.ndarray,
    y: np.ndarray,
    aug: np.ndarray,
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    grid_res: int,
    pos1: int,
    pos2: int,
    fig: Any,
    scatter: string,
    cmap: Any
):
    fig, v1, v2 = generate_nnm(
        nn_model,
        inverter,
        X_2d,
        grid_res=grid_res,
        # ax=ax,
        pos1=pos1,
        pos2=pos2,
        fig=fig,
        cmap=cmap,  
    )
    if scatter == "On":    
        fig = generate_map_w_scatterplots(
            X_2d,
            y,
            grid_res=grid_res,
            locally=False,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,     
        )
    elif scatter == "Locally":
        fig = generate_map_w_scatterplots(
            X_2d,
            y,
            grid_res=grid_res,
            locally=True,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,     
        )
    # elif closest_tp:

              
    # print(fig)
    return fig

