

import string
from typing import Union, Any
from matplotlib import colors, pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

import plotly.graph_objects as go

from utils.scatterplot import plot_points_on_decision_map
from utils.metrics import metric_distance_to_nearest_neighbor
from utils.utils import generate_map_w_scatterplots, make_grid, get_bounding_box
import models.sharp as sharp
import models.ssnp as ssnp
import models.nninv as nninv

def generate_ccm(
    model: MLPClassifier,
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    data: np.ndarray,
    grid_res: int,
    # ax: Axes,
    pos1: float,
    pos2: float, 
    closest_tp: string,
    fig: Any,
    cmap: Any,
):
    bounding_box = get_bounding_box(data)
    grid = make_grid(*bounding_box, pos1, pos2, grid_res)
    inverted_grid = inverter.inverse_transform(grid)

    res = model.predict_proba(inverted_grid)
     
    confidence = np.zeros(np.shape(res)[0])

    for i,lis in enumerate(res):
        confidence[i] = np.max(lis)

    cmapped = cmap(confidence)

    ret_values = (0,0)

    if closest_tp == "On":
        metric_matrix = metric_distance_to_nearest_neighbor(inverted_grid, nn_model)
        ret_values = (np.max(metric_matrix),np.min(metric_matrix))
        scaled_mm = minmax_scale(metric_matrix)
        cmapped[:,0] = cmapped[:,0]*scaled_mm
        cmapped[:,1] = cmapped[:,1]*scaled_mm
        cmapped[:,2] = cmapped[:,2]*scaled_mm

    coords = f"({pos1},{pos2})"
    # fig.clear()
    fig.imshow(
        cmapped.reshape((grid_res, grid_res, 4)),
        origin="lower",
        interpolation="none",
        resample=False,
    )
    fig.set_title(coords, fontsize=grid_res/5, x=0.5, y=1)
    fig.axis("off") 
    return fig, ret_values

def gen_and_save_ccm(
    X_2d: np.ndarray,
    y: np.ndarray,
    aug: np.ndarray,
    clf: MLPClassifier,
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    grid_res: int,
    pos1: int,
    pos2: int,
    fig: Any,
    scatter: string,
    closest_tp: string,
    cmap: Any
):
    fig.clear()
    fig, ret_values = generate_ccm(
        clf,
        nn_model,
        inverter,
        X_2d,
        grid_res=grid_res,
        # ax=ax,
        pos1=pos1,
        pos2=pos2,
        closest_tp=closest_tp,
        fig=fig,
        cmap=cmap,  
    )
    if scatter == "On":    
        cmap_points = plt.get_cmap("tab10")
        fig = plot_points_on_decision_map(
            X_2d,
            cmap_points(y),
            grid_res=grid_res,
            locally=False,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,     
        )
    elif scatter == "Locally":
        cmap_points = plt.get_cmap("tab10")
        fig = plot_points_on_decision_map(
            X_2d,
            cmap_points(y),
            grid_res=grid_res,
            locally=True,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,     
        )
    # elif closest_tp:

              
    # print(fig)
    return fig, ret_values

def generate_nnm(
    clf: MLPClassifier,
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    data: np.ndarray,
    grid_res: int,
    # ax: Axes,
    pos1: float,
    pos2: float, 
    class_confidence: string,
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
    ret_values = (np.max(metric_matrix),np.min(metric_matrix))
    
    if class_confidence == "On":
        res = clf.predict_proba(inverted_grid)
     
        confidence = np.zeros(np.shape(res)[0])

        for i,lis in enumerate(res):
            confidence[i] = np.max(lis)
        cmapped[:,0] = cmapped[:,0]*confidence
        cmapped[:,1] = cmapped[:,1]*confidence
        cmapped[:,2] = cmapped[:,2]*confidence

    coords = f"({pos1},{pos2})"
    # fig.clear()
    fig.imshow(
        cmapped.reshape((grid_res, grid_res, 4)),
        origin="lower",
        interpolation="none",
        resample=False,
    )
    fig.set_title(coords, fontsize=grid_res/5, x=0.5, y=1)
    fig.axis("off") 
    return fig, ret_values

def gen_and_save_nnm(
    X_2d: np.ndarray,
    y: np.ndarray,
    aug: np.ndarray,
    clf: MLPClassifier,
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    grid_res: int,
    pos1: int,
    pos2: int,
    fig: Any,
    scatter: string,
    class_confidence: string,
    cmap: Any
):
    fig.clear()
    fig, ret_values = generate_nnm(
        clf,
        nn_model,
        inverter,
        X_2d,
        grid_res=grid_res,
        # ax=ax,
        pos1=pos1,
        pos2=pos2,
        class_confidence=class_confidence,
        fig=fig,
        cmap=cmap,  
    )
    if scatter == "On":    
        cmap_points = plt.get_cmap("tab10")
        fig = plot_points_on_decision_map(
            X_2d,
            cmap_points(y),
            grid_res=grid_res,
            locally=False,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,     
        )
    elif scatter == "Locally":
        cmap_points = plt.get_cmap("tab10")
        fig = plot_points_on_decision_map(
            X_2d,
            cmap_points(y),
            grid_res=grid_res,
            locally=True,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,     
        )
    # elif closest_tp:

              
    # print(fig)
    return fig, ret_values