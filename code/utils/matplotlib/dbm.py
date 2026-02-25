

import math
import string
from typing import Union, Any
from matplotlib import colors, pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

import plotly.graph_objects as go

from code.utils.scatterplot import plot_points_on_decision_map
from code.utils.metrics import metric_distance_to_nearest_neighbor
from code.utils.utils import generate_grid_coords, generate_map_w_scatterplots, make_grid, get_bounding_box

import code.models.tensorflow.sharp as sharp
import code.models.tensorflow.ssnp as ssnp
import code.models.tensorflow.nninv as nninv

def generate_dbm(
    model: MLPClassifier,
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    data: np.ndarray,
    labels: np.ndarray,
    grid_res: int,
    # ax: Axes,
    pos1: float,
    pos2: float, 
    fig: Any,
    closest_tp: string,
    class_confidence: string,
    cmap: Any,
):
    bounding_box = get_bounding_box(data)
    grid = make_grid(*bounding_box, pos1, pos2, grid_res)
    inverted_grid = inverter.inverse_transform(grid)

    classes = model.predict(inverted_grid).astype(np.uint8)

    res = model.predict_proba(inverted_grid)
     
    confidence = np.zeros(np.shape(classes))

    for i,lis in enumerate(res):
        confidence[i] = np.max(lis)

    cmapped = cmap(classes)
    # print(np.shape(cmapped))
    ret_values = (0,0)

    if closest_tp == "On":
        metric_matrix = metric_distance_to_nearest_neighbor(inverted_grid, nn_model)
        ret_values = (np.max(metric_matrix),np.min(metric_matrix))
        scaled_mm = 1-minmax_scale(metric_matrix)
        cmapped[:,0] = cmapped[:,0]*scaled_mm
        cmapped[:,1] = cmapped[:,1]*scaled_mm
        cmapped[:,2] = cmapped[:,2]*scaled_mm

    if class_confidence == "On":
        res = model.predict_proba(inverted_grid)

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
    fig.set_title(coords, fontsize=10, x=0.5, y=1)
    fig.axis("off") 
    # fig.draw()
    return fig, ret_values

def gen_and_save_dbm(
    X_2d: np.ndarray,
    y: np.ndarray,
    aug: np.ndarray,
    classifier: ClassifierMixin,
    nn_model: NearestNeighbors,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    grid_res: int,
    pos1: int,
    pos2: int,
    fig: Any,
    scatter: string, 
    closest_tp: string,
    class_confidence: string,
    cmap: Any,
    reconstruct: string
):
    fig.clear()
    fig, ret_values = generate_dbm(
        classifier,
        nn_model,
        inverter,
        X_2d,
        y,
        grid_res=grid_res,
        # ax=ax,
        pos1=pos1,
        pos2=pos2,
        fig=fig,
        closest_tp=closest_tp,
        class_confidence=class_confidence,
        cmap=cmap,  
    )
    if scatter == "On":    
        fig = plot_points_on_decision_map(
            X_2d,
            cmap(y),
            grid_res=grid_res,
            locally=False,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,
        )
    elif scatter == "Locally":
        fig = plot_points_on_decision_map(
            X_2d,
            cmap(y),
            grid_res=grid_res,
            locally=True,
            map_extra_coords=[pos1,pos2],
            augmentation=aug,
            fig=fig,      
        )
    
    if reconstruct == "1":
        fig = plot_generated_images_grid_with_dbm(
            X_2d, 
            inverter, 
            grid_res, 
            pos1, 
            pos2, 
            fig
        )

              
    # print(fig)
    return fig, ret_values


# gets the decision map matrix and a NNInv model as parameters,
# generates a grid of images using the NNInv model on a uniform grid of coordinates,
# and plots them on top of the decision map, saving the result
def plot_generated_images_grid_with_dbm(data, inv_model, grid_res, pos1, pos2, ax,
                                        img_size=0.5, proximity=1.75,
                                        figsize=(10,8), cmap_images='gray'):
    bounding_box = get_bounding_box(data)

    # generate coordinates and images
    coords = generate_grid_coords(bounding_box, img_size, proximity, pos1, pos2)
    images = inv_model.inverse_transform(coords)
    images = images.reshape(-1, 28, 28)

    tcoords = np.array(coords).T

    tcoords[0] = minmax_scale(tcoords[0])*(grid_res-1)
    tcoords[1] = minmax_scale(tcoords[1])*(grid_res-1)

    coords = tcoords.T
    # print(minmax_scale(coords[]))
    # plot images on grid above DBM
    images = np.clip(images, 0, 1)
    for (x, y, z, w), img in zip(coords, images):
        img_obj = OffsetImage(img, cmap=cmap_images, zoom=img_size)
        ab = AnnotationBbox(img_obj, (x, y), frameon=False)
        ax.add_artist(ab)

    # ax.set_xlim(bounding_box[:2])
    # ax.set_ylim(bounding_box[-2:])
    ax.axis('off')

    return ax