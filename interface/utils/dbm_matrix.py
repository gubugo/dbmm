

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

from utils.utils import make_grid, get_bounding_box
import models.sharp as sharp
import models.ssnp as ssnp
import models.nninv as nninv

cmap = plt.get_cmap("tab10")

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

def gen_and_save_dbm_matrix(
    X_2d: np.ndarray,
    y: np.ndarray,
    classifier: ClassifierMixin,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    grid_res: int,
    v1: int,
    v2: int,
    fig: Any,
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

              
    # print(fig)
    return fig
