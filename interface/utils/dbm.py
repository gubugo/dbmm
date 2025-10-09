

from typing import Union
from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier
import plotly.express as px

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

def dbm_for_estimator(
    model: MLPClassifier,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    bounding_box: tuple[float, float, float, float],
    grid_res: int,
    # ax: Axes,
    v1: float,
    v2: float, 
    cmap=cmap,
):
    grid = make_grid(*bounding_box, v1, v2, grid_res)
    aux = inverter.inverse_transform(grid)

    classes = model.predict(aux).astype(np.uint8)

    cmapped = cmap(classes)*255

    fig = px.imshow(np.reshape(cmapped,(grid_res, grid_res, 4)))
    fig.update_layout(
      hovermode=False,
      xaxis=dict(visible=False),  # Hide x-axis
      yaxis=dict(visible=False),  # Hide y-axis
      margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
    )
    return fig

def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

def gen_and_save_dbm(
    X_2d: np.ndarray,
    classifier: ClassifierMixin,
    inverter: Union[sharp.ShaRP, ssnp.SSNP, nninv.NNInv],
    output_dir: str,
    grid_res: int,
    dataset_name: str,
    alg_name: str,
    v1: float,
    v2: float,
):
    # fig, ax = plt.subplots(figsize=(20, 20))
    fig = dbm_for_estimator(
        classifier,
        inverter,
        get_bounding_box(X_2d),
        grid_res=grid_res,
        # ax=ax,
        v1=v1,
        v2=v2,
        cmap=cmap if len(classifier.classes_) <= 10 else plt.get_cmap("tab20"),
    )
    # print(fig)
    return fig
