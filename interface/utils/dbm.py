

import math
import string
from typing import Union, Any
from matplotlib import colors, pyplot as plt
import numpy as np
from PIL import Image

from sklearn.base import ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

import plotly.graph_objects as go

from interface.utils.metrics import metric_distance_to_nearest_neighbor
from interface.utils.utils import generate_grid_coords, generate_map_w_scatterplots, make_grid, get_bounding_box

import interface.models.sharp as sharp
import interface.models.ssnp as ssnp
import interface.models.nninv as nninv

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
    
    cmapped = cmapped*255

    fig.add_trace(
        go.Image(z=np.reshape(cmapped,(grid_res, grid_res, 4)))
    )
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
    cmap: Any
):
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

              
    # print(fig)
    return fig, ret_values


def gen_images_grid_plotly(
    ssnp_model,
    clf,
    data: np.ndarray,
    grid_res: int,
    pos1: float,
    pos2: float,
    image_shape=(28, 28),
    latent_range=(-1.0, 1.0),
    img_size=1.6,
    proximity=20.8,
    cmap=[]
    # figsize=(900, 700),
    # projection_technique_name="t-SNE",
    # dataset_name="MNIST",
    # cmap="gray",
    # save_path="generated_images_grid_plotly.html"
):
    # cria figura
    fig = go.Figure()

    # gera coordenadas do grid
    bounding_box = get_bounding_box(data)
    grid = make_grid(*bounding_box, pos1, pos2, grid_res)
    classes = clf.predict(ssnp_model.inverse_transform(grid))
    coords = generate_grid_coords(latent_range, img_size, proximity, pos1, pos2)

    if len(coords) == 0:
        print("Nenhuma coordenada gerada para o grid.")
        return

    # gera imagens com o modelo
    images = ssnp_model.inverse_transform(coords)
    images = images.reshape(-1, *image_shape)

    # normaliza para 0–255
    images = (
        255 * (images - images.min()) /
        (images.max() - images.min())
    ).astype(np.uint8)

    # adiciona imagens ao gráfico
    cmapped = 255*cmap(classes)
    fig.add_trace(
        go.Image(z=np.reshape(cmapped,(grid_res, grid_res, 4)))
    )
    for values, img_array in zip(coords, images):
        # print(x)
        # print(y)
        # print(np.shape(img_array))
        img2 = np.stack([img_array, img_array, img_array], axis=-1)
        img = Image.fromarray(img2, mode="RGB")

        # converte pra RGB (plotly exige RGB)
        # img = Image.merge("RGB", (img, img, img))

        # adiciona imagem centralizada na coordenada
        fig.add_layout_image(
            dict(
                source=img,
                x=50*(values[0]+1)/2-1,
                y=50*(values[1]+1)/2-1,
                sizex=img_size,
                sizey=img_size,
                xref="x",
                yref="y",
                layer="above",
                opacity=1
            )
        )
    return fig
    # # layout final
    # fig.update_layout(
    #     width=figsize[0],
    #     height=figsize[1],
    #     xaxis=dict(showgrid=False, zeroline=False, range=list(latent_range)),
    #     yaxis=dict(showgrid=False, zeroline=False, range=list(latent_range),
    #                scaleanchor="x", scaleratio=1),
    #     plot_bgcolor="white",
    #     margin=dict(l=0, r=0, t=50, b=0)
    # )

    # # salvar HTML
    # output_dir = os.path.join("results", projection_technique_name, dataset_name)
    # os.makedirs(output_dir, exist_ok=True)
    # filepath = os.path.join(output_dir, save_path)
    # fig.write_html(filepath)

