import io
from typing import Any
from matplotlib import pyplot as plt
import numpy as np

import plotly.graph_objects as go
from sklearn.preprocessing import minmax_scale
from PIL import Image, ImageTk

from utils.scatterplot import apply_local_points_to_alpha

def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap('tab10')
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

def make_titles(start, step, size):
    titles = []
    for i in range(size):
        for j in range(size):
            title = f"({start[0]+i*step[0]},{start[1]+j*step[1]})"
            titles.append(title)

    return titles


def make_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, v1: float, v2: float, side_length: int
) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, side_length), np.linspace(y_min, y_max, side_length)
    )
    
    return np.array([[i[0], i[1], v1, v2] for i in np.c_[xx.ravel(), yy.ravel()]])


def get_bounding_box(X_proj: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = X_proj.min(axis=0)
    x_max, y_max = X_proj.max(axis=0)

    return x_min, x_max, y_min, y_max

# generate a grid of coordinates in the latent space
def generate_grid_coords(latent_range=(0.0, 1.0), img_size=0.05, proximity=1.6, v1=0.0, v2=0.0):
    num_cols = int(proximity*(latent_range[1]-latent_range[0]) / img_size)
    num_rows = int(proximity*(latent_range[1]-latent_range[0]) / img_size)
    xs = np.linspace(latent_range[0], latent_range[1], num_cols)
    ys = np.linspace(latent_range[0], latent_range[1], num_rows)
    xx, yy = np.meshgrid(xs, ys)
    coords = [[i[0], i[1], v1, v2] for i in np.stack([xx.ravel(), yy.ravel()], axis=1)]
    return coords

def generate_map_w_scatterplots(
    data: np.ndarray,
    labels: np.ndarray,
    grid_res: int,
    locally: bool,
    map_extra_coords: list,
    augmentation: np.ndarray,
    fig: Any,
):
    np.set_printoptions(suppress=True, precision=4)
    cmap_points = plt.get_cmap("tab10")
    cmap_colors = cmap_points(labels)

    if locally:
        cmap_colors = apply_local_points_to_alpha(cmap_colors, augmentation, map_extra_coords)

    groundtruth_colors = [f"rgba({i[0]:.5f}, {i[1]:.5f}, {i[2]:.5f}, {i[3]:.5f})" for i in cmap_colors]
    line_colors = [f"rgba(0, 0, 0, {i[3]:.5f})" for i in cmap_colors]
    fig.add_trace(
        go.Scatter(
            x=minmax_scale(data[:,0])*(grid_res-1), 
            y=minmax_scale(data[:,1])*(grid_res-1), 
            marker=dict(
                size=8,
                line=dict(
                    color=line_colors, width=1
                ),
            ),
            unselected=dict(
                marker=dict(
                    opacity=1  # Unselected points remain fully opaque
                )
            ),
            marker_color=groundtruth_colors,
            mode='markers', 
            visible=True,
            hoverinfo='skip',
            showlegend=False
        )
    )

    return fig

def plotly_to_image_tk(fig, max_w=700, max_h=700):
    png_bytes = fig.to_image(format="png", scale=2)  # requires kaleido
    im = Image.open(io.BytesIO(png_bytes))
    # resize to fit
    im.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(im)