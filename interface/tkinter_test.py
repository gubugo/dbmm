import os
import threading
import tkinter as tk
from tkinter import ttk

from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

from models import sharp
from training.auto_encoders import load_or_fit_model_ae
from training.classifier import load_or_fit_mlp_classifier
from training.inv_proj import load_or_fit_model_inv_proj
from utils.augmentations import get_augmentation_pca
from utils.dbm import gen_and_save_dbm, gen_images_grid_plotly
from utils.dbm_matrix import gen_and_save_dbm_matrix
from utils.maps import gen_and_save_ccm, gen_and_save_nnm
from utils.metrics import metric_distance_to_nearest_neighbor
from utils.utils import get_bounding_box, make_grid, make_titles, plotly_to_image_tk

from plotly.subplots import make_subplots

class LabeledScale(ttk.Frame):
    """A custom Tkinter widget that combines a ttk.Scale with a label for its value."""

    def __init__(self, parent, label_text="", from_value=0, to_value=100, value=0.0, command=None, **kwargs):
        super().__init__(parent)
        
        # Internal variable to hold the scale's value.
        self.var = tk.DoubleVar(self, value=value)

        self.command = command

        # Label for the scale widget.
        self.label = ttk.Label(self, text=f"{label_text}: {self.var.get():.2f}")
        self.label.pack(side="top", anchor="w", pady=(0, 5))

        # Scale widget. The 'variable' option links it to self.var.
        self.scale = ttk.Scale(
            self,
            from_=from_value,
            to=to_value,
            variable=self.var,
            command=self._update_label,
            **kwargs
        )
        self.scale.pack(fill="x", expand=True)
        
    def _update_label(self, value):
        """Callback to update the label text with the scale's current value."""
        self.label.config(text=f"{self.label.cget('text').split(':')[0]}: {float(value):.2f}")
        self.command()

    def get(self):
        """Get the current value of the scale."""
        return self.var.get()

    def set(self, value):
        """Set the current value of the scale."""
        self.var.set(value)
        self._update_label(value)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DBM Experiment — Tkinter")
        self.geometry("1600x800")
        self.minsize(1600, 800)

        # State
        self.results_2d = None
        self.labels = None
        self.augmentation_values = None
        self.inv_model = None
        self.clf = None
        self.nn_model = None
        self.limits = None

        self.nn_matrix = None
        self.nn_max_distance = None
        self.nn_min_distance = None

        self.generate_matrix = True
        self.data_loaded = False

        self.cmap_main = plt.get_cmap("tab10")
        self.cmap_nn   = plt.get_cmap("viridis")

        # UI Layout
        self._build_layout()

    def _build_layout(self):
        # Main grid: sidebar (0) and content (1)
        self.columnconfigure(0, weight=0)  # sidebar
        self.columnconfigure(1, weight=1)  # content
        self.rowconfigure(0, weight=1)

        # Sidebar
        sb = ttk.Frame(self, padding=12)
        sb.grid(row=0, column=0, sticky="ns")

        ttk.Label(sb, text="Dataset").grid(row=0, column=0, sticky="w")
        self.dataset = tk.StringVar(value="mnist")
        combobox1 = ttk.Combobox(sb, textvariable=self.dataset, state="readonly", values=["mnist", "fashionmnist"])
        combobox1.grid(row=1, column=0, sticky="ew", pady=(0,8))
        combobox1.bind("<<ComboboxSelected>>", self._combobox_wrap)

        ttk.Separator(sb).grid(row=2, column=0, sticky="ew", pady=8)

        self.x_v = LabeledScale(sb, label_text="x", from_value=-1.0, to_value=1.0, orient="horizontal", value=0.0, command=self.gen_dbm)
        self.x_v.grid(row=3, column=0, sticky="ew")
        self.y_v = LabeledScale(sb, label_text="y", from_value=-1.0, to_value=1.0, orient="horizontal", value=0.0, command=self.gen_dbm)
        self.y_v.grid(row=4, column=0, sticky="ew")

        ttk.Separator(sb).grid(row=5, column=0, sticky="ew", pady=8)

        style = ttk.Style()
        style.map("Toggle.TCheckbutton",
          background=[("selected", "green"), ("!selected", "red")],
          foreground=[("selected", "white"), ("!selected", "white")])
        self.images = tk.StringVar(value=0)
        self.toggle_button = ttk.Checkbutton(sb,text="Images Ontop",
                                variable=self.images,
                                command=self.gen_dbm,
                                style="Toggle.TCheckbutton").grid(row=7, column=0, sticky="w")

        ttk.Separator(sb).grid(row=8, column=0, sticky="ew", pady=8)

        ttk.Label(sb, text="Scatterplots").grid(row=9, column=0, sticky="w")
        self.scatter = tk.StringVar(value="Off")
        for i, lab in enumerate(["Off", "On", "Locally"]):
            ttk.Radiobutton(sb, text=lab, value=lab, variable=self.scatter, command=self.gen_dbm
                            ).grid(row=10+i, column=0, sticky="w")

        ttk.Separator(sb).grid(row=13, column=0, sticky="ew", pady=8)

        ttk.Label(sb, text="Nearest Training Point").grid(row=14, column=0, sticky="w")
        self.closest_tp = tk.StringVar(value="Off")
        for i, lab in enumerate(["Off", "On", "Exclusive"]):
            ttk.Radiobutton(sb, text=lab, value=lab, variable=self.closest_tp, command=self._closest_training_point_wrap
                            ).grid(row=15+i, column=0, sticky="w")

        ttk.Separator(sb).grid(row=18, column=0, sticky="ew", pady=8)

        ttk.Label(sb, text="Classifier Confidence").grid(row=19, column=0, sticky="w")
        self.class_conf = tk.StringVar(value="Off")
        for i, lab in enumerate(["Off", "On", "Exclusive"]):
            ttk.Radiobutton(sb, text=lab, value=lab, variable=self.class_conf, command=self._classifier_confidence_wrap
                            ).grid(row=20+i, column=0, sticky="w")

        ttk.Separator(sb).grid(row=23, column=0, sticky="ew", pady=8)

        ttk.Label(sb, text="DBM Grid Resolution").grid(row=24, column=0, sticky="w")
        self.resolution = tk.StringVar(value=50)
        combobox2 = ttk.Combobox(sb, textvariable=self.resolution, state="readonly", values=[50, 100, 150, 200, 300])
        combobox2.grid(row=25, column=0, sticky="ew", pady=(0,8))
        combobox2.bind("<<ComboboxSelected>>", self._combobox_wrap)
        
        # ttk.Button(sb, text="Compute Projection", command=self.compute_projection).grid(row=13, column=0, sticky="ew", pady=(10,4))
        # ttk.Button(sb, text="Update Plots", command=self.update_plots).grid(row=14, column=0, sticky="ew")

        # Content area
        content = ttk.Frame(self, padding=12)
        content.grid(row=0, column=1, sticky="nsew")
        content.rowconfigure(1, weight=1)
        content.rowconfigure(3, weight=1)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(content, textvariable=self.status).grid(row=0, column=0, sticky="w")

        ttk.Label(content, text="Decision Boundary Matrix").grid(row=1, column=0, sticky="w")
        self.img_label_matrix = ttk.Label(content)
        self.img_label_matrix.grid(row=2, column=0, sticky="nsew")

        ttk.Label(content, text="Neighbors / Confidence View").grid(row=0, column=1, sticky="w")
        self.img_label_right = ttk.Label(content)
        self.img_label_right.grid(row=2, column=1, sticky="nsew")

        self.thread = threading.Thread(target=self.gen_dbm, daemon=True)
        self.thread.start()

    def _combobox_wrap(self, _):
        self.gen_dbm()

    def _closest_training_point_wrap(self):
        closest_tp = self.closest_tp.get()
        class_conf = self.class_conf.get()
        if closest_tp == "On" and class_conf == "On":
            self.class_conf.set("Off")

        if closest_tp == "Exclusive" and class_conf == "Exclusive":
            self.class_conf.set("Off")

        self.gen_dbm()

    def _classifier_confidence_wrap(self):
        closest_tp = self.closest_tp.get()
        class_conf = self.class_conf.get()
        if closest_tp == "On" and class_conf == "On":
            self.closest_tp.set("Off")

        if closest_tp == "Exclusive" and class_conf == "Exclusive":
            self.closest_tp.set("Off")

        self.gen_dbm()

    def gen_dbm(self):
        print(" ")
        print("test")
        # print(v)
        print(self.x_v.get())
        print(self.y_v.get())
        print(self.resolution.get())
        print(self.class_conf.get())
        print(self.closest_tp.get())
        print(self.scatter.get())
        print(self.images.get())
        print(self.dataset.get())

        if not self.data_loaded:
            self.compute_projection()
        self.update_plots()


    def Load_data(self, path, dataset):
        X = np.load(os.path.join(path, dataset, "X.npy"))
        y = np.load(os.path.join(path, dataset, "y.npy"))
        return X, y

    def get_inv_proj_data_ae(self, output_dir, _model, dataset_name, model_name, method, epochs, random_state):
        data_dir = "./data/"
        X, y = self.Load_data(data_dir, dataset_name)

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

    def get_inv_proj_data_mlp(self, output_dir, _model, _inv_model, dataset_name, model_name, method, epochs, random_state):
        data_dir = "./data/"
        X, y = self.Load_data(data_dir, dataset_name)

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

    def get_nn_matrix(self, results_2d, _nn_model, _inv_model, grid_res, start, step, size):
        metric_matrix = np.zeros((size*size,grid_res*grid_res))
        bounding_box = get_bounding_box(results_2d)

        for i in range(size):
            for j in range(size):
                grid = make_grid(*bounding_box, start[0]+i*step[0], start[1]+j*step[1], grid_res)
                inverted_grid = _inv_model.inverse_transform(grid)

                metric_matrix[size*i+j] = metric_distance_to_nearest_neighbor(inverted_grid, _nn_model)
        
        max_v = np.max(metric_matrix)
        min_v = np.min(metric_matrix)

        return minmax_scale(metric_matrix), max_v, min_v

    def get_matrix_fig(self, results_2d, labels, _clf, _inv_model, grid_res, start, step, size):
        titles = make_titles(start,step,size)
        
        fig = make_subplots(rows=size, cols=size, horizontal_spacing=0.01, vertical_spacing=0.025, subplot_titles=titles)
        for i in range(size):
            for j in range(size):
                fig = gen_and_save_dbm_matrix(results_2d, labels, _clf, _inv_model, grid_res, i, j, fig, start, step)
        
        fig.update_layout(
            hovermode='closest',
            width=800,  # Set the width in pixels
            height=800,  # Set the height in pixels
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            margin=dict(l=1, r=1, t=17, b=12), # Remove margins
        )
        
        fig.update_annotations(font_size=11, yshift=0, font_color="black") # New coordinates
        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        
        return fig
    
    def compute_projection(self):
        """Run backend to obtain projection, models, etc."""
        # self.status.set("Computing projection…")
        # self.update_idletasks()

        # Attempt to call the AE path seen in the Streamlit version
        # Signature inferred from code slices:
        # get_inv_proj_data_ae(output_dir, _model, dataset_name, model_name, method, epochs, random_state)
        output_dir = "weights"
        dataset_name = self.dataset.get()
        model_name = "ae"
        method = "diagonal_normal"
        epochs = 10
        random_state = 420

        sharp_dims_classes = {}
        sharp_dims_classes["fashionmnist"] = [784, 10]
        sharp_dims_classes["mnist"] = [784, 10]
        sharp_dims_classes["har"]  = [561, 6]
        sharp_dims_classes["reuters"] = [5000, 6]

        dims = sharp_dims_classes[dataset_name][0]
        classes = sharp_dims_classes[dataset_name][1]

        # Build a default ShaRP model as in the code (with latent_dim=2 etc.)
        sharp_model = sharp.ShaRP(
            dims,
            classes,
            "diagonal_normal",
            latent_dim=2,
            variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
            var_leaky_relu_alpha=-0.0001,
            bottleneck_activation="linear",
            bottleneck_l1=0.0,
            bottleneck_l2=0.1,
        )

        self.results_2d, self.labels, self.augmentation_values, self.inv_model, self.clf, self.nn_model, self.limits = self.get_inv_proj_data_ae(
            output_dir,
            sharp_model,
            dataset_name,
            model_name,
            method,
            epochs,
            random_state
        )

        self.data_loaded = True


        # Pre-compute neighbor matrix used in left plot
        # start = (-1.0, -1.0)
        # step = (0.25, 0.25)
        # size = 9
        # grid_res = int(self.grid_res.get())
        # res2, err2 = safe_call(
        #     dbm_backend.get_nn_matrix,
        #     self.results_2d, self.nn_model, self.inv_model,
        #     grid_res, start, step, size
        # )
        # if err2:
        #     raise RuntimeError(err2)

        # self.nn_matrix, self.nn_max_distance, self.nn_min_distance = res2

    def update_plots(self):
        """Update both left (matrix) and right (neighbors/confidence) images."""
        # if self.results_2d is None:
        #     self.status.set("Please run 'Compute Projection' first.")
        #     return

        # self.status.set("Rendering plots…")
        # self.update_idletasks()

        # Left figure (matrix)
        start = (-1.0, -1.0)
        step = (0.25, 0.25)
        size = 9
        if self.generate_matrix:
            fig = self.get_matrix_fig(
                self.results_2d, self.labels, self.clf, self.inv_model,
                int(self.resolution.get()), start, step, size
            )
            self.generate_matrix = False
            img = plotly_to_image_tk(fig)
            self.image_matrix = img 
            self.img_label_matrix.configure(image=self.image_matrix, text="", compound=None)

        # Right figure (neighbors/confidence)

        fig2 = go.Figure()
        grid_res = int(self.resolution.get())
        x = float(self.x_v.get())
        y = float(self.y_v.get())
        scatter = bool(self.scatter.get())
        class_conf = bool(self.class_conf.get())
        closest_tp = self.closest_tp.get()

        fig2 = go.Figure()
        if self.images.get() == "0":
            print("yddd")
            if closest_tp == "Exclusive":
                fig2, ret_values = gen_and_save_nnm(self.results_2d, self.labels, self.augmentation_values, self.clf, self.nn_model, self.inv_model, grid_res, x, y, fig2, scatter, class_conf, self.cmap_nn)
            elif class_conf == "Exclusive":
                fig2, ret_values = gen_and_save_ccm(self.results_2d, self.labels, self.augmentation_values, self.clf, self.nn_model, self.inv_model, grid_res, x, y, fig2, scatter, closest_tp, self.cmap_nn)
            else:
                fig2, ret_values = gen_and_save_dbm(self.results_2d, self.labels, self.augmentation_values, self.clf, self.nn_model, self.inv_model, grid_res, x, y, fig2, scatter, closest_tp, class_conf, self.cmap_main)
        else:
            print("wefwef")
            fig2 = gen_images_grid_plotly(self.inv_model, self.clf, self.results_2d, grid_res, x, y, cmap=self.cmap_main)
            ret_values = (0.0,0.0)
        fig2.update_layout(
            title={
            'text': f"({x},{y})",
            # 'y':0.9,
            'x':0.575,
            'xanchor': 'center',
            'yanchor': 'top'},
            hovermode='closest',
            width=1000,  # Set the width in pixels
            height=650,  # Set the height in pixels
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            margin=dict(l=100, r=0, t=15, b=0), # Remove margins
        )

        img2 = plotly_to_image_tk(fig2)
        # if isinstance(img2, str):
        #     self.img_label_right.configure(text=img2, image="", compound=None)
        # else:
        self.image_matrix2 = img2 
        self.img_label_right.configure(image=self.image_matrix2, text="", compound=None)
        #     self.img_label_right.image = img2  # keep reference

        self.status.set("Plots updated.")

if __name__ == "__main__":
    App().mainloop()
