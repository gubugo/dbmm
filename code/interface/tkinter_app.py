import os
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

from code.models.tensorflow import sharp
from code.models.classifiers.MLP import load_or_fit_mlp_classifier
from code.training.auto_encoders import load_or_fit_model_ae
from code.training.inv_proj import load_or_fit_model_inv_proj
from code.utils.augmentations import get_augmentation_pca
from code.utils.data import get_dimensions_and_class, get_inv_proj_data_ae
from code.utils.matplotlib.dbm import gen_and_save_dbm, plot_generated_images_grid_with_dbm
from code.utils.matplotlib.dbm_matrix import gen_and_save_dbm_matrix
from code.utils.matplotlib.maps import gen_and_save_ccm, gen_and_save_nnm
from code.utils.metrics import metric_distance_to_nearest_neighbor
from code.utils.utils import get_bounding_box, make_grid, make_titles, plotly_to_image_tk

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
        self.scale = tk.Scale(
            self,
            from_=from_value,
            to=to_value,
            variable=self.var,
            resolution=0.01,
            command=self._update_label,
            **kwargs
        )
        self.scale.pack(fill="x", expand=True)
        
    def _update_label(self, value):
        """Callback to update the label text with the scale's current value."""
        self.label.config(text=f"{self.label.cget('text').split(':')[0]}: {float(value):.2f}")
        self.var.set(np.round(float(value), decimals=2))
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
        self.title("DBMM Interface")
        self.geometry("1600x900")
        self.minsize(1600, 900)

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
        self.matrix_dataset = ""
        self.right_fig = plt.figure()
        self.right_fig.set_size_inches(5,5)
        self.right_ax  = self.right_fig.add_subplot(111)
        self.img_fig = plt.figure()
        self.img_fig.set_size_inches(2,2)
        self.img_ax  = self.img_fig.add_subplot(111)
        self.img_ax.axis("off") 
        self.img_ax.margins(0) 
        self.img_axis_fig, self.img_axis_ax = plt.subplots(2,9)
        self.img_axis_fig.set_size_inches(8,2)
        for ax_row in self.img_axis_ax:
            for ax in ax_row:
                ax.set_xticklabels([])  # Hide x tick labels
                ax.set_yticklabels([])  # Hide y tick labels
                ax.set_xticks([])      # Hide x tick marks
                ax.set_yticks([])      # Hide y tick marks

        self.matrix_side_size = 9
        self.matrix_fig, self.matrix_ax = plt.subplots(self.matrix_side_size,self.matrix_side_size,figsize=(75/10, 75/10))
        # self.right_fig, self.right_ax = plt.subplot(1,1)
        
        # self.right_canvas.get_tk_widget().grid(row=1, column=1, sticky="nse")
        self.cmap_main = plt.get_cmap("tab10")
        self.cmap_nn   = plt.get_cmap("viridis")

        # UI Layout
        self.build_app_layout()

    def _side_bar_interface_construct(self):
        sb = ttk.Frame(self, padding=12)
        sb.grid(row=0, column=0, sticky="ns")

        ttk.Label(sb, text="Dataset").grid(row=0, column=0, sticky="w")
        self.dataset = tk.StringVar(value="mnist")
        combobox1 = ttk.Combobox(sb, textvariable=self.dataset, state="readonly", values=["mnist", "fashionmnist", "har", "reuters", "hate_speech"])
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

    def _matrix_interface_construct(self):
        self.content_area.grid(row=0, column=1, sticky="nsew")
        self.content_area.rowconfigure(1, weight=1)
        self.content_area.rowconfigure(3, weight=1)
        self.content_area.columnconfigure(0, weight=1)
        self.content_area.columnconfigure(1, weight=1)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.content_area, textvariable=self.status).grid(row=0, column=0, sticky="w")

        self.matrix_canvas = FigureCanvasTkAgg(self.matrix_fig, master=self.content_area)
        self.matrix_canvas.get_tk_widget().grid(column=0, row=1, sticky="WNES")
        self.matrix_canvas.draw()

    def _dbm_interface_construct(self):
        self.img_label_right = ttk.Frame(self.content_area, padding=12)
        self.img_label_right.grid(row=1, column=1, sticky="nsew")

        self.img_label_right2 = ttk.Label(self.img_label_right)
        self.img_label_right2.grid(row=1, column=0, sticky="nsew")

        self.right_canvas = FigureCanvasTkAgg(self.right_fig, master=self.img_label_right)
        self.right_canvas.get_tk_widget().grid(column=0, row=0, sticky="WNES")
        self.right_canvas.draw()
        self.right_canvas.mpl_connect('button_press_event', self.on_mouse_event)

        self.right_toolbar = NavigationToolbar2Tk(self.right_canvas, self.img_label_right2)
        self.right_toolbar.update()

    def _image_reconstruct_info_interface_construct(self):
        image_place = ttk.Frame(self.img_label_right, padding=12)
        image_place.grid(row=2, column=0, sticky="nsew")

        self.img_canvas = FigureCanvasTkAgg(self.img_fig, master=image_place)
        self.img_canvas.get_tk_widget().grid(column=0, row=0, sticky="WNES")
        self.img_canvas.draw()

        self.title_style = ttk.Style()
        self.title_style.configure("Bold.TLabel", font=("Helvetica", 12, "bold"))
        self.text_style = ttk.Style()
        self.text_style.configure("Normal.TLabel", font=("Helvetica", 9))

        # Column 1, Bottom Row, 2nd Column: Data 1: Nearest Training Point
        ntp_place = ttk.Frame(image_place, padding=12)
        ntp_place.grid(row=0, column=2, sticky="nsew")

        self.title_ctp = tk.StringVar(value="Nearest Neighbor Info:")
        ttk.Label(ntp_place, textvariable=self.title_ctp, style="Bold.TLabel").grid(row=0, column=0, sticky="WS")
        self.np_dist_ctp = tk.StringVar(value="Nearest Point Distance:")
        ttk.Label(ntp_place, textvariable=self.np_dist_ctp, style="Normal.TLabel").grid(row=1, column=0, sticky="WS")
        self.fp_dist_ctp = tk.StringVar(value="Farthest Point Distance:")
        ttk.Label(ntp_place, textvariable=self.fp_dist_ctp, style="Normal.TLabel").grid(row=2, column=0, sticky="WS")
        self.sp_dist_ctp = tk.StringVar(value="Selected Point Distance:")
        ttk.Label(ntp_place, textvariable=self.sp_dist_ctp, style="Normal.TLabel").grid(row=3, column=0, sticky="WS")

        # Column 1, Bottom Row, 3rd Column: Data 2: Class Confidence
        cc_place = ttk.Frame(image_place, padding=12)
        cc_place.grid(row=0, column=4, sticky="nsew")
        
        self.title_cc = tk.StringVar(value="Classifier Confidence Info:")
        ttk.Label(cc_place, textvariable=self.title_cc, style="Bold.TLabel").grid(row=0, column=0, sticky="WS")
        self.cc_class1 = tk.StringVar(value="Class ")
        ttk.Label(cc_place, textvariable=self.cc_class1, style="Normal.TLabel").grid(row=1, column=0, sticky="WS")
        self.cc_class2 = tk.StringVar(value="Class ")
        ttk.Label(cc_place, textvariable=self.cc_class2, style="Normal.TLabel").grid(row=2, column=0, sticky="WS")
        self.cc_class3 = tk.StringVar(value="Class ")
        ttk.Label(cc_place, textvariable=self.cc_class3, style="Normal.TLabel").grid(row=3, column=0, sticky="WS")

        # Column 1, bottom Row (image lines)
        # bottom_images = ttk.Frame(self.img_label_right, padding=12)
        # bottom_images.grid(row=3, column=0, sticky="nsew")

        # self.img_axis_canvas = FigureCanvasTkAgg(self.img_axis_fig, master=bottom_images)
        # self.img_axis_canvas.get_tk_widget().grid(column=0, row=0, sticky="WNES")
        # self.img_axis_canvas.draw()

    def build_app_layout(self):
        # Main grid: sidebar (0) and content (1)
        self.columnconfigure(0, weight=0)  # sidebar
        self.columnconfigure(1, weight=1)  # content
        self.rowconfigure(0, weight=1)

        self._side_bar_interface_construct()

        ## Content area
        self.content_area = ttk.Frame(self, padding=12)

        # Column 0 (Matrix)
        self._matrix_interface_construct()

        # Column 1 (DBM and stuff)
        self._dbm_interface_construct()

        # Column 1, bottom Row (img + info)
        self._image_reconstruct_info_interface_construct()

        self.gen_dbm()

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
        print(self.matrix_dataset)

        if (not self.data_loaded) or self.matrix_dataset != self.dataset.get():
            print("Here")
            self.compute_projection()
        self.update_plots()

    def on_mouse_event(self, event):
        if event.inaxes:  # Check if the mouse is within the plot area
            img = np.ones((28, 28, 4))
            metric_nn = [0]
            metric_cc = [0, 0, 0]
            top_cc_values = [0,0,0]

            bounding_box = get_bounding_box(self.results_2d)
            half_grid_res = int(self.resolution.get())//2
            grid_res = int(self.resolution.get())
            
            x = bounding_box[0]+(bounding_box[1]-bounding_box[0])*(int(event.xdata+0.5))/grid_res
            y = bounding_box[2]+(bounding_box[3]-bounding_box[2])*(int(event.ydata+0.5))/grid_res
            point_4d = np.reshape(np.array([x,y,float(self.x_v.get()),float(self.y_v.get())]),(1,4))
            img_1d = self.inv_model.inverse_transform(point_4d)
            img = np.reshape(img_1d,(28, 28))
            img = np.stack([img, img, img, np.ones(np.shape(img))], axis=-1)

            metric_nn = metric_distance_to_nearest_neighbor(img_1d, self.nn_model)
            metric_cc = self.clf.predict_proba(img_1d)
            metric_cc = metric_cc[0]
            top_cc_values = np.argpartition(metric_cc, -4)[-3:]
            top_cc_values = top_cc_values[np.argsort(metric_cc[top_cc_values])]
            self.img_ax.imshow(
                img,
                interpolation="none",
                resample=False,
            )
            self.img_ax.margins(0) 
            self.img_fig.set_size_inches(2,2)
            self.img_fig.savefig("image_recon.png", bbox_inches="tight", pad_inches=0.0)
            self.img_canvas.draw()
            
            self.sp_dist_ctp.set(f"Selected Point Distance:{metric_nn[0]:.5f}")

            self.cc_class1.set(f"Class {top_cc_values[-1]}:{metric_cc[top_cc_values[-1]]:.5f}")
            self.cc_class2.set(f"Class {top_cc_values[-2]}:{metric_cc[top_cc_values[-2]]:.5f}")
            self.cc_class3.set(f"Class {top_cc_values[-3]}:{metric_cc[top_cc_values[-3]]:.5f}")

            # AXES
            # dbm_coords_collumn = np.column_stack([np.full(9, x), np.full(9, y)])
            # mdbm_x_coords_collumn = np.column_stack([np.full(9, self.x_v.get())])
            # mdbm_y_coords_collumn = np.column_stack([np.full(9, self.y_v.get())])
            # extra_coords_collumn = np.linspace(-1, 1, 9).reshape((9,1))

            # grid_points_x = np.hstack([dbm_coords_collumn, extra_coords_collumn, mdbm_y_coords_collumn])
            # grid_points_y = np.hstack([dbm_coords_collumn, mdbm_x_coords_collumn, extra_coords_collumn])
            # both_axis = np.concatenate((grid_points_x,grid_points_y))

            # imgs_axis = self.inv_model.inverse_transform(both_axis)
            # imgs = np.reshape(imgs_axis,(18, 28, 28))
            # imgs = np.stack([imgs, imgs, imgs, np.ones(np.shape(imgs))], axis=-1)

            # for i in range(9):
            #     for j in range(2):
            #         self.img_axis_ax[j,i].imshow(
            #             imgs[i+9*j],
            #             interpolation="none",
            #             resample=False,
            #         )
            #         self.img_axis_ax[j,i].margins(0)
            #         self.img_axis_ax[j,i].set_title(f"{(i-4)/4}", fontsize=10, x=0.5, y=1)
            
            # self.img_axis_ax[0,0].set_ylabel('X axis DBMM', fontsize=7)
            # self.img_axis_ax[1,0].set_ylabel('Y axis DBMM', fontsize=7)

            # self.img_axis_fig.savefig("image_axis.png", bbox_inches="tight", pad_inches=0.0)
            # self.img_axis_canvas.draw()

            # DEBUG
            # x_data = event.xdata  # Data coordinates
            # y_data = event.ydata
            # x_pixel = event.x     # Pixel coordinates relative to the canvas
            # y_pixel = event.y
            # print(f"Data Coords: ({x_data:.2f}, {y_data:.2f}), Pixel Coords: ({x_pixel}, {y_pixel})")

    def get_matrix_fig(self, results_2d, _clf, _inv_model, grid_res, start, step, size, ax):
        bounding_box = get_bounding_box(results_2d)
        cmapped = np.zeros((size*size,grid_res*grid_res,4))
        for i in range(size):
            for j in range(size):
                grid = make_grid(*bounding_box, start[0]+i*step[0], start[1]+j*step[1], grid_res)
                inverted_grid = _inv_model.inverse_transform(grid)

                classes = _clf.predict(inverted_grid).astype(np.uint8)

                cmapped[size*i+j] = self.cmap_main(classes)

                ax[i,j].imshow(
                    cmapped[size*i+j].reshape((grid_res, grid_res, 4)),
                    origin="lower",
                    interpolation="none",
                    resample=False,
                )
                coords = f"({np.round(start[0]+i*step[0],2)},{np.round(start[1]+j*step[1],2)})"
                ax[i,j].axis("off") 
                ax[i,j].set_title(coords, fontsize=grid_res/(size), x=0.5, y=1-5/grid_res) 
                
        return ax

    
    def compute_projection(self):

        output_dir = "weights"
        dataset_name = self.dataset.get()
        self.matrix_dataset = dataset_name
        self.generate_matrix = True
        model_name = "sharp"
        method = "noise"
        epochs = 10
        random_state = 420

        dims, classes = get_dimensions_and_class(dataset_name)

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

        _, self.labels, self.augmentation_values, self.results_2d, self.clf, self.inv_model, self.nn_model = get_inv_proj_data_ae(
            output_dir,
            sharp_model,
            dataset_name,
            model_name,
            method,
            epochs
        )

        self.data_loaded = True



    def update_plots(self):
        """Update both left (matrix) and right (neighbors/confidence) images."""
        start = (-1.0, -1.0)
        step = (0.25, 0.25)
        size = 9
        grid_res = int(self.resolution.get())
        if self.generate_matrix:
            self.matrix_ax = self.get_matrix_fig(self.results_2d, self.clf, self.inv_model, grid_res, start, step, size, self.matrix_ax)
            self.generate_matrix = False

        # Right figure (neighbors/confidence)
        x = float(self.x_v.get())
        y = float(self.y_v.get())
        scatter = self.scatter.get()
        class_conf = self.class_conf.get()
        closest_tp = self.closest_tp.get()

        reconstruct = self.images.get()
        
        fig2 = self.right_ax
        # if self.images.get() == "0":
        if closest_tp == "Exclusive":
            fig2, ret_values = gen_and_save_nnm(self.results_2d, self.labels, self.augmentation_values, self.clf, self.nn_model, self.inv_model, grid_res, x, y, fig2, scatter, class_conf, self.cmap_nn)
        elif class_conf == "Exclusive":
            fig2, ret_values = gen_and_save_ccm(self.results_2d, self.labels, self.augmentation_values, self.clf, self.nn_model, self.inv_model, grid_res, x, y, fig2, scatter, closest_tp, self.cmap_nn)
        else:
            fig2, ret_values = gen_and_save_dbm(self.results_2d, self.labels, self.augmentation_values, self.clf, self.nn_model, self.inv_model, grid_res, x, y, fig2, scatter, closest_tp, class_conf, self.cmap_main, reconstruct)
        self.right_fig.savefig("dbm.png", bbox_inches="tight", pad_inches=0.0)
        self.right_canvas.draw()
        self.right_toolbar.update()

        self.np_dist_ctp.set(f"Nearest Point Distance: {ret_values[1]:.5f}")
        self.fp_dist_ctp.set(f"Farthest Point Distance:{ret_values[0]:.5f}")

        self.status.set("Plots updated.")

if __name__ == "__main__":
    App().mainloop()
