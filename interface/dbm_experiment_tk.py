#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter port of the Streamlit UI in dbm_experiment.py

This app loads the original module as a backend while shimming Streamlit so
we can reuse its cached functions (decorated with st.cache_resource) and
helpers without running the Streamlit UI.

Requirements (install if missing):
  pip install pillow plotly kaleido numpy pandas scikit-learn matplotlib

Notes:
- We render Plotly figures to PNG via kaleido and show them in Tkinter.
- If kaleido is not available, the app will display a helpful message.
- This is a best-effort conversion of the *interface*. Some Streamlit-only
  interactivity (like selection objects) is adapted to Tk widgets.
"""

import sys, os, io, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# -----------------------------
# Shim a minimal Streamlit API
# -----------------------------
class _StreamlitShim:
    def cache_resource(self, fn=None, **kwargs):
        # decorator passthrough (no caching in Tk app; backend can cache separately if needed)
        if fn is None:
            def deco(f): return f
            return deco
        return fn
    def __getattr__(self, name):
        # Provide dummies for attributes unexpectedly accessed
        if name == "session_state":
            return {}
        # sidebar is used as a namespace in original code; return self to allow st.sidebar.xxx attribute references
        if name == "sidebar":
            return self
        # no-op widgets
        def noop(*a, **kw): pass
        return noop

# Inject shim before importing backend
sys.modules.setdefault("streamlit", _StreamlitShim())

# -----------------------------
# Import backend module
# -----------------------------

# Support PyInstaller one-file bundles: resources are unpacked under sys._MEIPASS
def resource_path(rel_path):
    base_path = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return Path(base_path) / rel_path

import importlib.util
from pathlib import Path

BACKEND_PATH = resource_path("dbm_experiment.py")
if not BACKEND_PATH.exists():
    raise FileNotFoundError(f"Backend file not found: {BACKEND_PATH}")

spec = importlib.util.spec_from_file_location("dbm_backend", str(BACKEND_PATH))
dbm_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbm_backend)
# Make bundled local modules importable (e.g., `utils`, `models`, etc.)
sys.path.insert(0, str(resource_path('.')))

# Optional: ensure data assets are present similar to the original script
if not Path("data").exists():
    try:
        # run get_data.py next to the backend if present
        get_data = Path(__file__).parent / "get_data.py"
        if get_data.exists():
            import subprocess, sys
            subprocess.run([sys.executable, str(get_data)], check=False)
    except Exception:
        pass

# -----------------------------
# Utilities
# -----------------------------
def plotly_to_image_tk(fig, max_w=900, max_h=700):
    """Convert a Plotly Figure to a Tkinter PhotoImage (PNG via kaleido)."""
    try:
        import plotly.io as pio
        png_bytes = fig.to_image(format="png", scale=2)  # requires kaleido
        im = Image.open(io.BytesIO(png_bytes))
        # resize to fit
        im.thumbnail((max_w, max_h), Image.LANCZOS)
        return ImageTk.PhotoImage(im)
    except Exception as e:
        return f"[Plotly render error] {e}\nTip: pip install -U kaleido"

def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}\n" + traceback.format_exc()

# -----------------------------
# Tkinter Application
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DBM Experiment — Tkinter")
        self.geometry("1200x800")
        self.minsize(1000, 700)

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
        ttk.Combobox(sb, textvariable=self.dataset, state="readonly",
                     values=["mnist", "fashionmnist"]).grid(row=1, column=0, sticky="ew", pady=(0,8))

        ttk.Label(sb, text="Grid Resolution").grid(row=2, column=0, sticky="w")
        self.grid_res = tk.IntVar(value=9)
        ttk.Spinbox(sb, from_=5, to=31, increment=2, textvariable=self.grid_res, width=6).grid(row=3, column=0, sticky="w", pady=(0,8))

        ttk.Label(sb, text="Closest Type").grid(row=4, column=0, sticky="w")
        self.closest_tp = tk.StringVar(value="Exclusive")
        for i, lab in enumerate(["Exclusive", "Inclusive"]):
            ttk.Radiobutton(sb, text=lab, value=lab, variable=self.closest_tp).grid(row=5+i, column=0, sticky="w")

        self.scatter = tk.BooleanVar(value=True)
        ttk.Checkbutton(sb, text="Show scatter", variable=self.scatter).grid(row=7, column=0, sticky="w", pady=(8,0))

        self.class_confidence = tk.BooleanVar(value=True)
        ttk.Checkbutton(sb, text="Show class confidence", variable=self.class_confidence).grid(row=8, column=0, sticky="w")

        ttk.Separator(sb).grid(row=9, column=0, sticky="ew", pady=8)

        ttk.Label(sb, text="Probe (x, y) in latent space [-1,1]").grid(row=10, column=0, sticky="w")
        self.x_val = tk.DoubleVar(value=0.0)
        self.y_val = tk.DoubleVar(value=0.0)
        ttk.Scale(sb, from_=-1.0, to=1.0, orient="horizontal", variable=self.x_val).grid(row=11, column=0, sticky="ew")
        ttk.Scale(sb, from_=-1.0, to=1.0, orient="horizontal", variable=self.y_val).grid(row=12, column=0, sticky="ew")

        ttk.Button(sb, text="Compute Projection", command=self.compute_projection).grid(row=13, column=0, sticky="ew", pady=(10,4))
        ttk.Button(sb, text="Update Plots", command=self.update_plots).grid(row=14, column=0, sticky="ew")

        # Content area
        content = ttk.Frame(self, padding=12)
        content.grid(row=0, column=1, sticky="nsew")
        content.rowconfigure(1, weight=1)
        content.rowconfigure(3, weight=1)
        content.columnconfigure(0, weight=1)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(content, textvariable=self.status).grid(row=0, column=0, sticky="w")

        ttk.Label(content, text="Decision Boundary Matrix").grid(row=1, column=0, sticky="w")
        self.img_label_matrix = ttk.Label(content)
        self.img_label_matrix.grid(row=2, column=0, sticky="nsew")

        ttk.Label(content, text="Neighbors / Confidence View").grid(row=3, column=0, sticky="w")
        self.img_label_right = ttk.Label(content)
        self.img_label_right.grid(row=4, column=0, sticky="nsew")

    # -----------------------------
    # Backend calls
    # -----------------------------
    def compute_projection(self):
        """Run backend to obtain projection, models, etc."""
        self.status.set("Computing projection…")
        self.update_idletasks()
        try:
            # Attempt to call the AE path seen in the Streamlit version
            # Signature inferred from code slices:
            # get_inv_proj_data_ae(output_dir, _model, dataset_name, model_name, method, epochs, random_state)
            output_dir = "weights"
            dataset_name = self.dataset.get()
            model_name = "ae"
            method = "diagonal_normal"
            epochs = 5
            random_state = 42

            sharp_dims_classes = {}
            sharp_dims_classes["fashionmnist"] = [784, 10]
            sharp_dims_classes["mnist"] = [784, 10]
            sharp_dims_classes["har"]  = [561, 6]
            sharp_dims_classes["reuters"] = [5000, 6]

            dims = sharp_dims_classes[dataset_name][0]
            classes = sharp_dims_classes[dataset_name][1]

            # Build a default ShaRP model as in the code (with latent_dim=2 etc.)
            sharp_model = dbm_backend.sharp.ShaRP(
                sharp_dims_classes[dataset_name][0],
                sharp_dims_classes[dataset_name][1],
                "diagonal_normal",
                latent_dim=2,
                variational_layer_kwargs=dict(kl_weight=0.05, kl_mu_weight=0),
                var_leaky_relu_alpha=-0.0001,
                bottleneck_activation="linear",
                bottleneck_l1=0.0,
                bottleneck_l2=0.1,
            )

            # Try AE path first; fall back to MLP path
            fn = getattr(dbm_backend, "get_inv_proj_data_ae", None)
            if fn is None:
                fn = getattr(dbm_backend, "get_inv_proj_data_mlp", None)

            if fn is None:
                raise RuntimeError("Neither get_inv_proj_data_ae nor get_inv_proj_data_mlp found in backend.")

            res, err = safe_call(
                fn,
                output_dir,
                sharp_model,
                dataset_name,
                model_name,
                method,
                epochs,
                random_state
            )
            if err:
                raise RuntimeError(err)

            (self.results_2d,
             y_test,
             self.augmentation_values,
             self.inv_model,
             self.clf,
             self.nn_model,
             self.limits) = res

            self.labels = y_test

            # Pre-compute neighbor matrix used in left plot
            start = (-1.0, -1.0)
            step = (0.25, 0.25)
            size = 9
            grid_res = int(self.grid_res.get())
        #     res2, err2 = safe_call(
        #         dbm_backend.get_nn_matrix,
        #         self.results_2d, self.nn_model, self.inv_model,
        #         grid_res, start, step, size
        #     )
        #     if err2:
        #         raise RuntimeError(err2)

        #     self.nn_matrix, self.nn_max_distance, self.nn_min_distance = res2
            self.status.set("Projection computed. You can now update plots.")
        except Exception as e:
            self.status.set(f"Error computing projection: {e}")
            messagebox.showerror("Compute Error", str(e))

    def update_plots(self):
        """Update both left (matrix) and right (neighbors/confidence) images."""
        if self.results_2d is None:
            self.status.set("Please run 'Compute Projection' first.")
            return

        self.status.set("Rendering plots…")
        self.update_idletasks()

        # Left figure (matrix)
        try:
            start = (-1.0, -1.0)
            step = (0.25, 0.25)
            size = 9
            grid_res = int(self.grid_res.get())

            fig = dbm_backend.get_matrix_fig(
                self.results_2d, self.labels, self.clf, self.inv_model,
                grid_res, start, step, size
            )
            img = plotly_to_image_tk(fig)
            if isinstance(img, str):
                # error string
                self.img_label_matrix.configure(text=img, image="", compound=None)
            else:
                self.img_label_matrix.configure(image=img, text="", compound=None)
                self.img_label_matrix.image = img  # keep reference
        except Exception as e:
            self.img_label_matrix.configure(text=f"Error rendering left plot: {e}", image="", compound=None)

        # Right figure (neighbors/confidence)
        try:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            grid_res = int(self.grid_res.get())
            x = float(self.x_val.get())
            y = float(self.y_val.get())
            scatter = bool(self.scatter.get())
            class_conf = bool(self.class_confidence.get())

            # Try to call the backend helpers the same way as in Streamlit code branches
            # We use 'Exclusive' or 'Inclusive' behavior
            closest_tp = self.closest_tp.get()
            if closest_tp == "Exclusive":
                fn = getattr(dbm_backend, "gen_and_save_nnm", None)
            else:
                fn = getattr(dbm_backend, "gen_and_save_ccm", None)

            if fn is None:
                # Fallback: just plot neighbors scatter if available
                import plotly.express as px
                fig2 = px.scatter(x=self.results_2d[:,0], y=self.results_2d[:,1], color=self.labels)
            else:
                # The utility typically returns (fig2, ret_values)
                # Signature inferred from code: (results_2d, labels, clf, inv_model, nn_model, grid_res, x, y, fig2, scatter, class_confidence, cmap)
                # Use viridis as cmap through backend if available
                cmap_nn = getattr(dbm_backend, "cmap_nn", None)
                res, err = safe_call(fn,
                                     self.results_2d, self.labels, self.augmentation_values, self.clf, self.nn_model,
                                     self.inv_model, grid_res, x, y, fig2, scatter, class_conf, cmap_nn)
                if err:
                    raise RuntimeError(err)
                fig2, _ = res

            img2 = plotly_to_image_tk(fig2)
            if isinstance(img2, str):
                self.img_label_right.configure(text=img2, image="", compound=None)
            else:
                self.img_label_right.configure(image=img2, text="", compound=None)
                self.img_label_right.image = img2  # keep reference
        except Exception as e:
            self.img_label_right.configure(text=f"Error rendering right plot: {e}", image="", compound=None)

        self.status.set("Plots updated.")

if __name__ == "__main__":
    App().mainloop()
