"""
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def move_canvas():
    # Forget the old placement
    canvas.get_tk_widget().pack_forget()

    # Change its parent
    canvas._tkcanvas.master = frame2
    canvas.get_tk_widget().master = frame2

    # Repack into new frame
    canvas.get_tk_widget().pack(fill="both", expand=True)

    label.config(text="Matplotlib canvas moved to Frame 2")

# --- Tkinter setup ---
root = tk.Tk()
root.geometry("600x400")
root.title("Move Matplotlib Canvas Between Frames")

# Two side-by-side frames
frame1 = tk.Frame(root, bg="lightblue")
frame1.pack(side="left", fill="both", expand=True)

frame2 = tk.Frame(root, bg="lightgreen")
frame2.pack(side="right", fill="both", expand=True)

# --- Create a Matplotlib Figure ---
fig = Figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)

# Simple sine plot
x = np.linspace(0, 2 * np.pi, 200)
ax.plot(x, np.sin(x), label="sin(x)")
ax.legend()
ax.set_title("Matplotlib in Tkinter")

# Create the FigureCanvasTkAgg and embed it in frame1
canvas = FigureCanvasTkAgg(fig, master=frame1)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

# --- UI controls ---
btn = tk.Button(root, text="Move Canvas", command=move_canvas)
btn.place(x=250, y=10)

label = tk.Label(root, text="Canvas in Frame 1")
label.place(x=260, y=50)

root.mainloop()

"""
"""
import tkinter

import numpy as np

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

root = tkinter.Tk()
root.wm_title("Embedded in Tk")

fig = Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
ax = fig.add_subplot()
line, = ax.plot(t, 2 * np.sin(2 * np.pi * t))
ax.set_xlabel("time [s]")
ax.set_ylabel("f(t)")

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)


def update_frequency(new_val):
    # retrieve frequency
    f = float(new_val)

    # update data
    y = 2 * np.sin(2 * np.pi * f * t)
    line.set_data(t, y)

    # required to update canvas and attached toolbar!
    canvas.draw()


slider_update = tkinter.Scale(root, from_=1, to=5, orient=tkinter.HORIZONTAL,
                              command=update_frequency, label="Frequency [Hz]")

# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
button_quit.pack(side=tkinter.BOTTOM)
slider_update.pack(side=tkinter.BOTTOM)
toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

tkinter.mainloop()
"""
# import tkinter as tk

# class Example():
#     def __init__(self):
#         self.root = tk.Tk()
#         self.root.title("some application")

#         # menu left
#         self.menu_left = tk.Frame(self.root, width=150, bg="#ababab")
#         self.menu_left_upper = tk.Frame(self.menu_left, width=150, height=150, bg="red")
#         self.menu_left_lower = tk.Frame(self.menu_left, width=150, bg="blue")

#         self.test = tk.Label(self.menu_left_upper, text="test")
#         self.test.pack()

#         self.menu_left_upper.pack(side="top", fill="both", expand=True)
#         self.menu_left_lower.pack(side="top", fill="both", expand=True)

#         # right area
#         self.some_title_frame = tk.Frame(self.root, bg="#dfdfdf")

#         self.some_title = tk.Label(self.some_title_frame, text="some title", bg="#dfdfdf")
#         self.some_title.pack()

#         self.canvas_area = tk.Canvas(self.root, width=500, height=400, background="#ffffff")
#         self.canvas_area.grid(row=1, column=1)

#         # status bar
#         self.status_frame = tk.Frame(self.root)
#         self.status = tk.Label(self.status_frame, text="this is the status bar")
#         self.status.pack(fill="both", expand=True)

#         self.menu_left.grid(row=0, column=0, rowspan=2, sticky="nsew")
#         self.some_title_frame.grid(row=0, column=1, sticky="ew")
#         self.canvas_area.grid(row=1, column=1, sticky="nsew") 
#         self.status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

#         self.root.grid_rowconfigure(1, weight=1)
#         self.root.grid_columnconfigure(1, weight=1)

#         self.root.mainloop()

# Example()

import tkinter as tk

def disable_button_action():
    # This function is called when the button is clicked
    my_button.config(state=tk.DISABLED)
    print("Button disabled!")

def enable_button_action():
    # This function enables the button
    my_button.config(state=tk.NORMAL)
    print("Button enabled!")

# Create the main window
root = tk.Tk()
root.title("Disable Button Example")

# Create a button
my_button = tk.Button(root, text="Click to Disable", command=disable_button_action)
my_button.pack(pady=20)

# Create another button to enable the first one
enable_btn = tk.Button(root, text="Enable Button", command=enable_button_action)
enable_btn.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()