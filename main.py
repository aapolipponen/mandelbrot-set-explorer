import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import platform as sys_platform
import os
import sys
from compute_fractals import compute_mandelbrot, compute_perpendicular_mandelbrot, compute_multibrot, compute_julia, compute_perpendicular_julia, compute_modified_julia, compute_burning_ship, compute_tricorn, compute_celtic, compute_newtons_fractal, compute_minus_1i_fractal, compute_glynn, compute_bianca, compute_polynomial_roots, call_image_buffer, call_image

# Very hacky workaround corner
os.environ["PYOPENCL_NO_CACHE"] = "0"
sys.setrecursionlimit(5000)

xmin, xmax = -2.5, 1.5
ymin, ymax = -2, 2

max_iter = 5
threshold = 4.0
d = 2.0

zoom_factor = 1.5

tk_images = []

def compute_fractal():
    fractal = fractal_var.get()
    if fractal == "Mandelbrot":
        compute_mandelbrot(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Perpendicular Mandelbrot":
        compute_perpendicular_mandelbrot(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Multibrot":
        threshold_scale.set(10)
        compute_multibrot(width, height, image, image_buffer, max_iter, d, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack(side=tk.LEFT, padx=10)  # Show slider for d
    elif fractal == "Julia Set":
        compute_julia(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Perpendicular Julia Set":
        compute_perpendicular_julia(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Modified Trigonometric Julia Set":
        compute_modified_julia(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Burning Ship":
        compute_burning_ship(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Tricorn":
        compute_tricorn(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Celtic":
        compute_celtic(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Newton's fractal":
        compute_newtons_fractal(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "-1i Fractal":
        compute_minus_1i_fractal(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Glynn Fractal":
        compute_glynn(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Bianca Fractal":
        compute_bianca(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
    elif fractal == "Polynomial Roots":
        compute_polynomial_roots(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax)
        multibrot_scale.pack_forget()
        
    update_display()

def update_display():
    global tk_images
    pil_image = Image.fromarray(image)
    tk_image = ImageTk.PhotoImage(pil_image)
    tk_images.append(tk_image)  # Store the reference to prevent garbage collection
    canvas.itemconfig(image_on_canvas, image=tk_image)
    
def on_drag(event):
    global center_x, center_y, xmin, xmax, ymin, ymax
    dx = event.x - center_x
    dy = event.y - center_y
    x_shift = dx * (xmax - xmin) / width
    y_shift = dy * (ymax - ymin) / height
    xmin -= x_shift
    xmax -= x_shift
    ymin -= y_shift
    ymax -= y_shift
    center_x, center_y = event.x, event.y
    compute_fractal()
    update_display()

def handle_zoom(event):
    global xmin, xmax, ymin, ymax
    x_frac = event.x / width
    y_frac = event.y / height
    x_mandel = xmin + (xmax - xmin) * x_frac
    y_mandel = ymin + (ymax - ymin) * y_frac
    
    if sys_platform.system() == 'Linux':
        if event.num == 4:
            delta = 1 / zoom_factor
        elif event.num == 5:
            delta = zoom_factor
    elif sys_platform.system() == 'Windows':
        if event.delta > 0:
            delta = 1 / zoom_factor
        else:
            delta = zoom_factor
    else:
        return  # Do nothing for unknown platforms, mac support soon™?
    
    new_width = (xmax - xmin) * delta
    new_height = (ymax - ymin) * delta
    xmin = x_mandel - new_width / 2
    xmax = x_mandel + new_width / 2
    ymin = y_mandel - new_height / 2
    ymax = y_mandel + new_height / 2
    compute_fractal()
    update_display()

def adjust_viewing_window():
    global xmin, xmax, ymin, ymax
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    aspect_ratio_canvas = width / height
    aspect_ratio_mandelbrot = x_range / y_range

    if aspect_ratio_canvas > aspect_ratio_mandelbrot:
        # Expand the x range
        new_x_range = aspect_ratio_canvas * y_range
        x_center = (xmax + xmin) / 2
        xmin = x_center - new_x_range / 2
        xmax = x_center + new_x_range / 2
    else:
        # Expand the y range
        new_y_range = x_range / aspect_ratio_canvas
        y_center = (ymax + ymin) / 2
        ymin = y_center - new_y_range / 2
        ymax = y_center + new_y_range / 2

def handle_resize(event):
    global width, height, image, image_buffer
    width, height = event.width, event.height
    adjust_viewing_window()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    call_image_buffer(image)
    compute_fractal()
    update_display()

def fractal_changed():
    global selected_fractal
    selected_fractal = fractal_var.get()
    compute_fractal()

window = tk.Tk()
window.title("Mandelbrot Explorer")

toolbar_frame = tk.Frame(window)
toolbar_frame.pack(side=tk.TOP, fill=tk.X)

# Toolbar fractal switcher
fractal_types = ["Mandelbrot", "Perpendicular Mandelbrot", "Multibrot", "Julia Set", "Perpendicular Julia Set", "Modified Trigonometric Julia", "Burning Ship", "Tricorn", "Celtic", "Newton's fractal", "-1i Fractal", "Glynn fractal", "Bianca Fractal", "Polynomial Roots"]
fractal_var = tk.StringVar(window)
fractal_var.set(fractal_types[0])  # Default value = Mandelbrot

fractal_dropdown = tk.OptionMenu(toolbar_frame, fractal_var, *fractal_types, command=lambda _: fractal_changed())
fractal_dropdown.pack(side=tk.LEFT, padx=10)

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

width, height = screen_width, screen_height

image = call_image(height, width)

window.geometry(f"{width}x{height}")

center_x, center_y = width / 2, height / 2

image_buffer = call_image_buffer(image)

photo = ImageTk.PhotoImage(Image.fromarray(image, 'RGB'))
canvas = tk.Canvas(window, bg="black", width=width, height=height)
canvas.pack(fill=tk.BOTH, expand=tk.YES)  # Let the canvas expand when window is resized
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=photo)

adjust_viewing_window()

canvas.bind("<Configure>", handle_resize)

canvas.bind("<Button-1>", lambda event: canvas.scan_mark(event.x, event.y))
canvas.bind("<B1-Motion>", on_drag)

if sys_platform.system() == 'Linux':
    canvas.bind("<Button-4>", handle_zoom)
    canvas.bind("<Button-5>", handle_zoom)
elif sys_platform.system() == 'Windows':
    canvas.bind("<MouseWheel>", handle_zoom)

canvas.photos = []

def set_max_iter(val):
    global max_iter
    max_iter = int(float(val))
    compute_fractal()
    update_display()

def set_threshold(val):
    global threshold
    threshold = float(val)
    compute_fractal()
    update_display()

max_iter_scale = tk.Scale(toolbar_frame, from_=1, to=3000, resolution=1, orient=tk.HORIZONTAL, label="Max Iterations",
                          command=set_max_iter)
max_iter_scale.set(max_iter)
max_iter_scale.pack(side=tk.LEFT, padx=10)

threshold_scale = tk.Scale(toolbar_frame, from_=0.1, to=1000, resolution=0.1, orient=tk.HORIZONTAL, label="Threshold",
                           command=set_threshold)
threshold_scale.set(threshold)
threshold_scale.pack(side=tk.LEFT, padx=10)

def set_multibrot_power(val):
    global d
    d = float(val)
    compute_fractal()

multibrot_power = tk.DoubleVar(value=2.0)

multibrot_scale = tk.Scale(toolbar_frame, from_=-5, to=1, resolution=0.1, orient=tk.HORIZONTAL, label="Multibrot Power",
                   command=set_multibrot_power)

compute_fractal()
update_display()
window.mainloop()