import numpy as np
from PIL import Image, ImageTk
import pyopencl as cl
import tkinter as tk
import platform as sys_platform
import os
import sys
from backups.mpmath import mp

# Very hacky workaround corner
os.environ["PYOPENCL_NO_CACHE"] = "0"
sys.setrecursionlimit(5000)

mp.dps = 50 # Decimal place amount

width, height = 1000, 1000
xmin, xmax = -2.5, 1.5
ymin, ymax = -2, 2
max_iter = 100
threshold = 4.0

zoom_factor = 1.5
center_x, center_y = width / 2, height / 2

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

# OpenCL kernel code
kernel_code = """
__kernel void mandelbrot(__global uchar *image, const unsigned int width, const unsigned int height, 
                         const unsigned int max_iter, const double threshold, 
                         const double xmin, const double xmax, const double ymin, const double ymax) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    double zx = x * (xmax - xmin) / (width - 1) + xmin;
    double zy = y * (ymax - ymin) / (height - 1) + ymin;
    
    double cx = zx;
    double cy = zy;
    int iter = 0;
    
    double temp;
    while (zx*zx + zy*zy < threshold && iter < max_iter) {
        temp = zx*zx - zy*zy + cx;
        zy = 2*zx*zy + cy;
        zx = temp;
        iter++;
    }
    
    uchar r, g, b;
    if (iter == max_iter) {
        r = g = b = 0;
    } else {
        double smooth_val = iter + 1 - log(log(native_sqrt(zx*zx + zy*zy))) / log(2.0);
        double color = 0.5 + 0.5 * sin(smooth_val * 0.1);
        r = 9 * (1 - color) * color * color * color * 255;
        g = 15 * (1 - color) * (1 - color) * color * color * 255;
        b = 8.5 * (1 - color) * (1 - color) * (1 - color) * color * 255;
    }
    
    image[(y * width + x) * 3 + 0] = r;
    image[(y * width + x) * 3 + 1] = g;
    image[(y * width + x) * 3 + 2] = b;
}
"""

def mandelbrot_mpmath(xmin, xmax, ymin, ymax, width, height, max_iter):
    # Create an empty image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate pixel size
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height

    # Precompute color palette
    palette = np.array([(i % 8 * 32, i % 16 * 16, i % 32 * 8) for i in range(max_iter)], dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            zx, zy = x * dx + xmin, y * dy + ymin
            c = mp(zx, zy)
            z = mp(0, 0)
            for i in range(max_iter):
                if abs(z) > 2.0:
                    break 
                z = z * z + c

            # Use precomputed color palette
            image[y, x] = palette[i]
    
    return image

def should_use_mpmath(xmin, xmax, ymin, ymax):
    # Example: if the range of x or y is below a certain threshold, 
    # it indicates a high zoom level, so we switch to mpmath
    x_range = xmax - xmin
    y_range = ymax - ymin
    threshold = 1e-20  # example threshold value, adjust as needed
    return x_range < threshold or y_range < threshold

def compute_mandelbrot_gpu():
    global image
    global_size = (width, height)
    program.mandelbrot(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                       np.uint32(max_iter), np.double(threshold), 
                       np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)
    return image

def compute_mandelbrot_mpmath():
    return mandelbrot_mpmath(xmin, xmax, ymin, ymax, width, height, max_iter)

def compute_mandelbrot():
    # Check if we've crossed the precision threshold
    if should_use_mpmath(xmin, xmax, ymin, ymax):
        return compute_mandelbrot_mpmath()
    else:
        return compute_mandelbrot_gpu()
    
def update_display():
    pil_image = Image.fromarray(image)
    tk_image = ImageTk.PhotoImage(pil_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    window.mainloop()

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
    compute_mandelbrot()
    update_display()

def handle_zoom(event):
    global xmin, xmax, ymin, ymax
    x_frac = event.x / width
    y_frac = event.y / height
    x_mandel = xmin + (xmax - xmin) * x_frac
    y_mandel = ymin + (ymax - ymin) * y_frac
    if event.num == 4:
        delta = 1 / zoom_factor
    elif event.num == 5:
        delta = zoom_factor
    new_width = (xmax - xmin) * delta
    new_height = (ymax - ymin) * delta
    xmin = x_mandel - new_width / 2
    xmax = x_mandel + new_width / 2
    ymin = y_mandel - new_height / 2
    ymax = y_mandel + new_height / 2
    # Print canvas dimensions after zooming
    print(f"Canvas Dimensions after Zoom:")
    print(f"xmin: {xmin}, xmax: {xmax}")
    print(f"ymin: {ymin}, ymax: {ymax}\n")
    compute_mandelbrot()
    update_display()

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
double_support = "cl_khr_fp64" in device.get_info(cl.device_info.EXTENSIONS)
if not double_support:
    raise RuntimeError("Your GPU does not support double precision!")
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel_code).build()

image = np.zeros((height, width, 3), dtype=np.uint8)
image_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

window = tk.Tk()
window.title("Mandelbrot Explorer")

photo = ImageTk.PhotoImage(Image.fromarray(image, 'RGB'))
canvas = tk.Canvas(window, bg="black", width=width, height=height)
canvas.pack(pady=1)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=photo)

canvas.bind("<Button-1>", lambda event: canvas.scan_mark(event.x, event.y))
canvas.bind("<B1-Motion>", on_drag)
if sys_platform.system() == 'Linux':
    canvas.bind("<Button-4>", handle_zoom)
    canvas.bind("<Button-5>", handle_zoom)

canvas.photos = []

compute_mandelbrot()
update_display()