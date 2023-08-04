import numpy as np
from concurrent.futures import ThreadPoolExecutor

def mandelbrot(x, y, max_iter, threshold, xmin, xmax, ymin, ymax, width, height):
    zx, zy = x * (xmax - xmin) / (width - 1) + xmin, y * (ymax - ymin) / (height - 1) + ymin
    c = zx + zy * 1j
    z = c
    for i in range(max_iter):
        if abs(z) > threshold:
            break
        z = z * z + c
    else:
        i = max_iter
    return i

def compute_row(y, max_iter, threshold, xmin, xmax, ymin, ymax, width, height):
    return [mandelbrot(x, y, max_iter, threshold, xmin, xmax, ymin, ymax, width, height) for x in range(width)]

# Define the properties of the image
width, height = 2000, 2000
xmin, xmax = -2.5, 1.5
ymin, ymax = -2, 2

# Define the properties of the Mandelbrot set
max_iter = 512
threshold = 1.0

# Create a blank image with the desired resolution
image = np.zeros((height, width), dtype=np.uint8)

# Use a thread pool to compute the image
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(compute_row, y, max_iter, threshold, xmin, xmax, ymin, ymax, width, height) for y in range(height)]
    for y, future in enumerate(futures):
        image[y, :] = (np.array(future.result()) / max_iter * 255).astype(np.uint8)

# Write the image to a PGM file
with open('mandelbrot.pgm', 'wb') as f:
    f.write(b'P5\n')
    f.write(f"{width} {height}\n".encode())
    f.write(b'255\n')
    f.write(image.tobytes())
