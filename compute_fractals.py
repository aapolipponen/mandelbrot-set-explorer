import pyopencl as cl
import numpy as np

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
__kernel void julia(__global uchar *image, const unsigned int width, const unsigned int height, 
                    const unsigned int max_iter, const double threshold, 
                    const double xmin, const double xmax, const double ymin, const double ymax,
                    const double real_c, const double imag_c) {

    int x = get_global_id(0);
    int y = get_global_id(1);
    double zx = x * (xmax - xmin) / (width - 1) + xmin;
    double zy = y * (ymax - ymin) / (height - 1) + ymin;
    
    double cx = real_c;
    double cy = imag_c;
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
__kernel void burning_ship(__global uchar *image, const unsigned int width, const unsigned int height, 
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
        // Absolute value for the real and imaginary parts
        zx = fabs(zx);
        zy = fabs(zy);
        
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
__kernel void multibrot(__global uchar *image, const unsigned int width, const unsigned int height, 
                        const unsigned int max_iter, const double threshold, const double d,
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
        double r = sqrt(zx*zx + zy*zy);
        double theta = atan2(zy, zx);
        zx = pow(r, d) * cos(d * theta) + cx;
        zy = pow(r, d) * sin(d * theta) + cy;
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
__kernel void tricorn(__global uchar *image, const unsigned int width, const unsigned int height, 
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
        zy = -2*zx*zy - cy;
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
__kernel void celtic(__global uchar *image, const unsigned int width, const unsigned int height, 
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
        temp = fabs(zx*zx - zy*zy) + cx;
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
__kernel void perpendicular_mandelbrot(__global uchar *image, const unsigned int width, const unsigned int height, 
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
        zy = -2*zx*zy + cy;
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

def compute_mandelbrot(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.mandelbrot(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                       np.uint32(max_iter), np.double(threshold), 
                       np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_julia(width, height, image, image_buffer,max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)

    # User adjustment?
    real_c, imag_c = -0.7, 0.27015  

    program.julia(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                  np.uint32(max_iter), np.double(threshold), 
                  np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax),
                  np.double(real_c), np.double(imag_c))
    
    cl.enqueue_copy(queue, image, image_buffer)

def compute_burning_ship(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.burning_ship(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                         np.uint32(max_iter), np.double(threshold), 
                         np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_multibrot(width, height, image, image_buffer, max_iter, d, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.multibrot(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                      np.uint32(max_iter), np.double(threshold), np.double(d),
                      np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_tricorn(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.tricorn(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                    np.uint32(max_iter), np.double(threshold), 
                    np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_celtic(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.celtic(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                   np.uint32(max_iter), np.double(threshold), 
                   np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_perpendicular_mandelbrot(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.perpendicular_mandelbrot(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                                     np.uint32(max_iter), np.double(threshold), 
                                     np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def call_image_buffer(image):
    image_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)
    return image_buffer

def call_image(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return image

platforms = cl.get_platforms()
if not platforms:
    raise RuntimeError("No OpenCL platforms found!")
platform = platforms[0]
devices = platform.get_devices()
if not devices:
    raise RuntimeError("No OpenCL devices found on the platform!")
device = devices[0]
double_support = "cl_khr_fp64" in device.get_info(cl.device_info.EXTENSIONS)
if not double_support:
    raise RuntimeError("Your GPU does not support double precision!")
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel_code).build()
