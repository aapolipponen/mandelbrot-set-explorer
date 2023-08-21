import pyopencl as cl
import numpy as np

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

# OpenCL kernel code
kernel_code = """
inline uchar3 getColor(int iter, double zx, double zy, int max_iter) {
    uchar3 color;
    if (iter == max_iter) {
        color.x = color.y = color.z = 0;
    } else {
        double smooth_val = iter + 1 - log(log(native_sqrt(zx*zx + zy*zy))) / log(2.0);
        double col = 0.5 + 0.5 * sin(smooth_val * 0.1);
        color.x = 9 * (1 - col) * col * col * col * 255;
        color.y = 15 * (1 - col) * (1 - col) * col * col * 255;
        color.z = 8.5 * (1 - col) * (1 - col) * (1 - col) * col * 255;
    }
    return color;
}

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
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
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
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
}
__kernel void perpendicular_julia(__global uchar *image, const unsigned int width, const unsigned int height, 
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
        zy = -2*zx*zy + cy;
        zx = temp;
        iter++;
    }
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
}
__kernel void modified_julia(__global uchar *image, const unsigned int width, const unsigned int height, 
                                  const unsigned int max_iter, const double threshold, 
                                  const double xmin, const double xmax, const double ymin, const double ymax) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    double cx = x * (xmax - xmin) / (width - 1) + xmin;
    double cy = y * (ymax - ymin) / (height - 1) + ymin;

    // Initialize z to 2.5*exp(i*theta), where theta is between 0 and pi
    double theta = atan2(cy, cx);
    double r = 2.5;
    double zx = r * cos(theta);
    double zy = r * sin(theta);
    
    int iter = 0;

    double temp_x, temp_y;
    while (zx*zx + zy*zy < threshold && iter < max_iter) {
        // Compute z_next = cos(z) + 1/c
        // cos(z) = cos(a) * cosh(b) - i * sin(a) * sinh(b)
        temp_x = cos(zx) * cosh(zy) + 1/cx;
        temp_y = -sin(zx) * sinh(zy) + 1/cy;

        zx = temp_x;
        zy = temp_y;

        iter++;
    }
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
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
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
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
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
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
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
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
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
}
__kernel void newtons_fractal(__global uchar *image, const unsigned int width, const unsigned int height, 
                              const unsigned int max_iter, const double threshold, 
                              const double xmin, const double xmax, const double ymin, const double ymax) {

    int x = get_global_id(0);
    int y = get_global_id(1);
    double zx = x * (xmax - xmin) / (width - 1) + xmin;
    double zy = y * (ymax - ymin) / (height - 1) + ymin;
    
    int iter = 0;
    double temp_x, temp_y;
    while (zx*zx + zy*zy < threshold && iter < max_iter) {
        temp_x = zx;
        temp_y = zy;
        zx = temp_x - (temp_x*temp_x*temp_x - 3*temp_x*temp_y*temp_y - 1) / (3*temp_x*temp_x - 3*temp_y*temp_y);
        zy = temp_y - (3*temp_x*temp_x*temp_y - temp_y*temp_y*temp_y) / (3*temp_x*temp_x - 3*temp_y*temp_y);
        iter++;
    }
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
}
__kernel void minus_1i_fractal(__global uchar *image, const unsigned int width, const unsigned int height, 
                               const unsigned int max_iter, const double threshold, 
                               const double xmin, const double xmax, const double ymin, const double ymax) {

    int x = get_global_id(0);
    int y = get_global_id(1);
    double cx = x * (xmax - xmin) / (width - 1) + xmin;
    double cy = y * (ymax - ymin) / (height - 1) + ymin;
    
    double zx = cx;
    double zy = cy;
    int iter = 0;
    
    double temp;
    while (zx*zx + zy*zy < threshold && iter < max_iter) {
        temp = zx*zx - zy*zy - cx;
        zy = 2*zx*zy - 1.0;  // subtracting 1i here
        zx = temp;
        iter++;
    }
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
}
__kernel void bianca_fractal(__global uchar *image, const unsigned int width, const unsigned int height, 
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
        temp = fabs(zx) * (1 + zy) + cx;
        zy = fabs(zy) * (1 - zx) + cy;
        zx = temp;
        iter++;
    }
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
}
__kernel void glynn_fractal(__global uchar *image, const unsigned int width, const unsigned int height, 
                            const unsigned int max_iter, const double threshold, const double r, 
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
        double mag = pow(sqrt(zx*zx + zy*zy), r);
        double arg = atan2(zy, zx) * r;
        zx = mag * cos(arg) + cx;
        zy = mag * sin(arg) + cy;
        iter++;
    }
    
    uchar3 color = getColor(iter, zx, zy, max_iter);
    image[(y * width + x) * 3 + 0] = color.x;
    image[(y * width + x) * 3 + 1] = color.y;
    image[(y * width + x) * 3 + 2] = color.z;
}
__kernel void polynomial_roots(__global uchar *image, const unsigned int width, const unsigned int height, 
                               const unsigned int max_iter, const double threshold, 
                               const double xmin, const double xmax, const double ymin, const double ymax) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    double zx = x * (xmax - xmin) / (width - 1) + xmin;
    double zy = y * (ymax - ymin) / (height - 1) + ymin;

    // Our starting value is our current pixel
    double2 z = (double2)(zx, zy);
    
    int coeff[6];

    // Loop over all coefficients using nested loops
    for (coeff[0] = -4; coeff[0] <= 4; coeff[0]++) {
        for (coeff[1] = -4; coeff[1] <= 4; coeff[1]++) {
            for (coeff[2] = -4; coeff[2] <= 4; coeff[2]++) {
                for (coeff[3] = -4; coeff[3] <= 4; coeff[3]++) {
                    for (coeff[4] = -4; coeff[4] <= 4; coeff[4]++) {
                        for (coeff[5] = -4; coeff[5] <= 4; coeff[5]++) {

                            int iter;
                            double2 prev_z = z;
                            for (iter = 0; iter < max_iter; iter++) {
                                // Evaluate polynomial and its derivative
                                double2 p = (double2)(0, 0);
                                double2 dp = (double2)(0, 0);
                                double2 z_power = (double2)(1, 0);  // start at z^0

                                for (int i = 0; i <= 5; i++) {
                                    p += coeff[i] * z_power;
                                    if (i != 0)
                                        dp += i * coeff[i] * z_power;
                                    // Multiply z_power by z for the next iteration
                                    z_power = (double2)(z_power.x * z.x - z_power.y * z.y, z_power.x * z.y + z_power.y * z.x);
                                }

                                double dp_mag = dp.x * dp.x + dp.y * dp.y;
                                if (dp_mag < 1e-6)  // Avoid division by near-zero values
                                    break;

                                // Check for convergence
                                if (p.x * p.x + p.y * p.y < threshold || length(z - prev_z) < 1e-6)
                                    break;

                                // Newton's method update
                                z.x -= (p.x * dp.x + p.y * dp.y) / dp_mag;
                                z.y -= (p.y * dp.x - p.x * dp.y) / dp_mag;

                                prev_z = z;
                            }

                            // Set color based on number of iterations (this can be further enhanced)
                            uchar3 color;
                            color.x = iter % 256;
                            color.y = (iter * 2) % 256;
                            color.z = (iter * 3) % 256;

                            image[(y * width + x) * 3 + 0] = color.x;
                            image[(y * width + x) * 3 + 1] = color.y;
                            image[(y * width + x) * 3 + 2] = color.z;

                        }
                    }
                }
            }
        }
    }
}
"""

def compute_mandelbrot(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.mandelbrot(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                       np.uint32(max_iter), np.double(threshold), 
                       np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_multibrot(width, height, image, image_buffer, max_iter, d, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.multibrot(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                      np.uint32(max_iter), np.double(threshold), np.double(d),
                      np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_perpendicular_mandelbrot(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.perpendicular_mandelbrot(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
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

def compute_perpendicular_julia(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
   
    # User adjustment?
    real_c, imag_c = -0.7, 0.27015  
   
    program.perpendicular_julia(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                                np.uint32(max_iter), np.double(threshold), 
                                np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax),
                                np.double(real_c), np.double(imag_c))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_modified_julia(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.modified_julia(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                                np.uint32(max_iter), np.double(threshold), 
                                np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_burning_ship(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.burning_ship(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                         np.uint32(max_iter), np.double(threshold), 
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

def compute_newtons_fractal(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.newtons_fractal(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                            np.uint32(max_iter), np.double(threshold), 
                            np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_minus_1i_fractal(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.minus_1i_fractal(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                             np.uint32(max_iter), np.double(threshold), 
                             np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_glynn(width, height, image, image_buffer, max_iter, threshold, r, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.glynn_fractal(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                          np.uint32(max_iter), np.double(threshold), np.double(r),
                          np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_bianca(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.bianca_fractal(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                           np.uint32(max_iter), np.double(threshold), 
                           np.double(xmin), np.double(xmax), np.double(ymin), np.double(ymax))
    cl.enqueue_copy(queue, image, image_buffer)

def compute_polynomial_roots(width, height, image, image_buffer, max_iter, threshold, xmin, xmax, ymin, ymax):
    global_size = (width, height)
    program.polynomial_roots(queue, global_size, None, image_buffer, np.uint32(width), np.uint32(height), 
                             np.uint32(max_iter), np.float32(threshold), 
                             np.float32(xmin), np.float32(xmax), np.float32(ymin), np.float32(ymax))
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
