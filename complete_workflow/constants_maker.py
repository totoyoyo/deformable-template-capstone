import numpy as np
from sys import getsizeof
import scipy.linalg as sl
import scipy.sparse as ss
import functions_maker as func

# SD can be 15% of total side length
# so if side length is 100
# sd can be 15


"""
Functions to help pick kernel placement
"""


def kernel_on_every_pixel(img_dim_x, img_dim_y):
    IY, IX = np.meshgrid(np.arange(img_dim_y), np.arange(img_dim_x))

    x_2d = np.c_[IX.ravel(), IY.ravel()]

    return x_2d.astype('float32')


def get_spread_out_kernels(all_pixels, distance, randomize=False):
    if randomize:
        new_pixels = np.random.permutation(all_pixels)
    else:
        new_pixels = all_pixels
    to_return = np.array([new_pixels[0]])
    for pixel_point in new_pixels:
        diff = to_return - np.array([pixel_point])
        norms = np.linalg.norm(diff, axis=1)
        if not np.any(norms < distance):
            to_return = np.vstack((to_return, pixel_point))
    return to_return


def kernel_other_pixel(all_pixels, even=True):
    mod_out = 0 if even else 1
    filtered_pixels = all_pixels[np.sum(all_pixels, axis=1) % 2 == mod_out]
    return filtered_pixels


#Other needed functions
def gaussian_kernel_2d(x_val, center_val, sd2):
    """

    :param x_val: a 2d array with each row denoting a pixel
    :param center_val: ''
    :param sd2:
    :return: a 1d array of calculated gaussian (still needs to be reshaped)
    """
    diff = np.linalg.norm(x_val - center_val, axis=1)
    inter = (-((diff) ** 2)
             / (2 * sd2))
    out = np.exp(inter, dtype='float32')
    # Should be a float
    return out


def create_sparse_sigma_something_inverse(something_centers, k_something, some_sd2,
                                          error):
    _xx = np.repeat(something_centers, k_something, axis=0)
    _yy = np.tile(something_centers, (k_something, 1))
    the_inv = gaussian_kernel_2d(_xx, _yy, some_sd2).reshape((k_something, k_something))
    the_inv[np.abs(the_inv) < error] = 0.0
    s_inv = ss.csc_matrix(the_inv)
    return s_inv

class TrainingConstants:

    def __init__(self, images, ag, ap, t_sd2, d_sd2, init_sd, epochs=1000,
                 iterations=5,
                 train=True):
        if train:
            self.AG = ag
            self.AP = ap
            self.template_sd2 = t_sd2
            self.deform_sd2 = d_sd2
            self.init_sd = init_sd
            self.images = images
            self.image_nrow, self.image_ncol = self.images[0].shape
            self.image_total = self.image_nrow * self.image_ncol
            self.flat_images = list(map(lambda image: image.reshape(-1, 1),
                                        self.images))
            self.all_pixels = self.get_all_pixels()
            self.p_centers = kernel_other_pixel(self.all_pixels, even=True)

            self.g_centers = kernel_other_pixel(self.all_pixels, even=True)
            TD_SAME = self.template_sd2 == self.deform_sd2 \
                      and np.array_equal(self.p_centers, self.g_centers)
            self.kp = self.p_centers.shape[0]
            self.kg = self.g_centers.shape[0]
            self.alphas_init = np.zeros((self.kp, 1), dtype='float32')
            self.betas_init = np.zeros((self.kg, 2), dtype='float32')
            self.mup = np.zeros((self.kp, 1), dtype='float32')
            self.SPARSE_SIGMA_P_INV = create_sparse_sigma_something_inverse(
                self.p_centers, self.kp, self.template_sd2,
                1e-6)
            if TD_SAME:
                self.SPARSE_SIGMA_G_INV = self.SPARSE_SIGMA_P_INV
            else:
                self.SPARSE_SIGMA_G_INV = create_sparse_sigma_something_inverse(
                    self.g_centers, self.kg, self.deform_sd2,
                    1e-6)

            self.S_PIXEL_G_CENTERS_MATRIX = func.get_sparse_pixel_by_centers(
                all_pixels=self.all_pixels,
                all_centers=self.g_centers,
                sd2=self.deform_sd2,
                error=1e-6)
            self.epochs = epochs
            self.iterations = iterations

    def calculate_template(self, alphas):
        return (func.get_pixel_by_centers_matrix(self.all_pixels,
                                            self.p_centers,
                                            self.template_sd2) @ alphas)

    def calculate_kBp(self, betas):
        deformation = self.S_PIXEL_G_CENTERS_MATRIX.dot(betas)
        deformed_pixel = self.all_pixels - deformation
        out_matrix = func.get_sparse_pixel_by_centers(deformed_pixel,
                                                 all_centers=self.p_centers,
                                                 sd2=self.template_sd2)
        return out_matrix

    def get_all_pixels(self):
        IY, IX = np.meshgrid(np.arange(self.image_ncol),
                             np.arange(self.image_nrow))
        all_pixels = np.c_[IX.ravel(), IY.ravel()].astype('float32')
        return all_pixels

    def kBpa(self,betas, alphas):
        kBp = self.calculate_kBp(betas)
        return kBp @ alphas





