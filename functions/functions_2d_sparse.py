import constants.constants_2d_non_sparse_sigma as const
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import scipy.sparse as ss

from helpers.sparse_size_calculator import sparse_dia_memory_usage, \
    sparse_coo_memory_usage, sparse_other_memory_usage


def flatten_image(image):
    return image.flatten()

def unflatten_image(image):
    return image.reshape((-1,const.IMAGE_NCOLS))

def convert_to_1d(arr):
    if arr.ndim > 1:
        return arr.T[0]
    else:
        return arr

def faster_norm_squared(arr):
    return np.sum(arr * arr)

def convert_to_2d(mat):
    if mat.ndim < 2:
        return np.array([mat]).T
    else:
        return mat

def calculate_template(alphas):
    return (get_pixel_by_centers_matrix(const.ALL_PIXELS,
                                        const.P_CENTERS,
                                        const.TEMPLATE_SD2) @ alphas)


# Returns nparray dim (IMAGEDIM,)
def kBpa(betas, alphas):
    # Should be IMAGE_DIM by ALPHA_DIM
    kBp = calculate_kBp(betas)
    return kBp @ alphas


def to_minimize(beta, alpha, g_inv, sd2, image):  # Fix the static predictions
    image_difference = image - kBpa(beta, alpha)
    left = (1 / 2) * np.trace(beta.T @ g_inv @ beta)
    right = (1 / (2 * sd2)) * faster_norm_squared(image_difference)
    return left + right


def generate_to_minimize(alpha, g_inv, sd2, image):
    def tmp_min(beta_flat):
        beta_reshaped = betas_to_2D(beta_flat)
        return to_minimize(beta_reshaped, alpha, g_inv, sd2, image)
    return tmp_min


def betas_to_1D(betas):
    return betas.flatten()


def betas_to_2D(flat_beta):
    return flat_beta.reshape((-1,2))


def handle_save_plot(path, image_name):
    date_str = str(datetime.date.today())
    to_save = path + image_name
    save_counter = 0
    while (os.path.isfile(to_save
                          + "_"
                          + date_str
                          + "_"
                          + str(save_counter)
                          + ".jpg")):
        save_counter += 1
    plt.savefig(to_save
                + "_"
                + date_str
                + "_"
                + str(save_counter)
                + ".jpg")


def handle_save_arr(path, arr_name, arr):
    date_str = str(datetime.date.today())
    to_save = path + arr_name
    save_counter = 0
    while (os.path.isfile(to_save
                          + "_"
                          + date_str
                          + "_"
                          + str(save_counter)
                          + ".txt")):
        save_counter += 1
    np.savetxt(to_save
                + "_"
                + date_str
                + "_"
                + str(save_counter)
                + ".txt", arr)


# def manual_gaussian(indexes, sd):
#     """
#     :param precomputed_gaussian:
#     :param indexes: np.array of all indexes to lookup ex. np.array([[0,0],[1,1],...] )
#     :return:
#     """
#     gaussian_out = const.gaussian_kernel_one_point(indexes, sd)
#     return gaussian_out


def get_pixel_by_centers_matrix(all_pixels, all_centers, sd2):
    """

    :param all_pixels: an array of all pixels ex. [[0,0],[0,1]...
    :param all_centers:  an array of all centers ex. [[0,0],[0,1]...
    :return: (n_pixels, n_centers) array with evaluated gaussian values relative to each center
    """
    n_pixels = np.shape(all_pixels)[0]
    n_centers = np.shape(all_centers)[0]
    pixels_pixels = np.repeat(all_pixels,n_centers,axis=0)
    centers_centers = np.tile(all_centers,(n_pixels,1))
    diff_vector = pixels_pixels - centers_centers
    gaussian_out = const.gaussian_kernel_given_diffs(diff_vector, sd2)
    reshaped_gauss = gaussian_out.reshape((n_pixels,n_centers))
    return reshaped_gauss


def get_sparse_pixel_by_centers(all_pixels, all_centers, sd2,
                                sparse_type=ss.dia_matrix,
                                error=1e-6):
    out = get_pixel_by_centers_matrix(all_pixels=all_pixels,
                                      all_centers=all_centers,
                                      sd2=sd2)
    out[np.abs(out) < error] = 0.0
    s_out = sparse_type(out)
    return s_out


def get_pixel_by_centers_matrix_mul_only(all_pixels, all_centers, sd2):
    one_col2 = np.ones((2, 1))
    n_pixels = np.shape(all_pixels)[0]
    n_centers = np.shape(all_centers)[0]
    p_norm_squared = (all_pixels ** 2) @ one_col2
    c_norm_squared = (all_centers ** 2) @ one_col2
    p_norm_squared_repeated = p_norm_squared @ np.ones((1, n_centers))
    c_norm_squared_repeated = (c_norm_squared @ np.ones((1, n_pixels))).T
    p_dot_c = 2 * (all_pixels @ all_centers.T)
    big_matrix = p_norm_squared_repeated + c_norm_squared_repeated - p_dot_c
    K = np.exp( - big_matrix / (2 * sd2))
    return K


#
# PIXEL_G_CENTERS_MATRIX = get_pixel_by_centers_matrix(const.ALL_PIXELS,
#                                                      const.G_CENTERS,
#                                                      const.DEFORM_SD2)

S_PIXEL_G_CENTERS_MATRIX = get_sparse_pixel_by_centers(all_pixels=const.ALL_PIXELS,
                                                       all_centers=const.G_CENTERS,
                                                       sd2=const.DEFORM_SD2,
                                                       error=1e-6)


def calculate_kBp(betas):
    deformation = S_PIXEL_G_CENTERS_MATRIX.dot(betas)
    deformed_pixel = const.ALL_PIXELS - deformation
    out_matrix = get_sparse_pixel_by_centers(deformed_pixel,
                                             const.P_CENTERS,
                                             const.TEMPLATE_SD2,
                                             sparse_type=ss.csc_matrix)
    return out_matrix

def calculate_kBp_and_deformation(betas):
    deformation = S_PIXEL_G_CENTERS_MATRIX.dot(betas)
    deformed_pixel = const.ALL_PIXELS - deformation
    out_matrix = get_sparse_pixel_by_centers(deformed_pixel,
                                const.P_CENTERS,
                                const.TEMPLATE_SD2,
                                sparse_type=ss.csc_matrix)
    return out_matrix, deformed_pixel

#
# rx, ry = np.random.normal(loc=0.0, scale=1.8, size=const.IMAGE_NCOLS), \
#          np.random.normal(loc=0.0, scale=1.8, size=const.IMAGE_NROWS)
# bx, by = np.meshgrid(rx, ry)
# # Pair up elems from gx and gy to create array of pairs
# B_2D = np.c_[bx.ravel(), by.ravel()]
#
# res = calculate_kBp(B_2D)
#
# plt.imshow(res.todense(),cmap='jet')
# plt.show()
#
# print('done')

