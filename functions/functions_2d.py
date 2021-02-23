import constants.constants_2d_0 as const
from scipy.optimize import check_grad
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import numba as nb


# @jit(signature_or_function="float64(array(int32, 1d, C), array(float64, 2d, C))")
# @nb.jit(signature_or_function="float64[::1](float64[::1], float64[:,::1])")
# @nb.njit(parallel=True)
# def cal_deformation(pixel_location, betas):
#     counter = np.array([0.0, 0.0])
#     for index, center in enumerate(const.G_CENTERS):
#         counter += betas[index] * const.gaussian_kernel_2d(pixel_location, center, const.DEFORM_SD2)
#     return counter

# @nb.njit(parallel=True)
# def cal_deformation(pixel_location, betas):
#     counter = np.array([0.0, 0.0])
#     counter += betas[1] * const.gaussian_kernel_2d(pixel_location, np.array([1,2]), const.DEFORM_SD2)
#     return counter

# @nb.jit(parallel=True, forceobj=True)
# def cal_deformation(pixel_location, betas):
#     counter = np.array([0.0, 0.0])
#     for beta_index in range(const.KG):
#         value = const.gaussian_kernel_2d(pixel_location, const.G_CENTERS[beta_index], const.DEFORM_SD2)
#         counter += value * betas[beta_index]
#     return counter

def cal_deformation(pixel_location, betas):
    repeated_pixel_location = np.full((const.KG, 2), pixel_location)
    kernel_out = const.gaussian_kernel_2d_many(repeated_pixel_location,
                                               const.G_CENTERS,
                                               const.DEFORM_SD2)
    total_deformation = kernel_out @ betas
    return total_deformation


# cal_deformation(np.array([2.0,3.0]), const.BETAS_INIT)

def pixel_index_to_position(p_index):
    row = p_index // const.IMAGE_NCOLS
    col = p_index % const.IMAGE_NCOLS
    return np.array([row, col])


def position_to_pixel_index(position):
    return const.IMAGE_NCOLS * position[0] + position[1]


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
                                        COMPUTED_GAUSSIAN_P) @ alphas)


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


def precompute_gaussian(nrows, ncols, sd2=1):
    """
    Calculates a gaussian kernel
    :param nrows: number of rows in the image
    :param ncols: number of columns in the image
    :param sd2: SD^2 of the gaussian
    :return: (nrows, ncol) matrix with
    """
    rows = np.arange(nrows)
    cols = np.arange(ncols)
    xx, yy = np.meshgrid(rows, cols)
    return const.gaussian_kernel_naive(xx, yy, sd2)

def precompute_gaussian_big():
    rows = np.linspace(0,5,num=5001)
    cols = np.linspace(0,5,num=5001)
    xx, yy = np.meshgrid(rows, cols)
    return const.gaussian_kernel_naive_2(xx, yy, 1)


COMPUTED_GAUSSIAN_BIG = precompute_gaussian_big()


COMPUTED_GAUSSIAN_G = precompute_gaussian(const.IMAGE_NROWS, const.IMAGE_NCOLS, const.DEFORM_SD2)
COMPUTED_GAUSSIAN_P = precompute_gaussian(const.IMAGE_NROWS, const.IMAGE_NCOLS, const.TEMPLATE_SD2)


def lookup_gaussian(indexes, precomputed_gaussian):
    """
    :param precomputed_gaussian:
    :param indexes: np.array of all indexes to lookup ex. np.array([[0,0],[1,1],...] )
    :return:
    """
    int_indexes = np.rint(indexes).astype(int)
    row_lookup, col_lookup = int_indexes.T
    gaussian_out = precomputed_gaussian[row_lookup, col_lookup]
    return gaussian_out

def manual_gaussian(indexes, precomputed_gaussian):
    """
    :param precomputed_gaussian:
    :param indexes: np.array of all indexes to lookup ex. np.array([[0,0],[1,1],...] )
    :return:
    """
    gaussian_out = const.gaussian_kernel_one_point(indexes, 0.3)
    return gaussian_out

def lookup_big_gaussian(indexes, sd):
    z = indexes / sd
    new_indexes = (z * 1000).astype(int)
    clipped_indexes = np.clip(new_indexes, a_min=0, a_max=5000)
    row_lookup, col_lookup = clipped_indexes.T
    gaussian_out = COMPUTED_GAUSSIAN_BIG[row_lookup, col_lookup]
    return gaussian_out


def get_pixel_by_centers_matrix(all_pixels, all_centers, precomputed_gaussian):
    """

    :param all_pixels: an array of all pixels ex. [[0,0],[0,1]...
    :param all_centers:  an array of all centers ex. [[0,0],[0,1]...
    :return: (n_pixels, n_centers) array with evaluated gaussian values relative to each center
    """
    n_pixels = np.shape(all_pixels)[0]
    n_centers = np.shape(all_centers)[0]
    pixels_pixels = np.repeat(all_pixels,n_centers,axis=0)
    centers_centers = np.tile(all_centers,(n_pixels,1))
    vector = np.abs(pixels_pixels - centers_centers)

    gaussian_out = manual_gaussian(vector, precomputed_gaussian)
    reshaped_gauss = gaussian_out.reshape((n_pixels,n_centers))
    return reshaped_gauss

PIXEL_G_CENTERS_MATRIX = get_pixel_by_centers_matrix(const.ALL_PIXELS,
                                                     const.G_CENTERS,
                                                     COMPUTED_GAUSSIAN_G)

def calculate_kBp(betas):
    deformation = PIXEL_G_CENTERS_MATRIX @ betas
    deformed_pixel = const.ALL_PIXELS - deformation
    return get_pixel_by_centers_matrix(deformed_pixel,
                                const.P_CENTERS,
                                COMPUTED_GAUSSIAN_P)


out = lookup_big_gaussian(const.ALL_PIXELS,1)

print('done')
