import constants.constants_2d_0 as const
from scipy.optimize import check_grad
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import numba as nb



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


def precompute_1D_gaussian_big():
    inputs = np.linspace(0, 36, num=36000001)
    gauss_out = const.gaussian_kernel_input2_sd2(inputs,1)
    return gauss_out


gaussian_1d = precompute_1D_gaussian_big()


def lookup_big_gaussian(indexes, sd2):
    row_lookup, col_lookup = indexes.T
    squared = row_lookup ** 2 + col_lookup ** 2
    squared = squared / sd2
    new_indexes = (squared * 1000000).astype(int)
    clipped_indexes = np.clip(new_indexes, a_min=0, a_max=36000000)
    gaussian_out = gaussian_1d[clipped_indexes]
    return gaussian_out


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
    vector = np.abs(pixels_pixels - centers_centers)

    gaussian_out = lookup_big_gaussian(vector, sd2)
    reshaped_gauss = gaussian_out.reshape((n_pixels,n_centers))
    return reshaped_gauss

PIXEL_G_CENTERS_MATRIX = get_pixel_by_centers_matrix(const.ALL_PIXELS,
                                                     const.G_CENTERS,
                                                     const.DEFORM_SD2)
from timeit import default_timer as timer

def calculate_kBp(betas):
    deformation = PIXEL_G_CENTERS_MATRIX @ betas
    start = timer()
    deformed_pixel = const.ALL_PIXELS - deformation
    out_matrix = get_pixel_by_centers_matrix(deformed_pixel,
                                const.P_CENTERS,
                                const.TEMPLATE_SD2)
    end = timer()
    totaltime = end - start
    return totaltime, out_matrix


rx, ry = np.random.normal(loc=0.0, scale=1.8, size=const.IMAGE_NCOLS), \
         np.random.normal(loc=0.0, scale=1.8, size=const.IMAGE_NROWS)
bx, by = np.meshgrid(rx, ry)
# Pair up elems from gx and gy to create array of pairs
B_2D = np.c_[bx.ravel(), by.ravel()]


res = calculate_kBp(B_2D)
plt.imshow(res[1])
plt.show()