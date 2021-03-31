# old from constants---------------------
import numpy as np
import scipy.sparse as ss

def invert_to_dense(sparse_mat):
    dense = sparse_mat.todense()
    # u, s, vh = np.linalg.svd(dense)
    # raw_inv = np.linalg.inv(dense)
    pinv_dense = clean_pinv(dense)
    # dense[dense < 1e-3] = 0
    # pinv_norm = np.linalg.pinv(dense, rcond=1e-3, hermitian=True)
    return pinv_dense

def clean_pinv(mat):
    pinv_mat = np.linalg.pinv(mat, rcond=1e-4, hermitian=True)
    return pinv_mat


def to_sparse(dense_mat,
              error=1e-6):
    dense_mat[abs(dense_mat) < error] = 0.0
    out = ss.csc_matrix(dense_mat)
    return out

def gaussian_kernel_original(x_val, sd):
    diff = np.linalg.norm(x_val)
    inter = (-((diff) ** 2)
             / (2 * sd ** 2))
    out = np.exp(inter, dtype='float32')
    # Should be a float
    return out


def gaussian_kernel_given_diffs(diffs, sd2):
    """

    :param diffs: a 2d array with each row denoting vector (calculated diff)
    :param sd: sd not squared
    :return: A 1d array of calculated gaussians
    """
    norm2_squared = np.sum((diffs ** 2), axis=1)
    inter = (-((norm2_squared))
             / (2 * sd2))
    out = np.exp(inter)
    # Should be a float
    return out


def gaussian_kernel_naive(pixel_row, pixel_col, sd2):
    """
    Gaussian for precomputing gaussian
    :param pixel_row:
    :param pixel_col:
    :param sd2:
    :return:
    """
    diff = pixel_row ** 2 + pixel_col ** 2
    inter = (-(diff)
             / (2 * sd2))
    out = np.exp(inter, dtype='float32')
    return out


def gaussian_kernel_input2_sd2(input2, sd2):
    """
    Takes an array of squared norm of diffs
    :param input2: an 1D array of squared norm of diffs
    :param sd2:
    :return:
    """
    inter = (-(input2)
             / (2 * sd2))
    out = np.e ** inter
    return out

#----------------NEW STUFF--------------------------

# import constants_maker as constants
import datetime
import os
import matplotlib.pyplot as plt


def flatten_image(image):
    return image.flatten()

def unflatten_image(image, ncols):
    return image.reshape((-1, ncols))

def faster_norm_squared(arr):
    return np.sum(arr * arr)


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
    gaussian_out = gaussian_kernel_given_diffs(diff_vector, sd2)
    reshaped_gauss = gaussian_out.reshape((n_pixels,n_centers))
    return reshaped_gauss

def get_sparse_pixel_by_centers(all_pixels, all_centers, sd2,
                                sparse_type=ss.csc_matrix,
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

def generate_gaussian_kernel(sd2):
    sd = np.sqrt(sd2)
    c = np.ceil(sd * 4)
    x = np.arange(0, 2 * c + 1)
    RBF = np.exp(- (1 / (2 * sd2)) * (x - c) ** 2, dtype = 'float32')
    RBF2 = np.outer(RBF, RBF)
    return c.astype('int'), RBF2

def get_list_of_indexes_for_slicing(slice_length,total_length):
    i = 0
    l = []
    while(i < total_length):
        end = min(total_length, i + slice_length)
        indexes = [i,end]
        i = end
        l.append(indexes)
    return l

def make_image_of_betas_for_conv(betas, beta_centers, image_row, image_col):
    """
    :param betas:
    :param beta_centers:
    :param image_row:
    :param image_col:
    :return: 2 by image_row by image_col array (the 2 is the 2 channels of betas)
    """
    empty_array = np.zeros((2,image_row,image_col), dtype='float32')
    rows, cols = beta_centers.T
    empty_array[:, rows, cols] = betas.T
    return empty_array



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
