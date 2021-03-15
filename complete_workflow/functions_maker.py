# old from constants---------------------
import numpy as np
import scipy.sparse as ss

def invert_to_dense(sparse_mat):
    dense = sparse_mat.todense()
    inv_dense = np.linalg.pinv(dense, rcond=1e-6, hermitian=True)
    return inv_dense


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
