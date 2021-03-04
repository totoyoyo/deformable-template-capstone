import numpy as np
from sys import getsizeof

import scipy.linalg as sl
import scipy.sparse as ss

TEMPLATE_SD2 = np.float32(1)
DEFORM_SD2 = np.float32(1)
TD_SAME = True
SD_INIT = np.float32(1)

e32 = np.exp(SD_INIT, dtype='float32')
new = e32 * e32

# IMAGE_NROWS = 10
# IMAGE_NCOLS = 10
# IMAGE_TOTAL = IMAGE_NROWS * IMAGE_NCOLS
# IMAGE1 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float32')
# IMAGE2 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float32')
# # IMAGE1[4:6, 4:6] = 1.0
# # IMAGE2[6:8, 6:8] = 1.0
# IMAGE1[1:4, 1:4] = 1.0
# IMAGE2[5:9, 5:9] = 1.0


# IMAGE_NROWS = 70
# IMAGE_NCOLS = 70
# IMAGE_TOTAL = IMAGE_NROWS * IMAGE_NCOLS
# IMAGE1 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')
# IMAGE2 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')


IMAGE_NROWS = 50
IMAGE_NCOLS = 50
IMAGE_TOTAL = IMAGE_NROWS * IMAGE_NCOLS
IMAGE1 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS),dtype='float32')
IMAGE2 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS),dtype='float32')
IMAGE1[10:14, 10:14] = 1.0
IMAGE2[11:15, 11:15] = 1.0
# IMAGE3 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS),dtype='float64')

# IMAGE1[2:3, 2:3] = 1.0
# IMAGE2[4:5, 4:5] = 1.0



# IMAGE1[10:14, 10:14] = 1.0
# IMAGE2[11:15, 11:15] = 1.0


IMAGES = [IMAGE1, IMAGE2]

FLAT_IMAGES = list(map(lambda image: image.reshape(-1, 1),
                       IMAGES))


def kernel_on_every_pixel(img_dim_x, img_dim_y):
    rx, ry = np.arange(0, img_dim_x, 1), np.arange(0, img_dim_y, 1)
    gx, gy = np.meshgrid(rx, ry)

    # Pair up elems from gx and gy to create array of pairs
    X_2D = np.c_[gx.ravel(), gy.ravel()]

    return X_2D.astype('float32')


def kernel_on_every_pixel_fix(img_dim_x, img_dim_y):
    IY, IX = np.meshgrid(np.arange(img_dim_y), np.arange(img_dim_x))

    x_2d = np.c_[IX.ravel(), IY.ravel()]

    return x_2d.astype('float32')


P_CENTERS = kernel_on_every_pixel_fix(IMAGE_NROWS, IMAGE_NCOLS)

G_CENTERS = kernel_on_every_pixel_fix(IMAGE_NROWS, IMAGE_NCOLS)

KP = P_CENTERS.shape[0]
KG = G_CENTERS.shape[0]
ALPHAS_INIT = np.zeros((KP, 1),dtype='float32')
BETAS_INIT = np.zeros((KG, 2),dtype='float32')


def gaussian_kernel_2d(x_val, center_val, sd2):
    """

    :param x_val: a 2d array with each row denoting a pixel
    :param center_val: ''
    :param sd2:
    :return: a 1d array of calculated gaussian (still needs to be reshaped)
    """
    diff = np.linalg.norm(x_val - center_val,axis=1)
    inter = (-((diff) ** 2)
             / (2 * sd2))
    out = np.exp(inter, dtype='float32')
    # Should be a float
    return out

def gaussian_kernel_original(x_val, sd):
    diff = np.linalg.norm(x_val)
    inter = (-((diff) ** 2)
             / (2 * sd**2))
    out = np.exp(inter, dtype='float32')
    # Should be a float
    return out


def gaussian_kernel_given_diffs(diffs, sd2):
    """

    :param diffs: a 2d array with each row denoting vector (calculated diff)
    :param sd: sd not squared
    :return: A 1d array of calculated gaussians
    """
    norm2_squared = np.sum((diffs ** 2),axis=1)
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


AG = np.float32(1)
AP = np.float32(1)
MU_P = np.zeros((KP, 1), dtype='float32')


def create_sparse_sigma_something_inverse(something_centers, k_something, some_sd2,
                                          error):
    _xx = np.repeat(something_centers, k_something, axis=0)
    _yy = np.tile(something_centers, (k_something, 1))
    the_inv = gaussian_kernel_2d(_xx,_yy,some_sd2).reshape((k_something, k_something))
    the_inv[np.abs(the_inv) < error] = 0.0
    s_inv = ss.csc_matrix(the_inv)
    return s_inv

SPARSE_SIGMA_P_INV = create_sparse_sigma_something_inverse(P_CENTERS, KP, TEMPLATE_SD2,
                                                       1e-6)
if TD_SAME:
    SPARSE_SIGMA_G_INV = SPARSE_SIGMA_P_INV
else:
    SPARSE_SIGMA_G_INV = create_sparse_sigma_something_inverse(G_CENTERS, KG, DEFORM_SD2,
                                                       1e-6)

def invert_to_dense(sparse_mat):
    dense = sparse_mat.todense()
    inv_dense = sl.inv(dense)
    return inv_dense

def to_sparse(dense_mat,
              error=1e-6):
    dense_mat[abs(dense_mat) < error] = 0.0
    out = ss.csc_matrix(dense_mat)
    return out

IY, IX = np.meshgrid(np.arange(IMAGE_NCOLS),np.arange(IMAGE_NROWS))

ALL_PIXELS = np.c_[IX.ravel(),IY.ravel()].astype('float32')

# one_point = gaussian_kernel_one_point(ALL_PIXELS,0.2)
# import matplotlib.pyplot as plt
# plt.imshow(SIGMA_G,cmap='jet')
# plt.show()
#
# plt.imshow(SIGMA_G_INV,cmap='jet')
# plt.show()
#
# plt.imshow(SIGMA_P,cmap='jet')
# plt.show()
#
# plt.imshow(SIGMA_P_INV,cmap='jet')
# plt.show()


ONE_COL2 = np.array([1,1])

ONE_KP = np.ones((KP,))

ONE_L = np.ones((IMAGE_TOTAL,))