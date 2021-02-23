import numpy as np
from numba import njit

TEMPLATE_SD = 0.3
DEFORM_SD = 0.3
SD_INIT = 1

IMAGE_NROWS = 10
IMAGE_NCOLS = 10
IMAGE_TOTAL = IMAGE_NROWS * IMAGE_NCOLS
IMAGE1 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')
IMAGE2 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')
IMAGE1[4:6, 4:6] = 1.0
IMAGE2[6:8, 6:8] = 1.0

IMAGES = [IMAGE1, IMAGE2]
# IMAGES = [np.full((IMAGE_NROWS,IMAGE_NCOLS), 0.5),
#           np.full((IMAGE_NROWS,IMAGE_NCOLS), 0.4)]
FLAT_IMAGES = list(map(lambda image: image.reshape(-1, 1),
                       IMAGES))


def kernel_on_every_pixel(img_dim_x, img_dim_y):
    rx, ry = np.arange(0, img_dim_x, 1), np.arange(0, img_dim_y, 1)
    gx, gy = np.meshgrid(rx, ry)

    # Pair up elems from gx and gy to create array of pairs
    X_2D = np.c_[gx.ravel(), gy.ravel()]

    return X_2D.astype('float64')


kernel_on_every_pixel(IMAGE_NROWS, IMAGE_NCOLS)

# P_CENTERS = kernel_on_every_pixel(IMAGE_NROWS, IMAGE_NCOLS)

# G_CENTERS = kernel_on_every_pixel(IMAGE_NROWS, IMAGE_NCOLS)

P_CENTERS = np.array([[2, 2], [2, 4], [2, 6], [2, 8],
                      [4, 2], [4, 4], [4, 6], [4, 8],
                      [6, 2], [6, 4], [6, 6], [6, 8],
                      [8, 2], [8, 4], [8, 6], [8, 8]]).astype('float64')

G_CENTERS = np.array([[2, 2], [2, 4], [2, 6], [2, 8],
                      [4, 2], [4, 4], [4, 6], [4, 8],
                      [6, 2], [6, 4], [6, 6], [6, 8],
                      [8, 2], [8, 4], [8, 6], [8, 8]]).astype('float64')
#

#
# G_CENTERS = np.array([[0, 0], [1, 1],
#                       [2, 2], [3, 3],
#                       [4, 4], [5, 5],
#                       [6, 6], [7, 7],
#                       [8, 8], [9, 9],
#                       [10, 10], [11, 11],
#                       [12, 12], [13, 13],
#                       [14, 14], [15, 15]]).astype('float64')

KP = P_CENTERS.shape[0]
KG = G_CENTERS.shape[0]
ALPHAS_INIT = np.zeros((KP, 1)).astype('float64')
BETAS_INIT = np.zeros((KG, 2)).astype('float64')


@njit
def gaussian_kernel_2d(x_val, center_val, sd):
    diff = np.linalg.norm(x_val - center_val)
    inter = (-((diff) ** 2)
             / (2 * sd ** 2))
    out = np.e ** inter
    # Should be a float
    return out


def gaussian_kernel_2d_many(x_val, center_val, sd):
    diff = np.linalg.norm(x_val - center_val, axis=1)
    inter = (-((diff) ** 2)
             / (2 * sd ** 2))
    out = np.exp(inter)
    # Should be a float
    return out


def gaussian_kernel_naive(pixel_row, pixel_col, sd):
    diff = pixel_row ** 2 + pixel_col ** 2
    inter = (-(diff)
             / (2 * sd))
    out = np.e ** inter
    # Should be a float
    return out


yo = np.vectorize(gaussian_kernel_2d_many, signature='(a,b),(a,b),()->(b)')


def gaussian_on_betas(x_val, centers, sd):
    """
    :param x_values: np.array([,]) all pixel positions
    :param centers: np.array([[ , ],[ , ],...]) all centers
    :param betas: np.array([[ , ],[ , ],...])
    :param sd: scalar
    :return: deformation of the form np.array([,])
    """
    return 1


def kBp_in_one_go(pixel_positions, b_centers, a_centers, betas, d):
    """

    :param pixel_positions:
    :param b_centers:
    :param a_centers:
    :param betas:
    :param d:
    :return:
    """
    return None


AG = 5
AP = 1
MU_P = np.zeros((KP, 1)).astype('float64')

SIGMA_P = np.zeros((KP, KP)).astype('float64')
SIGMA_P_INV = np.zeros((KP, KP)).astype('float64')

SIGMA_G = np.zeros((KG, KG)).astype('float64')
SIGMA_G_INV = np.zeros((KG, KG)).astype('float64')

for i in range(KP):
    for j in range(KP):
        SIGMA_P_INV[i, j] = gaussian_kernel_2d(P_CENTERS[i],
                                               P_CENTERS[j],
                                               TEMPLATE_SD)

for i in range(KG):
    for j in range(KG):
        SIGMA_G_INV[i, j] = gaussian_kernel_2d(G_CENTERS[i],
                                               G_CENTERS[j],
                                               DEFORM_SD)
SIGMA_P = np.linalg.inv(SIGMA_P_INV)
SIGMA_G = np.linalg.inv(SIGMA_G_INV)


# myinput = np.full((KG,2),np.array([1,1]))
#
# out = gaussian_kernel_2d_many(myinput,G_CENTERS,1)


# def get_gaussian_g(x_val, center):
#     """
#     Looks up the gaussian for deformation
#
#     Maybe swap x_val and center?
#     :param x_val: np.array(_) x values (can be many)
#     :param center: np.array() center (only 1)
#     :return: np.array() of scalars for output
#     """
#     vector = np.abs(x_val - center)
#     row_lookup, col_lookup = vector.T
#     return COMPUTED_GAUSSIAN_G[row_lookup, col_lookup]



# def get_gaussian_extream(all_pixels, all_centers):
#     """
#
#     :param all_pixels: an array of all pixels ex. [[0,0],[0,1]...
#     :param all_centers:  an array of all centers ex. [[0,0],[0,1]...
#     :return: (n_pixels, n_centers) array with evaluated gaussian values
#     """
#     n_pixels = np.shape(all_pixels)[0]
#     n_centers = np.shape(all_centers)[0]
#     pixels_pixels = np.repeat(all_pixels,n_centers,axis=0)
#     centers_centers = np.tile(all_centers,(n_pixels,1))
#     vector = np.abs(pixels_pixels - centers_centers)
#     row_lookup, col_lookup = vector.T
#     gaussian_out = COMPUTED_GAUSSIAN_G[row_lookup, col_lookup]
#     reshaped_gauss = gaussian_out.reshape((n_pixels,n_centers))
#     return reshaped_gauss

pixels = np.array([[0,0],[0,1],[1,1]])
centers = np.array([[3,3],[2,2]])

# get_gaussian_extream(pixels,centers)

# def get_gaussian_p(x_val, center):
#     """
#     get gaussians for p
#     :param x_val: np.array(_) x values (can be many)
#     :param deformation: np.array() deformation (1 deformation)
#     :param center:
#     :return:
#     """
#     vector = np.abs(x_val - center)
#     rounded_vector = np.rint(vector)
#     row_lookup, col_lookup = rounded_vector.T
#     return COMPUTED_GAUSSIAN_P[row_lookup, col_lookup]


IY, IX = np.meshgrid(np.arange(IMAGE_NCOLS),np.arange(IMAGE_NROWS))

ALL_PIXELS = np.c_[IX.ravel(),IY.ravel()]