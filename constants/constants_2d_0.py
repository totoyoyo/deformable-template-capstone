import numpy as np

TEMPLATE_SD2 = 1
DEFORM_SD2 = 1
SD_INIT = 1

# IMAGE_NROWS = 10
# IMAGE_NCOLS = 10
# IMAGE_TOTAL = IMAGE_NROWS * IMAGE_NCOLS
# IMAGE1 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')
# IMAGE2 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')
# IMAGE1[4:6, 4:6] = 1.0
# IMAGE2[6:8, 6:8] = 1.0

IMAGE_NROWS = 50
IMAGE_NCOLS = 50
IMAGE_TOTAL = IMAGE_NROWS * IMAGE_NCOLS
IMAGE1 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')
IMAGE2 = np.zeros((IMAGE_NROWS, IMAGE_NCOLS)).astype('float64')
IMAGE1[20:30, 20:30] = 1.0
IMAGE2[30:40, 30:40] = 1.0

IMAGES = [IMAGE1, IMAGE2]
# IMAGES = [np.full((IMAGE_NROWS,IMAGE_NCOLS), 0.5),
#           np.full((IMAGE_NROWS,IMAGE_NCOLS), 0.4)]
FLAT_IMAGES = list(map(lambda image: image.reshape(-1, 1),
                       IMAGES))

# FLAT_IMAGES = list(map(lambda image: image.flatten(),
#                        IMAGES))
#
# FLAT_IMAGES1 = list(map(lambda image: image.reshape(-1, 1),
#                        IMAGES))
#
# FLAT_IMAGES2 = list(map(lambda image: image.flatten(),
#                        IMAGES))
#

def kernel_on_every_pixel(img_dim_x, img_dim_y):
    rx, ry = np.arange(0, img_dim_x, 1), np.arange(0, img_dim_y, 1)
    gx, gy = np.meshgrid(rx, ry)

    # Pair up elems from gx and gy to create array of pairs
    X_2D = np.c_[gx.ravel(), gy.ravel()]

    return X_2D.astype('float64')


def kernel_on_every_pixel_fix(img_dim_x, img_dim_y):
    IY, IX = np.meshgrid(np.arange(img_dim_y), np.arange(img_dim_x))

    x_2d = np.c_[IX.ravel(), IY.ravel()]

    return x_2d.astype('float64')


P_CENTERS = kernel_on_every_pixel_fix(IMAGE_NROWS, IMAGE_NCOLS)

G_CENTERS = kernel_on_every_pixel_fix(IMAGE_NROWS, IMAGE_NCOLS)

KP = P_CENTERS.shape[0]
KG = G_CENTERS.shape[0]
ALPHAS_INIT = np.zeros((KP, 1)).astype('float64')
BETAS_INIT = np.zeros((KG, 2)).astype('float64')


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
    out = np.exp(inter)
    # Should be a float
    return out

def gaussian_kernel_original(x_val, sd):
    diff = np.linalg.norm(x_val)
    inter = (-((diff) ** 2)
             / (2 * sd**2))
    out = np.exp(inter)
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
    out = np.exp(inter)
    return out


def gaussian_kernel_naive_sd(pixel_row, pixel_col, sd):
    """
    Same as above but with not squared sd
    :param pixel_row:
    :param pixel_col:
    :param sd:
    :return:
    """
    diff = pixel_row ** 2 + pixel_col ** 2
    inter = (-(diff)
             / (2 * sd ** 2))
    out = np.e ** inter
    # Should be a float
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



AG = 5
AP = 1
MU_P = np.zeros((KP, 1)).astype('float64')

# SIGMA_P = np.zeros((KP, KP)).astype('float64')
# SIGMA_P_INV = np.zeros((KP, KP)).astype('float64')
#
# SIGMA_G = np.zeros((KG, KG)).astype('float64')
# SIGMA_G_INV = np.zeros((KG, KG)).astype('float64')
#


p_xx = np.repeat(P_CENTERS,KP,axis=0)
p_yy = np.tile(P_CENTERS, (KP, 1))
SIGMA_P_INV = gaussian_kernel_2d(p_xx,p_yy,TEMPLATE_SD2).reshape((KP,KP))

g_xx = np.repeat(G_CENTERS,KG,axis=0)
g_yy = np.tile(G_CENTERS, (KG, 1))
SIGMA_G_INV = gaussian_kernel_2d(g_xx,g_yy,DEFORM_SD2).reshape((KG,KG))

SIGMA_P = np.linalg.inv(SIGMA_P_INV)
SIGMA_G = np.linalg.inv(SIGMA_G_INV)


IY, IX = np.meshgrid(np.arange(IMAGE_NCOLS),np.arange(IMAGE_NROWS))

ALL_PIXELS = np.c_[IX.ravel(),IY.ravel()]

# one_point = gaussian_kernel_one_point(ALL_PIXELS,0.2)


ONE_COL2 = np.array([1,1])

ONE_KP = np.ones((KP,))

ONE_L = np.ones((IMAGE_TOTAL,))

ONE_KG = np.ones((KG,))