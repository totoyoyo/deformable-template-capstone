import numpy as np
from numba import jit, njit

TEMPLATE_SD = 0.3
DEFORM_SD = 0.3
SD_INIT = 1

KP = 16
KG = 16
ALPHAS_INIT = np.zeros((KP, 1)).astype('float64')
BETAS_INIT = np.zeros((KG, 2)).astype('float64')

P_CENTERS = np.array([[7, 7], [0, 0], [1, 1],
                      [2, 2], [3, 3],
                      [4, 4], [5, 5],
                      [6, 6],
                      [8, 8], [9, 9],
                      [10, 10], [11, 11],
                      [12, 12], [13, 13],
                      [14, 14], [15, 15]]).astype('float64')

G_CENTERS = np.array([[0, 0], [1, 1],
                      [2, 2], [3, 3],
                      [4, 4], [5, 5],
                      [6, 6], [7, 7],
                      [8, 8], [9, 9],
                      [10, 10], [11, 11],
                      [12, 12], [13, 13],
                      [14, 14], [15, 15]]).astype('float64')

assert P_CENTERS.shape[0] == KP
assert G_CENTERS.shape[0] == KG

@njit
def gaussian_kernel_2d(x_val, center_val, sd):
    diff = np.linalg.norm(x_val - center_val)
    inter = (-((diff) ** 2)
             / (2 * sd ** 2))
    out = np.e ** inter
    # Should be a float
    return out

IMAGE_NROWS = 16
IMAGE_NCOLS = 16
IMAGE_TOTAL = IMAGE_NROWS * IMAGE_NCOLS
IMAGE1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                   ]).astype('float64')
IMAGE2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                   ]).astype('float64')
IMAGES = [IMAGE1,IMAGE2]
# IMAGES = [np.full((IMAGE_NROWS,IMAGE_NCOLS), 0.5),
#           np.full((IMAGE_NROWS,IMAGE_NCOLS), 0.4)]
FLAT_IMAGES = list(map(lambda image: image.reshape(-1,1),
                       IMAGES))

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
