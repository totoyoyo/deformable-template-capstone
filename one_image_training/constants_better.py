import numpy as np

TEMPLATE_SD = 1
DEFORM_SD = 1
SD_INIT = 1

ALPHAS_INIT = np.array([[0, 0]]).T
BETAS_INIT = np.array([[0, 0]]).T
KP = ALPHAS_INIT.size
KG = BETAS_INIT.size

IMAGE = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
IMAGE_DIM = IMAGE.size
PREDICT_INIT = np.array([[0, 0, 0, 0, 0]]).T
assert PREDICT_INIT.size == IMAGE_DIM

AG = 1
AP = 1
N = 1
MU_P = np.array([[0, 0]]).T

# Initializers
SIGMA_P = np.array([[0, 0],
                    [0, 0]])
SIGMA_P_INV = np.array([[0, 0],
                        [0, 0]])

SIGMA_G = np.array([[0, 0],
                    [0, 0]])
SIGMA_G_INV = np.array([[0, 0],
                        [0, 0]])

P_CENTERS = np.array([[0, 1]]).T
G_CENTERS = np.array([[0, 1]]).T