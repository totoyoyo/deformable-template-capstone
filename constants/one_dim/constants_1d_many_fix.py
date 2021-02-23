import numpy as np

TEMPLATE_SD = 0.3
DEFORM_SD = 0.3
SD_INIT = 1

KP = 15
KG = 15
ALPHAS_INIT = np.zeros((KP, 1))
BETAS_INIT = np.zeros((KG, 1))
P_CENTERS = np.array([[30, 32, 34, 36, 38,
                       40, 42, 44, 46, 48,
                       50, 52, 54, 56, 58]]).T.astype('float64')
G_CENTERS = np.array([[30, 32, 34, 36, 38,
                       40, 42, 44, 46, 48,
                       50, 52, 54, 56, 58]]).T.astype('float64')
assert P_CENTERS.size == KP
assert G_CENTERS.size == KG

# 40 to 59 is 1
IMAGE1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T.astype('float64')
IMAGE2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T.astype('float64')
IMAGES = [IMAGE1, IMAGE2]
# IMAGE = np.fromfunction(lambda i: 0.8 if (70. > i > 40.) else 0., shape=(100,))
IMAGE_DIM = IMAGE1.size
PREDICT_INIT = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T.astype('float64')
assert PREDICT_INIT.size == IMAGE_DIM
AG = 5
AP = 1
N = 2
MU_P = np.zeros((KP, 1))

# Initializers

SIGMA_P = np.zeros((KP, KP))
SIGMA_P_INV = np.zeros((KP, KP))

SIGMA_G = np.zeros((KG, KG))
SIGMA_G_INV = np.zeros((KG, KG))


# def gaussian_kernel(x, center, sd) -> int:
#     out = np.exp(-(np.linalg.norm(x - center) ** 2)
#                   / (2 * sd ** 2))
#     return out

def gaussian_kernel(x, center, sd) -> int:
    try:
        center_val = center[0]
    except:
        center_val = center
    try:
        x_val = x[0]
    except:
        x_val = x

    inter = (-((x_val - center_val) ** 2)
                  / (2 * sd ** 2))
    out = np.e ** inter
    return out


for i in range(KP):
    for j in range(KP):
        SIGMA_P_INV[i, j] = gaussian_kernel(P_CENTERS[i],
                                            P_CENTERS[j],
                                            TEMPLATE_SD)

for i in range(KG):
    for j in range(KG):
        SIGMA_G_INV[i, j] = gaussian_kernel(G_CENTERS[i],
                                            G_CENTERS[j],
                                            DEFORM_SD)

SIGMA_P = np.linalg.inv(SIGMA_P_INV)
SIGMA_G = np.linalg.inv(SIGMA_G_INV)

## Gradients