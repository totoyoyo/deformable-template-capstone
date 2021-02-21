from scipy import optimize

from constants.one_dim.constants_1d_2 import *

import matplotlib.pyplot as plt

# Deformation
def cal_deformation(x, b1d):
    sumer = 0
    for index, center in enumerate(G_CENTERS):
        sumer += b1d[index] * gaussian_kernel(x, center, DEFORM_SD)
    return sumer


def convert_to_1d(arr):
    return arr.T[0]


def convert_to_2d(mat):
    return np.array([mat]).T


def calculate_kBp(b1d):
    tmp_kbp = np.empty((IMAGE_DIM, KP))
    for i in range(IMAGE_DIM):
        for j in range(KP):
            the_def = cal_deformation(i, b1d)
            tmp_kbp[i, j] = gaussian_kernel(i - the_def,
                                            P_CENTERS[j],
                                            TEMPLATE_SD)
    return tmp_kbp


class Estimator1D1Image:

    def __init__(self):
        self.alphas = ALPHAS_INIT
        self.betas = BETAS_INIT
        self.sd2 = SD_INIT
        self.kBp = \
            calculate_kBp(convert_to_1d(self.betas))
        self.Gamma = SIGMA_G
        self.images = IMAGE
        self.YTY = np.linalg.norm(self.images) ** 2
        self.predictions = PREDICT_INIT

    def to_minimize(self, b1d):
        result = (1 / 2) * b1d.T @ np.linalg.inv(self.Gamma) @ b1d \
                 + (1 / (2 * self.sd2)) * np.linalg.norm(self.images - self.predictions) ** 2
        return result.item()

    # Depends on current beta, Gamma, sd2, predictions, images
    def best_betas(self):
        betas_in_1D = convert_to_1d(self.betas)
        out = optimize.minimize(self.to_minimize, betas_in_1D).x
        self.betas = convert_to_2d(out)
        return out

    # Depends on current best betas
    def _bbtl(self):
        self.best_betas()
        return np.matmul(self.betas, self.betas.T)

    def update_Gamma(self):
        self.Gamma = (1 / (N + AG)) * (N * self._bbtl() + AG * SIGMA_G)

    def update_kBp(self):
        self.best_betas()
        self.kBp = calculate_kBp(convert_to_1d(self.betas))

    def update_prediction(self):
        self.update_kBp()
        self.predictions = np.matmul(self.kBp, self.alphas)

    def ky_kk(self):
        b1d = self.best_betas()
        kBp = calculate_kBp(b1d)
        ky = np.matmul(kBp.T, self.images)
        kk = np.matmul(kBp.T, kBp)
        return ky, kk

    def update_alpha_and_sd2(self):
        ky, kk = self.ky_kk()
        new_alpha = np.matmul(np.linalg.inv(N * kk + self.sd2 * SIGMA_P_INV),
                              (N * ky + self.sd2 * np.matmul(SIGMA_P_INV, MU_P)))
        new_sd2 = (1 / (N * IMAGE_DIM * AP)) * (N * (self.YTY + self.alphas.T @ kk @ self.alphas
                                                     - 2 * self.alphas.T @ ky)
                                                + AP * SD_INIT)
        self.alphas = new_alpha
        self.sd2 = new_sd2
        self.update_prediction()

    def run_estimation(self, iterations):
        for _ in range(iterations):
            self.update_Gamma()
            self.update_alpha_and_sd2()
        print("here are alphas, sd2, and Gamma in that order")
        print(self.alphas)
        print(self.sd2)
        print(self.Gamma)
        print("here are betas")
        print(self.betas)
        print("here are predictions")
        print(self.predictions)


my_estimator = Estimator1D1Image()
my_estimator.run_estimation(10)
plt.plot(my_estimator.predictions)
plt.show()
plt.plot(my_estimator.images)
plt.show()