from scipy import optimize

from one_image_training.constants_1d_many_images import *

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


class Estimator1DNImages:

    def __init__(self):
        self.alphas = ALPHAS_INIT
        self.betas = [BETAS_INIT,BETAS_INIT]
        self.sd2 = SD_INIT
        self.kBps = \
            list(map((lambda beta: calculate_kBp(convert_to_1d(beta))), self.betas))
        self.Gamma = SIGMA_G
        self.Gamma_Inv = SIGMA_G_INV
        self.images = IMAGES
        self.number_of_images = N
        self.YTY = np.linalg.norm(self.images) ** 2
        self.predictions = [PREDICT_INIT,PREDICT_INIT]

    def to_minimize(self, b1d, n):
        result = (1 / 2) * b1d.T @ self.Gamma_Inv @ b1d \
                 + (1 / (2 * self.sd2)) * np.linalg.norm(self.images[n] - self.predictions[n]) ** 2
        return result.item()

    def update_all_betas(self):
        # Depends on current beta, Gamma, sd2, predictions, images
        def update_best_beta(n):
            betas_in_1D = convert_to_1d(self.betas[n])
            out = optimize.minimize(self.to_minimize, betas_in_1D, n).x
            self.betas[n] = convert_to_2d(out)

        for n in range(self.number_of_images):
            update_best_beta(n)

    # Depends on current best betas
    def _bbtl(self):
        self.update_all_betas()
        bbt = list(map((lambda beta: beta @ beta.T), self.betas))
        return (1/self.number_of_images) * sum(bbt)

    def update_Gamma(self):
        self.Gamma = (1 / (N + AG)) * (N * self._bbtl() + AG * SIGMA_G)
        self.Gamma_Inv = np.linalg.inv(self.Gamma)

    def update_kBps(self):
        self.kBps = list(map((lambda beta: calculate_kBp(convert_to_1d(beta))), self.betas))

    def update_predictions(self):
        self.update_kBps()
        self.predictions = list(map((lambda kBp: kBp @ self.alphas), self.kBps))

    def ky_kk(self):
        self.update_all_betas()
        self.update_kBps()
        ky = list(map((lambda kBp, image: kBp.T @ image), self.kBps, self.images))
        kk = list(map((lambda kBp: kBp.T @ kBp), self.kBps))
        return 1/self.number_of_images * sum(ky), 1/self.number_of_images * sum(kk)

    def update_alpha_and_sd2(self):
        kyl, kkl = self.ky_kk()
        new_alpha = np.linalg.inv(N * kkl + self.sd2 * SIGMA_P_INV) @\
                    (N * kyl + self.sd2 * (SIGMA_P_INV @ MU_P))
        new_sd2 = (1 / (N * IMAGE_DIM * AP)) * (N * (self.YTY + self.alphas.T @ kkl @ self.alphas
                                                     - 2 * self.alphas.T @ kyl)
                                                + AP * SD_INIT)
        self.alphas = new_alpha
        self.sd2 = new_sd2
        self.update_predictions()

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


my_estimator = Estimator1DNImages()
my_estimator.run_estimation(10)
# plt.plot(my_estimator.predictions)
# plt.show()
# plt.plot(my_estimator.images)
# plt.show()