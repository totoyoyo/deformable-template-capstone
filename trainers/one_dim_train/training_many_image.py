from scipy import optimize

from typing import *

# Useful Functions
from functions.one_dim_func.useful_functions_1d import *
from constants.one_dim.constants_1d_many_fix import *


# My gradiants


class Estimator1DNImages:

    def __init__(self):
        self.template = None
        self.alphas: np.ndarray = ALPHAS_INIT
        self.betas: List[np.ndarray] = [BETAS_INIT, BETAS_INIT]
        self.sd2: int = SD_INIT
        self.kBps: List[np.ndarray] = \
            list(map((lambda beta: calculate_kBp(convert_to_1d(beta))), self.betas))
        self.Gamma: np.ndarray = SIGMA_G
        self.Gamma_Inv = SIGMA_G_INV
        self.images: List[np.ndarray] = IMAGES
        self.number_of_images = N
        yty = list(map((lambda image: faster_norm_squared(image)), self.images))
        self.YTY = (1 / self.number_of_images) \
                   * sum(yty)  # Many images
        self.predictions = [PREDICT_INIT, PREDICT_INIT]
        self.Gamma_update_count = 0
        self.asd2_update_count = 0

    # def to_minimize(self, b1d, n):  # Fix the static predictions
    #     image_difference = self.images[n] - self.calculate_prediction(b1d)
    #     result = (1 / 2) * b1d.T @ self.Gamma_Inv @ b1d \
    #              + (1 / (2 * self.sd2)) \
    #              * faster_norm_squared(image_difference)
    #     return result.item()

    def update_all_betas(self):
        # Depends on current beta, Gamma, sd2, predictions, images
        def update_best_beta(n):
            betas_in_1d = convert_to_1d(self.betas[n])
            to_min, jac = generate_tomin_jac(self.alphas,
                                             self.Gamma_Inv,
                                             self.sd2,
                                             self.images[n])
            out = optimize.minimize(to_min,
                                    betas_in_1d,
                                    method='SLSQP',
                                    jac=jac).x
            self.betas[n] = convert_to_2d(out)
            print("beta at" + str(n))
            print(out)

        for n in range(self.number_of_images):
            update_best_beta(n)

    # Depends on current best betas
    def _bbtl(self):
        self.update_all_betas()
        bbt = list(map((lambda beta: beta @ beta.T), self.betas))
        return (1 / self.number_of_images) * sum(bbt)

    def update_Gamma(self):
        print("Updating Gamma", self.Gamma_update_count, "time")
        self.Gamma = (1 / (N + AG)) * (N * self._bbtl() + AG * SIGMA_G)
        self.Gamma_Inv = np.linalg.inv(self.Gamma)
        print("Finished Gamma", self.Gamma_update_count, "time")
        self.Gamma_update_count += 1

    def update_kBps(self):  # Can be part of minimization?
        self.kBps = list(map((lambda beta: calculate_kBp(convert_to_1d(beta))), self.betas))

    def update_predictions(self):
        self.predictions = list(map((lambda kBp: kBp @ self.alphas), self.kBps))

    def calculate_prediction(self, b1d):
        return calculate_kBp(b1d) @ self.alphas

    def ky_kk(self):
        # self.update_all_betas()
        self.update_kBps()
        ky = list(map((lambda kBp, image: kBp.T @ image), self.kBps, self.images))
        kk = list(map((lambda kBp: kBp.T @ kBp), self.kBps))
        return (1 / self.number_of_images) * sum(ky), (1 / self.number_of_images) * sum(kk)

    def update_alpha_and_sd2(self):
        print("Updating alpha", self.asd2_update_count, "time")
        kyl, kkl = self.ky_kk()
        new_alpha = np.linalg.inv(N * kkl + self.sd2 * SIGMA_P_INV) @ \
                    (N * kyl + self.sd2 * (SIGMA_P_INV @ MU_P))
        new_sd2 = (1 / (N * IMAGE_DIM * AP)) \
                  * (N * (self.YTY + self.alphas.T @ kkl @ self.alphas
                          - 2 * self.alphas.T @ kyl)
                     + AP * SD_INIT)
        self.alphas = new_alpha
        self.sd2 = new_sd2.item()
        self.update_predictions()
        print("Finish updating alpha", self.asd2_update_count, "time")
        self.asd2_update_count += 1

    def run_estimation(self, iterations):
        for _ in range(iterations):
            self.update_Gamma()
            self.update_alpha_and_sd2()
        self.calculate_template()
        print("here are alphas, sd2, and Gamma in that order")
        print(self.alphas)
        print(self.sd2)
        print(self.Gamma)
        print("here are betas")
        print(self.betas)
        print("here are predictions")
        print(self.predictions)
        print("here is the template")

    def calculate_template(self):
        self.template = calculate_template(self.alphas)

    def show_plots(self):
        path = "../../plots\\"
        for n in range(self.number_of_images):
            plt.plot(self.images[n])
            image_name = "image" + str(n)
            plt.title(image_name)
            handle_save(path, image_name)
            plt.show()

        for n in range(self.number_of_images):
            plt.plot(self.predictions[n])
            image_name = "Prediction" + str(n)
            plt.title(image_name)
            handle_save(path, image_name)
            plt.show()

        plt.plot(self.template)
        image_name = "Template"
        plt.title(image_name)
        handle_save(path, image_name)
        plt.show()



my_estimator = Estimator1DNImages()
my_estimator.run_estimation(5)
my_estimator.show_plots()
