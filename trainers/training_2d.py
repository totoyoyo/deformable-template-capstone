import functions.functions_2d as func
import constants.constants_2d_0 as const
import numpy as np
from scipy import optimize
# My gradiants
import typing
import matplotlib.pyplot as plt

class Estimator2DNImages:

    def __init__(self):
        self.template = None
        self.number_of_images = len(const.FLAT_IMAGES)
        self.alphas: np.ndarray = const.ALPHAS_INIT
        self.betas: List[np.ndarray] = [const.BETAS_INIT] * self.number_of_images
        self.sd2: int = const.SD_INIT
        self.kBps: List[np.ndarray] = \
            list(map((lambda beta: func.calculate_kBp(beta)),
                     self.betas))
        self.Gamma: np.ndarray = const.SIGMA_G
        self.Gamma_Inv = const.SIGMA_G_INV
        self.images: List[np.ndarray] = const.FLAT_IMAGES
        yty = list(map((lambda image: func.faster_norm_squared(image)), self.images))
        self.YTY = (1 / self.number_of_images) \
                   * sum(yty)  # Many images
        self.predictions = None
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
            curr_beta = self.betas[n].flatten()
            to_min = func.generate_to_minimize(self.alphas,
                                               self.Gamma_Inv,
                                               self.sd2,
                                               self.images[n])

            out = optimize.minimize(to_min,
                                    curr_beta,
                                    method='SLSQP').x

            self.betas[n] = func.betas_to_2D(out)
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
        self.Gamma = (1 / (self.number_of_images + const.AG)) \
                     * (self.number_of_images * self._bbtl()
                        + const.AG * const.SIGMA_G)
        self.Gamma_Inv = np.linalg.inv(self.Gamma)
        print("Finished Gamma", self.Gamma_update_count, "time")
        self.Gamma_update_count += 1

    def update_kBps(self):  # Can be part of minimization?
        self.kBps = list(map((lambda beta: func.calculate_kBp(beta)),
                             self.betas))

    def update_predictions(self):
        self.predictions = list(map((lambda kBp: kBp @ self.alphas), self.kBps))

    def ky_kk(self):
        # self.update_all_betas()
        self.update_kBps()
        ky = list(map((lambda kBp, image: kBp.T @ image), self.kBps, self.images))
        kk = list(map((lambda kBp: kBp.T @ kBp), self.kBps))
        return (1 / self.number_of_images) * sum(ky), \
               (1 / self.number_of_images) * sum(kk)

    def update_alpha_and_sd2(self):
        print("Updating alpha", self.asd2_update_count, "time")
        kyl, kkl = self.ky_kk()
        new_alpha = np.linalg.inv(self.number_of_images * kkl
                                  + self.sd2 * const.SIGMA_P_INV) @ \
                    (self.number_of_images * kyl + self.sd2 * (const.SIGMA_P_INV @ const.MU_P))
        new_sd2 = (1 / (self.number_of_images * const.IMAGE_TOTAL * const.AP)) \
                  * (self.number_of_images * (self.YTY + self.alphas.T @ kkl @ self.alphas
                          - 2 * self.alphas.T @ kyl)
                     + const.AP * const.SD_INIT)
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
        self.template = func.calculate_template(self.alphas)

    def show_plots(self):
        path = "..\\plots\\2D\\"
        for n in range(self.number_of_images):
            image_to_show = func.unflatten_image(self.images[n])
            plt.imshow(image_to_show)
            image_name = "image" + str(n)
            plt.title(image_name)
            func.handle_save(path, image_name)
            plt.show()

        for n in range(self.number_of_images):
            prediction_to_show = func.unflatten_image(self.predictions[n])
            plt.imshow(prediction_to_show)
            image_name = "Prediction" + str(n)
            plt.title(image_name)
            func.handle_save(path, image_name)
            plt.show()

        template_to_show = func.unflatten_image(self.template)
        plt.imshow(template_to_show)
        image_name = "Template"
        plt.title(image_name)
        func.handle_save(path, image_name)
        plt.show()


my_estimator = Estimator2DNImages()
my_estimator.run_estimation(1)
my_estimator.show_plots()
