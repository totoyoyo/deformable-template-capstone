import functions.functions_2d_fix as func
import constants.constants_2d_0 as const
import numpy as np
import time
from scipy import optimize
# My gradiants
import matplotlib.pyplot as plt
from helpers.solver_fix import solve

class Estimator2DNImages:

    def __init__(self):
        self.template = None
        self.start_time = time.time()
        self.estimation_time = 0
        self.number_of_images = len(const.FLAT_IMAGES)
        self.alphas: np.ndarray = const.ALPHAS_INIT
        self.betas = [const.BETAS_INIT] * self.number_of_images
        self.sd2: int = const.SD_INIT
        self.kBps = \
            list(map((lambda beta: func.calculate_kBp(beta)),
                     self.betas))
        self.Gamma: np.ndarray = const.SIGMA_G
        self.Gamma_Inv = const.SIGMA_G_INV
        self.images: typing.List[np.ndarray] = const.FLAT_IMAGES
        yty = list(map((lambda image: func.faster_norm_squared(image)), self.images))
        self.YTY = (1 / self.number_of_images) \
                   * sum(yty)  # Many images
        self.predictions = None
        self.Gamma_update_count = 0
        self.asd2_update_count = 0


    def update_all_betas(self):
        # Depends on current beta, Gamma, sd2, predictions, images
        def update_best_beta(n):
            curr_beta = self.betas[n].flatten()
            tmp_a = self.alphas.flatten()
            out = solve(self.Gamma_Inv,
                        const.ALL_PIXELS,
                        func.PIXEL_G_CENTERS_MATRIX,
                        const.P_CENTERS,
                        self.images[n],
                        const.ONE_COL2,
                        const.ONE_KP,
                        const.ONE_L,
                        tmp_a,
                        self.sd2,
                        const.TEMPLATE_SD2
                        )
            self.betas[n] = out['B']
            print("beta at" + str(n))
            print(out)

        for n in range(self.number_of_images):
            update_best_beta(n)

    # Depends on current best betas
    def _bbtl(self):
        self.update_all_betas()
        bbt = list(map((lambda beta: beta @ beta.T), self.betas))
        out = (1 / self.number_of_images) * sum(bbt)
        return out

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
        kyl = (1 / self.number_of_images) * sum(ky)
        return kyl.reshape(-1,1), \
               (1 / self.number_of_images) * sum(kk)

    def update_alpha_and_sd2(self):
        print("Updating alpha", self.asd2_update_count, "time")
        kyl, kkl = self.ky_kk()
        for x in range(5):
            a_left = np.linalg.inv(self.number_of_images * kkl
                                      + self.sd2 * const.SIGMA_P_INV)
            a_right = (self.number_of_images * kyl + self.sd2 * (const.SIGMA_P_INV @ const.MU_P))
            new_alpha = a_left @ a_right
            new_sd2 = (1 / (self.number_of_images * const.IMAGE_TOTAL * const.AP)) \
                      * (self.number_of_images * (self.YTY + self.alphas.T @ kkl @ self.alphas
                              - 2 * self.alphas.T @ kyl)
                         + const.AP * const.SD_INIT)
            self.alphas = new_alpha
            self.sd2 = new_sd2.item()
        # self.update_predictions()
        print("Finish updating alpha", self.asd2_update_count, "time")
        self.asd2_update_count += 1

    def save_data(self):
        path = "../outputs/"
        func.handle_save_arr(path, "alpha", self.alphas)
        # func.handle_save_arr(path, "beta", self.betas)
        func.handle_save_arr(path, "Gamma", self.Gamma)
        func.handle_save_arr(path, "sigma_squared", [self.sd2])
        func.handle_save_arr(path, "time", [self.estimation_time])


    def run_estimation(self, iterations):
        for _ in range(iterations):
            self.update_Gamma()
            self.update_alpha_and_sd2()
        self.calculate_template()
        self.update_predictions()
        end_time = time.time()
        total = end_time - self.start_time
        self.estimation_time = total

        # print("Here is the time")
        # print(total)
        # print("here are alphas, sd2, and Gamma in that order")
        # print("here are sd2")
        # print(self.alphas)
        # print(self.sd2)
        # print(self.Gamma)
        # print("here are betas")
        # print(self.betas)
        # print("here are predictions")
        # print(self.predictions)
        # print("here is the template")

    def calculate_template(self):
        self.template = func.calculate_template(self.alphas)

    def show_plots(self):
        path = "..\\plots\\2D\\"
        for n in range(self.number_of_images):
            image_to_show = func.unflatten_image(self.images[n])
            plt.imshow(image_to_show)
            image_name = "image" + str(n)
            plt.title(image_name)
            func.handle_save_plot(path, image_name)
            plt.show()

        for n in range(self.number_of_images):
            prediction_to_show = func.unflatten_image(self.predictions[n])
            plt.imshow(prediction_to_show)
            image_name = "Prediction" + str(n)
            plt.title(image_name)
            func.handle_save_plot(path, image_name)
            plt.show()

        template_to_show = func.unflatten_image(self.template)
        plt.imshow(template_to_show)
        image_name = "Template"
        plt.title(image_name)
        func.handle_save_plot(path, image_name)
        plt.show()
#
#
my_estimator = Estimator2DNImages()
my_estimator.run_estimation(10)
my_estimator.show_plots()
my_estimator.save_data()