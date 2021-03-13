import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['GOTO_NUM_THREADS'] = '10'
os.environ['OMP_NUM_THREADS'] = '10'

import numpy as np
import functions_2d_new as func
import constants_2d_new as const
import time
import scipy.linalg as sl
import numpy.linalg as nl

# My gradiants
import matplotlib.pyplot as plt
import pytorch_batch_conv as pt_op
import time

float_one = np.float32(1)
batch_size = 100


class Estimator2DNImages:

    def __init__(self):
        self.template = None
        self.start_time = time.time()
        self.estimation_time = 0
        self.number_of_images = len(const.FLAT_IMAGES)
        self.alphas: np.ndarray = const.ALPHAS_INIT
        self.betas = [const.BETAS_INIT] * self.number_of_images
        self.sd2: int = const.SD_INIT
        # KBP ARE SPARSE
        self.kBps = None
        # self.Gamma: np.ndarray = const.SIGMA_G
        self.Gamma_Inv = const.SPARSE_SIGMA_G_INV
        self.images = const.FLAT_IMAGES
        yty = list(map((lambda image: func.faster_norm_squared(image)), self.images))
        self.YTY = (1 / self.number_of_images) \
                   * sum(yty)  # Many images
        self.predictions = None
        self.Gamma_update_count = 0
        self.asd2_update_count = 0


    def update_all_betas(self):
        # Depends on current beta, Gamma, sd2, predictions, images
        dense_gamma_inv = self.Gamma_Inv.todense()
        list_of_start_end_indexes = func.get_list_of_indexes_for_slicing(batch_size,
                                                                    self.number_of_images)
        for start_end in list_of_start_end_indexes:
            start = start_end[0]
            end = start_end[1]
            curr_beta = self.betas[start:end]
            curr_images = self.images[start:end]
            start_time = time.time()
            optimizer = pt_op.PyTorchOptimizer(alphas=self.alphas,
                                               curr_beta=curr_beta,
                                               g_inv=dense_gamma_inv,
                                               sdp2=const.TEMPLATE_SD2,
                                               sdl2=self.sd2,
                                               images=curr_images)
            out = optimizer.optimize_betas(1000)
            self.betas[start:end] = out
            print("--- %s seconds ---" % (time.time() - start_time))
            print(f"beta at {start} to {end} (exclusive)")
            # print(out)


    # Depends on current best betas
    def _bbtl(self):
        self.update_all_betas()
        bbt = list(map((lambda beta: beta @ beta.T), self.betas))
        out = (1 / self.number_of_images) * sum(bbt)
        return out

    def update_Gamma(self):
        print("Updating Gamma", self.Gamma_update_count, "time")
        coef = (self.number_of_images + const.AG)
        left = self.number_of_images * self._bbtl()
        # FIX THIS
        right = const.AG * const.invert_to_dense(const.SPARSE_SIGMA_G_INV)
        new_gamma = (left + right)/coef
        # self.Gamma = (1 / (self.number_of_images + const.AG)) \
        #              * (self.number_of_images * self._bbtl()
        #                 + const.AG * const.SIGMA_G)
        # tmp_inv = nl.pinv(new_gamma, hermitian=True)
        tmp_inv = nl.pinv(new_gamma, rcond=1e-6, hermitian=True)
        self.Gamma_Inv = const.to_sparse(tmp_inv)
        print("Finished Gamma", self.Gamma_update_count, "time")
        self.Gamma_update_count += 1

    def update_kBps(self):  # Can be part of minimization?
        self.kBps = list(map((lambda beta: func.calculate_kBp(beta)),
                             self.betas))

    def update_predictions(self):
        self.predictions = list(map((lambda kBp: kBp.dot(self.alphas)), self.kBps))

    def ky_kk(self):
        # self.update_all_betas()
        self.update_kBps()
        ky = list(map((lambda kBp, image: kBp.transpose().dot(image)), self.kBps, self.images))
        kk = list(map((lambda kBp: kBp.transpose().dot(kBp)), self.kBps))
        kyl = sum(ky) / self.number_of_images
        kyl_reshaped = kyl.reshape(-1,1)
        kk_out = sum(kk) / self.number_of_images
        # kk_tmp = list(map((lambda kBp: kBp.transpose().dot(kBp).toarray()), self.kBps))
        # kk_out2 = ((1 / self.number_of_images) * sum(kk_tmp)).astype('float32')
        return kyl_reshaped, \
               kk_out

    def update_alpha_and_sd2(self):
        print("Updating alpha", self.asd2_update_count, "time")
        start_time = time.time()
        kyl, kkl = self.ky_kk()
        p_inverse = const.SPARSE_SIGMA_P_INV.todense()
        for x in range(2):
            a_left_before_inv = self.number_of_images * kkl \
                                + self.sd2 * p_inverse
            # a_left_before_inv[abs(a_left_before_inv) < 1e-6] = 0.0
            a_left = nl.pinv(a_left_before_inv,rcond=1e-6,hermitian=True)
            # a_left = nl.pinv(a_left_before_inv,hermitian=True)
            a_right = (self.number_of_images * kyl + self.sd2 * (p_inverse @ const.MU_P))
            new_alpha = a_left @ a_right
            new_sd2_coef = (self.number_of_images * const.IMAGE_TOTAL + const.AP)
            new_sd2_bigterm_1 = self.alphas.T @ kkl @ self.alphas
            new_sd2_bigterm_2 = - 2 * self.alphas.T @ kyl
            new_sd2 = (self.number_of_images *
                       (self.YTY + new_sd2_bigterm_1
                        + new_sd2_bigterm_2)
                       + const.AP * const.SD_INIT)/new_sd2_coef
            self.alphas = new_alpha
            self.sd2 = new_sd2.item()
        # self.update_predictions()
        print("Finish updating alpha", self.asd2_update_count, "time")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.asd2_update_count += 1

    def save_data(self):
        path = "../outputs/"
        func.handle_save_arr(path, "alpha", self.alphas)
        # func.handle_save_arr(path, "beta", self.betas)
        # func.handle_save_arr(path, "Gamma", self.Gamma)
        func.handle_save_arr(path, "sigma_squared", [self.sd2])
        func.handle_save_arr(path, "time", [self.estimation_time])


    def run_estimation(self, iterations):
        self.update_alpha_and_sd2()
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
        tmp_template = func.calculate_template(self.alphas)
        tmp_template[tmp_template < 1e-5] = 0.0
        self.template = tmp_template

    def show_plots(self):
        path = "..\\plots\\2D\\"
        # for n in range(self.number_of_images):
        #     image_to_show = func.unflatten_image(self.images[n])
        #     plt.imshow(image_to_show)
        #     image_name = "image" + str(n)
        #     plt.title(image_name)
        #     func.handle_save_plot(path, image_name)
        #     plt.show()

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
print("start training")
my_estimator = Estimator2DNImages()
my_estimator.run_estimation(10)
my_estimator.show_plots()
# my_estimator.save_data()