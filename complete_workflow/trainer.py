import numpy as np
import functions_maker as func
import constants_maker as const
import time
import numpy.linalg as nl
import pathlib
# My gradiants
import matplotlib.pyplot as plt
import pytorch_train_classify as pt_op
import time
import save

float_one = np.float32(1)
batch_size = 2


class Estimator2DNImages:

    def __init__(self, cons_obj=const.TrainingConstants(),
                 template_name='template0',
                 training_output_path=pathlib.Path()):
        self.template_name = template_name
        self.template = None
        self.cons_obj = cons_obj
        self.start_time = time.time()
        self.estimation_time = 0
        self.number_of_images = len(cons_obj.images)
        self.alphas: np.ndarray = cons_obj.alphas_init
        self.betas = [cons_obj.betas_init] * self.number_of_images
        self.sd2: int = cons_obj.init_sd
        # KBP ARE SPARSE
        self.kBps = None
        # self.Gamma: np.ndarray = const.SIGMA_G
        self.Gamma_Inv = cons_obj.SPARSE_SIGMA_G_INV
        self.images = cons_obj.flat_images
        yty = list(map((lambda image: func.faster_norm_squared(image)), self.images))
        self.YTY = (1 / self.number_of_images) \
                   * sum(yty)  # Many images
        self.predictions = None
        self.Gamma_update_count = 0
        self.asd2_update_count = 0
        self.epochs = cons_obj.epochs
        self.training_output_path = training_output_path


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
            print(f"beta at {start} to {end} (exclusive)")
            optimizer = pt_op.PyTorchOptimizer(alphas=self.alphas,
                                               curr_beta=curr_beta,
                                               g_inv=dense_gamma_inv,
                                               sdp2=self.cons_obj.template_sd2,
                                               sdl2=self.sd2,
                                               images=curr_images)
            out = optimizer.optimize_betas(self.epochs)
            self.betas[start:end] = out
            print("--- %s seconds ---" % (time.time() - start_time))
            # print(out)


    # Depends on current best betas
    def _bbtl(self):
        self.update_all_betas()
        bbt = list(map((lambda beta: beta @ beta.T), self.betas))
        out = (1 / self.number_of_images) * sum(bbt)
        return out

    def update_Gamma(self):
        print("Updating Gamma", self.Gamma_update_count, "time")
        coef = (self.number_of_images + self.cons_obj.AG)
        left = self.number_of_images * self._bbtl()
        print("Betas calculated, updating gamma")
        start_time = time.time()
        # FIX THIS
        right = self.cons_obj.AG * \
                func.invert_to_dense(self.cons_obj.SPARSE_SIGMA_G_INV)
        new_gamma = (left + right)/coef
        # self.Gamma = (1 / (self.number_of_images + const.AG)) \
        #              * (self.number_of_images * self._bbtl()
        #                 + const.AG * const.SIGMA_G)
        # tmp_inv = nl.pinv(new_gamma, hermitian=True)
        tmp_inv = nl.pinv(new_gamma, rcond=1e-6, hermitian=True)
        self.Gamma_Inv = func.to_sparse(tmp_inv)
        print("Finished Gamma", self.Gamma_update_count, "time")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.Gamma_update_count += 1

    def update_kBps(self):  # Can be part of minimization?
        self.kBps = list(map((lambda beta: self.cons_obj.calculate_kBp(beta)),
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
        p_inverse = self.cons_obj.SPARSE_SIGMA_P_INV.todense()
        for x in range(2):
            a_left_before_inv = self.number_of_images * kkl \
                                + self.sd2 * p_inverse
            # a_left_before_inv[abs(a_left_before_inv) < 1e-6] = 0.0
            a_left = nl.pinv(a_left_before_inv,rcond=1e-6,hermitian=True)
            # a_left = nl.pinv(a_left_before_inv,hermitian=True)
            a_right = (self.number_of_images * kyl + self.sd2 *
                       (p_inverse @ self.cons_obj.mup))
            new_alpha = a_left @ a_right
            new_sd2_coef = (self.number_of_images *
                            self.cons_obj.image_total + self.cons_obj.AP)
            new_sd2_bigterm_1 = self.alphas.T @ kkl @ self.alphas
            new_sd2_bigterm_2 = - 2 * self.alphas.T @ kyl
            new_sd2 = (self.number_of_images *
                       (self.YTY + new_sd2_bigterm_1
                        + new_sd2_bigterm_2)
                       + self.cons_obj.AP * self.cons_obj.init_sd) / new_sd2_coef
            self.alphas = new_alpha
            self.sd2 = new_sd2.item()
        # self.update_predictions()
        print("Finish updating alpha", self.asd2_update_count, "time")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.asd2_update_count += 1

    def save_data(self):
        path = self.training_output_path / self.template_name
        alphas = self.alphas
        save.handle_saving_npdata(parent_path=path,
                                  npdata=alphas,
                                  data_name="alphas",
                                  suffix=".data")
        g_inv = self.Gamma_Inv.todense()
        save.handle_saving_npdata(parent_path=path,
                                  npdata=g_inv,
                                  data_name="g_inv",
                                  suffix=".data")
        sd2 = self.sd2
        save.handle_saving_npdata(parent_path=path,
                                  npdata=[sd2],
                                  data_name="sd2",
                                  suffix=".data")
        for n in range(self.number_of_images):
            beta = self.betas[n]
            beta_name = "beta" + str(n)
            save.handle_saving_npdata(parent_path=path,
                                      npdata=beta,
                                      data_name=beta_name,
                                      suffix=".data")
        time_taken = self.estimation_time
        save.handle_saving_npdata(parent_path=path,
                                  npdata=[time_taken],
                                  data_name="time",
                                  suffix=".time")


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

    def calculate_template(self):
        tmp_template = self.cons_obj.calculate_template(self.alphas)
        tmp_template[tmp_template < 1e-5] = 0.0
        self.template = tmp_template

    def save_images(self):
        path = self.training_output_path / self.template_name
        path.mkdir(parents=True, exist_ok=True)
        for n in range(self.number_of_images):
            prediction_to_show = func.unflatten_image(
                self.predictions[n],
                self.cons_obj.image_ncol)
            image_name = "prediction" + str(n)
            save.handle_saving_plots(path,
                                     prediction_to_show,
                                     image_name)
            save.handle_saving_npdata(parent_path=path,
                                      npdata=prediction_to_show,
                                      data_name=image_name,
                                      suffix=".data")
        template_to_show = func.unflatten_image(self.template,
                                                self.cons_obj.image_ncol)
        image_name = "template"
        save.handle_saving_plots(path,
                                 template_to_show,
                                 image_name)
        save.handle_saving_npdata(parent_path=path,
                                  npdata=template_to_show,
                                  data_name=image_name,
                                  suffix=".data")

    def save_all(self):
        self.save_images()
        self.save_data()


# my_estimator = Estimator2DNImages()
# my_estimator.run_estimation(5)
# my_estimator.show_plots()
# my_estimator.save_data()