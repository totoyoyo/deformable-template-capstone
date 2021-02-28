import numpy as np
from functions.functions_2d_fix import calculate_kBp_and_deformation, \
    kBpa

from scipy.optimize import check_grad


class BetasOptimizer():
    def __init__(self, alphas, image, g_inv, sdp2, sdl2, K, C_a):
        self.alphas = alphas
        self.KTonsd2 = K.T * (1/(sdl2 + sdp2))
        self.diag_a_Ca = alphas * C_a
        self.image = image.reshape(-1,1)
        self.g_inv = g_inv
        self.sdl2 = sdl2

    def gradient(self,betas):
        kBp, deformed = calculate_kBp_and_deformation(betas)
        prediction = kBp @ self.alphas
        diff_from_image = self.image - prediction
        diff_from_image_times_pred = diff_from_image * prediction

        # right_right = np.diag(diff_from_image_times_pred.ravel()) @ deformation
        right_right = diff_from_image_times_pred * deformed

        # right_left = np.diag(diff_from_image.ravel()) @ kBp @ self.diag_a_Ca
        right_left = diff_from_image * (kBp @ self.diag_a_Ca)
        right_before_coefficient = right_left - right_right

        right = self.KTonsd2 @ right_before_coefficient
        right2 = self.KTonsd2 @ right_left - self.KTonsd2 @ right_right
        left = self.g_inv @ betas
        out = left + right

        return out

    def loss(self,betas):
        pred = kBpa(betas, self.alphas)
        assert pred.ndim == 2
        image_difference = self.image - pred
        left = (1 / 2) * np.trace(betas.T @ self.g_inv @ betas)
        right = (1 / (2 * self.sdl2)) * np.sum((image_difference) ** 2)
        out = left + right
        return out

    def gradient_flat_betas(self,flat_betas):
        betas = flat_betas.reshape(-1, 2)
        grad = self.gradient(betas)
        flat_grad = grad.ravel()
        return flat_grad

    def loss_flat_betas(self,flat_betas):
        betas = flat_betas.reshape(-1, 2)
        loss = self.loss(betas)
        return loss



def test_gradient(iter, KG, KP, image, K, C_a):
    error_counter = 0.0
    beta_list = []
    alpha_list = []
    sig_inv_list = []
    sd_list = []
    sdp_list = []
    for i in range(iter):
        random_beta = (np.random.rand(KG, 2) * 2) - 1
        beta_list.append(random_beta)

        random_alpha = (np.random.rand(KP, 1) * 2) - 1
        alpha_list.append(random_alpha)

        sigma_g_inv_maker = np.random.rand(KG, KG)
        random_sigma_g_inv = sigma_g_inv_maker @ sigma_g_inv_maker.T
        sig_inv_list.append(random_sigma_g_inv)

        random_sd = np.random.uniform(0, 3)
        sd_list.append(random_sd)

        random_sdp = np.random.uniform(0, 3)
        sdp_list.append(random_sdp)

        optimizer = BetasOptimizer(random_alpha, image,
                                   random_sigma_g_inv,
                                   random_sdp, random_sd,
                                   K, C_a)
        flat_beta = random_beta.ravel()

        tmp_error = check_grad(optimizer.loss_flat_betas,
                               optimizer.gradient_flat_betas,
                               flat_beta)
        error_counter += tmp_error
    return {'error': error_counter, 'betas': beta_list,
            'alphas': alpha_list, 'Sigma_inv': sig_inv_list,
            'sd2': sd_list}
