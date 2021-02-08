import datetime
import os
import matplotlib.pyplot as plt

from constants.constants_1d_many_fix import *
from scipy.optimize import check_grad
import functools


def cal_deformation(x, b1d):
    counter = 0.0
    for index, center in enumerate(G_CENTERS):
        counter += b1d[index] * gaussian_kernel(x, center, DEFORM_SD)
    return counter


def convert_to_1d(arr):
    if arr.ndim > 1:
        return arr.T[0]
    else:
        return arr


def faster_norm_squared(arr):
    return np.sum(arr * arr)


def convert_to_2d(mat):
    if mat.ndim < 2:
        return np.array([mat]).T
    else:
        return mat


def calculate_kBp(b1d):
    tmp_kbp = np.empty((IMAGE_DIM, KP))
    for i in range(IMAGE_DIM):
        for j in range(KP):
            the_def = cal_deformation(i, b1d)
            tmp_kbp[i, j] = gaussian_kernel(i - the_def,
                                            P_CENTERS[j],
                                            TEMPLATE_SD)
    return tmp_kbp


def calculate_template(alphas):
    a1d = convert_to_1d(alphas)
    template = np.empty((IMAGE_DIM, 1))
    for i in range(IMAGE_DIM):
        template[i, 0] = sum([alpha * gaussian_kernel(i, P_CENTERS[index], TEMPLATE_SD)
                              for index, alpha in enumerate(a1d)])
    return template


# Returns nparray dim (IMAGEDIM,1)
def kBpa(beta, alpha):
    b1d = convert_to_1d(beta)
    # Should be IMAGE_DIM by ALPHA_DIM
    kBp = calculate_kBp(b1d)
    return convert_to_2d(kBp @ alpha)


def grad_for_optimization(beta_1d, alpha, g_inv, sdl2, image):
    # Gradient of left (gives vector)
    alpha1d = convert_to_1d(alpha)

    def grad_left(beta):
        return g_inv @ beta

    # Gradient of right (gives vector)
    def grad_right(beta):
        # Should be matrix with dim (IMAGE_DIM,1)
        my_kBpa = kBpa(beta, alpha1d)
        # Dim (IMAGEDIM,1)
        grad_wrt_kBpa = (-(1 / (sdl2)) * (image - my_kBpa))
        # grad_kBpa should be (BETA_DIM,IMAGE_DIM)
        return convert_to_1d(grad_kBpa(beta, alpha1d) @ grad_wrt_kBpa)

    # Gradient of gaussian (gives scalar)
    def grad_gaussian(x, center, sd):
        # Should be scalar
        return gaussian_kernel(x, center, sd) * - (x - center) / (sd ** 2)

    # Gradient of zbx wrt to a beta
    def grad_zbx_wrt_a_beta(image_index, b_index):
        # Should be scalar
        return gaussian_kernel(image_index, G_CENTERS[b_index], DEFORM_SD)

    def grad_kBpa_wrt_nth_beta(beta, alphas, beta_index):
        # Should be array of IMAGE_DIM length
        store_grad = np.empty(IMAGE_DIM)
        for image_index in range(IMAGE_DIM):
            counter = 0.0
            for alpha_index in range(alphas.size):
                # Should be scalar
                counter += alphas[alpha_index] \
                           * (grad_gaussian((image_index - cal_deformation(image_index, beta)),
                                            P_CENTERS[alpha_index],
                                            TEMPLATE_SD)) \
                           * (- (grad_zbx_wrt_a_beta(image_index, beta_index)))
            store_grad[image_index] = counter
        return store_grad

    def grad_kBpa(beta, alpha):
        store_grad = np.empty([beta.size, IMAGE_DIM])
        # Should be betas by image_dim matrix
        # Careful, maybe we need to tke transpose
        for beta_index in range(beta.size):
            store_grad[beta_index] = grad_kBpa_wrt_nth_beta(beta, alpha, beta_index)
        return store_grad

    return grad_left(beta_1d) + grad_right(beta_1d)


def generate_jacobian_callable(alpha, g_inv, sd2, image):
    def jac(beta_1d):
        return grad_for_optimization(beta_1d, alpha, g_inv, sd2, image)

    return jac


def to_minimize(b1d, alpha, g_inv, sd2, image):  # Fix the static predictions
    image_difference = image - kBpa(b1d, alpha)
    result = (1 / 2) * b1d.T @ g_inv @ b1d \
             + (1 / (2 * sd2)) \
             * faster_norm_squared(image_difference)
    return result


def generate_to_minimize(alpha, g_inv, sd2, image):
    def tmp_min(beta_1d):
        return to_minimize(beta_1d, alpha, g_inv, sd2, image)

    return tmp_min


def generate_tomin_jac(alpha, g_inv, sd2, image):
    return generate_to_minimize(alpha, g_inv, sd2, image), \
           generate_jacobian_callable(alpha, g_inv, sd2, image)


# to_min, jac = generate_tomin_jac(ALPHAS_INIT,
#                       SIGMA_G_INV,
#                       1,
#                       IMAGE1)


def test_gradient(iter):
    error_counter = 0.0
    beta_list = []
    alpha_list = []
    sig_inv_list = []
    sd_list = []
    for i in range(iter):
        random_beta = (np.random.rand(KG) * 2) - 1
        beta_list.append(random_beta)

        random_alpha = (np.random.rand(KP) * 2) - 1
        alpha_list.append(random_alpha)

        sigma_g_inv_maker = np.random.rand(KG, KG)
        random_sigma_g_inv = sigma_g_inv_maker @ sigma_g_inv_maker.T
        sig_inv_list.append(random_sigma_g_inv)

        random_sd = np.random.uniform(0, 1)
        sd_list.append(random_sd)

        to_min, jac = generate_tomin_jac(random_alpha, random_sigma_g_inv, random_sd,
                                         IMAGE1)
        tmp_error = check_grad(to_min, jac, random_beta)
        error_counter += tmp_error
    return beta_list, alpha_list, sig_inv_list, sd_list, error_counter


def handle_save(path, image_name):
    date_str = str(datetime.date.today())
    to_save = path + image_name
    save_counter = 0
    while (os.path.isfile(to_save
                          + "_"
                          + date_str
                          + "_"
                          + str(save_counter)
                          + ".jpg")):
        save_counter += 1
    plt.savefig(to_save
                + "_"
                + date_str
                + "_"
                + str(save_counter)
                + ".jpg")

# print(calculate_template(np.linspace(0,1,KP)))
