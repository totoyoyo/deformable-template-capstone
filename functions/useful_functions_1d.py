from constants.constants_1d_many_images import *

def cal_deformation(x, b1d):
    counter = 0
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


def kBpa(beta, alpha):
    b1d = convert_to_1d(beta)
    kBp = calculate_kBp(b1d)
    return kBp @ alpha


def grad_for_optimization(beta_1d, alpha, g_inv, sdl, image):
    # Gradient of left (gives vector)
    def grad_left(beta):
        return g_inv @ beta

    # Gradient of right (gives vector)
    def grad_right(beta):
        # Should be vector with length IMAGE_DIM
        grad_wrt_kBpa = (-(1 / (sdl ** 2)) * (image - kBpa(beta, alpha)))
        return -(1 / (sdl ** 2)) * grad_kBpa(beta, alpha) @ grad_wrt_kBpa

    # Gradient of gaussian (gives scalar)
    def grad_gaussian(x, center, sd):
        # Should be scalar
        return gaussian_kernel(x, center, sd) * - (x - center) / (sd ** 2)

    # Gradient of zbx wrt to a beta
    def grad_zbx_wrt_a_beta(image_index, b_index):
        # Should be scalar
        return gaussian_kernel(image_index, G_CENTERS[b_index], DEFORM_SD)

    def grad_kBpa_wrt_nth_beta(beta, alphas, beta_index):
        # Should be image_dim matrix
        store_grad = np.empty(IMAGE_DIM)
        for image_index in range(IMAGE_DIM):
            counter = 0
            for alpha_index in range(alphas):
                # Should be scalar
                counter += alphas[alpha_index] \
                           * (grad_gaussian((image_index - cal_deformation(image_index, beta)),
                                            P_CENTERS[alpha_index],
                                            TEMPLATE_SD)) \
                           * (- (grad_zbx_wrt_a_beta(image_index, beta_index)))
            store_grad[image_index] = counter
        return store_grad

    def grad_kBpa(beta, alpha):
        store_grad = np.empty(beta.size)

        # Should be betas by image_dim matrix
        # Careful, maybe we need to tke transpose
        for beta_index in range(beta.size):
            store_grad[beta_index] = grad_kBpa_wrt_nth_beta(beta, alpha, beta_index)
        return store_grad

    return grad_left(beta_1d) + grad_right(beta_1d)