import numpy as np
import torch
import constants_2d_new as const
import functions_2d_new as func
import torch.nn.functional as tnf


# Convert from diag to coo and then to tensor
# Also learn how to clear gpu memory
# func.S_PIXEL_G_CENTERS_MATRIX.

padding, kernel = func.generate_gaussian_kernel(const.DEFORM_SD2)

#INTERMEDIATE FUNCTIONS
def make_beta_tensors_for_conv(betas):
    batch_size = betas.size()[0]
    empty_tensor = torch.zeros((batch_size,2, const.IMAGE_NROWS, const.IMAGE_NCOLS),
                               device=torch.device('cuda'))
    rows, cols = const.G_CENTERS.T
    empty_tensor[:, :, rows, cols] = torch.transpose(betas, 1, 2)
    return empty_tensor

def do_batch_convolution(beta_images, kernel):
    t_kernel = torch.from_numpy(kernel).cuda()
    kernel_row = t_kernel.size()[0]
    kernel_col = t_kernel.size()[1]
    big_kernel = t_kernel.expand(2, 1, kernel_row, kernel_col)
    out = tnf.conv2d(input=beta_images, weight=big_kernel, padding=padding, groups=2)
    return out

def conv_res_to_deformation(res, batch_size):
    flat_res = res.reshape((batch_size, 2, -1))
    deformation = torch.transpose(flat_res, 1, 2)
    return deformation


def get_deformation_from_conv(betas):
    batch_size = betas.size()[0]
    beta_img = make_beta_tensors_for_conv(betas)
    res = do_batch_convolution(beta_img, kernel=kernel)
    deformation = conv_res_to_deformation(res=res, batch_size=batch_size)
    return deformation


def get_pt_dense_K(sparse_scipy_matrix):
    dense_numpy = sparse_scipy_matrix.todense()
    torch_tensor = torch.from_numpy(dense_numpy).float().cuda()
    return torch_tensor

# torch_K = get_pt_sparse_K(func.S_PIXEL_G_CENTERS_MATRIX)
# torch_K_dense = get_pt_dense_K(func.S_PIXEL_G_CENTERS_MATRIX)

torch_C_a = torch.from_numpy(const.P_CENTERS).cuda()
torch_all_pixel = torch.from_numpy(const.ALL_PIXELS).cuda()
n_pixels = torch_all_pixel.size()[0]
n_centers = torch_C_a.size()[0]
ones_pixels = torch.ones((1, n_pixels), dtype=torch.float32,
                         device=torch.device('cuda'))
ones_centers = torch.ones((1, n_centers), dtype=torch.float32,
                          device=torch.device('cuda'))
one_col2 = torch.ones((2, 1), dtype=torch.float32,
                      device=torch.device('cuda'))
# torch_all_images = torch.tensor(const.FLAT_IMAGES, dtype=torch.float32,
#                                   device=torch.device('cuda'))

class CalcDeformation(torch.nn.Module):

    def forward(self,betas):
        batch_size = betas.size()[0]
        beta_img = make_beta_tensors_for_conv(betas)
        res = do_batch_convolution(beta_img, kernel=kernel)
        deformation = conv_res_to_deformation(res=res, batch_size=batch_size)
        return deformation


class KBpA(torch.nn.Module):

    def __init__(self, alphas, curr_betas, sdp2, all_pixels, all_p_centers):
        super().__init__()
        #beta shape (images, centers,2)
        self.betas = torch.nn.Parameter(curr_betas)
        self.alphas = alphas
        self.sdp2 = sdp2
        self.all_pixels = all_pixels
        self.all_p_centers = all_p_centers
        # self.calc_deformation = CalcDeformation().cuda()

    def forward(self):
        # deformation (images, pixels, 2)
        # batch_size = self.betas.size()[0]
        # beta_images_tensor = make_beta_tensors_for_conv(self.betas)
        # deformation_images = do_batch_convolution(beta_images_tensor, kernel)
        # deformation1 = conv_res_to_deformation(res=deformation_images,batch_size=batch_size)
        # deformation = (torch_K_dense @ self.betas)
        # deformation = self.calc_deformation(self.betas)
        deformation = get_deformation_from_conv(self.betas)
        deformed_pixel = self.all_pixels - deformation
        return torch.exp(
            - (torch.square(deformed_pixel) @ one_col2 @ ones_centers
               + (torch.square(self.all_p_centers) @ one_col2 @ ones_pixels).T
               - 2 * (deformed_pixel @ self.all_p_centers.T))
            / (2 * self.sdp2)) \
               @ self.alphas

    def string(self):
        return f'b = {self.betas.item()}'


class PyTorchOptimizer:
    def __init__(self, alphas, curr_beta, g_inv, sdp2, sdl2, images):
        self.alphas = torch.from_numpy(alphas).float().cuda()
        self.curr_betas = torch.tensor(curr_beta, dtype=torch.float32,
                                       device=torch.device('cuda'))
        # self.image = torch_all_images
        self.image = torch.tensor(images, dtype=torch.float32,
                                  device=torch.device('cuda'))
        self.g_inv = torch.from_numpy(g_inv).float().cuda()
        self.sdp2 = sdp2
        self.sdl2 = sdl2

    def optimize_betas(self, iter):
        image_predictor = KBpA(alphas=self.alphas,
                               curr_betas=self.curr_betas,
                               sdp2=self.sdp2,
                               all_pixels=torch_all_pixel,
                               all_p_centers=torch_C_a
                               ).cuda()
        criterion = torch.nn.MSELoss(reduction='sum').cuda()
        # optimizer = torch.optim.SGD(image_predictor.parameters(),
        #                             lr=0.001,
        #                             nesterov=True)

        optimizer = torch.optim.AdamW(image_predictor.parameters())
        # optimizer = torch.optim.RMSprop(image_predictor.parameters())
        # optimizer = torch.optim.LBFGS(params=image_predictor.parameters())
        for i in range(iter):
            pred = image_predictor()

            # Dont forget to multiply by the sd
            loss_right = (1 / (2 * self.sdl2)) * criterion(self.image, pred)

            loss_left = None
            for betas in image_predictor.parameters():
                beta_t = torch.transpose(betas, 1, 2)
                before_trace = beta_t @ self.g_inv @ betas
                diag = torch.diagonal(before_trace,offset=0,dim1=-2,dim2=-1)
                loss_left = (1 / 2) * torch.sum(diag)

            loss = loss_left + loss_right
            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            optimizer.step()

        for betas in image_predictor.parameters():
            to_numpy = betas.detach().cpu().numpy()
            list_of_numpy = list(to_numpy)
            torch.cuda.empty_cache()
            return list_of_numpy

