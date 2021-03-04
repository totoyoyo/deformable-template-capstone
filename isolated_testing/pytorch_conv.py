import numpy as np
import torch
import constants_2d_new as const
import functions_2d_new as func
import torch.nn.functional as tnf
# Convert from diag to coo and then to tensor
# Also learn how to clear gpu memory
# func.S_PIXEL_G_CENTERS_MATRIX.

def double_unsqueeze(tensor):
    u0 = torch.unsqueeze(tensor, 0)
    u1 = torch.unsqueeze(u0, 0)
    return u1

padding, kernel = func.generate_gaussian_kernel(const.DEFORM_SD2)

unsqueezed_kernel = double_unsqueeze(torch.from_numpy(kernel)).cuda()

def get_deformation(betas):
    dim0 = betas[:, 0]
    shape = dim0.size()
    usual_shape = shape
    beta0_img = torch.reshape(dim0,(const.IMAGE_NROWS,const.IMAGE_NCOLS))
    unsqueezed_beta0 = double_unsqueeze(beta0_img)
    out0_img = tnf.conv2d(unsqueezed_beta0, unsqueezed_kernel, padding=padding)
    deformation0 = torch.reshape(out0_img,(usual_shape[0],1))

    dim1 = betas[:, 1]
    shape = dim1.size()
    beta1_img = torch.reshape(dim1, (const.IMAGE_NROWS, const.IMAGE_NCOLS))
    unsqueezed_beta1 = double_unsqueeze(beta1_img)
    out1_img = tnf.conv2d(unsqueezed_beta1, unsqueezed_kernel, padding=padding)
    deformation1 = torch.reshape(out1_img, (usual_shape[0],1))
    # for_cat = torch.unsqueeze(deformation0, 1)
    both_deformation = torch.cat((deformation0, deformation1),
                                 dim=1)
    return both_deformation


torch_C_a = torch.from_numpy(const.P_CENTERS).cuda()
torch_all_pixel = torch.from_numpy(const.ALL_PIXELS).cuda()
n_pixels = torch_all_pixel.size()[0]
n_centers = torch_C_a.size()[0]
ones_pixels = torch.ones((1, n_pixels), dtype=torch.float32,
                         device=torch.device('cuda'))
ones_centers = torch.ones((1, n_centers),dtype=torch.float32,
                          device=torch.device('cuda'))
one_col2 = torch.ones((2, 1),dtype=torch.float32,
                          device=torch.device('cuda'))


class KBpA(torch.nn.Module):

    def __init__(self, alphas, curr_betas, sdp2, all_pixels, all_p_centers):
        super().__init__()
        self.betas = torch.nn.Parameter(curr_betas)
        self.alphas = alphas
        self.sdp2 = sdp2
        self.all_pixels = all_pixels
        self.all_p_centers = all_p_centers

    def forward(self):
        deformation = get_deformation(self.betas)
        deformed_pixel = self.all_pixels - deformation
        p_norm_squared = torch.square(deformed_pixel) @ one_col2
        c_norm_squared = torch.square(self.all_p_centers) @ one_col2
        p_norm_squared_repeated = p_norm_squared @ ones_centers
        c_norm_squared_repeated = (c_norm_squared @ ones_pixels).T
        p_dot_c = 2 * (deformed_pixel @ self.all_p_centers.T)
        big_matrix = p_norm_squared_repeated + c_norm_squared_repeated - p_dot_c
        kBp = torch.exp(- big_matrix / (2 * self.sdp2))
        out = kBp @ self.alphas
        return out

    # def forward(self):
    #     deformation = get_deformation(self.betas)
    #     deformed_pixel = self.all_pixels - deformation
    #     rep_pixels = deformed_pixel.repeat_interleave(repeats=n_centers,
    #                                                   dim=0)
    #     tiled_centers = self.all_p_centers.repeat(n_pixels, 1)
    #     diff_vector = rep_pixels - tiled_centers
    #     diff_squared = torch.square(diff_vector)
    #     summed_square = torch.sum(diff_squared, dim=1)
    #     gaussian_out = torch.exp((-1/(2 * self.sdp2)) * summed_square)
    #     reshaped_gauss = torch.reshape(gaussian_out, (n_pixels,n_pixels))
    #     out = reshaped_gauss @ self.alphas
    #     return out

    def string(self):
        return f'b = {self.betas.item()}'


class PyTorchOptimizer():
    def __init__(self, alphas, image, curr_beta, g_inv, sdp2, sdl2):
        self.alphas = torch.from_numpy(alphas).float().cuda()
        self.curr_betas = torch.from_numpy(curr_beta).float().cuda()
        self.image = torch.from_numpy(image.reshape(-1, 1)).float().cuda()
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
        optimizer = torch.optim.Adam(image_predictor.parameters())
        # optimizer = torch.optim.RMSprop(image_predictor.parameters())
        # optimizer = torch.optim.LBFGS(params=image_predictor.parameters())
        for i in range(iter):
            pred = image_predictor()

            # Dont forget to multiply by the sd
            loss_right = (1/(2 * self.sdl2)) * criterion(self.image, pred)

            loss_left = None
            for betas in image_predictor.parameters():
                loss_left = (1 / 2) * torch.trace(betas.T @
                                                  self.g_inv @
                                                  betas)

            loss = loss_left + loss_right
            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            optimizer.step()

        for betas in image_predictor.parameters():
            to_numpy = betas.detach().cpu().numpy()
            return to_numpy