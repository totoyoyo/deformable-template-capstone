import numpy as np
import torch
import constants.constants_2d_0 as const
import functions.functions_2d_fix as func

torch_K = torch.from_numpy(func.PIXEL_G_CENTERS_MATRIX).type(torch.FloatTensor)
torch_C_a = torch.from_numpy(const.P_CENTERS).type(torch.FloatTensor)
torch_all_pixel = torch.from_numpy(const.ALL_PIXELS).type(torch.FloatTensor)


class KBpA(torch.nn.Module):

    def __init__(self, alphas, curr_betas, sdp2, K, all_pixels, all_p_centers):
        super().__init__()
        self.betas = torch.nn.Parameter(curr_betas)
        self.alphas = alphas
        self.sdp2 = sdp2
        self.K = K
        self.all_pixels = all_pixels
        self.all_p_centers = all_p_centers

    def forward(self):
        one_col2 = torch.ones((2, 1))
        deformed_pixel = self.all_pixels - (self.K @ self.betas)
        n_pixels = deformed_pixel.size()[0]
        n_centers = self.all_p_centers.size()[0]
        p_norm_squared = torch.square(deformed_pixel) @ one_col2
        c_norm_squared = torch.square(self.all_p_centers) @ one_col2
        p_norm_squared_repeated = p_norm_squared @ torch.ones((1, n_centers))
        c_norm_squared_repeated = (c_norm_squared @ torch.ones((1, n_pixels))).T
        p_dot_c = 2 * (deformed_pixel @ self.all_p_centers.T)
        big_matrix = p_norm_squared_repeated + c_norm_squared_repeated - p_dot_c
        kBp = torch.exp(- big_matrix / (2 * self.sdp2))
        out = kBp @ self.alphas
        return out

    def string(self):
        return f'b = {self.betas.item()}'


class PyTorchOptimizer():
    def __init__(self, alphas, image, curr_beta, g_inv, sdp2, sdl2):
        self.alphas = torch.from_numpy(alphas).type(torch.FloatTensor)
        self.curr_betas = torch.from_numpy(curr_beta).type(torch.FloatTensor)
        # self.KTonsd2 = torch.from_numpy((1/(sdl2 * sdp2)) * K.T)
        # self.diag_a_Ca = torch.from_numpy(alphas * C_a)
        self.image = torch.from_numpy(image.reshape(-1, 1)).type(torch.FloatTensor)
        self.g_inv = torch.from_numpy(g_inv).type(torch.FloatTensor)
        self.sdp2 = sdp2
        self.sdl2 = sdl2

    def optimize_betas(self, iter):
        image_predictor = KBpA(alphas=self.alphas,
                               curr_betas=self.curr_betas,
                               sdp2=self.sdp2,
                               K=torch_K,
                               all_pixels=torch_all_pixel,
                               all_p_centers=torch_C_a
                               )
        criterion = torch.nn.MSELoss(reduction='sum')
        # optimizer = torch.optim.Adam(image_predictor.parameters(),
        #                                 lr=1e-4)
        optimizer = torch.optim.Adagrad(image_predictor.parameters())
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
            to_numpy = betas.detach().numpy()
            return to_numpy