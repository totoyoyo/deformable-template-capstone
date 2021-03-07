import numpy as np
import torch
import constants_2d_new as const
import functions_2d_new as func


# Convert from diag to coo and then to tensor
# Also learn how to clear gpu memory
# func.S_PIXEL_G_CENTERS_MATRIX.

def get_pt_sparse_K(sparse_scipy_matrix):
    sp_coo = sparse_scipy_matrix.tocoo()
    indices = np.vstack((sp_coo.row, sp_coo.col))
    values = sp_coo.data
    pt_sparse = torch.sparse_coo_tensor(indices=indices,
                                        values=values,
                                        size=sp_coo.shape,
                                        device=torch.device('cuda'))
    return pt_sparse


def get_pt_dense_K(sparse_scipy_matrix):
    dense_numpy = sparse_scipy_matrix.todense()
    torch_tensor = torch.from_numpy(dense_numpy).float().cuda()
    return torch_tensor

torch_K = get_pt_sparse_K(func.S_PIXEL_G_CENTERS_MATRIX)
torch_K_dense = get_pt_dense_K(func.S_PIXEL_G_CENTERS_MATRIX)


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
torch_all_images = torch.tensor(const.FLAT_IMAGES, dtype=torch.float32,
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
        deformation = torch_K_dense @ self.betas
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

    def string(self):
        return f'b = {self.betas.item()}'


class PyTorchOptimizer():
    def __init__(self, alphas, curr_beta, g_inv, sdp2, sdl2):
        self.alphas = torch.from_numpy(alphas).float().cuda()
        self.curr_betas = torch.tensor(curr_beta, dtype=torch.float32,
                                       device=torch.device('cuda'))
        self.image = torch_all_images
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
            return list_of_numpy
