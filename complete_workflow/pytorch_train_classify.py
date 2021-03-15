import torch
import torch.nn.functional as tnf
import functions_maker as func

# Convert from diag to coo and then to tensor
# Also learn how to clear gpu memory
# func.S_PIXEL_G_CENTERS_MATRIX.


#INTERMEDIATE FUNCTIONS

class PyTorchConstants:

    def __init__(self,
                 const_object):
        self.const_object=const_object
        self.padding, self.kernel = func.generate_gaussian_kernel(
            const_object.deform_sd2
        )
        self.torch_C_a = torch.from_numpy(const_object.p_centers).cuda()
        self.torch_all_pixel = torch.from_numpy(const_object.all_pixels).cuda()
        self.n_pixels = self.torch_all_pixel.size()[0]
        self.n_centers = self.torch_C_a.size()[0]
        self.ones_pixels = torch.ones((1, self.n_pixels), dtype=torch.float32,
                                      device=torch.device('cuda'))
        self.ones_centers = torch.ones((1, self.n_centers), dtype=torch.float32,
                                       device=torch.device('cuda'))
        self.one_col2 = torch.ones((2, 1), dtype=torch.float32,
                                   device=torch.device('cuda'))

    def make_beta_tensors_for_conv(self,betas):
        batch_size = betas.size()[0]
        empty_tensor = torch.zeros((batch_size, 2,
                                    self.const_object.image_nrow,
                                    self.const_object.image_ncol),
                                   device=torch.device('cuda'))
        rows, cols = self.const_object.g_centers.T
        empty_tensor[:, :, rows, cols] = torch.transpose(betas, 1, 2)
        return empty_tensor

    def do_batch_convolution(self, beta_images, kernel):
        t_kernel = torch.from_numpy(kernel).cuda()
        kernel_row = t_kernel.size()[0]
        kernel_col = t_kernel.size()[1]
        big_kernel = t_kernel.expand(2, 1, kernel_row, kernel_col)
        out = tnf.conv2d(input=beta_images, weight=big_kernel,
                         padding=self.padding, groups=2)
        return out

    def conv_res_to_deformation(self, res, batch_size):
        flat_res = res.reshape((batch_size, 2, -1))
        deformation = torch.transpose(flat_res, 1, 2)
        return deformation

    def get_deformation_from_conv(self,betas):
        batch_size = betas.size()[0]
        beta_img = self.make_beta_tensors_for_conv(betas)
        res = self.do_batch_convolution(beta_img, kernel=self.kernel)
        deformation = self.conv_res_to_deformation(res=res, batch_size=batch_size)
        return deformation


class PyTorchOptimizer:
    def __init__(self, alphas, curr_beta, g_inv, sdp2, sdl2, images,
                 pytorch_const):
        self.pytorch_const = pytorch_const
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
                               all_pixels=self.pytorch_const.torch_all_pixel,
                               all_p_centers=self.pytorch_const.torch_C_a,
                               pytorch_const=self.pytorch_const).cuda()
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

class KBpA(torch.nn.Module):

    def __init__(self, alphas, curr_betas, sdp2, all_pixels, all_p_centers,
                 pytorch_const):
        super().__init__()
        # beta shape (images, centers,2)
        self.pytorch_const = pytorch_const
        self.betas = torch.nn.Parameter(curr_betas)
        self.alphas = alphas
        self.sdp2 = sdp2
        self.all_pixels = all_pixels
        self.all_p_centers = all_p_centers

    def forward(self):
        deformation = self.pytorch_const.get_deformation_from_conv(self.betas)
        deformed_pixel = self.all_pixels - deformation
        return torch.exp(
            - (torch.square(deformed_pixel)
               @ self.pytorch_const.one_col2
               @ self.pytorch_const.ones_centers
               + (torch.square(self.all_p_centers)
                  @ self.pytorch_const.one_col2 @ self.pytorch_const.ones_pixels).T
               - 2 * (deformed_pixel @ self.all_p_centers.T))
            / (2 * self.sdp2)) \
               @ self.alphas

    def string(self):
        return f'b = {self.betas.item()}'