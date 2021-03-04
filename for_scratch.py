import torch
# torch.cuda.is_available()
#
#
# big_zero = torch.tensor([[1,2,3],
#                         [4,5,6]]).repeat((5,1))
#
#
#
#
# big_zero.size()

# import trainers.training_2d_pytorch

import numpy as np
import gc
import scipy.sparse as ss

#
# yo = np.ones((100**2,100**2),dtype='float32')
# print('done1')
# yo2 = np.ones((100**2,100**2),dtype='float32')
# print('done2')
# yo3 = np.ones((100**2,100**2),dtype='float32')
# print('done3')
# yo4 = np.ones((100**2,100**2),dtype='float32')
#
# haha = gc.isenabled()

# data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
# offsets = np.array([0, -1, 2])
# d = ss.dia_matrix((data, offsets), shape=(4, 4))
# a = d.toarray()
#
import torch
import torch.sparse as ts

# A = ss.coo_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]], dtype='float32')
# indices = np.vstack((A.row,A.col))
# values = A.data
# i = torch.LongTensor(indices)
# v = torch.FloatTensor(values)
# shape = A.shape
#
# s = torch.sparse_coo_tensor(indices=indices,
#                             values=values,
#                             size=shape)
# out = s.to_dense()
#
# one = np.float32(1.0)
# e = np.exp(one,dtype='float32')

np1 = np.array([[1],[2],[3]])
np2 = np.array([[4],[5],[6]])

lnp = [np1,np2]

t = list(map(lambda x: torch.from_numpy(x),lnp))

# sd2 = 2
# c = 6  # actually should depend on bandwith l
#
# x   = np.arange(0, 2*c + 1)
# RBF = np.exp( -  (1/(2*sd2))  * (x - c)**2 )
# RBF2 = np.outer(RBF,RBF)
# import matplotlib.pyplot as plt
#
# plt.imshow(RBF2, cmap = 'jet')
# plt.colorbar();
# plt.show()


import matplotlib.pyplot as plt


#
# sd2 = 9
# c = 9  # actually should depend on bandwith l
#
# x   = np.arange(0, 2*c + 1)
# RBF = np.exp( -  (1/(2*sd2))  * (x - c)**2 )
# RBF2 = np.outer(RBF,RBF)
# import matplotlib.pyplot as plt
#
# plt.imshow(RBF2, cmap = 'jet')
# plt.colorbar();
# plt.show()


def generate_gaussian_kernel(sd2):
    sd = np.sqrt(sd2)
    c = np.ceil(sd * 4)
    x = np.arange(0, 2 * c + 1)
    RBF = np.exp(- (1 / (2 * sd2)) * (x - c) ** 2)
    RBF2 = np.outer(RBF, RBF)
    return c, RBF2


import torch.nn.functional as tnf

center_kernel, np_kernel = generate_gaussian_kernel(4)

t_kernel = torch.from_numpy(np_kernel).type(torch.float32)
k_view = t_kernel.view(1,1,-1)

n = 28
m = 28
a = np.zeros((n,m))
a[6:15,6:22] = 1
a[8:13,8:20] = 0
a[22:24,22:24] = 4

t_a = torch.from_numpy(a).type(torch.float32)
t_a = torch.unsqueeze(t_a,0)
t_a = torch.unsqueeze(t_a,0)



pixels = torch.tensor([[0,0],[0,1],[0,2]])
centers = torch.tensor([[0,0],[0,1],[0,2]])
rep_pixels = pixels.repeat_interleave(repeats=3, dim = 0)
tiled_centers = centers.repeat(3,1)

print('done')