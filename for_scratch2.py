import torch
import numpy as np

A = [[[0,1,2], [3,4,5], [6,7,8]], [[9,10,11], [12,13,14], [15,16,17]]]

A = torch.tensor(A)

B = [[1,0,0], [0,1,0], [0,0,1]]

B = torch.tensor(B)

AB = B @ A

C = A - B


AA = [np.array([[1,2,3],
               [4,5,6]]),
     np.array([[1,2,3],
               [4,5,6]])]

A_t = torch.tensor(AA)

A_t_numpy = list(A_t.numpy())


t1 = np.matrix([[1, 2], [3, 4]]).T
t2 = np.matrix([[1,3]]).T
t3 = t1 @ t2


a = list(range(10))
b = a[0:5]
b[1:2] = [2,4]
print(a)


def get_list_of_indexes_for_slicing(slice_length,total_length):
    i = 0
    l = []
    while(i < total_length):
        end = min(total_length, i + slice_length)
        indexes = [i,end]
        i = end
        l.append(indexes)
    return l


t1 = get_list_of_indexes_for_slicing(2,10)
t2 = get_list_of_indexes_for_slicing(1,10)
t3 = get_list_of_indexes_for_slicing(5,2)

for i in t2:
    print(i[0])
    print(i[1])

import random

a = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]])
# la = list(a)
# filtered = list(filter(lambda array: (array[0] + array[1]) % 2 == 0, la))
# filtered_a = np.array(filtered)
new_a = a[np.sum(a,axis=1) % 2 == 0]



betas_sample = [np.array([[1,2],
                          [3,4],
                          [5,6]]),
                np.array([[7, 8],
                          [9,10],
                          [11,12]]),
                np.array([[13,14],
                          [15,16],
                          [17,18]])
                ]

IMAGE_DIM = (3,3)
G_CENTERS = np.array([[0,1],
                      [1,2],
                      [2,2]])
to_fill = np.array([1,2,3])

rows, cols = G_CENTERS.T
betas = np.array([[5,6],
                 [7,8],
                 [9,10]])
empty_array = np.zeros((2,3,3))
empty_array[:,rows,cols] = betas.T




# empty_array[rows, cols] = to_fill

betas_torch = torch.Tensor(betas_sample)
kernel_t = torch.Tensor([[[1,2,3],
                         [4,5,6],
                         [7,8,9]],
                         [[11, 22, 33],
                         [44, 55, 66],
                         [77, 88, 99]]
                         ])

bigger_kernel = kernel_t.expand(3,2,3,3)

def conv_res_to_deformation(res, batch_size):
    flat_res = res.reshape((batch_size, 2, -1))
    deformation = torch.transpose(flat_res, 1, 2)
    return deformation

flat_b_k = bigger_kernel.reshape((3,2,-1))
flat_b_k_2 = torch.transpose(flat_b_k,1,2)

def make_image_of_betas_for_conv(betas, beta_centers, image_row, image_col):
    """
    :param betas:
    :param beta_centers:
    :param image_row:
    :param image_col:
    :return: 2 by image_row by image_col array (the 2 is the 2 channels of betas)
    """
    empty_array = np.zeros((2,image_row,image_col), dtype='float32')
    rows, cols = beta_centers.T
    empty_array[:, rows, cols] = betas.T
    return empty_array



empty_tensor = torch.zeros((10,2,3,3))
#beta shape (images, centers,2)
#out has shape (images, 2, centers)
out = empty_tensor[:,:, rows, cols]

firstdim = empty_tensor.size()[0]
#dim (batch, pixels, 2)


kernel_t = torch.Tensor([[1,2,3],
                         [4,5,6],
                         [7,8,9],
                         ])

bigger_kernel = kernel_t.expand(3,1,3,3)

print('done')


g = np.ones((5000, 5000))
np.savetxt("g64.txt",X=g)
np.savetxt("g32.txt",X=g.astype('float32'))
np.save("g64",g)
np.save("g32",g.astype('float32'))

np.savez("g64z",g)
np.savez("g32z",g.astype('float32'))



