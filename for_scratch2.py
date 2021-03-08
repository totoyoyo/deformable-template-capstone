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

def get_spread_out_kernels(all_pixels, distance, randomize = False):
    if randomize:
        new_pixels = random.sample(all_pixels,len(all_pixels))
    else:
        new_pixels = all_pixels
    to_return = []
    for pixel_point in new_pixels:
        if not any(np.linalg.norm(existing - pixel_point) < distance for existing in to_return):
            to_return.append(pixel_point)
    return to_return


out =


print('done')