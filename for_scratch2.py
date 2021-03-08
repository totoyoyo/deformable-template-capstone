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


four = np.float32(4)
onfor = 1.0/four

for n in range(5.0):
    print(n)

print('done')