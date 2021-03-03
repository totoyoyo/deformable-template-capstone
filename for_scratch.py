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


yo = np.ones((100**2,100**2),dtype='float32')
print('done1')
yo2 = np.ones((100**2,100**2),dtype='float32')
print('done2')
yo3 = np.ones((100**2,100**2),dtype='float32')
print('done3')
yo4 = np.ones((100**2,100**2),dtype='float32')

haha = gc.isenabled()
