import os

os.environ["CUDA_VISIBLE_DEVICES"] ="1,2"

import torch

mydevice = torch.cuda.device_count()
avai = torch.cuda.is_available()

print(mydevice)
print(avai)