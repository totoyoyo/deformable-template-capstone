import torch
import matplotlib
import scipy
import torch.nn.functional as tnf
import numpy as np
import datetime
import os
import matplotlib.pyplot
import scipy.sparse
import numpy.linalg
import glob
from PIL import Image
import scipy.linalg
import time

t= torch.Tensor([1])

f = open("myfile.txt", "w")
f.write("Hello!")
f.close()
print("done writing")