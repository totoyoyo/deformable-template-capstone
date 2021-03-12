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
import file_writing_no_path as fwnp

t= torch.Tensor([1])

f = open("myfile.txt", "w")
f.write("Hello!")
f.close()
print("done writing")

import os

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

print("This file directory only")
print(os.path.dirname(full_path))

print("The file should be here")
print(os.path.dirname(full_path) + "/myfile_two.txt")

f = open(os.path.dirname(full_path) + "/myfile_two.txt", "w")
f.write("Hello again!")
f.close()
print("done writing second file")