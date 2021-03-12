import glob
from PIL import Image
import numpy as np

png = []
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


for image_path in glob.glob(os.path.dirname(full_path) + "/data/cropped/*.jpg"):
    img = Image.open(image_path).convert('L')
    img = img.resize((100,100))
    np_img = np.array(img)
    np_img[np_img == 255] = 0
    png.append(np_img * 2/255)


# img.show()