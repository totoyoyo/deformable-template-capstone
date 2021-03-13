import glob
from PIL import Image
import numpy as np

png = []
for image_path in glob.glob("../data/cropped/*.jpg"):
    img = Image.open(image_path).convert('L')
    img = img.resize((100,100))
    np_img = np.array(img)
    np_img[np_img == 255] = 0
    png.append(np_img / 255)


# img.show()