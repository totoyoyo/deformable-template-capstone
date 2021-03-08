import glob
from PIL import Image
import numpy as np

png = []
for image_path in glob.glob("../data/coins/*.jpg"):
    img = Image.open(image_path).convert('L')
    img = img.resize((100,100))
    np_img = np.array(img)
    png.append(np_img/255)


img.show()