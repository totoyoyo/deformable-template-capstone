from pathlib import Path
import matplotlib.pyplot as plt
main_path = Path(__file__).resolve().parent
from PIL import Image
import numpy as np


def load_train_images_digits(template_path=Path()):
    image_folder = template_path / "train"
    png_list = []
    for image_path in image_folder.glob("*.png"):
        image = plt.imread(image_path)
        png_list.append(image)
        print(image.shape)
        print(image.dtype)
    return png_list

def load_train_images_coins(template_path=Path()):
    image_folder = template_path / "train"
    jpg_list = []
    for image_path in image_folder.glob("*.jpg"):
        img = Image.open(image_path).convert('L')
        img = img.resize((100, 100))
        np_img = np.array(img)
        np_img[np_img == 255] = 0
        jpg_list.append(np_img / 255)
    return jpg_list