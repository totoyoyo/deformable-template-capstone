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


def load_train_images(template_path=Path(), coins=False):
    types = ('*.jpg', '*.png')
    files_grabbed = []
    for files in types:
        files_grabbed.extend(template_path.glob(files))
    img_list = []
    for image_path in files_grabbed:
        img = get_coin_array(image_path) if coins else get_digit_array(image_path)
        img_list.append(img)
    return img_list


def load_classify_images(input_data=Path(), coins=False):
    types = ('**/test/*.jpg', '**/test/*.png')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(input_data.glob(files))
    image_dict_list = []
    for image_path in files_grabbed:
        template_path = image_path.parent.parent
        template_name = template_path.stem
        if coins:
            arr = get_coin_array(image_path)
        else:
            arr = get_digit_array(image_path)
        image_name = image_path.stem
        img_dict = {
            'name' : image_name,
            'arr' : arr,
            'true_template_name' : template_name
        }
        print(img_dict)
        image_dict_list.append(img_dict)
    return image_dict_list


def get_coin_array(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((100, 100))
    np_img = np.array(img)
    np_img[np_img == 255] = 0
    scaled = (np_img / 255)
    return scaled

def get_digit_array(image_path):
    image = plt.imread(image_path)
    print(image.shape)
    print(image.dtype)
    return image




# p = main_path / "input_coins"
#
# out = load_classify_images(p)