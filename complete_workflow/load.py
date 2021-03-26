from pathlib import Path

import matplotlib.pyplot as plt

main_path = Path(__file__).resolve().parent
from PIL import Image
import numpy as np


def load_train_images_digits(template_path=Path()):
    image_folder = template_path / "train"
    png_list = []
    for image_path in image_folder.glob("*.png"):
        arr = get_digit_array(image_path)
        png_list.append(arr)
    return png_list


def load_train_images_coins(template_path=Path()):
    image_folder = template_path / "train"
    jpg_list = []
    for image_path in image_folder.glob("*.png"):
        arr = get_coin_array(image_path)
        jpg_list.append(arr)
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
    types = ('**/test/*.absolute_nonsense', '**/test/*.png')  # the tuple of file types
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
        image_name = image_path.name
        img_dict = {
            'name': image_name,
            'arr': arr,
            'true_template_name': template_name
        }
        # print(img_dict)
        image_dict_list.append(img_dict)
    return image_dict_list


def get_coin_array(image_path):
    img1 = plt.imread(image_path)
    name = image_path.name
    print("Loading image: " + name)
    return img1


def get_digit_array(image_path):
    image = plt.imread(image_path)
    name = image_path.name
    print("Loading image: " + name)
    return image


def load_hyperparameters(training_output_path=Path()):
    dict_out = {}
    for data_path in training_output_path.glob("*.data"):
        data_name = data_path.stem
        data = np.loadtxt(data_path)
        dict_out[data_name] = data.item()
    return dict_out


def load_template(template_output_path=Path()):
    dict_out = {}
    for data_path in template_output_path.glob("*.data"):
        data_name = data_path.stem
        data = np.loadtxt(data_path)
        dict_out[data_name] = data
    for g_inv_path in template_output_path.glob("*.npy"):
        g_inv = np.load(g_inv_path)
    return dict_out, g_inv

# out = load_hyperparameters(main_path/'train_output6')

# p = main_path / "input_coins"
#
# out = load_classify_images(p)
