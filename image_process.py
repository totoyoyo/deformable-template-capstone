
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from pathlib import Path


def process_img_and_save(img_path : Path, denoise_h=20,
                sobel=True, kernel_size=3):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (350, 350),
                     interpolation=cv2.INTER_AREA)
    denoised = cv2.fastNlMeansDenoising(src=img, dst=None, h=denoise_h)
    if sobel:
        filtered = apply_sobel(denoised, kernel_size)
    else:
        filtered = apply_laplacian(denoised, kernel_size)
    masked = apply_circle_mask(filtered)
    resized_to_100 = cv2.resize(masked, (100,100),
                    interpolation=cv2.INTER_AREA)
    # resized_to_100 = masked
    rescaled = ((resized_to_100 / resized_to_100.max()) * 255).astype("uint8")
    im = Image.fromarray(rescaled)
    img_orig_name = img_path.stem
    img_folder = img_path.parent
    im.save(img_folder / (img_orig_name + "_p_l" + str(kernel_size) + ".png"))
    return rescaled


def apply_sobel(img, kernel_size):
    sobel_x = cv2.Sobel(img, dx=1, dy=0, ddepth=cv2.CV_64F,
                        ksize=kernel_size, borderType=cv2.BORDER_REFLECT)
    sobel_y = cv2.Sobel(img, dx=0, dy=1, ddepth=cv2.CV_64F,
                        ksize=kernel_size, borderType=cv2.BORDER_REFLECT)

    sobel_both = np.sqrt((sobel_x ** 2 + sobel_y ** 2))
    return sobel_both

def apply_laplacian(img, kernel_size):
    laplacian = np.abs(cv2.Laplacian(img, ddepth=cv2.CV_64F,
                                     ksize=kernel_size, borderType=cv2.BORDER_REFLECT))
    return laplacian

def apply_circle_mask(img, radius=150):
    hh, ww = img.shape
    ycen = hh // 2
    xcen = ww // 2
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, center=(ycen, xcen), radius=radius,
                      color=1, thickness=-1)
    masked = np.where(mask == 1, img, 0)
    return masked

def canny_process_img_and_save(img_path, l_t=150, h_t=300,
                               kernel_size=3):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (350, 350),
                     interpolation=cv2.INTER_AREA)
    icanny_out = cv2.Canny(img, threshold1=l_t,
                           threshold2=h_t, L2gradient=True, apertureSize=kernel_size)
    masked = apply_circle_mask(icanny_out)
    resized_to_100 = cv2.resize(masked, (100,100),
                    interpolation=cv2.INTER_AREA)
    rescaled = ((resized_to_100 / resized_to_100.max()) * 255).astype("uint8")
    im = Image.fromarray(rescaled)
    img_orig_name = img_path.stem
    img_folder = img_path.parent
    im.save(img_folder / (img_orig_name +"_p_can" + ".png"))
    return rescaled

main_path = Path(__file__).resolve().parent
image_folder = main_path / "orig_coin_5_classes"
for image_path in image_folder.glob("**/*.jpg"):
    process_img_and_save(image_path,sobel=False,kernel_size=5)

# Sobel noisy
#Canny is very sensitive to thresholds
# Laplacian h=5 is best