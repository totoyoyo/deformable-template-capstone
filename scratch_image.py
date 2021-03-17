import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path = "coin.jpg"

pil_img = Image.open("coin.jpg").convert('L')
pil_img = pil_img.resize((100, 100))
np_img = np.array(pil_img)
# np_img[np_img == 255] = 0
# scaled1 = (np_img / 255)

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv_img_n = cv2.resize(img, (100,100),
                    interpolation=cv2.INTER_NEAREST)
cv_img_l = cv2.resize(img, (100,100),
                    interpolation=cv2.INTER_LINEAR)
cv_img_a = cv2.resize(img, (100,100),
                    interpolation=cv2.INTER_AREA)
cv_img_c = cv2.resize(img, (100,100),
                    interpolation=cv2.INTER_CUBIC)
cv_img_la = cv2.resize(img, (100,100),
                    interpolation=cv2.INTER_LANCZOS4)

# plt.imshow(np_img, cmap='gray')
# plt.title("Pil")
# plt.show()
#
plt.imshow(img, cmap='gray')
plt.title("Opencv orig")
plt.show()
#
# plt.imshow(cv_img_n, cmap='gray')
# plt.title("Opencv nearest")
# plt.show()
#
# plt.imshow(cv_img_l, cmap='gray')
# plt.title("Opencv linear")
# plt.show()
#
plt.imshow(cv_img_a, cmap='gray')
plt.title("Opencv area")
plt.show()
#
# plt.imshow(cv_img_c, cmap='gray')
# plt.title("Opencv cubic")
# plt.show()
#
# plt.imshow(cv_img_la, cmap='gray')
# plt.title("Opencv lanc")
# plt.show()

my_h= 20
dstf = cv2.fastNlMeansDenoising(src=img,dst=None,h=my_h)
plt.imshow(dstf, cmap='gray')
plt.title("Opencv lanc" + str(my_h))
plt.show()


from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


my_tv_weight = 0.1
dst = denoise_tv_chambolle(image=img,
                           weight=my_tv_weight)
plt.imshow(dst, cmap='gray')
plt.title("Tv lanc" + str(my_tv_weight))
plt.show()

#
# dst = denoise_bilateral(image=img)
# plt.imshow(dst, cmap='gray')
# plt.title("Bilateral" + str(my_tv_weight))
# plt.show()
#
# dst = denoise_wavelet(image=img)
# plt.imshow(dst, cmap='gray')
# plt.title("Wavelet" + str(my_tv_weight))
# plt.show()


# cv_img_a2 = cv2.resize(dstf, (100,100),
#                     interpolation=cv2.INTER_AREA)
#
# plt.imshow(cv_img_a2, cmap='gray')
# plt.title("Opencv area2")
# plt.show()

laplacian = np.abs(cv2.Laplacian(dstf,ddepth=cv2.CV_64F,
                          ksize=3,borderType=cv2.BORDER_REFLECT))/840
plt.imshow(laplacian, cmap='jet')
plt.title("Laplace")
plt.colorbar()
plt.show()

sobel_k = 3
sobel_x = cv2.Sobel(dstf, dx=1, dy=0, ddepth=cv2.CV_64F,
                           ksize=sobel_k, borderType=cv2.BORDER_REFLECT)
sobel_y = cv2.Sobel(dstf, dx=0, dy=1, ddepth=cv2.CV_64F,
                           ksize=sobel_k, borderType=cv2.BORDER_REFLECT)

grad_approx = cv2.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)
plt.imshow(grad_approx, cmap='jet')
plt.title("sobel approx")
plt.colorbar()
plt.show()

grad_correct = np.sqrt((sobel_x **2 + sobel_y**2))
plt.imshow(grad_correct, cmap='jet')
plt.title("sobel real")
plt.colorbar()
plt.show()
