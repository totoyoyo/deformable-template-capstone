import glob
import matplotlib.pyplot as plt

png = []
for image_path in glob.glob("/raw_data_peeranat/data/threes/*.png"):
    image = plt.imread(image_path)
    png.append(image * 2)
    print(image.shape)
    print(image.dtype)

