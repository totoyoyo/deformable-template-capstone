import glob
import matplotlib.pyplot as plt

png = []
for image_path in glob.glob("../data/for_training_new/*.png"):
    image = plt.imread(image_path)
    png.append(image * 2)
    print(image.shape)
    print(image.dtype)

