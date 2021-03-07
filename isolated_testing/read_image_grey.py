import glob
from PIL import Image

png = []
for image_path in glob.glob("../data/coins/*.jpg"):
    img = Image.open(image_path).convert('LA')


img.show()