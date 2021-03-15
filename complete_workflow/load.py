from pathlib import Path
import matplotlib.pyplot as plt
main_path = Path(__file__).resolve().parent

def load_train_images_digits(template_path=Path()):
    image_folder = template_path / "train"
    png_list = []
    for image_path in image_folder.glob("*.png"):
        image = plt.imread(image_path)
        png_list.append(image)
        print(image.shape)
        print(image.dtype)
    return png_list
