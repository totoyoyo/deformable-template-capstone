import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def handle_saving_plots(parent_path, array_to_plot, plotname):
    save_path = parent_path / (plotname + ".png")
    plt.imshow(array_to_plot)
    plt.title(plotname)
    plt.colorbar()
    plt.savefig(save_path)
    plt.show()

def handle_saving_npdata(parent_path, npdata, data_name, suffix):
    np.savetxt(parent_path / (data_name+suffix),
               X=npdata)

def handle_duplicate_names(parent_path, wanted_name):
    new_path = parent_path / wanted_name
    counter = 0
    while(new_path.is_dir() and counter < 20):
        new_path = parent_path / (wanted_name + (str(counter)))
        counter += 1
    return new_path





main_path = Path(__file__).resolve().parent
a = np.array([1,2,3,4])

# handle_saving_plots(main_path / 'hey' / "something.png",
#                     a,
#                     "YoYo",
#                     2,
#                     2)

# handle_saving_npdata(main_path,[25],"sd",".data")

yo = handle_duplicate_names(main_path, "train_output")
