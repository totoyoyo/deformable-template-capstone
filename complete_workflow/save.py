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




main_path = Path(__file__).resolve().parent
a = np.array([1,2,3,4])

# handle_saving_plots(main_path / 'hey' / "something.png",
#                     a,
#                     "YoYo",
#                     2,
#                     2)

handle_saving_npdata(main_path,[25],"sd",".data")