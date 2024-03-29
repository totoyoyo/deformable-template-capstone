import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import json

original_stdout = sys.stdout

def redirect_stdout_to_txt(file_path):
    sys.stdout = open(file_path, "w")

def bring_back_stdout():
    if sys.stdout is not original_stdout:
        sys.stdout.close()
    sys.stdout = original_stdout


def handle_saving_plots(parent_path, array_to_plot, plotname):
    save_path = parent_path / (plotname + ".png")
    plt.imshow(array_to_plot)
    plt.title(plotname)
    plt.colorbar()
    plt.savefig(save_path)
    plt.clf()
    # plt.show()

def handle_saving_npdata(parent_path, npdata, data_name, suffix):
    np.savetxt(parent_path / (data_name+suffix),
               X=npdata)

def handle_saving_compressed_npdata(parent_path, npdata, data_name, suffix):
    np.save(parent_path / (data_name),
            npdata)


def handle_duplicate_names(parent_path, wanted_name):
    new_path = parent_path / wanted_name
    counter = 0
    while(new_path.is_dir() and counter < 20):
        new_path = parent_path / (wanted_name + (str(counter)))
        counter += 1
    return new_path


def handle_saving_parameters(parent_path,ag, ap, t_sd2,
                         d_sd2, init_sd, epochs,
                         iterations):
    parent_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(parent_path / ("ag" + ".data"),
               X=[ag])
    np.savetxt(parent_path / ("ap" + ".data"),
               X=[ap])
    np.savetxt(parent_path / ("t_sd2" + ".data"),
               X=[t_sd2])
    np.savetxt(parent_path / ("d_sd2" + ".data"),
               X=[d_sd2])
    np.savetxt(parent_path / ("init_sd" + ".data"),
               X=[init_sd])
    np.savetxt(parent_path / ("epochs" + ".data"),
               X=[epochs])
    np.savetxt(parent_path / ("iterations" + ".data"),
               X=[iterations])
    dict_hyper = {
        "ag" : ag,
        "ap" : ap,
        "t_sd2": t_sd2,
        "d_sd2": d_sd2,
        "init_sd": init_sd,
        "epochs": epochs,
        "iterations": iterations
    }
    with open(parent_path / ("hyper" + ".json"), 'w') as fp:
        json.dump(dict_hyper, fp)




def handle_save_classification_results(path, res_df):
    pass



main_path = Path(__file__).resolve().parent
a = np.array([1,2,3,4])

# handle_saving_plots(main_path / 'hey' / "something.png",
#                     a,
#                     "YoYo",
#                     2,
#                     2)

# handle_saving_npdata(main_path,[25],"sd",".data")

yo = handle_duplicate_names(main_path, "train_output")
