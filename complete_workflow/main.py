# ---------------Constants are here----------

# turn on to limit cpu usage
SERVER = False

# Save print to file
SAVE_PRINTS = False

# use coins
COINS = False

# If not doing training, be sure to set DEFAULT_CLASSIFY_PATH below
DO_TRAIN = False
DO_CLASSIFY = True

# Constants for the algorithm, most important are the two SD2 (sd-squared for gaussian)
TEMPLATE_SD2 = 2
DEFORM_SD2 = 2
AG = 2.5
AP = 100
EPOCHS = 1000
ITERATIONS = 5
INIT_SD2 = 1

import os

if SERVER:
    os.environ['OPENBLAS_NUM_THREADS'] = '4'

import classifier
import trainer
import constants_maker as const
import load
import save
import pathlib

DEFAULT_CLASSIFY_PATH = pathlib.Path(__file__).resolve().parent / 'train_output'


# ------------- CONSTANTS END ------------

def do_training_and_save(template_name, const_object,
                         training_output_path):
    to_train = trainer.Estimator2DNImages(cons_obj=const_object,
                                          template_name=template_name,
                                          training_output_path=training_output_path)
    to_train.run_estimation(const_object.iterations)
    to_train.save_all()


def make_constant_object_train(template_path, ag=AG, ap=AP, t_sd2=TEMPLATE_SD2,
                               d_sd2=DEFORM_SD2, init_sd=INIT_SD2, epochs=EPOCHS,
                               iterations=ITERATIONS):
    if not COINS:
        images = load.load_train_images_digits(template_path=template_path)
    else:
        images = load.load_train_images_coins(template_path=template_path)
    obj = const.TrainingConstants(images=images, ag=len(images) / 4, ap=ap, t_sd2=t_sd2,
                                  d_sd2=d_sd2, init_sd=init_sd,
                                  epochs=epochs, iterations=iterations)
    return obj


def train():
    if not COINS:
        data_path = pathlib.Path(__file__).resolve().parent / 'input_data'
    else:
        data_path = pathlib.Path(__file__).resolve().parent / 'input_coins'
    main_path = pathlib.Path(__file__).resolve().parent
    training_output_path = save.handle_duplicate_names(main_path, "train_output")
    save.handle_saving_parameters(training_output_path,
                                  ag=AG, ap=AP, t_sd2=TEMPLATE_SD2,
                                  d_sd2=DEFORM_SD2, init_sd=INIT_SD2, epochs=EPOCHS,
                                  iterations=ITERATIONS)
    if SAVE_PRINTS:
        save.redirect_stdout_to_txt(training_output_path / "printed_during_train.txt")
    for template_path in data_path.glob('template*'):
        template_name = template_path.stem
        const_obj = make_constant_object_train(template_path=template_path)
        do_training_and_save(template_name=template_name,
                             const_object=const_obj,
                             training_output_path=training_output_path)
    if SAVE_PRINTS:
        save.bring_back_stdout()
    return training_output_path


def make_constant_object_classify(input_path, training_output_path):
    if not COINS:
        images = load.load_train_images_digits(template_path=input_path)
    else:
        images = load.load_train_images_coins(template_path=input_path)

    hyper_dict = load.load_hyperparameters(training_output_path)
    obj = const.TrainingConstants(images=images, ag=hyper_dict['ag'],
                                  ap=hyper_dict['ap'], t_sd2=hyper_dict['t_sd2'],
                                  d_sd2=hyper_dict['d_sd2'], init_sd=hyper_dict['init_sd'],
                                  epochs=hyper_dict['epochs'], iterations=hyper_dict['iterations'])
    return obj


def classify():
    if not COINS:
        input_path = pathlib.Path(__file__).resolve().parent / 'input_data'
    else:
        input_path = pathlib.Path(__file__).resolve().parent / 'input_coins'

    train_result_path = TRAIN_OUTPUT_PATH
    if SAVE_PRINTS:
        save.redirect_stdout_to_txt(train_result_path / "printed_during_classify.txt")
    print("Classifying!")

    sample_template_path = next(input_path.glob("template*"))
    const_obj_classify = make_constant_object_classify(sample_template_path,
                                                       train_result_path)

    img_to_classify = load.load_classify_images(input_path, coins=COINS)

    image_classifier = classifier.ImageClassifier(cons_obj=const_obj_classify,
                                                  images=img_to_classify,
                                                  output=train_result_path)

    for template_path in train_result_path.glob('template*'):
        template_name = template_path.stem
        curr_template = classifier.TemplateClass(trained_template_path=template_path,
                                                 name=template_name)
        image_classifier.template_search(epochs=EPOCHS,
                                         template=curr_template)
    res = image_classifier.compute_and_save_results()
    if SAVE_PRINTS:
        save.bring_back_stdout()
    return res


if __name__ == '__main__':
    TRAIN_OUTPUT_PATH = train() if DO_TRAIN else DEFAULT_CLASSIFY_PATH
    if DO_CLASSIFY:
        classify_results = classify()

"""
Dictionary should contain (name, alphas, sd2, Gamma inv)
"""

"""
Note, Gamma_Inv is NOT sparse!!!
For 5000x5000 Gamma_Inv (for 100x100 images)
Size of sparse representation is 299991156 bytes
size of dense representation is  200000000 bytes
"""

""" Could be a dataframe?"""
"""row: name alphas, sd2, gamma_inv"""

"""
Now,we have a list of images with labels on them
"""

"""
Classification should take a list of dictionaries and
a list of images (without the labels)
"""

"""
return a list of labels for the images, and a list of likelihoods for all classes
ie each image should have a list of likelihoods
"""

### do evaluation

"""
Given list of labels and likelihoods 
"""
