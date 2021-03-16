SERVER = False
SAVE_PRINTS = False
import os

if SERVER:
    os.environ['OPENBLAS_NUM_THREADS'] = '4'

import classifier
import pytorch_train_classify
import trainer
import constants_maker as const
import load
import save
import pathlib
import sys

### do some training

DO_TRAIN = True
DO_CLASSIFY = True
DEFAULT_CLASSIFY_PATH = pathlib.Path(__file__).resolve().parent / 'train_output'

AG = 5
TEMPLATE_SD2 = 4
AP = 1
DEFORM_SD2 = 1
EPOCHS = 1
ITERATIONS = 1
INIT_SD2 = 1
COINS = False


"""
Should be list of dictionaries
"""

# original_stdout = sys.stdout

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
    obj = const.TrainingConstants(images=images, ag=len(images), ap=ap, t_sd2=t_sd2,
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
        save.redirect_stdout_to_txt(training_output_path / "printed.txt")
        # sys.stdout = open(training_output_path / "printed.txt", "w")
    for template_path in data_path.glob('template*'):
        template_name = template_path.stem
        const_obj = make_constant_object_train(template_path=template_path)
        do_training_and_save(template_name=template_name,
                             const_object=const_obj,
                             training_output_path=training_output_path)
    if SAVE_PRINTS:
        save.bring_back_stdout()
    print("yo")
    return training_output_path


TRAIN_OUTPUT_PATH = train() if DO_TRAIN else DEFAULT_CLASSIFY_PATH


def make_constant_object_classify(input_path, training_output_path):
    if not COINS:
        images = load.load_train_images_digits(template_path=input_path)
    else:
        images = load.load_train_images_coins(template_path=input_path)
    obj = const.TrainingConstants(images=images, ag=len(images), ap=ap, t_sd2=t_sd2,
                                  d_sd2=d_sd2, init_sd=init_sd,
                                  epochs=epochs, iterations=iterations)
    return obj


def classify():
    if not COINS:
        input_path = pathlib.Path(__file__).resolve().parent / 'input_data'
    else:
        input_path = pathlib.Path(__file__).resolve().parent / 'input_coins'

    train_result_path = TRAIN_OUTPUT_PATH
    const_obj_classify = make_constant_object_classify(input_path/ 'template1')
    img_to_classify = load.load_classify_images(input_path,coins=COINS)

    image_classifier = classifier.ImageClassifier(cons_obj=const_obj_classify,
                                                  images=img_to_classify,
                                                  output=train_result_path)

    for template_path in train_result_path.glob('template*'):
        template_name = template_path.stem
        curr_template = classifier.TemplateClass(trained_template_path=None,
                                                 name=template_name)
        image_classifier.template_search(epochs=1000,
                                         template=curr_template)

    res = image_classifier.compute_and_save_results()
    save.handle_save_classification_results(train_result_path, res)

if DO_CLASSIFY:
    classify()



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
