import classify
import trainer
import constants_maker as const
import functions
import load
import save
import pathlib
### do some training

DO_TRAIN = True
DO_CLASSIFY = True
AG = None
TEMPLATE_SD2 = None
AP = None
DEFORM_SD2 = None
EPOCHS = None
ITERATIONS = None
INIT_SD2 = None

templates = 0
"""
Should be list of dictionaries
"""


def do_training_and_save(template_name, const_object):
    main_path = pathlib.Path(__file__).resolve().parent
    training_output_path = save.handle_duplicate_names(main_path, "train_output")
    to_train = trainer.Estimator2DNImages(cons_obj=const_object,
                                          template_name=template_name,
                                          training_output_path=training_output_path)
    to_train.run_estimation(const_object.iterations)
    to_train.save_all()


def make_constant_object(template_path, ag=AG, ap=AP, t_sd2=TEMPLATE_SD2,
                         d_sd2=DEFORM_SD2, init_sd=INIT_SD2, epochs=EPOCHS,
                         iterations=ITERATIONS):
    images = load.load_train_images_digits(template_path=template_path)
    obj = const.TrainingConstants(images=images, ag=ag, ap=ap, t_sd2=t_sd2,
                                  d_sd2=d_sd2, init_sd=init_sd,
                                  epochs=epochs,iterations=iterations)
    return obj



def train():
    data_path = pathlib.Path(__file__).resolve().parent / 'input_data'
    for template_path in data_path.glob('template*'):
        template_name = template_path.stem
        const_obj = make_constant_object(template_path=template_path)
        do_training_and_save(template_name=template_name,
                             const_object=const_obj)


train()

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


### do some classification

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
