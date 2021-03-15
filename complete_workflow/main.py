import classify
import train
import constants_maker as const
import functions
import load
import save

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



def do_training():
    for x in templates:
        curr_constants = const.TrainingConstants()



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
