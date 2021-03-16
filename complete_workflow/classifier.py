import numpy as np
from typing import List
import functions_maker as func
import constants_maker as const
import time
import pandas as pd

"""
Takes a list of images:

returns

"""

loaded_image = {
    "name": "something",
    "arr": np.array([]),
    "true_template_name": "something"
}

#KEEP THIS AT 1!!
#WE NEED TO KEEP TRACK OF LOSS FOR EACH IMAGE
batch_size = 1

class TemplateClass:

    def __init__(self):
        self.alpha = None
        self.betas = None
        self.g_inv = None
        self.sd2 = None
        self.name = None


import pytorch_train_classify as pt_op


class ImageClassifier:

    def __init__(self, cons_obj : const.TrainingConstants,
                 images=[loaded_image]):
        self.images = images
        img_names = [image['name'] for image in self.images]
        self.number_of_images = len(self.images)
        self.cons_obj = cons_obj
        self.epochs = 1000
        self.df_out = pd.DataFrame(columns=img_names)
        self.neg_probability = []

    def template_search(self, epochs, template: TemplateClass):
        list_of_start_end_indexes = func. \
            get_list_of_indexes_for_slicing(batch_size,
                                            self.number_of_images)
        pytorch_constant = pt_op.PyTorchConstants(const_object=self.cons_obj)
        res = []
        npix = self.cons_obj.all_pixels
        to_add = (npix / 2) * np.log(2 * np.pi * (template.sd2))
        for start_end in list_of_start_end_indexes:
            start = start_end[0]
            end = start_end[1]
            curr_images = self.images[start:end]
            raw_images = [image['arr'] for image in curr_images]
            start_time = time.time()
            print(f"images at {start} to {end} (exclusive)")
            optimizer = pt_op.PyTorchClassify(alphas=template.alpha,
                                              g_inv=template.g_inv,
                                              sdp2=self.cons_obj.template_sd2,
                                              sdl2=template.sd2,
                                              images=raw_images,
                                              pytorch_const=pytorch_constant)
            out = optimizer.optimize_betas(epochs)
            prob = -(to_add + out)
            res.append(prob)
            print("--- %s seconds ---" % (time.time() - start_time))
        template_name = template.name
        self.df_out.loc[template_name] = res
