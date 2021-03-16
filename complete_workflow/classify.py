import numpy as np
from typing import List
import functions_maker as func
import pytorch_train_classify as pt_op

"""
Takes a list of images:

returns

"""

loaded_image = {
    "name": "something",
    "arr": np.array([]),
    "true_template_name": "something"
}

batch_size = 2


class TemplateClass:

    def __init__(self):
        self.alpha = None
        self.betas = None
        self.g_inv = None
        self.name = None


class ImageClassifier:
    """

    """

    def __init__(self, images, template: TemplateClass):
        self.images = [loaded_image]
        self.templates = template

    def template_search(self, epochs, template: TemplateClass):
        list_of_start_end_indexes = func. \
            get_list_of_indexes_for_slicing(batch_size,
                                            self.number_of_images)
        pytorch_constant = pt_op.PyTorchConstants(const_object=self.cons_obj)
        for start_end in list_of_start_end_indexes:
            start = start_end[0]
            end = start_end[1]
            curr_beta = self.betas[start:end]
            curr_images = self.images[start:end]
            start_time = time.time()
            print(f"beta at {start} to {end} (exclusive)")
            optimizer = pt_op.PyTorchOptimizer(alphas=self.alphas,
                                               curr_beta=curr_beta,
                                               g_inv=dense_gamma_inv,
                                               sdp2=self.cons_obj.template_sd2,
                                               sdl2=self.sd2,
                                               images=curr_images,
                                               pytorch_const=pytorch_constant)
            out = optimizer.optimize_betas(self.epochs)
            self.betas[start:end] = out
            print("--- %s seconds ---" % (time.time() - start_time))
