import numpy as np
from typing import List
import functions_maker as func
import constants_maker as const
import time
import pandas as pd
import save
from pathlib import Path
import load
import scipy.stats as stat
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

class TemplateClass:

    def __init__(self, trained_template_path, name):
        self.alpha = None
        self.g_inv = None
        self.sd2 = None
        self.name = name
        self.load_data(trained_template_path)

    def load_data(self, path=Path()):
        dict_out, ginv = load.load_template(path)
        self.alpha = dict_out['alphas'].reshape(-1,1)
        self.g_inv = ginv
        self.sd2 = dict_out['sd2'].item()



import pytorch_train_classify as pt_op


class ImageClassifier:

    def __init__(self, cons_obj : const.TrainingConstants,
                 images=[loaded_image],
                 output = None):
        self.images = images
        img_names = [image['name'] for image in self.images]
        self.number_of_images = len(self.images)
        self.cons_obj = cons_obj
        self.epochs = 1000
        self.df_out = pd.DataFrame(columns=img_names)
        self.neg_probability = []
        self.classify_output_path = output
        self.save_orig_image(self.images)

    def save_orig_image(self,images):
        for image in images:
            im_name = image['name']
            im_array = image['arr']
            self.save_image(im_array, img_name=im_name,
                            template_name="orig",
                            path=self.classify_output_path)


    def template_generate(self, template: TemplateClass, iters):
        cov = func.clean_pinv(template.g_inv)
        n_kernels = np.shape(cov)[0]
        alphas = template.alpha
        sd2 = template.sd2
        m = np.zeros(n_kernels)
        template_path = self.classify_output_path / template.name
        for i in range(iters):
            name = "gen" + str(i)
            sample_beta = np.random.multivariate_normal(mean=m,cov=cov,size=2, tol=1e-5,
                                                        check_valid="ignore").T
            # dist = stat.multivariate_normal(cov=cov)
            # sample_beta = dist.rvs(size=2).T
            img = self.compute_prediction_image(betas=sample_beta, alphas=alphas)
            self.save_image(img,img_name=name,
                            template_name=template.name,
                            path=template_path)

    def template_search(self, epochs, template: TemplateClass):
        list_of_start_end_indexes = func. \
            get_list_of_indexes_for_slicing(slice_length=1,
                                            total_length=self.number_of_images)
        pytorch_constant = pt_op.PyTorchConstants(const_object=self.cons_obj)
        res = []
        npix = self.cons_obj.image_total
        to_add = (npix / 2) * np.log(2 * np.pi * template.sd2)
        # to_add = 0
        for start_end in list_of_start_end_indexes:
            start = start_end[0]
            end = start_end[1]
            curr_images = self.images[start:end]
            raw_images = [image['arr'] for image in curr_images]
            flat_images = list(map(lambda image: image.reshape(-1, 1),
                                    raw_images))
            name_images = [image['name'] for image in curr_images][0]
            start_time = time.time()
            print(f"images at {start} to {end} (exclusive)")
            optimizer = pt_op.PyTorchClassify(alphas=template.alpha,
                                              g_inv=template.g_inv,
                                              sdp2=self.cons_obj.template_sd2,
                                              sdl2=template.sd2,
                                              images=flat_images,
                                              pytorch_const=pytorch_constant)
            betas, out = optimizer.optimize_betas(epochs)
            prediction_image = self.compute_prediction_image(betas[0], template.alpha)
            self.save_image(prediction_image,img_name=name_images,
                            template_name=template.name,
                            path=self.classify_output_path)
            prob = -(to_add + out)
            res.append(prob)
            print("--- %s seconds ---" % (time.time() - start_time))
        template_name = template.name
        self.df_out.loc[template_name] = res
        self.template_generate(template=template, iters=5)

    def compute_prediction_image(self, betas, alphas):
        return self.cons_obj.kBpa(betas, alphas)

    def compute_and_save_results(self):
        out = self.df_out.transpose()
        max_template = out.idxmax(axis=1)
        large2 = out.apply(lambda row: row.nlargest(2).values[-1], axis=1)

        large1 = out.apply(lambda row: row.nlargest(1).values[-1], axis=1)

        diff = large1 - large2
        out['Diff'] = diff
        out['Max'] = max_template
        out['Truth'] = [image['true_template_name'] for image in self.images]
        out['Correct'] = out['Truth'] == out['Max']
        total_images = out.shape[0]
        total_correct = sum(out['Correct'])
        out["Rate"] = total_correct/total_images

        out.to_csv(path_or_buf=self.classify_output_path / "final_results.csv")
        return out

    def save_image(self, image, img_name, template_name, path):
        path.mkdir(parents=True, exist_ok=True)
        image_to_save = func.unflatten_image(image,self.cons_obj.image_ncol)
        image_name = img_name + "_" + template_name
        save.handle_saving_plots(path,
                                 image_to_save,
                                 image_name)
        save.handle_saving_npdata(parent_path=path,
                                  npdata=image_to_save,
                                  data_name=image_name,
                                  suffix=".classify_out")
