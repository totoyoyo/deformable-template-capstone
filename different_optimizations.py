from trainers.training_many_image_est import *

# all_ops = ['Nelder-Mead',
#            'Powell',
#            'CG',
#            'BFGS',
#            'Newton-CG',
#            'L-BFGS-B',
#            'TNC',
#            'COBYLA',
#            'SLSQP',
#            'trust-constr',
#            'dogleg',
#            'trust-ncg',
#            'trust-exact',
#            'trust-krylov']

fastest = ['BFGS',
           'L-BFGS-B',
           'COBYLA',
           'SLSQP']



my_estimators = list(map(lambda method: Estimator1DNImagesMethod(method),fastest))

times = list(map(lambda estimator: estimator.remember_time(10), my_estimators))