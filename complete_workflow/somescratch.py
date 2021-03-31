import numpy as np

c = np.array([[2,-1,0],
              [-1,2,-1],
              [0,-1,2]])

m = np.zeros(3)

out1 = np.random.multivariate_normal(m, cov=c, size=2).T

import scipy.stats as stat
dist = stat.multivariate_normal(cov=c)
out2 = dist.rvs(size=2).T