import numpy as np

c = np.array([[2,-1,0],
              [-1,2,-1],
              [0,-1,2]])

m = np.zeros(3)

out = np.random.multivariate_normal(m, cov=c, size=2).T