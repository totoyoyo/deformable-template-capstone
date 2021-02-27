"""
Sample code automatically generated on 2021-02-27 18:17:25

by geno from www.geno-project.org

from input

parameters
  matrix Ginv
  matrix P
  matrix K
  matrix Ca
  vector image
  vector oneCOL2
  vector oneROWKp
  vector oneROWL
  vector A
  scalar sdl2
  scalar sdp2
variables
  matrix B
min
  1/2*tr(B'*Ginv*B)+1/(2*sdl2)*norm2(image-exp(((P-K*B).^2*oneCOL2*oneROWKp'+(Ca.^2*oneCOL2*oneROWL')'-2*(P-K*B)*Ca')/((-2)*sdp2))*A).^2


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
import numpy as np


try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)

# from scipy.optimize import minimize
# USE_GENO_SOLVER = False

class GenoNLP:
    def __init__(self, Ginv, P, K, Ca, image, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2):
        self.Ginv = Ginv
        self.P = P
        self.K = K
        self.Ca = Ca
        self.image = image
        self.oneCOL2 = oneCOL2
        self.oneROWKp = oneROWKp
        self.oneROWL = oneROWL
        self.A = A
        self.sdl2 = sdl2
        self.sdp2 = sdp2
        assert isinstance(Ginv, np.ndarray)
        dim = Ginv.shape
        assert len(dim) == 2
        self.Ginv_rows = dim[0]
        self.Ginv_cols = dim[1]
        assert isinstance(P, np.ndarray)
        dim = P.shape
        assert len(dim) == 2
        self.P_rows = dim[0]
        self.P_cols = dim[1]
        assert isinstance(K, np.ndarray)
        dim = K.shape
        assert len(dim) == 2
        self.K_rows = dim[0]
        self.K_cols = dim[1]
        assert isinstance(Ca, np.ndarray)
        dim = Ca.shape
        assert len(dim) == 2
        self.Ca_rows = dim[0]
        self.Ca_cols = dim[1]
        assert isinstance(image, np.ndarray)
        dim = image.shape
        assert len(dim) == 1
        self.image_rows = dim[0]
        self.image_cols = 1
        assert isinstance(oneCOL2, np.ndarray)
        dim = oneCOL2.shape
        assert len(dim) == 1
        self.oneCOL2_rows = dim[0]
        self.oneCOL2_cols = 1
        assert isinstance(oneROWKp, np.ndarray)
        dim = oneROWKp.shape
        assert len(dim) == 1
        self.oneROWKp_rows = dim[0]
        self.oneROWKp_cols = 1
        assert isinstance(oneROWL, np.ndarray)
        dim = oneROWL.shape
        assert len(dim) == 1
        self.oneROWL_rows = dim[0]
        self.oneROWL_cols = 1
        assert isinstance(A, np.ndarray)
        dim = A.shape
        assert len(dim) == 1
        self.A_rows = dim[0]
        self.A_cols = 1
        if isinstance(sdl2, np.ndarray):
            dim = sdl2.shape
            assert dim == (1, )
        self.sdl2_rows = 1
        self.sdl2_cols = 1
        if isinstance(sdp2, np.ndarray):
            dim = sdp2.shape
            assert dim == (1, )
        self.sdp2_rows = 1
        self.sdp2_cols = 1
        self.B_rows = self.Ginv_rows
        self.B_cols = self.P_cols
        self.B_size = self.B_rows * self.B_cols
        # the following dim assertions need to hold for this problem
        assert self.oneROWL_rows == self.K_rows == self.P_rows == self.image_rows
        assert self.A_rows == self.Ca_rows == self.oneROWKp_rows
        assert self.Ca_cols == self.B_cols == self.P_cols == self.oneCOL2_rows
        assert self.K_cols == self.B_rows == self.Ginv_cols == self.Ginv_rows

    def getBounds(self):
        bounds = []
        bounds += [(-inf, inf)] * self.B_size
        return bounds

    def getStartingPoint(self):
        self.BInit = np.random.randn(self.B_rows, self.B_cols)
        return self.BInit.reshape(-1)

    def variables(self, _x):
        B = _x
        B = B.reshape(self.B_rows, self.B_cols)
        return B

    def fAndG(self, _x):
        B = self.variables(_x)
        T_0 = (self.P - (self.K).dot(B))
        t_1 = (1 / 2)
        T_2 = np.exp(-((1 / (2 * self.sdp2)) * ((np.multiply.outer(((T_0 ** 2)).dot(self.oneCOL2), self.oneROWKp) + np.multiply.outer(self.oneROWL, ((self.Ca ** 2)).dot(self.oneCOL2))) - (2 * (T_0).dot(self.Ca.T)))))
        t_3 = (self.image - (T_2).dot(self.A))
        t_4 = (4 / ((4 * self.sdp2) * self.sdl2))
        f_ = ((np.trace(((B.T).dot(self.Ginv)).dot(B)) / 2) + ((np.linalg.norm(t_3) ** 2) / (2 * self.sdl2)))
        g_0 = (((t_1 * (self.Ginv).dot(B)) + (t_1 * (self.Ginv.T).dot(B))) + ((t_4 * (((self.K.T).dot((t_3[:, np.newaxis] * T_2)) * self.A[np.newaxis, :])).dot(self.Ca)) - (t_4 * ((self.K.T).dot(((t_3 * (T_2).dot((self.A * self.oneROWKp)))[:, np.newaxis] * T_0)) * self.oneCOL2[np.newaxis, :]))))
        g_ = g_0.reshape(-1)
        return f_, g_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(Ginv, P, K, Ca, image, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2):
    start = timer()
    NLP = GenoNLP(Ginv, P, K, Ca, image, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2)
    x0 = NLP.getStartingPoint()
    bnds = NLP.getBounds()
    tol = 1E-6
    # These are the standard GENO solver options, they can be omitted.
    options = {'tol' : tol,
               'constraintsTol' : 1E-4,
               'maxiter' : 1000,
               'verbosity' : 1  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.0.3')
        result = minimize(NLP.fAndG, x0,
                          bounds=bnds, options=options)
    else:
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=bnds)

    # assemble solution and map back to original problem
    x = result.x
    B = NLP.variables(x)
    solution = {}
    solution['success'] = result.success
    solution['message'] = result.message
    solution['fun'] = result.fun
    solution['grad'] = result.jac
    solution['B'] = B
    solution['elapsed'] = timer() - start
    return solution

def generateRandomData():
    np.random.seed(0)
    Ginv = np.random.randn(3, 3)
    P = np.random.randn(3, 3)
    K = np.random.randn(3, 3)
    Ca = np.random.randn(3, 3)
    image = np.random.randn(3)
    oneCOL2 = np.random.randn(3)
    oneROWKp = np.random.randn(3)
    oneROWL = np.random.randn(3)
    A = np.random.randn(3)
    sdl2 = np.random.randn(1)
    sdp2 = np.random.randn(1)
    return Ginv, P, K, Ca, image, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2

if __name__ == '__main__':
    print('\ngenerating random instance')
    Ginv, P, K, Ca, image, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2 = generateRandomData()
    print('solving ...')
    solution = solve(Ginv, P, K, Ca, image, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable B = ', solution['B'])
        print('solving took %.3f sec' % solution['elapsed'])
