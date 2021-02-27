"""
Sample code automatically generated on 2021-02-27 14:02:42

by geno from www.geno-project.org

from input

parameters
  matrix Ginv
  matrix P
  matrix K
  matrix Ca
  vector image
  vector y
  vector oneCOL2
  vector oneROWKp
  vector oneROWL
  vector A
  scalar sdl2
  scalar sdp2
variables
  scalar B
min
  1/2*tr(B*B*Ginv)+1/(2*sdl2)*norm2(image-exp(((P-B*K).^2*oneCOL2*oneROWKp'+(Ca.^2*oneCOL2*oneROWL')'-2*(P-B*K)*Ca')/(2*sdp2))*A).^2


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

class GenoNLP:
    def __init__(self, Ginv, P, K, Ca, image, y, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2):
        self.Ginv = Ginv
        self.P = P
        self.K = K
        self.Ca = Ca
        self.image = image
        self.y = y
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
        assert isinstance(y, np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
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
        self.B_rows = 1
        self.B_cols = 1
        self.B_size = self.B_rows * self.B_cols
        # the following dim assertions need to hold for this problem
        assert self.Ginv_rows == self.Ginv_cols
        assert self.oneROWL_rows == self.K_rows == self.P_rows == self.image_rows
        assert self.A_rows == self.Ca_rows == self.oneROWKp_rows
        assert self.Ca_cols == self.K_cols == self.P_cols == self.oneCOL2_rows

    def getBounds(self):
        bounds = []
        bounds += [(-inf, inf)] * self.B_size
        return bounds

    def getStartingPoint(self):
        self.BInit = np.random.randn(self.B_rows, self.B_cols)
        return self.BInit.reshape(-1)

    def variables(self, _x):
        B = _x
        return B

    def fAndG(self, _x):
        B = self.variables(_x)
        T_0 = (self.P - (B * self.K))
        t_1 = np.trace(self.Ginv)
        t_2 = (1 / (2 * self.sdp2))
        t_3 = ((T_0 ** 2)).dot(self.oneCOL2)
        t_4 = ((self.Ca ** 2)).dot(self.oneCOL2)
        T_5 = np.exp((t_2 * ((np.multiply.outer(self.oneROWKp, t_3) + np.multiply.outer(t_4, self.oneROWL)) - (2 * (self.Ca).dot(T_0.T)))))
        t_6 = (self.image - (self.A).dot(T_5))
        t_7 = ((4 * self.sdp2) * self.sdl2)
        f_ = ((((B ** 2) * t_1) / 2) + ((np.linalg.norm((self.image - (np.exp((t_2 * ((np.multiply.outer(t_3, self.oneROWKp) + np.multiply.outer(self.oneROWL, t_4)) - (2 * (T_0).dot(self.Ca.T)))))).dot(self.A))) ** 2) / (2 * self.sdl2)))
        g_0 = ((B * t_1) - (((4 * np.trace(((self.K).dot(self.Ca.T)).dot(((self.A[:, np.newaxis] * T_5) * t_6[np.newaxis, :])))) / t_7) - ((4 * np.trace((self.K).dot(((self.oneCOL2[:, np.newaxis] * T_0.T) * (t_6 * ((self.oneROWKp * self.A)).dot(T_5))[np.newaxis, :])))) / t_7)))
        g_ = g_0
        return f_, g_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(Ginv, P, K, Ca, image, y, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2):
    start = timer()
    NLP = GenoNLP(Ginv, P, K, Ca, image, y, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2)
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
    y = np.random.randn(3)
    oneCOL2 = np.random.randn(3)
    oneROWKp = np.random.randn(3)
    oneROWL = np.random.randn(3)
    A = np.random.randn(3)
    sdl2 = np.random.randn(1)
    sdp2 = np.random.randn(1)
    return Ginv, P, K, Ca, image, y, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2

if __name__ == '__main__':
    print('\ngenerating random instance')
    Ginv, P, K, Ca, image, y, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2 = generateRandomData()
    print('solving ...')
    solution = solve(Ginv, P, K, Ca, image, y, oneCOL2, oneROWKp, oneROWL, A, sdl2, sdp2)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        print('optimal variable B = ', solution['B'])
        print('solving took %.3f sec' % solution['elapsed'])
