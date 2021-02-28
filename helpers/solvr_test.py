"""
Sample code automatically generated on 2021-02-28 17:53:24

by www.matrixcalculus.org

from input

d/dx A*exp(x) = A*diag(exp(x))

where

A is a symmetric matrix
x is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(A, x):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(x, np.ndarray)
    dim = x.shape
    assert len(dim) == 1
    x_rows = dim[0]
    assert A_cols == x_rows

    t_0 = np.exp(x)
    functionValue = (A).dot(t_0)
    gradient = (A * t_0[np.newaxis, :])

    return functionValue, gradient

def checkGradient(A, x):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3)
    f1, _ = fAndG(A, x + t * delta)
    f2, _ = fAndG(A, x - t * delta)
    f, g = fAndG(A, x)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData():
    A = np.random.randn(3, 3)
    A = 0.5 * (A + A.T)  # make it symmetric
    x = np.random.randn(3)

    return A, x

if __name__ == '__main__':
    A, x = generateRandomData()
    functionValue, gradient = fAndG(A, x)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A, x)
