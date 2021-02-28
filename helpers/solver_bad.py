"""
Sample code automatically generated on 2021-02-28 15:38:58

by www.matrixcalculus.org

from input

d/dB (1/(2*sdl2)) * norm2(image - (exp( (-1/(2 * sdp2)) * ((((P - (K * B)).^2) * M1) + ((Ca.^2) * M2)' - (2 * (P - (K * B)) * Ca')))) * A)^2 = 1/(sdp2*sdl2)*K'*diag(image-exp(-1/(2*sdp2)*((P-K*B).^2*M1+(Ca.^2*M2)'-2*(P-K*B)*Ca'))*A)*exp(-1/(2*sdp2)*((P-K*B).^2*M1+(Ca.^2*M2)'-2*(P-K*B)*Ca'))*diag(A)*Ca-1/(sdp2*sdl2)*K'*diag(image-exp(-1/(2*sdp2)*((P-K*B).^2*M1+(Ca.^2*M2)'-2*(P-K*B)*Ca'))*A)*((exp(-1/(2*sdp2)*((P-K*B).^2*M1+(Ca.^2*M2)'-2*(P-K*B)*Ca'))*diag(A)*M1').*(P-K*B))

where

A is a vector
B is a matrix
Ca is a matrix
K is a matrix
M1 is a matrix
M2 is a matrix
P is a matrix
image is a vector
sdl2 is a scalar
sdp2 is a scalar

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(A, B, Ca, K, M1, M2, P, image, sdl2, sdp2):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 1
    A_rows = dim[0]
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert isinstance(Ca, np.ndarray)
    dim = Ca.shape
    assert len(dim) == 2
    Ca_rows = dim[0]
    Ca_cols = dim[1]
    assert isinstance(K, np.ndarray)
    dim = K.shape
    assert len(dim) == 2
    K_rows = dim[0]
    K_cols = dim[1]
    assert isinstance(M1, np.ndarray)
    dim = M1.shape
    assert len(dim) == 2
    M1_rows = dim[0]
    M1_cols = dim[1]
    assert isinstance(M2, np.ndarray)
    dim = M2.shape
    assert len(dim) == 2
    M2_rows = dim[0]
    M2_cols = dim[1]
    assert isinstance(P, np.ndarray)
    dim = P.shape
    assert len(dim) == 2
    P_rows = dim[0]
    P_cols = dim[1]
    assert isinstance(image, np.ndarray)
    dim = image.shape
    assert len(dim) == 1
    image_rows = dim[0]
    if isinstance(sdl2, np.ndarray):
        dim = sdl2.shape
        assert dim == (1, )
    if isinstance(sdp2, np.ndarray):
        dim = sdp2.shape
        assert dim == (1, )
    assert P_rows == M2_cols == image_rows == K_rows
    assert Ca_cols == P_cols == M1_rows == B_cols
    assert K_cols == B_rows
    assert A_rows == Ca_rows == M1_cols
    assert Ca_cols == P_cols == B_cols == M2_rows
    assert Ca_cols == P_cols == M1_rows == B_cols == M2_rows

    T_0 = (P - (K).dot(B))
    T_1 = np.exp(-((1 / (2 * sdp2)) * ((((T_0 ** 2)).dot(M1) + ((Ca ** 2)).dot(M2).T) - (2 * (T_0).dot(Ca.T)))))
    t_2 = (image - (T_1).dot(A))
    t_3 = (1 / (sdp2 * sdl2))
    functionValue = ((np.linalg.norm(t_2) ** 2) / (2 * sdl2))
    gradient = ((t_3 * (((K.T).dot((t_2[:, np.newaxis] * T_1)) * A[np.newaxis, :])).dot(Ca)) - (t_3 * (K.T).dot((t_2[:, np.newaxis] * (((T_1 * A[np.newaxis, :])).dot(M1.T) * T_0)))))

    return functionValue, gradient

def checkGradient(A, B, Ca, K, M1, M2, P, image, sdl2, sdp2):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3, 3)
    f1, _ = fAndG(A, B + t * delta, Ca, K, M1, M2, P, image, sdl2, sdp2)
    f2, _ = fAndG(A, B - t * delta, Ca, K, M1, M2, P, image, sdl2, sdp2)
    f, g = fAndG(A, B, Ca, K, M1, M2, P, image, sdl2, sdp2)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    A = np.random.randn(3)
    B = np.random.randn(3, 3)
    Ca = np.random.randn(3, 3)
    K = np.random.randn(3, 3)
    M1 = np.random.randn(3, 3)
    M2 = np.random.randn(3, 3)
    P = np.random.randn(3, 3)
    image = np.random.randn(3)
    sdl2 = np.random.randn(1)
    sdp2 = np.random.randn(1)

    return A, B, Ca, K, M1, M2, P, image, sdl2, sdp2

if __name__ == '__main__':
    A, B, Ca, K, M1, M2, P, image, sdl2, sdp2 = generateRandomData()
    functionValue, gradient = fAndG(A, B, Ca, K, M1, M2, P, image, sdl2, sdp2)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A, B, Ca, K, M1, M2, P, image, sdl2, sdp2)
