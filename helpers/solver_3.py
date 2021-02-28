"""
Sample code automatically generated on 2021-02-28 17:44:07

by www.matrixcalculus.org

from input

d/dB ((1/2) * tr(B' * Ginv  * B) + (1/(2*sdl2)) * norm2(image - (exp(((((P - (K * B)).^2) * oneCOL2 * oneROWKp') + ((Ca.^2) * oneCOL2 * oneROWL')' - (2 * (P - (K * B)) * Ca')) / (-2 * sdp2))) * A)^2) = Ginv*B+1/(sdp2*sdl2)*K'*diag(image-exp(-1/(2*sdp2)*((P-K*B).^2*oneCOL2*oneROWKp'+oneROWL*(Ca.^2*oneCOL2)'-2*(P-K*B)*Ca'))*A)*exp(-1/(2*sdp2)*((P-K*B).^2*oneCOL2*oneROWKp'+oneROWL*(Ca.^2*oneCOL2)'-2*(P-K*B)*Ca'))*diag(A)*Ca-1/(sdp2*sdl2)*K'*diag((image-exp(-1/(2*sdp2)*((P-K*B).^2*oneCOL2*oneROWKp'+oneROWL*(Ca.^2*oneCOL2)'-2*(P-K*B)*Ca'))*A).*(exp(-1/(2*sdp2)*((P-K*B).^2*oneCOL2*oneROWKp'+oneROWL*(Ca.^2*oneCOL2)'-2*(P-K*B)*Ca'))*(A.*oneROWKp)))*(P-K*B)*diag(oneCOL2)

where

A is a vector
B is a matrix
Ca is a matrix
Ginv is a symmetric matrix
K is a matrix
P is a matrix
image is a vector
oneCOL2 is a vector
oneROWKp is a vector
oneROWL is a vector
sdl2 is a scalar
sdp2 is a scalar

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(A, B, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2):
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
    assert isinstance(Ginv, np.ndarray)
    dim = Ginv.shape
    assert len(dim) == 2
    Ginv_rows = dim[0]
    Ginv_cols = dim[1]
    assert isinstance(K, np.ndarray)
    dim = K.shape
    assert len(dim) == 2
    K_rows = dim[0]
    K_cols = dim[1]
    assert isinstance(P, np.ndarray)
    dim = P.shape
    assert len(dim) == 2
    P_rows = dim[0]
    P_cols = dim[1]
    assert isinstance(image, np.ndarray)
    dim = image.shape
    assert len(dim) == 1
    image_rows = dim[0]
    assert isinstance(oneCOL2, np.ndarray)
    dim = oneCOL2.shape
    assert len(dim) == 1
    oneCOL2_rows = dim[0]
    assert isinstance(oneROWKp, np.ndarray)
    dim = oneROWKp.shape
    assert len(dim) == 1
    oneROWKp_rows = dim[0]
    assert isinstance(oneROWL, np.ndarray)
    dim = oneROWL.shape
    assert len(dim) == 1
    oneROWL_rows = dim[0]
    if isinstance(sdl2, np.ndarray):
        dim = sdl2.shape
        assert dim == (1, )
    if isinstance(sdp2, np.ndarray):
        dim = sdp2.shape
        assert dim == (1, )
    assert Ginv_rows == B_rows == K_cols == Ginv_cols
    assert P_cols == B_cols == oneCOL2_rows == Ca_cols
    assert image_rows == K_rows == oneROWL_rows == P_rows
    assert Ca_rows == oneROWKp_rows == A_rows

    T_0 = (P - (K).dot(B))
    T_1 = np.exp(-((1 / (2 * sdp2)) * ((np.multiply.outer(((T_0 ** 2)).dot(oneCOL2), oneROWKp) + np.multiply.outer(oneROWL, ((Ca ** 2)).dot(oneCOL2))) - (2 * (T_0).dot(Ca.T)))))
    t_2 = (image - (T_1).dot(A))
    t_3 = (1 / (sdp2 * sdl2))
    functionValue = ((np.trace(((B.T).dot(Ginv)).dot(B)) / 2) + ((np.linalg.norm(t_2) ** 2) / (2 * sdl2)))
    gradient = (((Ginv).dot(B) + (t_3 * (((K.T).dot((t_2[:, np.newaxis] * T_1)) * A[np.newaxis, :])).dot(Ca))) - (t_3 * ((K.T).dot(((t_2 * (T_1).dot((A * oneROWKp)))[:, np.newaxis] * T_0)) * oneCOL2[np.newaxis, :])))

    return functionValue, gradient

def checkGradient(A, B, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-10
    delta = np.random.randn(3, 3)
    f1, _ = fAndG(A, B + t * delta, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2)
    f2, _ = fAndG(A, B - t * delta, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2)
    f, g = fAndG(A, B, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    A = np.random.randn(3)
    B = np.random.randn(3, 3)
    Ca = np.random.randn(3, 3)
    Ginv = np.random.randn(3, 3)
    Ginv = 0.5 * (Ginv + Ginv.T)  # make it symmetric
    K = np.random.randn(3, 3)
    P = np.random.randn(3, 3)
    image = np.random.randn(3)
    oneCOL2 = np.random.randn(3)
    oneROWKp = np.random.randn(3)
    oneROWL = np.random.randn(3)
    sdl2 = np.random.randn(1)
    sdp2 = np.random.randn(1)

    return A, B, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2

if __name__ == '__main__':
    A, B, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2 = generateRandomData()
    functionValue, gradient = fAndG(A, B, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A, B, Ca, Ginv, K, P, image, oneCOL2, oneROWKp, oneROWL, sdl2, sdp2)
