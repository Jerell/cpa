import math
import numpy as np

# [CH4, CO, CO2, N2]

# critical temperature
Tc = np.array([190.56, 132.85, 304.12, 126.2])
# critical pressure
Pc = np.array([45.992, 34.94, 73.74, 33.98])
w = np.array([0.011, 0.045, 0.225, 0.037])
na = np.array([15.68, 9.62, 34.04, 0.00])
ma = np.array([15.68, 9.62, 34.04, 0.00])
ba = np.array([6.48, 66.2, 14.04, 0.000928699])
da = np.array([0.025, 0.03, 0.061, 0.00])
ha = np.array([30.56, 48.16, 34.67, 30.0])
ua = np.array([13.45, 14.84, 20.79, 0.0])
ka = np.array([1.0, 1.0, 1.0, 1.13])

c0 = 0.48
c1 = 1.574
c2 = -0.176
eps = 1
sig = 0

omega = 0.08664035
psi = 0.42748023


def CA(y, T, n):
    """
    parameters:
        Double precision,dimension(4),intent(in)::y

        Double precision,intent(in)::T,n

    output:
        At: float64
        D: float64
        ap: NDArray
        aij: NDArray[float64]
    """
    ap = np.array(
        [
            (psi * (83.14 * 83.14 * Tc[i] * Tc[i]) / Pc[i])
            * (1 + ((c0 + c1 * w[i] + c2 * w[i] * w[i]) * (1 - math.sqrt(Tc[i] / T))))
            ** 2
            for i in range(4)
        ]
    )

    aij = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            aij[i, j] = y[i] * y[j] * math.sqrt(ap[i] * ap[j])

    At = np.sum(aij)
    D = At * n * n

    return At, D, ap, aij


def CB(n, y):
    """
    parameters:
        Double precision, intent(in)::n

        Double precision,dimension(4),intent(in)::y

    output:
        bi: NDArray,
        Bt: float64,
        B: float64,
        bij: NDArray
    """
    bi = np.array([omega * 83.14 * Tc[i] / Pc[i] for i in range(4)])

    bij = np.array([[0.5 * (bi[i] + bi[j]) for j in range(4)] for i in range(4)])

    Bt = np.sum(bi * y)
    B = n * Bt
    return bi, Bt, B, bij
