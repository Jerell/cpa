import math
import numpy as np


# Double precision,dimension(4)::fBi,fDi,Xa !Parametros de la ecuacion, estan en pagina 290 del libro


def Fugg(Aij, B, Bij, D, n, y, V, P, T, Xa):
    """
    parameters:

        Double precision,dimension(4,4),intent(in)::Aij,Bij

        Double precision,intent(in)::n,P,T

        Double precision,intent(in)::A,B,D,V

        Double precision,dimension(4),intent(in)::y,ai,bi

    output:
        Fu: NDArray,
        FCPA: NDArray,
        FSRK: NDArray,
        phi: NDArray,
        lnp: NDArray,
        Z: float64
    """

    arbi = np.array([sum(n * y[i] * Bij[i, :]) for i in range(4)])
    print(arbi)
    fBi = np.array([(2 * arbi[i] - B) / n for i in range(4)])
    fDi = np.array([sum(2 * y[i] * n * Aij[i, :]) for i in range(4)])

    g = math.log(1 + (B / V))
    FF = (1 / (8.314 * B)) * (math.log(1 + (B / V)))
    Fd = (-1 / T) * FF
    fv = -(1 / 8.314) * (1 / (V * (V + B)))
    fbm = (-1 / B) * (FF + V * fv)
    gb = -1 / (V - B)
    Fb = -n * gb - (D / T) * fbm
    Fn = -g

    FSRK = np.array([Fn + Fb * fDi[i] + Fd * fBi[i] for i in range(4)])
    dlng = [
        0.475 * V * ((1 / (V - 0.475 * B)) ** 2) * fBi[i] * (1 - 1.9 * (B / (4 * V)))
        for i in range(4)
    ]
    FCPA = np.array([np.log(Xa) - 0.5 * (y * n) * (1 - Xa) * dlng[i] for i in range(4)])
    Z = 1e5 * P * V / (n * 8.314 * T)
    lnp = np.array([1e-6 * (FCPA[i] + FSRK[i]) / (8.314 * T) for i in range(4)])
    phi = np.array([np.exp(lnp[i]) for i in range(4)])
    Fu = np.array([P * y[i] * phi[i] / Z for i in range(4)])

    return Fu, FCPA, FSRK, phi, lnp, Z
