import numpy as np
import math

import matriz
import volstab

beta = np.zeros(4)
epsi = np.zeros(4)


def Solu(P, T, n, B, D, y):
    """
    paramaters:

        Double precision,intent(in)::B,A,D,n,P,T

        Double precision,dimension(4),intent(in)::bi,ai,y

        Double precision,dimension(4,4),intent(in)::bij,aij

    output:

    """
    Delta = np.empty((4, 4))
    bij = np.empty((4, 4))
    aij = np.empty((4, 4))

    Xa = np.ones(4)
    dXa = np.empty(4)
    gXa = np.empty(4)

    i = 1  # iter
    j = 1
    bep = math.sqrt(sum(beta + epsi) ** 2)

    Jacx = np.empty((4, 5))
    Vest, Pcz, tol = volstab.Vol(n, T, P, D, B, Xa)

    V = -1

    if bep == 0:
        Xa = np.ones(4)
        V = Vest
    else:
        while i < 1000:
            Delta = Par(P, T, n, y, B, bij)
            V = Vest
            for i in range(4):
                gXa[i] = 1 + (n / V) * np.sum(Xa[i] * Delta[i, :]) - (1 / Xa[i])
            for im in range(4):  # Definicion matriz cuya determinante es el Jacobiano
                for jm in range(4):
                    if i == j:
                        Jacx[im, jm] = (
                            (n / V) * (Xa[i] + 1e-6) * Delta[im, jm]
                            - (1 / (Xa[im] + 1e-6))
                            - (n / V) * (Xa[im] - 1e-6) * Delta[im, jm]
                            + (1 / (Xa[im] - 1e-6))
                        ) / 2e-6
                    else:
                        Jacx[im, jm] = (
                            (n / V) * (Xa[im] + 1e-6) * Delta[im, jm]
                            - (n / V) * (Xa[im] - 1e-6) * Delta[im, jm]
                        ) / 2e-6
                # Vamos a intentar definir primero la parte de las variables de la matriz primero y luego añadir las soluciones
                # como array=Jacx(:,5) para meterlo a gsselm
                Jacx[:, 5] = -gXa
                SINGULAR, X = matriz.Gsselm(Jacx, 4, dXa)
                opt = math.sqrt(np.sum(dXa**2))
                Xa = Xa + dXa
                i += 1
    return Xa, dXa, V


def Par(P, T, Vnec, n, B, bij):
    """
    parameters:
        Double precision,intent(in)::P,T,Vnec,B,n

        Double precision,dimension(4,4),intent(in)::bij

        Double precision,dimension(4),intent(in)::y

    output:

        Double precision,dimension(4,4),intent(out)::Delta
    """
    eta = 1.9 * B * n / (4 * Vnec)
    Delta = np.empty((4, 4))

    betm = np.empty((4, 4))
    epsim = np.empty((4, 4))

    for i in range(4):
        for j in range(4):
            epsim[i, j] = 0.5 * (epsi[i] + epsi[j])
            betm[i, j] = math.sqrt(beta[i] * beta[j])

    for i in range(4):
        for j in range(4):
            Delta[i, j] = (
                math.exp(epsim[i, j] / (8.314 * T)) * bij[i, j] * betm[i, j] / (1 - eta)
            )
    return Delta
