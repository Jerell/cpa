import math
import numpy as np


def Vol(n, T, P, D, B, Xa):
    """
    parameters:

        Double precision, intent(in)::n,T,P,D,B

        Double precision,Dimension(4),intent(in)::Xa

    output:

    """
    tol = 1
    Vast = n * 8.314 * T / (P * 1e5)

    Pcz = 0
    Vest = 0

    i = 1
    while i < 1000:
        Ppa = P
        tol = np.abs(
            Pca(n, B, D, T, Vast, Xa) - Ppa
        )  # Primero tolerancia para saber si el while se hace

        Hv = P - Pca(n, B, D, T, Vast, Xa)  # Definimos FO
        dHv = (
            -Pca(n, B, D, T, (Vast + 1e-6), Xa) + Pca(n, B, D, T, (Vast - 1e-6), Xa)
        ) / 2e-6  # Su derivada numerica

        Vast = Vast - (Hv / dHv)  # *1E-5 !Metodo de Newton

        i += 1

        Vest = Vast * 1e-6
        Pcz = Pca(n, B, D, T, Vast, Xa)
    return Vest, Pcz, tol


def Pca(n, B, D, T, V, Xa):
    """
    parameters:

        Double precision, intent(in)::n,T,P,D,B

        Double precision,Dimension(4),intent(in)::Xa
    """
    return (
        83.14
        * T
        * (
            (n / V)
            - (
                -((n * B) / (V * (V - B)))
                + ((D / T) / (83.14 * V * (V + B)))
                + (
                    (1 / (2 * V))
                    * (1 + (0.475 * B) / ((V - 0.475 * B) ** 2))
                    * (n * np.sum(1 - Xa))
                )
            )
        )
    )
