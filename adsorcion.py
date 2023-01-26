import numpy as np
import constants
import fugac
import volstab
import CPA


def datos():
    T = float(input("Temperature of the system (Kelvin)"))
    P = float(input("Pressure (Bar)"))

    y = np.array([])
    y[0] = float(input("Mole fraction CH4"))
    y[1] = float(input("Mole fraction CO"))
    y[2] = float(input("Mole fraction CO2"))
    y[3] = float(input("Mole fraction N2"))

    n = float(input("Total moles"))

    At, D, ap, aij = constants.CA(y, T, n)
    bi, Bt, B, bij = constants.CB(n, y)

    print("Constants At, D, Bt, B")
    print(At, D, Bt, B)

    Xb = np.array([1, 1, 1, 1])
    Vest, Pcz, tol = volstab.Vol(n, T, P, D, B, Xb)
    print("P calculated, error, V SRK")
    print(Pcz, tol, Vest)

    # Solu
    Xa, dXa, V = CPA.Solu(1.0, 1.0, 1.0, 1.0, 1.0, np.array([1.0, 1.0, 1.0, 1.0]))
    print("Volume:", V)
    print("Non associated components", Xa)
    # ----

    Fu, FCPA, FSRK, phi, lnp, Z = fugac.Fugg(aij, B, bij, D, n, y, V, P, T, Xa)

    print("Z", Z)
    print("FSRK", FSRK)
    print("FCPA", FCPA)
    print("phi", phi)
    print("Fugacity(bar)", Fu)
