import unittest
import numpy as np

import constants
import fugac
import volstab
import matriz


class TestConstants(unittest.TestCase):
    def test_CA(self):
        At, D, ap, aij = constants.CA([1.0, 1.0, 1.0, 1.0], 1.0, 1.0)
        self.assertEqual(At, 1896588436.5456827)

    def test_CB(self):
        bi, Bt, B, bij = constants.CB(1.0, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(B, 113.69461537971584)


class TestFugac(unittest.TestCase):
    def test_Fugg(self):
        At, D, ap, aij = constants.CA([1.0, 1.0, 1.0, 1.0], 1.0, 1.0)
        bi, Bt, B, bij = constants.CB(1.0, [1.0, 1.0, 1.0, 1.0])

        Fu, FCPA, FSRK, phi, lnp, Z = fugac.Fugg(
            aij,
            B,
            bij,
            D,
            1.0,
            np.array([1.0, 1.0, 1.0, 1.0]),
            1.0,
            1.0,
            1.0,
            np.array([1.0, 1.0, 1.0, 1.0]),
        )
        self.assertEqual(Z, 12027.904738994466)


class TestVolstab(unittest.TestCase):
    def test_Vol(self):
        Vest, Pcz, tol = volstab.Vol(
            1.0, 1.0, 1.0, 1.0, 1.0, np.array([1.0, 1.0, 1.0, 1.0])
        )
        self.assertEqual(tol, 1.0)


class TestMatriz(unittest.TestCase):
    def test_Gsselm(self):
        singular, x = matriz.Gsselm(np.ones((2, 3)), 1, np.ones(2))
        self.assertEqual(x[0], 1)


if __name__ == "__main__":
    unittest.main()
