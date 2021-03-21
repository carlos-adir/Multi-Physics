

import numpy as np
from numpy import linalg as la


def calculateA_mono(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    A = np.zeros((2 * Npoints, 2 * Npoints))

    A[:Npoints, :Npoints] = Mmv / dt + a**2 * dt * Kmu
    A[:Npoints, Npoints:] = a * KmT
    A[Npoints:, :Npoints] = a * KTv
    A[Npoints:, Npoints:] = MTT / dt + a * KTT

    return A


def calculateC_mono(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    C = np.zeros((2 * Npoints, 3 * Npoints))

    C[:Npoints, :Npoints] = -Kmu
    C[:Npoints, Npoints:2 * Npoints] = Mmv / dt - a * (1 - a) * dt * Kmu
    C[:Npoints, 2 * Npoints:] = -(1 - a) * KmT
    C[Npoints:, :Npoints] = 0
    C[Npoints:, Npoints:2 * Npoints] = -(1 - a) * KTv
    C[Npoints:, 2 * Npoints:] = MTT / dt - (1 - a) * KTT

    return C

def calculateD_mono(Npoints):

    D = np.zeros(2 * Npoints)

    return D


def calculateA_isot(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    A = np.zeros((2 * Npoints, 2 * Npoints))

    A[:Npoints, :Npoints] = Mmv / dt + a**2 * dt * Kmu
    A[:Npoints, Npoints:] = 0
    A[Npoints:, :Npoints] = KTv
    A[Npoints:, Npoints:] = MTT / dt + a * KTT

    return A


def calculateC_isot(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    C = np.zeros((2 * Npoints, 3 * Npoints))

    C[:Npoints, :Npoints] = -Kmu
    C[:Npoints, Npoints:2 * Npoints] = Mmv / dt - a * (1 - a) * dt * Kmu
    C[:Npoints, 2 * Npoints:] = -KmT
    C[Npoints:, :Npoints] = 0
    C[Npoints:, Npoints:2 * Npoints] = 0
    C[Npoints:, 2 * Npoints:] = MTT / dt - (1 - a) * KTT

    return C

def calculateD_isot(Npoints):

    D = np.zeros(2 * Npoints)

    return D


def calculateA_adia(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    A = np.zeros((2 * Npoints, 2 * Npoints))

    invMTT = la.inv(MTT)
    Kmadb = Kmu - KmT @ invMTT @ KTv
    Ktadb = KTv - (1 - a) * dt * KTT @ invMTT @ KTv

    A[:Npoints, :Npoints] = Mmv / dt + a**2 * dt * Kmadb
    A[:Npoints, Npoints:] = 0
    A[Npoints:, :Npoints] = a * Ktadb
    A[Npoints:, Npoints:] = MTT / dt + a * KTT

    return A


def calculateC_adia(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    C = np.zeros((2 * Npoints, 3 * Npoints))

    invMTT = la.inv(MTT)
    Kmadb = Kmu - KmT @ invMTT @ KTv
    Ktadb = KTv - (1 - a) * dt * KTT @ invMTT @ KTv

    C[:Npoints, :Npoints] = -Kmu
    C[:Npoints, Npoints:2 * Npoints] = Mmv / dt - a * (1 - a) * dt * Kmadb
    C[:Npoints, 2 * Npoints:] = -KmT
    C[Npoints:, :Npoints] = 0
    C[Npoints:, Npoints:2 * Npoints] = -(1 - a) * Ktadb
    C[Npoints:, 2 * Npoints:] = MTT / dt - (1 - a) * KTT

    return C

def calculateD_adia(Npoints):

    D = np.zeros(2 * Npoints)

    return D


if __name__ == "__main__":
    print("You should import this file instead of run it")
