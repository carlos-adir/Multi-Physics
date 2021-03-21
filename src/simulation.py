"""
Code made for the cours of coupled to solve a problem of a bar using finite elements.
It solves a problem of a bar, that can move in only one direction, and it's boundary are fixed:
displacement u: u(x = 0) = 0 and u(x = L) = 0 for all t (time)
and given the inicial conditions, we have:
    v(x, t = 0) = sin( x * pi/L)
The equations that we are using are:
    * The mechanical equation:
        (d^2 u/dt^2) = (d^2 u/dx^2) - (dT/dx)
    * The thermodynamic equation:
        (dT/dt) = (d^2 T/dx^2) - w * (d^2 u/dt*dx)
        
Using elements finis, and the monotic system to solve, we get the matrix:
[ II           ]   [ dot(u) ]   [      -II      ]   [ u ]   [ 0 ]
[     Mmv      ] * [ dot(v) ] + [ Kmu       KmT ] * [ v ] = [ 0 ]
[          MTT ]   [ dot(T) ]   [      KTv  KTT ]   [ T ]   [ 0 ]
Then when we make the time discretization to solve the problem, we get:
[ 1   A12      ]   [ u_{n+1} ]   [  1   C12      ]   [ u_{n} ]   [ 0 ]
[     A22  A23 ] * [ v_{n+1} ] = [ C21  C22  C23 ] * [ v_{n} ] + [ 0 ]
[     A32  A33 ]   [ T_{n+1} ]   [      C32  C33 ]   [ T_{n} ]   [ 0 ]
Which we have:
A12 = -a^2 * dt
A22 = Mmv/dt + a^2 * dt * Kmu
A23 = a * Kmt
A32 = a * KTv
A33 = MTT/dt + a * KTT
------
C12 = a * (1-a) * dt
C21 = -Kmu
C22 = Mmv/dt - a * (1-a) * dt * Kmu
C23 = -(1-a) * KmT
C32 = -(1-a) * KTv
C33 = MTT/dt - (1-a) * KTT

For the others methods, it changes.
"""

import numpy as np
from numpy import linalg as la
import finiteelement as fe
import multiphysics

import time


def calculateX(i, u, v, T):
    _, Npoints = u.shape
    X = np.zeros(3 * Npoints)
    X[:Npoints] = u[i, :]
    X[Npoints:2 * Npoints] = v[i, :]
    X[2 * Npoints:] = T[i, :]

    return X


if __name__ == "__main__":
    w = 3
    BarLength = 100
    TotalTime = 400
    Nx = 100  # Number of elements in space
    Nt = 590  # Number of elements in time

    # Schema de Crank-Nicholson for time integration
    a = 1 / 2

    # schema = "monolitic"
    # schema = "isothermal"
    # schema = "adiabatic"
    data_folder = "../data/"
    schemas = ["monolitic", "isothermal", "adiabatic"]

    for schema in schemas:
        if schema == "monolitic":
            savename_sufix = "mono"
            calculateA = multiphysics.calculateA_mono
            calculateC = multiphysics.calculateC_mono
            calculateD = multiphysics.calculateD_mono
        elif schema == "isothermal":
            savename_sufix = "isot"
            calculateA = multiphysics.calculateA_isot
            calculateC = multiphysics.calculateC_isot
            calculateD = multiphysics.calculateD_isot
        elif schema == "adiabatic":
            savename_sufix = "adia"
            calculateA = multiphysics.calculateA_adia
            calculateC = multiphysics.calculateC_adia
            calculateD = multiphysics.calculateD_adia
        else:
            raise Exception("Schema not recognized")

        # So, we impose only the value of w

        # THe differential equation are
        #
        #  d^2 u/dt^2 = d^2 u/dx^2 - dT/dx
        #  dT/dt = d^2 T/dx^2 - w * d^2 u/ dt dx
        #

        # Initial conditions Dirichlet homogenes
        #
        #  u(0, t) = u(L, t) = 0
        #  T(0, t) = T(L, t) = 0
        #
        #  u(x, 0) = 0
        #
        #  T(x, 0) = 0
        #

        ##########################
        #       Simulation       #
        ##########################

        dt = TotalTime / Nt

        # Construct vectors and matrices
        x_vec = np.linspace(0, BarLength, Nx + 1)
        t_vec = np.linspace(0, TotalTime, Nt + 1)

        u = np.zeros((Nt + 1, Nx + 1))
        v = np.zeros((Nt + 1, Nx + 1))
        T = np.zeros((Nt + 1, Nx + 1))

        # Initial condition
        v[0] = np.sin(np.pi * x_vec / BarLength)  # v(x, 0) = sin(pi*x/L)

        # Periods
        Tis = 2 * BarLength
        Tau = Tis / np.sqrt(1 + w)
        print("Isotherm period = %.3f" % Tis)
        print("Approximation period = %.3f" % Tau)

        ####################
        # Begin simulation #
        ####################
        print("Begin making matrix")
        start_time = time.time()
        II = np.eye(Nx + 1)
        Mmv = fe.Matrix1D(x_vec, 0, 0)
        Kmu = fe.Matrix1D(x_vec, 1, 1)
        KmT = -fe.Matrix1D(x_vec, 1, 0)
        MTT = fe.Matrix1D(x_vec, 0, 0)
        KTv = w * fe.Matrix1D(x_vec, 0, 1)
        KTT = fe.Matrix1D(x_vec, 1, 1)
        print("    Calculated the base matrix")
        print("    Making bigger matrix")

        if True:  # Put boundary condition
            BCfixed = []
            # Because u(x = 0, t) = 0, and so v(x = 0, t) = 0
            BCfixed.append(0)
            # Because u(x = L, t) = 0, and so v(x = 0, t) = 0
            BCfixed.append(Nx)
            # Because T(x = 0, t) = 0
            BCfixed.append(Nx + 1)
            # Because T(x = L, t) = 0
            BCfixed.append(2 * Nx + 1)

        if True:
            A = calculateA(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT)
            A1 = np.delete(A, BCfixed, axis=0)
            A11 = np.delete(A1, BCfixed, axis=1)

            C = calculateC(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT)
            C1 = np.delete(C, BCfixed, axis=0)

            invA11 = la.inv(A11)
            G = invA11 @ C1

            D = calculateD(Nx + 1)
            D1 = np.delete(D, BCfixed, axis=0)

            H = invA11 @ D1

        time1 = time.time() - start_time
        print("Time making matrix = %.3f ms" % (1000 * time1))

        print("Begin simulation")
        start_time = time.time()
        progress = 0
        steps = 20  # We want 0%, 20%, 40%, ..., 100%
        for i in range(Nt):
            if progress / 100 < i / Nt:
                if progress % steps == 0:
                    print("    Calcul progress = %02d%%" % progress)
                progress += 1
            Xi = calculateX(i, u, v, T)
            solution = G @ Xi + H
            v[i + 1, 1:-1] = solution[:Nx - 1]
            T[i + 1, 1:-1] = solution[Nx - 1:]
            u[i + 1] = u[i] + dt * ((1 - a) * v[i] + a * v[i + 1])

        time2 = time.time() - start_time
        print("Time simulation = %.3f ms" % (1000 * time2))
        print("---------------------------")
        print("Total time calculation = %.3f ms" % (1000 * (time1 + time2)))

        np.save(data_folder + "x.npy", x_vec)
        np.save(data_folder + "t.npy", t_vec)
        np.save(data_folder + "U_" + savename_sufix + ".npy", u)
        np.save(data_folder + "V_" + savename_sufix + ".npy", v)
        np.save(data_folder + "T_" + savename_sufix + ".npy", T)
