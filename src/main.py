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
import monolitic as mono
import isotherm as isot
import adiabatic as adia

import time


if __name__ == "__main__":
    w = 0
    BarLength = 100
    TotalTime = 400
    Nx = 100  # Number of elements in space
    Nt = 100  # Number of elements in time

    x_vec = np.linspace(0, BarLength, Nx+1)
    t_vec = np.linspace(0, TotalTime, Nt+1)

    u0 = np.zeros(x_vec.shape)
    v0 = np.sin(np.pi * x_vec / BarLength)  # v(x, 0) = sin(pi*x/L)
    T0 = np.zeros(x_vec.shape)

    data_folder = ""
    schemas = ["monolitic", "isothermal", "adiabatic"]


    if True:  # Periods
        Tis = 2 * BarLength
        Tau = Tis / np.sqrt(1 + w)
        print("Isotherm period = %.3f" % Tis)
        print("Approximation period = %.3f" % Tau)


    np.save(data_folder + "x.npy", x_vec)
    np.save(data_folder + "t.npy", t_vec)

    for schema in schemas:
        if schema == "monolitic":
            savename_sufix = "mono"
            solve = mono.solve
        elif schema == "isothermal":
            savename_sufix = "isot"
            solve = isot.solve
        elif schema == "adiabatic":
            savename_sufix = "adia"
            solve = adia.solve
        else:
            raise Exception("Schema not recognized")

        print("##############################")
        print("#    Using the schema %s   #" % savename_sufix )
        print("##############################")

        ##########################
        #       Simulation       #
        ##########################
        
        u, v, T = solve(x_vec, t_vec, u0, v0, T0, w)


        ##########################
        #        Save info       #
        ##########################

        np.save(data_folder + "U_" + savename_sufix + ".npy", u)
        np.save(data_folder + "V_" + savename_sufix + ".npy", v)
        np.save(data_folder + "T_" + savename_sufix + ".npy", T)
