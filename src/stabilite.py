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
from matplotlib import pyplot as plt
from tqdm import trange

import finiteelement as fe
import multiphysics
import monolitic as mono
import isotherm as isot
import adiabatic as adia

import time

def is_stable(u, v, T):
    if np.max(np.abs(u)) > 35:
        return False
    if np.max(np.abs(T)) > 15:
        return False
    return True



if __name__ == "__main__":
    BarLength = 100
    TotalTime = 400
    Nx = 25  # Number of elements in space
    
    x_vec = np.linspace(0, BarLength, Nx+1)

    u0 = np.zeros(x_vec.shape)
    v0 = np.sin(np.pi * x_vec / BarLength)  # v(x, 0) = sin(pi*x/L)
    T0 = np.zeros(x_vec.shape)

    data_folder = ""
    schema = "adiabatic"
    # schema = "monolitic"

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


    ##########################
    #       Simulation       #
    ##########################

    dx = x_vec[1]-x_vec[0]


    Ntests = 50
    Ntests_ws = Ntests
    Ntests_dts = Ntests
    wmin = 0
    wmax = 10
    dtmin = 0.1
    dtmax = 10
    # ws = np.array([0., 0.2, 0.4, 0.6, 0.8])
    # ws = np.array([0, 1, 10, 100, 1000])
    ws = np.linspace(wmin, wmax, Ntests_ws)
    # Nts = np.array([50, 100, 200, 300, 400])
    # Nts = np.arange(10, 100, 10, dtype="int8")
    dts = dx*np.linspace(dtmin, dtmax, Ntests_dts)
    Nts = np.zeros(Ntests_dts, dtype="int32")
    for j, dt in enumerate(dts):
        Nts[j] = int(np.ceil(TotalTime/dt))
        if Nts[j] <= 0:
            raise Exception("Not possible")

    stable_table = np.zeros((Ntests_ws, Ntests_dts), dtype=bool)
    print("Shape stable_table = %s" % str(stable_table.shape))
    for i in trange(len(ws)):
        w = ws[i]
        for j, dt in enumerate(dts):
            Nt = Nts[j]
            t_vec = np.linspace(0, TotalTime, Nt+1)
            u, v, T = solve(x_vec, t_vec, u0, v0, T0, w, verbose=False)
            stable_table[i, j] = is_stable(u, v, T)

    dw = ws[1]-ws[0]
    ddt = dts[1] - dts[0]
    ws = np.linspace(wmin-dw/2, wmax+dw/2, Ntests_ws+1)
    dts = np.linspace(dtmin - ddt/2, dtmax+ddt/2, Ntests_dts+1)
   
    fig, ax = plt.subplots()
    ax.pcolormesh(ws, dts/dx, stable_table)
    ax.set(xlabel="w")
    ax.set(ylabel="dt/dx")
    plt.show()


