#!/usr/bin/python

import numpy as np


def Lin_shp(idx, xi, Xe):
    if (idx == 1):
        return np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])
    elif (idx == 2):
        return (np.array([-np.ones(len(xi)), np.ones(len(xi))]) / (Xe[1] - Xe[0]))


def Lin_eval(xi, Xe):
    return ((Xe[0] * (-xi + 1.0) + Xe[1] * (xi + 1.0)) / 2.0)


def Cst_eval(xi, Xe):
    return Xe * np.ones(len(xi))


def Jac(Xe):
    return ((Xe[1] - Xe[0]) / 2.0)


def GaussIntegration(npg=3):
    if npg == 1:
        xi = [0]
        wi = [2]
    elif npg == 2:
        a = 1 / np.sqrt(3)
        xi = [-a, a]
        wi = [1, 1]
    elif npg == 3:
        b = np.sqrt(3 / 5)
        xi = [-b, 0, b]
        wi = [5 / 9, 8 / 9, 5 / 9]
    else:
        raise Exception("Couldn't find the npg wantted")

    xi = np.array(xi)
    wi = np.array(wi)
    return xi, wi


def Linear_form(x, op, lin_fields, cst_fields):
    Nnodes = len(x)
    Nelem = Nnodes - 1
    T10 = np.array([np.arange(0, Nnodes - 1, 1), np.arange(1, Nnodes, 1)]).T
    h = x[1:len(x)] - x[:len(x) - 1]

    xi, wi = GaussIntegration()
    A = np.zeros(Nnodes)
    for i in range(Nelem):
        mapp = T10[i, :]
        Xe = x[mapp]
        loc_jac = Jac(Xe)

        M = Lin_shp(op + 1, xi, Xe)
        D = np.array(wi)
        if len(lin_fields) != 0:
            D *= Lin_eval(xi, lin_fields[mapp])
        if len(cst_fields) != 0:
            D *= Cst_eval(xi, cst_fields[i])
        A[mapp[0]:mapp[1] + 1] += (loc_jac * np.dot(M, D.T))
    return A


def Matrix1D(x, op1, op2, lin_fields=[], cst_fields=[]):
    Nnodes = len(x)
    Nelem = Nnodes - 1
    T10 = np.array([np.arange(0, Nnodes - 1, 1), np.arange(1, Nnodes, 1)]).T
    h = x[1:len(x)] - x[:len(x) - 1]

    xi, wi = GaussIntegration()
    A = np.zeros((Nnodes, Nnodes))
    for i in range(Nelem):
        mapp = T10[i, :]
        Xe = x[mapp]
        loc_jac = Jac(Xe)

        M1 = Lin_shp(op1 + 1, xi, Xe)
        M2 = Lin_shp(op2 + 1, xi, Xe)
        D = np.array(wi)
        if len(lin_fields) != 0:
            D *= Lin_eval(xi, lin_fields[mapp])
        if len(cst_fields) != 0:
            D *= Cst_eval(xi, cst_fields[i])
        A[mapp[0]:mapp[1] + 1, mapp[0]:mapp[1] +
            1] += (loc_jac * np.dot(M1, np.dot(np.diag(D), M2.T)))
    return A


if __name__ == "__main__":
    print("You should import this file instead of run it")
