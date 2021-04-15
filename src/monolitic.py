import numpy as np
from numpy import linalg as la
import finiteelement as fe
import time
    

def calculateA(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    A = np.zeros((2*Npoints, 2*Npoints))
    
    A[:Npoints, :Npoints] = Mmv/dt + a**2 *dt*Kmu
    A[:Npoints, Npoints:] = a*KmT
    A[Npoints:, :Npoints] = a*KTv
    A[Npoints:, Npoints:] = MTT/dt + a*KTT

    return A

def calculateC(dt, a, Mmv, MTT, Kmu, KmT, KTv, KTT):
    Npoints, _ = Mmv.shape
    C = np.zeros((2*Npoints, 3*Npoints))

    C[:Npoints, :Npoints] = -Kmu
    C[:Npoints, Npoints:2*Npoints] = Mmv/dt - a*(1-a)*dt*Kmu
    C[:Npoints, 2*Npoints:] = -(1-a)*KmT
    C[Npoints:, :Npoints] = 0
    C[Npoints:, Npoints:2*Npoints] = -(1-a)*KTv
    C[Npoints:, 2*Npoints:] = MTT/dt - (1-a)*KTT

    return C

def calculateD(Npoints):

    D = np.zeros(2*Npoints)

    return D


def calculateX(i, u, v, T):
    _, Npoints = u.shape
    X = np.zeros(3*Npoints)
    X[:Npoints] = u[i, :]
    X[Npoints:2*Npoints] = v[i, :]
    X[2*Npoints:] = T[i, :]

    return X

def solve(x, t, u0, v0, T0, w, alpha=1/2, steps=20, verbose=True):
    
    Nx = len(x)-1
    Nt = len(t)-1

    dt = t[1]-t[0]
    dts = t[1:]-t[:-1]
    if np.all(np.abs(dts - dt) < 1e-13):
        dt_constant = True
    else:
        raise Exception("Not expecting that the time step changes")

    
    try:
        if not 0 < steps < 100:
            raise Exception("Steps nod valid")
    except Exception:
        steps = 20
        
    if verbose:
        print("Begin making matrix")
        start_time = time.time()

    u = np.zeros((Nt+1, Nx+1))
    v = np.zeros((Nt+1, Nx+1))
    T = np.zeros((Nt+1, Nx+1))

    u[0, :] = u0[:]
    v[0, :] = v0[:]
    T[0, :] = T0[:]

    if 1:
        Mmv = fe.Matrix1D(x, 0, 0)
        Kmu = fe.Matrix1D(x, 1, 1)
        KmT = -fe.Matrix1D(x, 1, 0)
        MTT = fe.Matrix1D(x, 0, 0)
        KTv = w*fe.Matrix1D(x, 0, 1)
        KTT = fe.Matrix1D(x, 1, 1)

    if verbose:
        print("    Calculated the base matrix")
        print("    Making bigger matrix")

    BCfixed = np.array([0, Nx, Nx+1, 2*Nx+1])
    BCfixed2 = np.array([0, Nx, Nx+1, 2*Nx+1, 2*Nx+2, 3*Nx+2])
    
    if 1:
        A = calculateA(dt, alpha, Mmv, MTT, Kmu, KmT, KTv, KTT)
        A1 = np.delete(A, BCfixed, axis=0)
        A11 = np.delete(A1, BCfixed, axis=1)
        A11_inv = la.inv(A11)
        # print("    Making LUFactorization")
        # L, U = LUFactorization(A11)   
        C = calculateC(dt, alpha, Mmv, MTT, Kmu, KmT, KTv, KTT)
        C1 = np.delete(C, BCfixed, axis=0)
        C11 = np.delete(C1, BCfixed2, axis=1)
        D = calculateD(Nx+1)

    if verbose:
        time1 = time.time()-start_time
        print("Time making matrix = %.3f s" % time1)

        
        
        print("Begin simulation")
        start_time = time.time()
        progress = 0
    for i in range(Nt):
        if verbose:
            if progress/100 < i/Nt:
                if progress%steps == 0:
                    print("    Calcul progress = %02d%%" %progress)
                progress += 1
                
        Xi = calculateX(i, u, v, T)
        B = C @ Xi + D
        B1 = np.delete(B, BCfixed)


        solution = A11_inv @ B1

        v[i+1, 1:-1] = solution[:Nx-1]
        T[i+1, 1:-1] = solution[Nx-1:]
        u[i+1] = u[i] + dt*((1-alpha)*v[i]+alpha*v[i+1])

    if verbose:
        time2 = time.time() - start_time
        print("Time simulation = %.3f s" % time2)
        print("---------------------------")
        print("Total time calculation = %.3f s" % (time1+time2))

    return u, v, T



if __name__ == "__main__": 
    data_folder = ""
    savename_sufix = "mono"
    w = 1 
    BarLength = 100  # Bar length
    TotalTime = 142  # adimentionalized time

    ##########################
    #       Simulation       #
    ##########################
    
    Nx = 100  # Number of elements in space
    Nt = 100  # Number of time intervals
    
    x_vec = np.linspace(0, BarLength, Nx+1)
    t_vec = np.linspace(0, TotalTime, Nt+1)
    u0 = np.zeros(x_vec.shape)
    v0 = np.sin(np.pi*x_vec/BarLength)
    T0 = np.zeros(x_vec.shape)

    u, v, T = solve(x_vec, t_vec, u0, v0, T0, w)

    ##########################
    #        Save info       #
    ##########################

    np.save(data_folder + "x.npy", x_vec)
    np.save(data_folder + "t.npy", t_vec)
    np.save(data_folder + "U_" + savename_sufix + ".npy", u)
    np.save(data_folder + "V_" + savename_sufix + ".npy", v)
    np.save(data_folder + "T_" + savename_sufix + ".npy", T)





