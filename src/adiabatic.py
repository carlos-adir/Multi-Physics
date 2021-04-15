import numpy as np
from numpy import linalg as la
import finiteelement as fe
import time

def solve(x, t, u0, v0, T0, w, alpha=1/2, steps = 20, verbose=True):
    """
    
    """
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

    if True:  # Because dt is constant

        MTTinv = la.inv(MTT)
        Kmadb = Kmu - KmT @ MTTinv @ KTv
        Ktadb = KTv + alpha**2 *dt*KTT @ MTTinv @ KTv

        A11_big = Mmv/dt + alpha**2 * dt * Kmadb
        A21_big = alpha*Ktadb
        A22_big = MTT/dt + alpha*KTT

        B11_big = -Kmu
        B12_big = Mmv/dt - alpha*(1-alpha)*dt*Kmadb
        B13_big = -(1-alpha)*KmT
        B22_big = -(1-alpha)*Ktadb
        B23_big = MTT/dt - (1-alpha)*KTT


        A11 = A11_big[1:-1, 1:-1]
        A22 = A22_big[1:-1, 1:-1]
        A21 = A21_big[1:-1, 1:-1]
        A11inv = la.inv(A11)
        A22inv = la.inv(A22)

        B11 = B11_big[1:-1, 1:-1]
        B12 = B12_big[1:-1, 1:-1]
        B13 = B13_big[1:-1, 1:-1]
        B22 = B22_big[1:-1, 1:-1]
        B23 = B23_big[1:-1, 1:-1]

    
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
        

        v[i+1, 1:-1] = A11inv @ (B11 @ u[i, 1:-1] + B12 @ v[i, 1:-1] + B13 @ T[i, 1:-1] )
        T[i+1, 1:-1] = A22inv @ (B23 @ T[i, 1:-1] - A21 @ v[i+1, 1:-1])
        u[i+1] = u[i] + dt*((1-alpha)*v[i]+alpha*v[i+1])

    if verbose:
        time2 = time.time() - start_time
        print("Time simulation = %.3f s" % time2)
        print("---------------------------")
        print("Total time calculation = %.3f s" % (time1+time2))


    return u, v, T



if __name__ == "__main__": 
    data_folder = ""
    savename_sufix = "adia"
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
    #       Simulation       #
    ##########################
    
    np.save(data_folder + "x.npy", x_vec)
    np.save(data_folder + "t.npy", t_vec)
    np.save(data_folder + "U_" + savename_sufix + ".npy", u)
    np.save(data_folder + "V_" + savename_sufix + ".npy", v)
    np.save(data_folder + "T_" + savename_sufix + ".npy", T)