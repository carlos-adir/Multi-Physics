import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider


if __name__ == "__main__":
    data_folder = ""
    dimention4 = True

    # sufix = "mono"
    sufix = "isot"
    # sufix = "adia"
    u = np.load(data_folder + "U_" + sufix + ".npy")
    v = np.load(data_folder + "V_" + sufix + ".npy")
    T = np.load(data_folder + "T_" + sufix + ".npy")
    x_vec = np.load(data_folder + "x.npy")
    t_vec = np.load(data_folder + "t.npy")

    BarLength = x_vec[-1]
    Nx = len(x_vec) - 1
    TotalTime = t_vec[-1]
    Nt = len(t_vec) - 1

    print("BarLength = ", BarLength)
    print("TotalTime = ", TotalTime)
    print("Number elements in space = ", Nx)
    print("Number elements in time = ", Nt)

    data = "u"
    # data = "v"
    # data = "T"
    if data == "u":
        des = u  # Desired graph
        des_name = "Displacement (u)"  # Desired name
        des_title = "Value of the Displacement"
        color_dimension = u
    if data == "v":
        des = v  # Desired graph
        des_name = "Speed (v)"  # Desired name
        des_title = "Value of the Speed"
        color_dimension = v
    if data == "T":
        des = T  # Desired graph
        des_name = "Temperature (T)"  # Desired name
        des_title = "Value of the Temperature"
        color_dimension = T
    if dimention4:
        color_dimension = T  # change to desired fourth dimension
        color_dimension = u

    # Validation of the entry
    print("U.shape =", u.shape)
    print("V.shape =", v.shape)
    print("T.shape =", T.shape)
    if u.shape != v.shape:
        raise Exception("U and V doesn't have the same shape")
    if u.shape != T.shape:
        raise Exception("U and T doesn't have the same shape")
    if u.shape[0] != Nt + 1:
        print("U.shape = ", u.shape)
        print("t_vec.shape = ", t_vec.shape)
        raise Exception("U and t_vec doesn't have the same length")
    if u.shape[1] != Nx + 1:
        raise Exception("U and x_vec doesn't have the same length")

    fig = plt.figure()

    ax3d = plt.subplot2grid((2, 4), (0, 0), rowspan=2,
                            colspan=2, projection='3d')
    ax_xconst = plt.subplot2grid((2, 4), (0, 2), colspan=2)
    ax_tconst = plt.subplot2grid((2, 4), (1, 2), colspan=2)

    if 1:
        x, t = np.meshgrid(x_vec, t_vec)

    if dimention4:
        ############
        #  Colors  #
        ############
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = matplotlib.colors.Normalize(minn, maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        fcolors = m.to_rgba(color_dimension)

        fig.colorbar(m, ax=ax3d, orientation='vertical', fraction=.1)

    if 1:
        ###########
        # Sliders #
        ###########
        dt = TotalTime / Nt
        dx = BarLength / Nx
        tmin = 0
        tmax = TotalTime
        xmin = 0
        xmax = BarLength
        desmin = np.min(des)
        desmax = np.max(des)
        ddes = desmax - desmin
        desmin = desmin - 0.2 * ddes
        desmax = desmax + 0.2 * ddes
        t0 = (tmin + tmax) / 2
        x0 = (xmin + xmax) / 2
        axcolor = 'lightgoldenrodyellow'
        ax_time = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        ax_posi = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
        slider_time = Slider(ax_time, 'Time t', tmin,
                             tmax, valinit=t0, valstep=dt)
        slider_posi = Slider(ax_posi, 'Position x', xmin,
                             xmax, valinit=x0, valstep=dx)

        def update_time(val):
            t0 = slider_time.val
            index_time = int(t0 / dt)

            ax_tconst.clear()
            ax_tconst.plot(x_vec, des[index_time, :], color="r")
            ax_tconst.set_xlim([xmin, xmax])
            ax_tconst.set_ylim([desmin, desmax])
            # ax_tconst.set_zlim([-30, 30])
            ax_tconst.set(xlabel="Position (x)")
            ax_tconst.set(ylabel=des_name)
            ax_tconst.set(title="Values when t = %.1f" % t0)
            ax_tconst.grid()
            update_3d()

        def update_posi(val):
            x0 = slider_posi.val
            index_posi = int(x0 / dx)

            ax_xconst.clear()
            ax_xconst.plot(t_vec, des[:, index_posi], color="b")
            ax_xconst.set_xlim([tmin, tmax])
            ax_xconst.set_ylim([desmin, desmax])
            # ax_xconst.set_zlim([-30, 30])
            ax_xconst.set(xlabel="Time (t)")
            ax_xconst.set(ylabel=des_name)
            ax_xconst.set(title="Values when x = %.1f" % x0)
            ax_xconst.grid()
            update_3d()

        def update_3d():
            x0 = slider_posi.val
            index_posi = int(x0 / dx)
            t0 = slider_time.val
            index_time = int(t0 / dt)

            ax3d.clear()  # clear current axes
            ax3d.set_xlim([0, TotalTime])
            ax3d.set_ylim([0, BarLength])
            ax3d.set_zlim([-30, 30])
            if dimention4:
                surface = ax3d.plot_surface(
                    t, x, des, facecolors=fcolors, alpha=0.3)
            else:
                surface = ax3d.plot_surface(t, x, des)
            ax3d.set(xlabel="Time (t)")
            ax3d.set(ylabel="Bar position (x)")
            ax3d.set(zlabel=des_name)
            ax3d.plot3D(t_vec, BarLength * np.ones(t_vec.shape),
                        des[:, index_posi], color="b")
            ax3d.plot3D(np.zeros(x_vec.shape), x_vec,
                        des[index_time, :], color="r")
            ax3d.plot3D(t_vec, x0 * np.ones(t_vec.shape),
                        des[:, index_posi], color="k", linewidth=2)
            ax3d.plot3D(t0 * np.ones(x_vec.shape), x_vec,
                        des[index_time, :], color="k", linewidth=2)

        slider_time.on_changed(update_time)
        slider_posi.on_changed(update_posi)
        update_time(0)
        update_posi(0)

    # fig.suptitle(des_title)
    plt.subplots_adjust(bottom=0.2, wspace=0.5, hspace=0.5)
    plt.show()
