from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('seaborn-pastel')

if __name__ == "__main__":
    data_folder = "../data/"
    reference = "isot"

    try:
        x_vec = np.load(data_folder + "x.npy")
        t_vec = np.load(data_folder + "t.npy")
        BarLength = x_vec[-1]
        Nx = len(x_vec) - 1
        TotalTime = t_vec[-1]
        Nt = len(t_vec) - 1
    except Exception:
        raise Exception("Data x.npy or t.npy was not found")
    if True:
        # Opening data
        U_ref = np.load(data_folder + "U_" + reference + ".npy")
        V_ref = np.load(data_folder + "V_" + reference + ".npy")
        T_ref = np.load(data_folder + "T_" + reference + ".npy")

    fig = plt.figure()
    Umin = np.min(U_ref)
    Umax = np.max(U_ref)
    Tmin = np.min(T_ref)
    Tmax = np.max(T_ref)
    dU = Umax - Umin
    rangex = (0, BarLength)
    rangey = (Umin - 0.1 * dU, Umax + 0.1 * dU)
    axs = fig.add_subplot(111)
    div = make_axes_locatable(axs)
    cax = div.append_axes('right', '5%', '10%')
    axs.set(xlim=rangex)
    axs.set(ylim=rangey)

    cmap = matplotlib.cm.viridis
    # norm = matplotlib.colors.Normalize(vmin=Tmin, vmax=Tmax)
    norm = matplotlib.colors.Normalize(T_ref.min(), T_ref.max())
    cb1 = matplotlib.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap,
                                           orientation='vertical')

    # axs.set(title='Frame 0')
    # def init():
    #     print("type(line) = ", type(line))
    #     line.set_data([], [])
    #     return line,
    title = axs.text(0.0005 * BarLength, 0.002 * dU, "",
                     transform=axs.transAxes, ha="left")

    def animate(i):
        x = x_vec
        y = U_ref[i, :]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # im.set_data(x, y)
        # im.set_clim(Tmin, Tmax)

        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping

        lc.set_array(T_ref[i, :])
        lc.set_linewidth(2)
        line = axs.add_collection(lc)
        title.set_text("Time = %.1f" % t_vec[i])
        # tx = axs.set_title("Frame ")
        return line, title

    anim = FuncAnimation(fig, animate, frames=Nt, interval=20, blit=True)

    plt.show()
    # anim.save('corda.gif', writer='pillow')
