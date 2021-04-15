from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    ###########################################
    # Opening informations from file
    #####################
    data_folder = ""
    try:
        U_mono = np.load(data_folder + "U_mono.npy")
        V_mono = np.load(data_folder + "V_mono.npy")
        T_mono = np.load(data_folder + "T_mono.npy")
        mono_results = True
    except Exception:
        mono_results = False
    try:
        U_isot = np.load(data_folder + "U_isot.npy")
        V_isot = np.load(data_folder + "V_isot.npy")
        T_isot = np.load(data_folder + "T_isot.npy")
        isot_results = True
    except Exception:
        isot_results = False
    try:
        U_adia = np.load(data_folder + "U_adia.npy")
        V_adia = np.load(data_folder + "V_adia.npy")
        T_adia = np.load(data_folder + "T_adia.npy")
        adia_results = True
    except Exception:
        adia_results = False

    try:
        x_vec = np.load(data_folder + "x.npy")
        t_vec = np.load(data_folder + "t.npy")
    except Exception:
        raise Exception("Data x.npy or t.npy was not found")

    if not (mono_results or isot_results or isot_results):
        raise Exception("No data included")

    ###########################################
    # Trait
    #######################################
    BarLength = x_vec[-1]
    Nx = len(x_vec) - 1
    TotalTime = t_vec[-1]
    Nt = len(t_vec) - 1
    Tis = 2 * BarLength

    if mono_results and isot_results:
        if U_mono.shape != U_isot.shape:
            raise Exception("U_mono et U_isot - Not the same shape")
    if mono_results and adia_results:
        if U_mono.shape != U_adia.shape:
            raise Exception("U_mono et U_adia - Not the same shape")
    if isot_results and adia_results:
        if U_isot.shape != U_adia.shape:
            raise Exception("U_isot et U_adia - Not the same shape")

    N_time, N_points = U_mono.shape

    index_position_Ls2 = (N_points // 2) - 1  # Index of the position L/2
    index_position_Ls4 = (N_points // 4) - 1  # Index of the position L/4
    index_time_50 = int(np.ceil(50 * N_time / TotalTime))
    index_time_300 = int(np.ceil(300 * N_time / TotalTime))

    if index_time_300 > N_time:
        raise Exception("Not too much time simuled")

    x = np.linspace(0, BarLength, N_points)
    t = np.linspace(0, TotalTime, N_time)

    fig, axs = plt.subplots(3, 2)

    if mono_results:
        axs[0, 0].plot(t, U_mono[:, index_position_Ls2], label="Mono")
    if isot_results:
        axs[0, 0].plot(t, U_isot[:, index_position_Ls2], label="Isot")
    if adia_results:
        axs[0, 0].plot(t, U_adia[:, index_position_Ls2], label="Adia")
    axs[0, 0].set(xlabel=r"Time $\tau$")
    axs[0, 0].set(ylabel=r"Displacement $u$")
    axs[0, 0].set_title(r"With the position $x = L/2$")
    axs[0, 0].set_xlim([0, TotalTime])
    axs[0, 0].legend()

    if mono_results:
        axs[0, 1].plot(t, T_mono[:, index_position_Ls4], label="Mono")
    if isot_results:
        axs[0, 1].plot(t, T_isot[:, index_position_Ls4], label="Isot")
    if adia_results:
        axs[0, 1].plot(t, T_adia[:, index_position_Ls4], label="Adia")
    axs[0, 1].set(xlabel=r"Time $\tau$")
    axs[0, 1].set(ylabel=r"Temperature $T$")
    axs[0, 1].set_title(r"With the position $x = L/4$")
    axs[0, 1].set_xlim([0, TotalTime])
    axs[0, 1].legend()

    if mono_results:
        axs[1, 0].plot(x, U_mono[index_time_50, :], label="Mono")
    if isot_results:
        axs[1, 0].plot(x, U_isot[index_time_50, :], label="Isot")
    if adia_results:
        axs[1, 0].plot(x, U_adia[index_time_50, :], label="Adia")
    axs[1, 0].set(xlabel=r"Position $x$")
    axs[1, 0].set(ylabel=r"Displacement $u$")
    axs[1, 0].set_title(r"With the time $\tau = 50$")
    axs[1, 0].set_xlim([0, BarLength])
    axs[1, 0].legend()

    if mono_results:
        axs[1, 1].plot(x, T_mono[index_time_50, :], label="Mono")
    if isot_results:
        axs[1, 1].plot(x, T_isot[index_time_50, :], label="Isot")
    if adia_results:
        axs[1, 1].plot(x, T_adia[index_time_50, :], label="Adia")
    axs[1, 1].set(xlabel=r"Position $x$")
    axs[1, 1].set(ylabel=r"Temperature $T$")
    axs[1, 1].set_title(r"With the time $\tau = 50$")
    axs[1, 1].set_xlim([0, BarLength])
    axs[1, 1].legend()

    if mono_results:
        axs[2, 0].plot(x, U_mono[index_time_300, :], label="Mono")
    if isot_results:
        axs[2, 0].plot(x, U_isot[index_time_300, :], label="Isot")
    if adia_results:
        axs[2, 0].plot(x, U_adia[index_time_300, :], label="Adia")
    axs[2, 0].set(xlabel=r"Position $x$")
    axs[2, 0].set(ylabel=r"Displacement $u$")
    axs[2, 0].set_title(r"With the time $\tau = 300$")
    axs[2, 0].set_xlim([0, BarLength])
    axs[2, 0].legend()

    if mono_results:
        axs[2, 1].plot(x, T_mono[index_time_300, :], label="Mono")
    if isot_results:
        axs[2, 1].plot(x, T_isot[index_time_300, :], label="Isot")
    if adia_results:
        axs[2, 1].plot(x, T_adia[index_time_300, :], label="Adia")
    axs[2, 1].set(xlabel=r"Position $x$")
    axs[2, 1].set(ylabel=r"Temperature $T$")
    axs[2, 1].set_title(r"With the time $\tau = 300$")
    axs[2, 1].set_xlim([0, BarLength])
    axs[2, 1].legend()

    plt.subplots_adjust(left=0.06)
    plt.subplots_adjust(bottom=0.07)
    plt.subplots_adjust(right=0.97)
    plt.subplots_adjust(top=0.88)
    plt.subplots_adjust(hspace=0.56)
    plt.show()
