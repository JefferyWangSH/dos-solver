import numpy as np
import time, datetime
from grids import FrequencyGrids, MomentumGrids
from model import FreePropagator, Kernel, GreenFunc
from lattice_momentum import LatticeMomentum
from matplotlib import pyplot as plt


if "__main__":
    """
        The main program
    """

    # setting up params for grids
    lattice_size = 16
    freq_range = [-30.0, 30.0]
    freq_num = int(1e3)

    infinitesimal_imag = 0.95

    # setting up model params
    mass = 1.0
    fermi_momentum = np.pi/2
    fermi_surface = fermi_momentum**2/(2*mass)
    static_gap = 1.0
    corr_length = 40.0

    # record cpu time
    time_begin = time.time()
    time_end = time.time()
    print(" The program gets started at %s .\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # initialize grids
    freq_grid = FrequencyGrids(num_of_grids = freq_num, freqency_range = freq_range)
    k_grid = MomentumGrids(num_of_grids_per_length = lattice_size)
    time_end = time.time()
    print(" Grids initialized in %.2f s.\n" % (time_end-time_begin))
    time_begin = time.time()

    # initialize free propagator
    free_propagator = FreePropagator(freq_grid, k_grid)
    free_propagator.SetModelParms(mass, fermi_surface)
    free_propagator.SetInfinitesimalImag(infinitesimal_imag = infinitesimal_imag)
    free_propagator.init()
    time_end = time.time()
    print(" Generate free propagator in %.2f s.\n" % (time_end-time_begin))
    time_begin = time.time()

    # generate kernel
    kernel = Kernel(momentum_grids = k_grid)
    kernel.SetDisorderParams(static_gap = static_gap, corr_length = corr_length)
    kernel.init()
    time_end = time.time()
    print(" Generate kernel in %.2f s.\n" % (time_end-time_begin))
    time_begin = time.time()

    # compute self energy and green's function
    green_func = GreenFunc(kernel = kernel, free_propagator = free_propagator)
    green_func.ComputeSelfEnergy()
    green_func.ComupteGreenFunc()
    time_end = time.time()
    print(" Complete the computation of Green's function in %.2f s.\n" % (time_end-time_begin))
    time_begin = time.time()

    # collecting spectrum A(k,omega)
    spectrum_mat = np.zeros((green_func.MomentumDim(), green_func.FrequencyDim()))
    for i in range(spectrum_mat.shape[0]):
        for j in range(spectrum_mat.shape[1]):
            spectrum_mat[i,j] = -2 * green_func.GreenFuncMat()[i,j].imag

    # collecting density of states
    dos_list = spectrum_mat.sum(axis=0)/spectrum_mat.shape[0]

    # testing the normalization condition
    print(" Normalize condition for DOS is %.3f \n" % (sum(dos_list)*(freq_range[1]-freq_range[0])/freq_num))

    # plot the figure
    print(" Ploting the figure of DOS ... \n")
    plt.figure()
    plt.grid(linestyle='-.')
    plt.plot(freq_grid.Grids(), dos_list)
    plt.xlabel("${\omega}$", fontsize=13)
    plt.ylabel("${N(\omega)}$", fontsize=13)
    plt.tight_layout()
    plt.savefig("./out/out.pdf")
    plt.show()
    
