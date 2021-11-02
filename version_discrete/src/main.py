import numpy as np
import time, datetime
from grids import FrequencyGrids, MomentumGrids
from model import FreePropagator, Kernel, GreenFunc
from lattice_momentum import LatticeMomentum
from matplotlib import pyplot as plt


"""
    subroutine for the evaluation of density of states
"""
def dos_run():
    # initialize grids
    freq_grid = FrequencyGrids(num_of_grids = freq_num, freqency_range = freq_range)
    k_grid = MomentumGrids(num_of_grids_per_length = lattice_size)

    # initialize free propagator
    free_propagator = FreePropagator(freq_grid, k_grid)
    free_propagator.SetModelParms(hopping = hopping, chemical_potential = fermi_surface)
    free_propagator.SetInfinitesimalImag(infinitesimal_imag = infinitesimal_imag)
    free_propagator.init()

    # generate kernel
    kernel = Kernel(momentum_grids = k_grid)
    kernel.SetDisorderParams(static_gap = static_gap, corr_length = corr_length)
    kernel.init()

    # compute self energy and green's function
    green_func = GreenFunc(kernel = kernel, free_propagator = free_propagator)
    green_func.ComputeSelfEnergy()
    green_func.ComupteGreenFunc()

    # collecting spectrum A(k,omega)
    spectrum_mat = np.zeros((green_func.MomentumDim(), green_func.FrequencyDim()))
    for i in range(spectrum_mat.shape[0]):
        for j in range(spectrum_mat.shape[1]):
            spectrum_mat[i,j] = -2 * green_func.GreenFuncMat()[i,j].imag

    # collecting density of states
    dos_list = spectrum_mat.sum(axis=0)/spectrum_mat.shape[0]
    print(" finished! ")
    return freq_grid.Grids(), dos_list


if "__main__":
    """
        The main program
    """

    # setting up params for grids
    lattice_size = 63
    freq_range = [-12.0, 12.0]
    freq_num = int(1e3)

    # some comments for the strategy of choicing the imaginary value:
    # a large imaginary value tends to smooth out the effect of finite lattice size, thus leading to a smooth dos curve.
    # hence, as the imaginary valude decreases, the lattice size should be correspondingly increased 
    # to avoid the sharp-peak behaviour of results.
    infinitesimal_imag = 0.1

    # setting up model params
    # choice free lattice model as our 0th order results of pertubation theory.
    # for a free lattice model, the motion of electrons is described 
    # by the hopping term between the nearest-neighbour (to the lowest order) sites.
    # the dispersion relation reads:
    #     E = - 2t ( cos(kx) + cos(ky) ) - mu
    # where k = (kx, ky) is discrete lattice momentum, t is hopping constant and mu being the chemical potential (fermi surface).
    # band width w = 8t.
    hopping = 1.0
    fermi_surface = 0.0

    # the gap should be sufficent low compared with hopping constant, such that the pertubation theory works.
    static_gap = 0.2
    corr_length = float(0.1*lattice_size)

    corr_length_range = list(lattice_size * np.array([ 0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 24.0 ]))
    data_list = []
    for corr_length in corr_length_range:
        data_list.append(dos_run())
    
    # plot the figure of dos
    plt.figure()
    plt.grid(linestyle='-.')
    for i in range(len(data_list)):
        omega_list, dos_list = data_list[i]
        # frequencies omega are measured in unit of half band width of free theory (4t)
        omega_list = omega_list/(4*hopping)
        corr_per_length = corr_length_range[i]/lattice_size
        str_corr_per_length = "{:.2f}".format(corr_per_length)
        plt.plot(omega_list, dos_list, label="${\\xi/L}$ = "+str_corr_per_length)
    plt.xlabel("${\omega/4t}$", fontsize=13)
    plt.ylabel("${N(\omega)}$", fontsize=13)
    plt.title("${L = 64 \\times 64}$,  ${\Delta_{0}/t = 0.2}$")
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.savefig("./version_discrete/results/out.pdf")
    plt.show()


    """
        codes for testing
    """
    # # record cpu time
    # time_begin = time.time()
    # time_end = time.time()
    # print(" The program gets started at %s .\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # # initialize grids
    # freq_grid = FrequencyGrids(num_of_grids = freq_num, freqency_range = freq_range)
    # k_grid = MomentumGrids(num_of_grids_per_length = lattice_size)
    # time_end = time.time()
    # print(" Grids initialized in %.2f s.\n" % (time_end-time_begin))
    # time_begin = time.time()

    # # initialize free propagator
    # free_propagator = FreePropagator(freq_grid, k_grid)
    # free_propagator.SetModelParms(hopping = hopping, chemical_potential = fermi_surface)
    # free_propagator.SetInfinitesimalImag(infinitesimal_imag = infinitesimal_imag)
    # free_propagator.init()
    # time_end = time.time()
    # print(" Generate free propagator in %.2f s.\n" % (time_end-time_begin))
    # time_begin = time.time()

    # # generate kernel
    # kernel = Kernel(momentum_grids = k_grid)
    # kernel.SetDisorderParams(static_gap = static_gap, corr_length = corr_length)
    # kernel.init()
    # time_end = time.time()
    # print(" Generate kernel in %.2f s.\n" % (time_end-time_begin))
    # time_begin = time.time()

    # # compute self energy and green's function
    # green_func = GreenFunc(kernel = kernel, free_propagator = free_propagator)
    # green_func.ComputeSelfEnergy()
    # green_func.ComupteGreenFunc()
    # time_end = time.time()
    # print(" Complete the computation of Green's function in %.2f s.\n" % (time_end-time_begin))
    # time_begin = time.time()

    # # collecting spectrum A(k,omega)
    # spectrum_mat = np.zeros((green_func.MomentumDim(), green_func.FrequencyDim()))
    # for i in range(spectrum_mat.shape[0]):
    #     for j in range(spectrum_mat.shape[1]):
    #         spectrum_mat[i,j] = -2 * green_func.GreenFuncMat()[i,j].imag

    # # collecting density of states
    # dos_list = spectrum_mat.sum(axis=0)/spectrum_mat.shape[0]

    # # testing the normalization condition
    # print(" Normalize condition for DOS is %.3f \n" % (sum(dos_list)*(freq_range[1]-freq_range[0])/freq_num))

    # # plot the figure
    # print(" Ploting the figure of DOS ... \n")
    # plt.figure()
    # plt.grid(linestyle='-.')
    # plt.plot(freq_grid.Grids(), dos_list)
    # plt.xlabel("${\omega/t}$", fontsize=13)
    # plt.ylabel("${N(\omega)}$", fontsize=13)
    # plt.tight_layout()
    # plt.savefig("./version_discrete/results/out.pdf")
    # plt.show()
