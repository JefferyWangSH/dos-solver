import numpy as np
from matplotlib import pyplot as plt

from dos_solver import DosParams, DosSolver
from grids import FrequencyGrids, MomentumGrids
from model import FreePropagator, Kernel, GreenFunc
from dos_io import write_dos, read_dos


if "__main__":
    """
        The main program
    """

    # params for dos calculation
    params = DosParams()

    # setting up params for grids
    params.lattice_size = 100
    params.freq_range = [-6.0, 6.0]
    params.freq_num = int(1e3)

    # some comments for the strategy of choicing the imaginary value:
    # a large imaginary value tends to smooth out the effect of finite lattice size, thus leading to a smooth dos curve.
    # hence, as the imaginary valude decreases, the lattice size should be correspondingly increased 
    # to avoid the sharp-peak behaviour of results.
    params.infinitesimal_imag = 0.08

    # setting up model params
    # choice free lattice model as our 0th order results of pertubation theory.
    # for a free lattice model, the motion of electrons is described 
    # by the hopping term between the nearest-neighbour (to the lowest order) sites.
    # the dispersion relation reads:
    #     E = - 2t ( cos(kx) + cos(ky) ) - mu
    # where k = (kx, ky) is discrete lattice momentum, t is hopping constant and mu being the chemical potential (fermi surface).
    # band width w = 8t.
    params.hopping = 1.0
    params.fermi_surface = 0.0

    # the gap should be sufficent low compared with hopping constant, such that the pertubation theory works.
    params.static_gap = 0.1
    params.corr_length = float(0.1*params.lattice_size)
    
    # scan range of correlation
    corr_length_range = list(params.lattice_size * np.array([ 0.0, 0.05, 0.1, 1.0, 2.0 ]))

    # do the calculation
    solver = DosSolver()
    solver.setKernel("gaussian")

    data_list = []
    for corr_length in corr_length_range:
        params.corr_length = corr_length

        # run with given params
        solver.run(params)
        data_list.append(solver.Data())
        print(" Finished in {:.2f} s. \n".format(solver.Timer()))

        # file output
        out_file_path = "./results/{:s}/L{:d}Cor{:.2f}.dat".format(solver.KernelType(), params.lattice_size, params.corr_length/params.lattice_size)
        write_dos(solver.Data(), out_file_path)


    # # read calculation data from file
    # # for benchmark usage
    # data_list = []
    # for id, corr_length in enumerate(corr_length_range):
    #     in_file_path = "./results/lorentz/L{:d}Cor{:.2f}.dat".format(params.lattice_size, corr_length/params.lattice_size)
    #     data_list.append(read_dos(in_file_path))


    # plot dos figure
    plt.figure()
    plt.grid(linestyle='-.')

    for i, data in enumerate(data_list):
        omega_list, dos_list = zip(*data)
        omega_list = np.array(omega_list) / (4*params.hopping)
        dos_list = np.array(dos_list)

        # with frequencies measured in unit of half band width of free theory (4t)
        corr_per_length = corr_length_range[i] / params.lattice_size
        str_corr_per_length = "{:.2f}".format(corr_per_length)
        plt.plot(omega_list, dos_list, label="${\\xi/L}$ = "+str_corr_per_length)

    # set up figure style and legends
    x_label = "${\omega/4t}$"
    y_label = "${N(\omega)}$"
    title = "${L = %d \\times %d}$,  ${\Delta_{0}/t = %.1f}$" % (params.lattice_size, params.lattice_size, params.static_gap/params.hopping)
    font = {'family' : 'Times New Roman', 'weight' : 'bold', 'size' : 13, }
    
    plt.xlabel(x_label, font)
    plt.ylabel(y_label, font)
    plt.title(title)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("./results/phase_disordered_sc.pdf", dpi=200)
    plt.show()


    """
        benchmark between different kernels
    """

    # fmt_path = "./results/{:s}/L100Cor0.10.dat"
    # freq_lorentz, dos_lorentz = zip(*read_dos(fmt_path.format("lorentz")))
    # freq_gaussian, dos_gaussian = zip(*read_dos(fmt_path.format("gaussian")))

    # plt.figure()
    # plt.grid(linestyle='-.')
    # plt.plot(freq_lorentz, dos_lorentz, label="Lorentz")
    # plt.plot(freq_gaussian, dos_gaussian, label="Gaussian")
    # plt.title("benchmark")
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.show()


    """
        testing codes
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
