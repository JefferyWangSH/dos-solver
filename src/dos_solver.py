import time
import numpy as np
from grids import FrequencyGrids, MomentumGrids
from model import FreePropagator, Kernel, GreenFunc


class DosParams:
    def __init__(self) -> None:
        self.lattice_size = 100
        self.freq_range = [-6.0, 6.0]
        self.freq_num = int(1e3)
        self.infinitesimal_imag = 0.08

        self.hopping = 1.0
        self.fermi_surface = 0.0

        self.static_gap = 0.1
        self.corr_length = float(0.1*self.lattice_size)


class DosSolver:
    def __init__(self) -> None:
        pass
    
    def run(self, dos_params) -> None:
        assert(isinstance(dos_params, DosParams))
        # record cpu time
        begin_time = time.time()

        # initialize grids
        freq_grid = FrequencyGrids(dos_params.freq_num, dos_params.freq_range)
        k_grid = MomentumGrids(dos_params.lattice_size)

        # initialize free propagator
        free_propagator = FreePropagator(freq_grid, k_grid)
        free_propagator.SetModelParms(dos_params.hopping, dos_params.fermi_surface)
        free_propagator.SetInfinitesimalImag(dos_params.infinitesimal_imag)
        free_propagator.init()

        # generate kernel
        kernel = Kernel(momentum_grids = k_grid)
        kernel.SetDisorderParams(dos_params.static_gap, dos_params.corr_length)
        kernel.init()

        # compute self energy and green's function
        green_func = GreenFunc(kernel = kernel, free_propagator = free_propagator)
        green_func.ComputeSelfEnergy()
        green_func.ComupteGreenFunc()

        # collecting spectrum A(k,omega)
        spectrum_mat = -2 * np.imag(green_func.GreenFuncMat())

        # collecting density of states
        dos_list = spectrum_mat.sum(axis=0)/spectrum_mat.shape[0]
        
        end_time = time.time()
        print(" Finished in {:.2f} s. \n".format(end_time-begin_time))

        # save the results
        self.data = zip(freq_grid.Grids(), dos_list)

