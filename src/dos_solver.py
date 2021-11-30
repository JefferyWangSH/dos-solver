import time
import numpy as np
from dos_params import DosParams
from grids import FrequencyGrids, MomentumGrids
from model import FreePropagator, Kernel, GreenFunc


class DosSolver:
    def __init__(self) -> None:
        self._freq_list = ()
        self._dos_list = ()
        self._timer = 0.0
    
    def clear(self) -> None:
        self._freq_list = ()
        self._dos_list = ()
        self._timer = 0.0

    def setKernel(self, kernel_func) -> None:
        self._kernel = kernel_func

    def setFreeBand(self, free_band_func) -> None:
        self._free_band = free_band_func

    def Data(self):
        return zip(self._freq_list, self._dos_list)

    def Timer(self) -> float:
        return self._timer

    def run(self, dos_params) -> None:
        assert(isinstance(dos_params, DosParams))

        # record cpu time
        begin_time = time.time()
        self.clear()

        # initialize grids
        freq_grid = FrequencyGrids(dos_params.freq_num, dos_params.freq_range)
        k_grid = MomentumGrids(dos_params.lattice_size)

        # initialize free propagator
        free_propagator = FreePropagator(freq_grid, k_grid)
        free_propagator.SetFreeBand(self._free_band)
        free_propagator.compute(dos_params = dos_params)

        # generate kernel
        kernel = Kernel(momentum_grids = k_grid)
        kernel.SetKernel(self._kernel)
        kernel.compute(dos_params = dos_params)

        # compute self energy and green's function
        green_func = GreenFunc(kernel = kernel, free_propagator = free_propagator)
        green_func.ComputeSelfEnergy()
        green_func.ComupteGreenFunc()

        # collecting spectrum A(k,omega)
        spectrum_mat = -2 * np.imag(green_func.GreenFuncMat())

        # collecting density of states
        dos_list = spectrum_mat.sum(axis=0)/spectrum_mat.shape[0]
        end_time = time.time()

        # save the results
        self._timer = end_time - begin_time
        self._freq_list = freq_grid.Grids()
        self._dos_list = dos_list
        
        