import numpy as np
from dos_params import DosParams
from grids import FrequencyGrids, MomentumGrids


class FreePropagator:
    """
        Free Feynman propagator
    """
    def __init__(self, frequency_grids, momentum_grids):
        assert(isinstance(frequency_grids, FrequencyGrids))
        assert(isinstance(momentum_grids, MomentumGrids))
        self._freq_grids = frequency_grids
        self._momentum_grids = momentum_grids
        self._arrary2d_particle = np.zeros((self.MomentumDim(), self.FrequencyDim()), dtype=complex)
        self._arrary2d_hole = np.zeros((self.MomentumDim(), self.FrequencyDim()), dtype=complex)

    def SetFreeBand(self, free_band_func) -> None:
        self._free_band = free_band_func

    def compute(self, dos_params):
        assert(isinstance(dos_params, DosParams))
        # use broadcast property of numpy and accelate the creation of free propagator matrix
        kx_trans = np.array(np.mat([ k[0] for k in self._momentum_grids.MomentumGrids() ]).transpose())
        ky_trans = np.array(np.mat([ k[1] for k in self._momentum_grids.MomentumGrids() ]).transpose())
        tmp_ek = self._free_band(kx_trans, ky_trans, dos_params)
        tmp_omega = self._freq_grids.Grids() + dos_params.infinitesimal_imag * 1.0j
        
        # the plus sign here indicates that,
        # there exist a particle-hole symmetry in our model of phase-disordered supercondutivity.
        self._arrary2d_hole = (tmp_omega + tmp_ek)**-1
        self._arrary2d_particle = (tmp_omega - tmp_ek)**-1

    def FrequencyDim(self) -> int:
        return self._freq_grids.GridsNum()

    def MomentumDim(self) -> int:
        return self._momentum_grids.GridsNum()

    def ParticleMat(self):
        return self._arrary2d_particle

    def HoleMat(self):
        return self._arrary2d_hole

    # TODO: reload [][] operator


class Kernel:
    """
        Kernel between self energy and free Feynman propagator.
    """
    def __init__(self, momentum_grids):
        assert(isinstance(momentum_grids, MomentumGrids))
        self._momentum_grids = momentum_grids
        self._array2d = np.zeros((self.Dim(), self.Dim()))

    def SetKernel(self, kernel_func) -> None:
        self._kernel = kernel_func

    # this step, the generation of kernel, should be the most computational expensive part of the program,
    # in case of large size of lattice.
    def compute(self, dos_params) -> None:
        assert(isinstance(dos_params, DosParams))
        # accelarate the fabrication of kernel by using operations between arrays.
        # again, the broadcast property of numpy arrays is used.
        tmp_px = np.array([ k.data()[0] for k in self._momentum_grids.MomentumGrids() ])
        tmp_py = np.array([ k.data()[1] for k in self._momentum_grids.MomentumGrids() ])
        tmp_kx = np.array(np.mat(tmp_px).transpose())
        tmp_ky = np.array(np.mat(tmp_py).transpose())
        self._array2d = self._kernel(tmp_kx, tmp_ky, tmp_px, tmp_py, dos_params)

    def Dim(self) -> int:
        return self._momentum_grids.GridsNum()

    def Mat(self):
        return self._array2d


# class SelfEnergy:
#     pass


class GreenFunc:
    """
        Retarded Green's function of interacting system, 
        evaluated by computing the self energy correction using pertubation theroy 
    """
    def __init__(self, kernel, free_propagator):
        assert(isinstance(kernel, Kernel))
        assert(isinstance(free_propagator, FreePropagator))
        self._momentum_grids = free_propagator._momentum_grids
        self._freq_grids = free_propagator._freq_grids
        self._kernel = kernel
        self._free_propagator = free_propagator
        self._array2d_self_energy = np.zeros((self._free_propagator.MomentumDim(), self._free_propagator.FrequencyDim()), dtype=complex)
        self._array2d_green_func = np.zeros((self._free_propagator.MomentumDim(), self._free_propagator.FrequencyDim()), dtype=complex)


    def ComputeSelfEnergy(self) -> None:
        self._array2d_self_energy = self._kernel.Mat().dot(self._free_propagator.HoleMat())

    def ComupteGreenFunc(self) -> None:
        self._array2d_green_func = self._free_propagator.ParticleMat() / (1-self._free_propagator.ParticleMat()*self._array2d_self_energy)

    def SelfEnergyMat(self):
        return self._array2d_self_energy

    def GreenFuncMat(self):
        return self._array2d_green_func

    def FrequencyDim(self) -> int:
        return self.GreenFuncMat().shape[1]

    def MomentumDim(self) -> int:
        return self.GreenFuncMat().shape[0]


# class PhaseDisorderedSC:
#     pass