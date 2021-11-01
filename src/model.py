import numpy as np
from grids import FrequencyGrids, MomentumGrids
from lattice_momentum import LatticeMomentum


class FreePropagator:
    """
        Free Feynman propagator
    """
    def __init__(self, frequency_grids, momentum_grids):
        assert(isinstance(frequency_grids, FrequencyGrids))
        assert(isinstance(momentum_grids, MomentumGrids))
        self._freq_grids = frequency_grids
        self._momentum_grids = momentum_grids
        self._mass = 1.0
        self._chemical_potential = 1.0
        self._infinitesimal_imag = 1.0
        self._arrary2d = np.zeros((self.MomentumDim(), self.FrequencyDim()), dtype=complex)
    
    def SetModelParms(self, mass, chemical_potential) -> None:
        self._mass = mass
        self._chemical_potential = chemical_potential

    def SetInfinitesimalImag(self, infinitesimal_imag) -> None:
        assert(isinstance(infinitesimal_imag, float))
        self._infinitesimal_imag = infinitesimal_imag
    
    # def FreePropagator(self, k, omega):
    #     assert(isinstance(k, LatticeMomentum))
    #     assert(isinstance(omega, complex) or isinstance(omega, float))
    #     # the plus sign here indicates that,
    #     # there exist a partical-hole symmetry in our model of phase-disordered supercondutivity.
    #     return (omega + k.abs()**2/(2*self._mass) - self._chemical_potential)**-1

    def init(self):
        # use broadcast property of numpy and accelate the creation of free propagator matrix
        tmp_k = np.array(np.mat([k.abs() for k in self._momentum_grids.MomentumGrids()]).transpose())
        tmp_omega = self._freq_grids.Grids() + self._infinitesimal_imag * 1.0j
        self._arrary2d = (tmp_omega + tmp_k**2/(2*self._mass) - self._chemical_potential)**-1

        # for i, k in enumerate(self._momentum_grids.MomentumGrids()):
        #     for j, omega in enumerate(self._freq_grids.Grids()):
        #         omega = complex(omega, self._infinitesimal_imag)
        #         # the plus sign here indicates that,
        #         # there exist a partical-hole symmetry in our model of phase-disordered supercondutivity.
        #         self._arrary2d[i,j] = (omega + k.abs()**2/(2*self._mass) - self._chemical_potential)**-1

        # for i in range(self.MomentumDim()):
        #     for j in range(self.FrequencyDim()):
        #         k = self._momentum_grids[i].abs()
        #         omega = complex(self._freq_grids[j], self._infinitesimal_imag)

        #         # the plus sign here indicates that,
        #         # there exist a partical-hole symmetry in our model of phase-disordered supercondutivity.
        #         self._arrary2d[i,j] = (omega + k**2/(2*self._mass) - self._chemical_potential)**-1

    def FrequencyDim(self) -> int:
        return self._freq_grids.GridsNum()

    def MomentumDim(self) -> int:
        return self._momentum_grids.GridsNum()

    def Mat(self):
        return self._arrary2d

    # TODO: reload [][] operator


class Kernel:
    """
        Kernel between self energy and free Feynman propagator,
        in case of phase-disordered superconductivity, corresponding to the dynamical Coopergap 
    """
    def __init__(self, momentum_grids):
        assert(isinstance(momentum_grids, MomentumGrids))
        self._momentum_grids = momentum_grids
        self._static_gap = 1.0
        self._corr_length = 1.0
        self._array2d = np.zeros((self.Dim(), self.Dim()))

    def SetDisorderParams(self, static_gap, corr_length) -> None:
        assert(isinstance(static_gap, float))
        assert(isinstance(corr_length, float))
        self._static_gap = static_gap
        self._corr_length = corr_length

    # TODO: accelarate assignment of values, avoiding `for`
    def init(self) -> None:
        for i, k in enumerate(self._momentum_grids.MomentumGrids()):
            for j, p in enumerate(self._momentum_grids.MomentumGrids()):
                self._array2d[i,j] = (k+p).abs()
        self._array2d = (1+(self._array2d*self._corr_length)**2)**-1.5 * np.pi/2 * (self._static_gap*self._corr_length)**2 / self.Dim()

        # for i, k in enumerate(self._momentum_grids.MomentumGrids()):
        #     for j, p in enumerate(self._momentum_grids.MomentumGrids()):
        #         self._array2d[i,j] = (1+((k+p).abs()*self._corr_length)**2)**-1.5

        # for i in range(self.Dim()):
        #     for j in range(self.Dim()):
        #         k = self._momentum_grids[i]
        #         p = self._momentum_grids[j]
        #         self._array2d[i,j] = (1+((k+p).abs()*self._corr_length)**2)**-1.5
        # self._array2d *= np.pi/2 * (self._static_gap*self._corr_length)**2 / self.Dim()
    
    def Dim(self) -> int:
        return self._momentum_grids.GridsNum()

    def Mat(self):
        return self._array2d


# class SelfEnergy:
#     pass


class GreenFunc:
    """
        Retarded Green's function of interacting system, 
        evaluated with pertubation theroy by computing self energy correction.
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
        self._array2d_self_energy = self._kernel.Mat().dot(self._free_propagator.Mat())

    def ComupteGreenFunc(self) -> None:
        self._array2d_green_func = self._free_propagator.Mat() / (1-self._free_propagator.Mat()*self._array2d_self_energy)

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