import numpy as np
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
        self._hopping = 1.0
        self._chemical_potential = 1.0
        self._infinitesimal_imag = 1.0
        self._arrary2d_particle = np.zeros((self.MomentumDim(), self.FrequencyDim()), dtype=complex)
        self._arrary2d_hole = np.zeros((self.MomentumDim(), self.FrequencyDim()), dtype=complex)
    
    def SetModelParms(self, hopping, chemical_potential) -> None:
        self._hopping = hopping
        self._chemical_potential = chemical_potential

    def SetInfinitesimalImag(self, infinitesimal_imag) -> None:
        assert(isinstance(infinitesimal_imag, float))
        self._infinitesimal_imag = infinitesimal_imag

    def init(self):
        # use broadcast property of numpy and accelate the creation of free propagator matrix
        tmp_ek = np.array(np.mat([ k.energy(self._hopping, self._chemical_potential) for k in self._momentum_grids.MomentumGrids() ]).transpose())
        tmp_omega = self._freq_grids.Grids() + self._infinitesimal_imag * 1.0j
        self._arrary2d_particle = (tmp_omega - tmp_ek)**-1
        # the plus sign here indicates that,
        # there exist a particle-hole symmetry in our model of phase-disordered supercondutivity.
        self._arrary2d_hole = (tmp_omega + tmp_ek)**-1

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
        In case of phase-disordered superconductivity, 
        this corresponds to the fourier transformation of space correlation of Cooper gap. 
        E.g. In two dimensional space
            for exponential decayed correlation ~ exp( -r/(xi) )
                Kernel = 2 pi * Delta0^2 * ( (xi)^2 / V ) * ( 1 + (k+p)^2 (xi)^2 )^-1.5
            which is 2d lorentz-type.

            for gaussian-type decayed correlation ~ exp( -(r/(xi))^2 ) 
                Kernel = 2 pi * Delta0^2 * ( (xi)^2 / V ) * exp( -0.5 (k+p)^2 (xi)^2 )
            which is also gaussian-type.
        
        In practice, different types of kernel does not have significant impact on the final results of density of states.
        This is mainly because that :
            When the characteristic length of the gap correlation is large enough so that the gap of d.o.s. exists,
            the kernel degenerates to a delta function and it is the lattice momentum p = -k that dominates the physics of the system.
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

    # this step, the generation of kernel, should be the most computational expensive part of the program,
    # in case of large size of lattice.
    def init(self) -> None:
        # accelarate the fabrication of kernel by using operations between arrays.
        # again, the broadcast property of numpy arrays is used.
        tmp_px = np.array([ k.data()[0] for k in self._momentum_grids.MomentumGrids() ])
        tmp_py = np.array([ k.data()[1] for k in self._momentum_grids.MomentumGrids() ])
        tmp_kx = np.array(np.mat(tmp_px).transpose())
        tmp_ky = np.array(np.mat(tmp_py).transpose())
        self._array2d = ((tmp_kx+tmp_px) - 2*np.pi*((tmp_kx+tmp_px+np.pi)//(2*np.pi)))**2 \
                      + ((tmp_ky+tmp_py) - 2*np.pi*((tmp_ky+tmp_py+np.pi)//(2*np.pi)))**2

        # lorentz correlation
        self._array2d = (1+self._array2d*(self._corr_length)**2)**-1.5 * 2*np.pi * (self._static_gap*self._corr_length)**2 / self.Dim()

        # # gaussian correlation
        # self._array2d = np.exp((-0.5*self._array2d*(self._corr_length)**2)) * 2*np.pi * (self._static_gap*self._corr_length)**2 / self.Dim()

    
    def Dim(self) -> int:
        return self._momentum_grids.GridsNum()

    def Mat(self):
        return self._array2d


# class SelfEnergy:
#     pass


class GreenFunc:
    """
        Retarded Green's function of interacting system, 
        evaluated with pertubation theroy by computing the self energy correction.
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