import numpy as np


class LatticeMomentum:
    """
        Class for 2d lattice momentum in the 1st BZ,
        to which the periodical boundary condition is applied.
    """
    def __init__(self, lattice_momentum) -> None:
        assert(isinstance(lattice_momentum, type(np.array(2))))
        self._latice_momentum = lattice_momentum
        self.wrap()

    def __add__(self, lattice_momentum):
        assert(isinstance(lattice_momentum, LatticeMomentum))
        return LatticeMomentum(self.data() + lattice_momentum.data())

    def __getitem__(self, index) -> float:
        assert(isinstance(index, int))
        assert(index == 0 or index == 1)
        return self.data()[index]

    def data(self) -> np.array(2):
        return self._latice_momentum

    def abs(self) -> float:
        return (self[0]**2 + self[1]**2)**0.5

    """
        Confine the lattice momentum in the 1st BZ, and wrap at the boundaries due to PBC
    """
    def wrap(self):
        self._latice_momentum = np.array( [self.data()[i] - 2*np.pi*((self.data()[i]+np.pi)//(2*np.pi)) for i in range(len(self.data()))] )


