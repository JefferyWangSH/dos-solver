import numpy as np
from momentum import LatticeMomentum


class FrequencyGrids:
    """
        Grids of real frequnecy (1d)
    """
    def __init__(self, num_of_grids, freqency_range) -> None:
        assert(isinstance(num_of_grids, int))
        assert(isinstance(freqency_range, list))
        assert(len(freqency_range) == 2)
        assert(freqency_range[0] < freqency_range[1])
        self._num_of_grids = num_of_grids
        self._min_frequency = freqency_range[0]
        self._max_frequency = freqency_range[1]
        self._grids = np.linspace(self._min_frequency, self._max_frequency, self._num_of_grids)

    def __getitem__(self, index) -> float:
        assert(isinstance(index, int))
        assert(index >= 0 and index < self.GridsNum())
        return self.Grids()[index]

    def Grids(self):
        return self._grids

    def Min(self) -> float:
        return self._min_frequency

    def Max(self) -> float:
        return self._max_frequency

    def GridsNum(self) -> int:
        return self._num_of_grids


class MomentumGrids:
    """
        Grids in two-dimensional 1st Brillouin zone
    """
    def __init__(self, num_of_grids_per_length) -> None:
        assert(isinstance(num_of_grids_per_length, int))
        self._num_of_grids_per_length = num_of_grids_per_length
        self._num_of_grids = self._num_of_grids_per_length**2
        self._grids_interval_per_length = 2*np.pi / self._num_of_grids_per_length
        self._momentum_grids = np.array( [self.Grid2Momentum(id) for id in range(self._num_of_grids)] )

    def __getitem__(self, index) -> LatticeMomentum:
        assert(isinstance(index, int))
        assert(index >= 0 and index < self.GridsNum())
        return self.MomentumGrids()[index]

    def GridsNumPerLength(self) -> int:
        return self._num_of_grids_per_length

    def GridsNum(self) -> int:
        return self._num_of_grids
    
    def GridsInterval(self) -> float:
        return self._grids_interval_per_length

    def MomentumGrids(self):
        return self._momentum_grids

    """
        Convert grid index to specific 2d momentum k in the 1st BZ 
    """
    def Grid2Momentum(self, grid_id) -> LatticeMomentum:
        assert(isinstance(grid_id, int))
        assert(grid_id >= 0 and grid_id < self.GridsNum())
        kx = grid_id // self.GridsNumPerLength()
        ky = grid_id % self.GridsNumPerLength()
        return LatticeMomentum( np.array([-np.pi, -np.pi]) + self.GridsInterval() * np.array([kx, ky]) )
    
