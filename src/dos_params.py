class DosParams:
    """
        Params class for dos calculation
    """
    def __init__(self) -> None:
        self.lattice_size = 100
        self.freq_range = [-6.0, 6.0]
        self.freq_num = int(1e3)
        self.infinitesimal_imag = 0.08

        self.hopping = 1.0
        self.fermi_surface = 0.0
        self.static_gap = 0.1
        self.corr_length = float(0.1*self.lattice_size)

        