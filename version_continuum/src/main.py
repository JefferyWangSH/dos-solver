import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


# f = lambda y, x: x*y**2
# print(integrate.dblquad(f, 0.0, 2.0, lambda x: 0, lambda x: 1))

def self_energy_integrand(p, theta) -> complex:
    """
        integrand function, the integration of which performs the self energy correction, 
        integrating over the dummy 2d momentum labeled by `p` and `theta`.
    """
    pass


if "__main__":
    """
        The main program
    """

    # set up params
    mass = 1.0
    fermi_surface = -10.0
    static_gap = 1.0
    corr_length = 5.0       # in uint of lattice ( or system ) length

    momentum_cut_off = 5,0  # cut-off of the integration of momentum
    infinitesimal_imag = 0.1

    pass


