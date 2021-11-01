import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


# f = lambda y, x: x*y**2
# print(integrate.dblquad(f, 0.0, 2.0, lambda x: 0, lambda x: 1))

def kernel() -> float:
    """
        kernel function, which connects free feynman propagator with matsubara green's function of interaction systems
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

