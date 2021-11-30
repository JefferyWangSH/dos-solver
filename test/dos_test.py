import numpy as np
from matplotlib import pyplot as plt

# import sys
# sys.path.append("src")
# from dos_solver import DosParams
# from grids import FrequencyGrids, MomentumGrids
# from model import FreePropagator, Kernel, GreenFunc
# from dos_io import write_dos, read_dos

# def free_propagator_hole(k, omega):
#     """
#         free feynman propagator of hole excitation
#     """
#     omega_complex = omega + infinitesimal_imag*1.0j
#     return 1/(omega_complex + k**2/(2*mass) + fermi_surface )


# def self_energy(k, omega):
#     """
#         functional form of the approximated self-energy correction.
#     """
#     return 2*np.pi*static_gap**2*free_propagator_hole(k, omega) * (1 - free_propagator_hole(k, omega)/(2*mass*corr_length**2))



if "__main__":
    """
        Take approximate treatment of self-energy, and benchmark with exact results.
    """
    
    # set up model params
    freq_cutoff, k_cutoff = 6.0, 6.0
    freq_num, k_num = int(1e3), int(1e3)
    infinitesimal_imag = 0.05
    
    # continuum model with dispersion relation E = p^2/(2m) - mu
    mass = 0.5
    fermi_surface = -2.5
    static_gap = 0.1
    corr_length = 10.0

    # generate grids
    k_grids = np.linspace(0.0, k_cutoff, k_num)
    freq_grids = np.linspace(-freq_cutoff, freq_cutoff, freq_num)

    # generate feynman propagators: for both particle and hole
    freq_grids_complex = freq_grids + infinitesimal_imag*1.0j
    k_grids_trans = np.array(np.mat(k_grids).transpose())
    dispersion_trans = k_grids_trans**2/(2*mass) + fermi_surface
    free_propagator_particle = (freq_grids_complex - dispersion_trans)**-1
    free_propagator_hole = (freq_grids_complex + dispersion_trans)**-1

    # generate matrix of self-energy
    self_energy = 2*np.pi*static_gap**2 * free_propagator_hole * (1 - free_propagator_hole/(2*mass*corr_length**2))

    # generate matrix of green's function
    green_function = free_propagator_particle / (1-free_propagator_particle*self_energy)

    # generate matrix of spectrum function
    spectrum = -2 * np.imag(green_function)

    # compress to obatin density of state
    # TODO: check it out 
    dos_list = (k_grids*spectrum).sum(axis=0)/spectrum.shape[0]

    plt.figure()
    plt.plot(freq_grids, dos_list)
    plt.show()
