import numpy as np
from matplotlib import pyplot as plt


def calculate_dos_exact():
    # generate grids for momentums of 2d
    k_grids_1d = np.linspace(-k_cutoff, k_cutoff, k_num)
    px_grids_2d = np.array([px for px in k_grids_1d for py in k_grids_1d])
    py_grids_2d = np.array([py for px in k_grids_1d for py in k_grids_1d])
    kx_grids_2d = np.array(np.mat(px_grids_2d).transpose())
    ky_grids_2d = np.array(np.mat(py_grids_2d).transpose())
    freq_grids = np.linspace(-freq_cutoff, freq_cutoff, freq_num)
    
    # generate kernel (gaussian-type)
    kernel = np.exp(-0.5*((kx_grids_2d+px_grids_2d)**2 + (ky_grids_2d+py_grids_2d)**2)*corr_length**2) \
             * 2*np.pi * (static_gap * corr_length)**2 *(2*k_cutoff/k_num)**2

    # generate feynman propagator
    ek_grids = (kx_grids_2d**2+ky_grids_2d**2)/(2*mass) + fermi_surface
    freq_grids_complex = freq_grids + infinitesimal_imag*1.0j
    free_propagator_particle = (freq_grids_complex - ek_grids)**-1
    free_propagator_hole = (freq_grids_complex + ek_grids)**-1

    # calculate self energy correction and green's function
    self_energy = kernel.dot(free_propagator_hole)
    green_func = free_propagator_particle / (1 - free_propagator_particle * self_energy)

    # allocate spectrum function
    spectrum = -2 * np.imag(green_func)

    # compress to obatin density of state
    dos_list = spectrum.sum(axis=0)/spectrum.shape[0]
    return freq_grids, dos_list


def calculate_dos_approximate():
    # generate grids
    k_grids = np.linspace(0.0, k_cutoff, k_num)
    freq_grids = np.linspace(-freq_cutoff, freq_cutoff, freq_num)

    # generate feynman propagators: for both particle and hole
    freq_grids_complex = freq_grids + infinitesimal_imag*1.0j
    k_grids_trans = np.array(np.mat(k_grids).transpose())
    dispersion_trans = k_grids_trans**2/(2*mass) + fermi_surface
    free_propagator_particle = (freq_grids_complex - dispersion_trans)**-1
    free_propagator_hole = (freq_grids_complex + dispersion_trans)**-1

    # generate approximated self-energy
    self_energy = 2*np.pi*static_gap**2 * free_propagator_hole * (1 - free_propagator_hole/(2*mass*corr_length**2))

    # compute green's function
    green_function = free_propagator_particle / (1-free_propagator_particle*self_energy)

    # allocate spectrum function
    spectrum = -2 * np.imag(green_function)

    # compress to obatin density of state
    # TODO: check it out 
    dos_list = (np.array(np.mat(k_grids).transpose())*spectrum).sum(axis=0)/spectrum.shape[0]
    return freq_grids, dos_list



if "__main__":
    """
        Take approximate treatment of self-energy, and benchmark with exact results.
    """
    
    # set up model params
    freq_cutoff, k_cutoff = 8.0, 6.0
    freq_num, k_num = int(1e3), int(1e2)
    infinitesimal_imag = 0.3
    
    # continuum model with dispersion relation E = p^2/(2m) - mu
    mass = 1.0
    fermi_surface = -5.0
    static_gap = 0.1
    corr_length = 10.0

    # exact, relatively speaking, results from calculation
    freq_exact, dos_exact = calculate_dos_exact()

    # approximated results
    freq_approx, dos_approx = calculate_dos_approximate()

    # plot and comparison
    plt.figure()
    plt.grid(linestyle="-.")
    plt.plot(freq_exact, dos_exact, label="Exact")
    plt.plot(freq_approx, dos_approx, label="Approximate")
    plt.ylim(bottom=0.0)
    plt.xlabel("${\omega}$", fontsize=13)
    plt.ylabel("${N(\omega)}$", fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("./test/compare.pdf", dpi=200)