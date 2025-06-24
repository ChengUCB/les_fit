import sys
import os
import numpy as np
import torch
import torch.nn as nn

import les
from mace.calculators import MACECalculator
from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.bussi import Bussi
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import mol

import scipy.constants as constants


atoms = read('mol_liquid_custom_periodic_optimised.xyz', '0')

#atoms = read('opt_structure.xyz', 0)
calculator = MACECalculator(model_path='../../SPICE_small-new-2025-05-27.model', device='cuda')
atoms.set_calculator(calculator)

from ase.optimize import BFGS
optimizer = BFGS(atoms)
optimizer.run(fmax=0.2)
write('opt_structure.xyz', atoms)

temperature = 298.15 # in K
pressure_in_bar = 1.01325 # in bar

# Set initial velocities using Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(atoms, temperature * units.kB)

# Define the NPT ensemble with isotropic compression


NPTdamping_timescale = 100 * units.fs  # Time constant for NPT dynamics
NVTdamping_timescale = 10 * units.fs  # Time constant for NVT dynamics (NPT includes both)

dyn = NPT(atoms, timestep=1 * units.fs, temperature_K=temperature,
          ttime=NVTdamping_timescale, pfactor=None,
          externalstress=0.0,)
dyn.run(1000)

from ase.md.nose_hoover_chain import NoseHooverChainNVT, IsotropicMTKNPT

pressure_GPa = 10

dyn = IsotropicMTKNPT(
        atoms,
        timestep=1 * units.fs,
        temperature_K=temperature,
        pressure_au=1.01325 * units.bar,
        #pressure_GPa=float(pressure_GPa),
        tdamp=50 * units.fs,
        pdamp=500 * units.fs,
        tchain=5,
        pchain=5,
        tloop=2,
        ploop=2)


def print_energy(a=atoms):
    """Function to print the potential, kinetic and total energy per atom."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    mass_amu = a.get_masses().sum()
    mass_g = mass_amu / mol
    vol_A3 = a.get_volume()  # A^3
    A3_to_cm3 = 1e-24  # 1 A^3 = 1e-24 cm^3
    vol_cm3 = vol_A3 * A3_to_cm3 
    system_density = mass_g / vol_cm3  #g/cm^3
 
    print('Energy per atom: Epot = %.4f eV  Ekin = %.4f eV  T=%3.0f K  Etot = %.4f eV   Density = %.4f ' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, system_density))

    # save density to a separate file
    with open('density.txt', 'a') as f:
        f.write(f"{system_density}\n")

dyn.attach(MDLogger(dyn, atoms, f'log_{temperature}K_1bar_mace.log', header=True, stress=True,
           peratom=True, mode="w"), interval=100)


dyn.attach(lambda: dyn.atoms.write('npt-mace.xyz', append=True), interval=100)
dyn.attach(print_energy, interval=10)

# Run the MD simulation
n_steps = 300000
dyn.run(n_steps)


