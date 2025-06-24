import sys
import torch
import time
import torch.nn as nn
from mace.calculators import MACECalculator
from ase import Atoms
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase import units
from ase.io import read, write
import numpy as np
from ase.units import mol
import scipy.constants as constants

temperature = 298 # in K

atoms = read('./single_molecule.pdb', '0')

calculator = MACECalculator(model_paths='../../SPICE_small-new-2025-05-27.model', device='cuda', default_dtype="float32")
atoms.set_calculator(calculator)

# No need to define a unit cell for vacuum
atoms.cell = None
atoms.pbc = False  # Ensure no periodic boundary conditions

from ase.optimize import BFGS
optimizer = BFGS(atoms, maxstep=0.1)
optimizer.run(fmax=0.05)

# Set initial velocities using Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(atoms, temperature * units.kB)
NVTdamping_timescale = 10 * units.fs  # Time constant for NVT dynamics (NPT includes both)


from ase.md import VelocityVerlet
from ase.md import Langevin
#from ase.md import NoseHoover

timestep = 1 * units.fs
friction = 0.1 / (1000 * units.fs)  # Adjust friction coefficient as needed
dyn = Langevin(atoms, timestep, temperature_K=temperature, friction=friction)
#ttime = 20 * units.fs  # Thermostat time constant
#dyn = NoseHoover(atoms, timestep, temperature=temperature * units.kB, ttanh=ttime)


def print_energy(a=atoms):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4f eV  Ekin = %.4f eV  T=%3.0f K  Etot = %.4f  ' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


dyn.attach(MDLogger(dyn, atoms, f'log_{temperature}K_vac.log', header=True, stress=False,
           peratom=True, mode="w"), interval=100)

dyn.attach(print_energy, interval=100)

# Run the MD simulation
n_steps = 300000
dyn.run(n_steps)

