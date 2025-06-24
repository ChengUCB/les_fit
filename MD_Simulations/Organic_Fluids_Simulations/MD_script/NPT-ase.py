import os
import numpy as np
from mace.calculators import MACECalculator
from ase import units
from ase.md.npt import NPT
from ase.md import MDLogger
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import mol
from ase.optimize import BFGS
from ase.md.nose_hoover_chain import IsotropicMTKNPT

# Simulation parameters
TEMPERATURE = 298.15  # K
PRESSURE_BAR = 1.01325  # bar
TIMESTEP = 1 * units.fs
N_STEPS = 300000
OPTIMIZATION_FMAX = 0.2

# Load initial structure
atoms = read('INIT.xyz', '0')

# Setup MACE calculator
calculator = MACECalculator(
    model_path='../../SPICE_small-new-2025-05-27.model', 
    device='cuda'
)
atoms.set_calculator(calculator)

# Optimize structure
optimizer = BFGS(atoms)
optimizer.run(fmax=OPTIMIZATION_FMAX)
write('opt_structure.xyz', atoms)

# Set initial velocities
MaxwellBoltzmannDistribution(atoms, TEMPERATURE * units.kB)

# Initial NPT equilibration
npt_dyn = NPT(
    atoms, 
    timestep=TIMESTEP, 
    temperature_K=TEMPERATURE,
    ttime=10 * units.fs, 
    pfactor=None,
    externalstress=0.0
)
npt_dyn.run(5000)

# Main NPT simulation with Nosé-Hoover chains
dyn = IsotropicMTKNPT(
    atoms,
    timestep=TIMESTEP,
    temperature_K=TEMPERATURE,
    pressure_au=PRESSURE_BAR * units.bar,
    tdamp=50 * units.fs,
    pdamp=500 * units.fs,
    tchain=5,
    pchain=5,
    tloop=2,
    ploop=2
)

def print_energy(atoms=atoms):
    """Print potential, kinetic, and total energy per atom with density."""
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    
    # Calculate density
    mass_amu = atoms.get_masses().sum()
    mass_g = mass_amu / mol
    vol_A3 = atoms.get_volume()
    vol_cm3 = vol_A3 * 1e-24  # Convert A³ to cm³
    density = mass_g / vol_cm3  # g/cm³
    
    temperature = ekin / (1.5 * units.kB)
    etot = epot + ekin
    
    print(f'Energy per atom: Epot = {epot:.4f} eV  Ekin = {ekin:.4f} eV  '
          f'T = {temperature:.0f} K  Etot = {etot:.4f} eV  Density = {density:.4f} g/cm³')
    
    # Save density to file
    with open('density.txt', 'a') as f:
        f.write(f"{density}\n")

# Attach observers
dyn.attach(
    MDLogger(
        dyn, atoms, 
        f'log_{TEMPERATURE}K_1bar_mace.log', 
        header=True, stress=True, peratom=True, mode="w"
    ), 
    interval=100
)
dyn.attach(lambda: dyn.atoms.write('npt-mace.xyz', append=True), interval=100)
dyn.attach(print_energy, interval=10)

# Run simulation
dyn.run(N_STEPS)
