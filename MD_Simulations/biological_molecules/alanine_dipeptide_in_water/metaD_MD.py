import os
import time
from torch import cuda
import numpy as np
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin, MDLogger
from ase.calculators.plumed import Plumed
from mace.calculators import MACECalculator

# --- Set up PLUMED and output directory ---
os.environ["PLUMED_KERNEL"] = os.path.expanduser("~/plumed_build-prefix/lib/libplumedKernel.so")
output_dir = "output_md_restart"
os.makedirs(output_dir, exist_ok=True)
cuda.empty_cache()  # Free CUDA memory if using GPU

# --- Simulation parameters ---
model = "SPICE_small-MACELES-OFF.model"
temperature = 310  # Target temperature in Kelvin
device = "cuda"    # Device used for MACECalculator

# --- Load starting structure and assign calculator ---
atoms = read(os.path.join(output_dir, "nvt_metad.traj"), -1)
atoms.set_pbc([True, True, True])  # Enable periodic boundary conditions
calculator = MACECalculator(
    model_paths=model,
    device=device,
    default_dtype="float32"
)
atoms.calc = calculator

# --- Relax structure with LBFGS optimization ---
opt = LBFGS(atoms)
opt.run(fmax=0.3)  # Stop optimization when the maximum force drops below 0.3 eV/Å

# --- Initialize velocities via Maxwell-Boltzmann distribution ---
MaxwellBoltzmannDistribution(atoms, temperature * units.kB)

# --- NPT Equilibration using Berendsen Barostat ---
npt_steps = 10000
traj_npt_path = os.path.join(output_dir, "npt_equil.traj")
traj_writer_npt = Trajectory(traj_npt_path, 'w', atoms)

dyn_npt = NPTBerendsen(
    atoms,
    timestep=1 * units.fs,
    temperature_K=temperature,
    taut=20 * units.fs,
    pressure_au=1.01325 * units.bar,  # 1 atm in ASE units
    taup=200 * units.fs,
    compressibility_au=4.57e-5 / units.bar,
)

log_npt_path = os.path.join(output_dir, f"log_{temperature}K_npt.log")
mdlogger_npt = MDLogger(
    dyn_npt, atoms, log_npt_path,
    header=True,
    stress=False,
    peratom=True,
    mode="w"
)
dyn_npt.attach(mdlogger_npt, interval=500)
dyn_npt.attach(traj_writer_npt.write, interval=500)

# Print per-atom energies and instantaneous temperature function
def print_energy(a=atoms):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temp_inst = ekin / (1.5 * units.kB)
    print(f"Energy per atom: Epot = {epot:.4f} eV  Ekin = {ekin:.4f} eV  "
          f"T = {float(temp_inst):.0f} K  Etot = {epot+ekin:.4f} eV")

dyn_npt.attach(print_energy, interval=200)
dyn_npt.run(npt_steps)

# --- NVT Metadynamics Setup (Read equilibrated structure) ---
atoms = read(traj_npt_path, -1)
atoms.calc = calculator  # Reattach calculator, needed after reload

# Define PLUMED input
plumed_input = [
    "UNITS LENGTH=A TIME=fs ENERGY=kj/mol",
    "RESTART",
    "phi:   TORSION ATOMS=5,7,9,15",  # monitor phi torsion
    "psi:   TORSION ATOMS=7,9,15,17", # monitor psi torsion
    "metad: METAD ARG=phi,psi PACE=500 HEIGHT=2.5 SIGMA=500 BIASFACTOR=30 TEMP=310 ADAPTIVE=DIFF FILE=HILLS",
    "PRINT ARG=phi,psi,metad.bias FILE=COLVAR STRIDE=500",
]

# Attach PLUMED bias as calculator
atoms.calc = Plumed(
    calc=calculator,
    input=plumed_input,
    timestep=1.0 * units.fs,
    atoms=atoms,
    kT=temperature * units.kB,
)

# Set up NVT Langevin dynamics for Metadynamics run
dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=temperature,
    friction=1.0,
)

traj_nvt_path = os.path.join(output_dir, "nvt_metad.traj")
traj_writer_nvt = Trajectory(traj_nvt_path, 'w', atoms)
log_nvt_path = os.path.join(output_dir, f"log_{temperature}K_nvt.log")

mdlogger_nvt = MDLogger(
    dyn, atoms, log_nvt_path,
    header=True,
    stress=False,
    peratom=True,
    mode="w"
)

# Attach loggers and energy printout
dyn.attach(mdlogger_nvt, interval=100)
dyn.attach(traj_writer_nvt.write, interval=100)
dyn.attach(print_energy, interval=100)

# --- Production NVT + Metadynamics run ---
n_steps = 4_000_000
dyn.run(n_steps)
