import sys
from torch import cuda
import time
import os
import numpy as np
from mace.calculators import MACECalculator
from ase import Atoms, units
from ase.md.nptberendsen import NPTBerendsen
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
from ase.io import read, write
from ase.calculators.plumed import Plumed
from ase.io.trajectory import Trajectory
from ase.md import MDLogger


os.environ["PLUMED_KERNEL"]  = "~/plumed_build-prefix/lib/libplumedKernel.so"

mace_type = "macelesoff"
temperature = 310
OUTPUT_DIR = "output_md_restart"
pdb = "alanine-dipeptide-solvated.pdb"
model = "SPICE_small-MACELES-OFF.model" if mace_type == "macelesoff" else "MACE-OFF23_medium.model"
device = "cuda" if cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
cuda.empty_cache()

atoms = read("./output_md_restart/nvt_metad_macelesoff.traj", "-1")
atoms.set_pbc([True, True, True])

calculator = MACECalculator(
    model_paths=f"{model}",
    device=device,
    default_dtype="float32"
)
atoms.calc = calculator

#opt = LBFGS(atoms)
#opt.run(fmax=0.3)

MaxwellBoltzmannDistribution(atoms, temperature * units.kB)

npt_steps = 10000
traj_npt = os.path.join(OUTPUT_DIR, f"npt_equil{mace_type}.traj")
traj_writer_npt = Trajectory(traj_npt, 'w', atoms)

dyn_npt = NPTBerendsen(
    atoms,
    timestep=1 * units.fs,
    temperature_K=temperature,
    taut=20 * units.fs,
    pressure_au=1.01325 * units.bar,
    taup=200 * units.fs,
    compressibility_au=4.57e-5 / units.bar
)

log_npt = os.path.join(OUTPUT_DIR, f"log_{temperature}K_npt_{mace_type}.log")
dyn_npt.attach(MDLogger(dyn_npt, atoms, log_npt, header=True, stress=False, peratom=True, mode="w"), interval=500)
dyn_npt.attach(traj_writer_npt.write, interval=500)

state = {"last_time": time.time()}
def print_energy(a=atoms):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temp_inst = ekin / (1.5 * units.kB)
    print(f"Energy per atom: Epot = {epot:.4f} eV  Ekin = {ekin:.4f} eV  T = {float(temp_inst):.0f} K  Etot = {epot+ekin:.4f} eV")


def report_speed(interval_steps):
    now = time.time()
    elapsed = now - state["last_time"]
    sim_ps = (interval_steps * 1.0) * 1e-3
    ns_per_day = (sim_ps / elapsed) * 86400.0 * 1e-3
    print(f"[{interval_steps} steps] avg speed: {ns_per_day:.3f} ns/day")
    state["last_time"] = now


dyn_npt.attach(print_energy, interval=200)
dyn_npt.attach(lambda: report_speed(200), interval=200)
dyn_npt.run(npt_steps)

atoms = read(traj_npt, index=-1)
atoms.calc = calculator

plumed_input = [
    "UNITS LENGTH=A TIME=fs ENERGY=kj/mol",
    "RESTART",
    "phi:   TORSION ATOMS=5,7,9,15",
    "psi:   TORSION ATOMS=7,9,15,17",
    "metad: METAD ARG=phi,psi PACE=500 HEIGHT=2.5 SIGMA=500 BIASFACTOR=30 TEMP=310 ADAPTIVE=DIFF FILE=HILLS",
    "PRINT ARG=phi,psi,metad.bias FILE=COLVAR STRIDE=500"
]

atoms.calc = Plumed(calc=calculator, input=plumed_input, timestep=1.0 * units.fs, atoms=atoms, kT=temperature * units.kB)

dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=temperature, friction=1.0)

traj_nvt = os.path.join(OUTPUT_DIR, f"nvt_metad_{mace_type}.traj")
traj_writer_nvt = Trajectory(traj_nvt, 'w', atoms)
log_nvt = os.path.join(OUTPUT_DIR, f"log_{temperature}K_nvt_{mace_type}.log")

dyn.attach(MDLogger(dyn, atoms, log_nvt, header=True, stress=False, peratom=True, mode="w"), interval=100)
dyn.attach(traj_writer_nvt.write, interval=100)
dyn.attach(print_energy, interval=100)
dyn.attach(lambda: report_speed(100), interval=100)

n_steps = 4000000
dyn.run(n_steps)

