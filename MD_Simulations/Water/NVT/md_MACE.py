import sys
import torch
from ase.io import read

import os
src = os.path.join('/global/home/users/dongjinkim/software/les', 'src')
if src not in sys.path:
    sys.path.insert(0, src)
from mace.calculators import MACECalculator

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write

###### variables ######
model_path = 'MACELES-OFF_small.model'
DEVICE = 'cuda'
temperature = 300
timestep = 0.25 # to ensure to capture the fast O-H vibrations
nsteps = 200000
trajectory_file = f'md_h2o.traj'
logfile = f'md_h2o.log'

###### load model ######
calculator = MACECalculator(model_paths=model_path, device=DEVICE)

###### load init_config ######
init_config = read('./liquid-64.xyz', index=0)
atoms = init_config.copy()
atoms.set_calculator(calculator)

from ase.optimize import BFGS
optimizer = BFGS(atoms)
optimizer.run(fmax=0.03)

###### set NVT velocity ######
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
MaxwellBoltzmannDistribution(atoms, temperature_K= temperature )

from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.logger import MDLogger
from ase import Atoms, units
from ase.io.trajectory import Trajectory
equil_dyn = NPT(atoms, 
                    timestep * units.fs, 
                    temperature_K= temperature,
                    ttime = 10 * units.fs,
                    pfactor=None,
                    externalstress=0.0
                        )

equil_dyn.run(1000)



md_logger = MDLogger(equil_dyn, atoms, logfile=logfile, 
                     header=True, stress=True, mode='w')
equil_dyn.attach(md_logger, interval=100)
traj = Trajectory(trajectory_file, 'w', atoms)
equil_dyn.attach(traj.write, interval=1)

from ase.io import write
def save_xyz(atoms):
    write('./md_out.xyz', atoms, format='extxyz', append=True)
equil_dyn.attach(save_xyz, interval=100, atoms=atoms)
equil_dyn.run(nsteps)
print("complete.")
