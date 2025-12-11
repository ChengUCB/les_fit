#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append('/global/home/users/dongjinkim/software/cace/')

import numpy as np
import torch
import torch.nn as nn
import logging

import cace
from cace.tools import torch_geometric
from cace.data import AtomicData


root = './best_model.pth'
average_E0 = {1: -5.853064337340629, 8: -2.926532168670322} # avg
cace_nnp = torch.load(root, map_location='cuda')

if len(sys.argv) < 2:
    print("Usage: python test.py path/to/test_SYSTEM.xyz")
    sys.exit(1)


to_read = '../../data-sets/test-H2O_RPBE-D3.xyz' 

channel_num = int(sys.argv[1])

test_write = f"../test_cace_{channel_num}.xyz"


evaluator = cace.tasks.EvaluateTask(model_path=cace_nnp, device='cuda',
                                    energy_key = 'CACE_energy',
                                    forces_key = 'CACE_forces',
                                    other_keys=['q', 'tot_q'],
                                    atomic_energies = average_E0,
                                    )
from ase.io import read
dataset = read(to_read, index=':')

result = evaluator(data=dataset, batch_size=1, compute_stress=False, xyz_output=test_write)


import numpy as np
dataset = read(test_write, index=':')
# pred_energy = result['energy']
# pred_forces = result['forces']
pred_energy = np.array([atoms.info['CACE_energy'] for atoms in dataset])
pred_forces = ref_forces = np.vstack([atoms.get_array('CACE_forces') for atoms in dataset])
ref_energy = np.array([atoms.info['energy'] for atoms in dataset])
ref_forces = np.vstack([atoms.get_array('forces') for atoms in dataset])

def rmse(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return np.sqrt(np.mean((a - b)**2))

atoms_list = read(to_read, index=":")
n_atoms = np.array([len(atoms) for atoms in atoms_list])

ref_e_per_atom  = ref_energy  / n_atoms
pred_e_per_atom = pred_energy / n_atoms

rmse_per_atom = rmse(ref_e_per_atom, pred_e_per_atom)
energy_rmse = rmse(ref_energy, pred_energy)
force_rmse  = rmse(ref_forces, pred_forces)


# print(f"Energy RMSE: {energy_rmse:.4f} eV")
print(f"Per-atom energy RMSE: {rmse_per_atom*1000:.3f} meV/atom")
print(f"Force  RMSE: {force_rmse:.5f} eV/Å")

summary_file = "../cace_test_summary.txt"
header = "# channel_num  per_atom_RMSE_meV_per_atom  force_RMSE_eV_per_A\n"

line = f"{channel_num}  {rmse_per_atom*1000:.3f}  {force_rmse:.5f}\n"

write_header = not os.path.exists(summary_file)

with open(summary_file, "a") as f:
    if write_header:
        f.write(header)
    f.write(line)
