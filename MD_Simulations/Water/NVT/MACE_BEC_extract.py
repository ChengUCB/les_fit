import mace
import sys
import os
src = os.path.join('/global/home/users/dongjinkim/software/les', 'src')
if src not in sys.path:
    sys.path.insert(0, src)
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


from tqdm import tqdm 
import gc
from ase.io import read, write
import pickle

device = torch_tools.init_device('cuda')
model = torch.load('MACELES-OFF_small.model', map_location='cuda')
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
# print(z_table)
try:
    heads = model.heads
except AttributeError:
    heads = None
# print(heads)

r_max = model.r_max
factor = 1

traj_iter = read('./md_h2o.traj', index=':')



total_dP_list = []
for i, atoms in tqdm(enumerate(traj_iter), total=len(traj_iter)):
    config = data.config_from_atoms(atoms)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=float(r_max), heads=heads)],
        batch_size=1,
        shuffle=False,
        drop_last=False,)
    batch = next(iter(data_loader)).to(device)
    dict = batch.to_dict() 
    
    output = model(dict, compute_stress=False, compute_bec=True)
    BEC = output['BEC'] * factor
    velocity = torch.tensor(atoms.get_velocities(), dtype=BEC.dtype, device=BEC.device)

    dP = torch.bmm(BEC, velocity.unsqueeze(-1)).squeeze(-1)
    total_dP = torch.sum(dP, dim=0)
    total_dP_list.append(total_dP.detach().cpu())

    del config, data_loader, batch, dict, output, BEC, velocity, dP, total_dP
    torch.cuda.empty_cache()
    gc.collect()

    if (i+1) % 100000 == 0:
        print(f'{i+1} frames are done.')
        with open(f'bec_{i+1}.pkl', 'wb') as f:
            pickle.dump({'total_dp': torch.stack(total_dP_list).numpy()}, f)

total_dP_stack = np.array(torch.stack(total_dP_list))
print('save dict')

dict = {'total_dp': total_dP_stack}

with open('bec_dict.pkl', 'wb') as f:
    pickle.dump(dict, f)
