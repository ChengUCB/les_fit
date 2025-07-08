import os
os.environ["DGL_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # you can set this to include/exclude gpus
import gzip
import json
import os
import numpy as np
# import pytorch_lightning as pl
import torch.backends.mps
from matgl.layers import AtomRef
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph, collate_fn_pes
from matgl.models import TensorNet_LR, CHGNet_LR, CACE_LR
from matgl.utils.training import ModelLightningModule, PotentialLightningModule, xavier_init
from pymatgen.core import Lattice, Structure
from torch.optim.lr_scheduler import CosineAnnealingLR


from tqdm import tqdm
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from functools import partial

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import lightning.pytorch as pl

# Import EMA callback
from lightning.pytorch.callbacks import StochasticWeightAveraging


from matgl.config import DEFAULT_ELEMENTS
element_types = DEFAULT_ELEMENTS


from copy import deepcopy

class EMACallback(pl.Callback):
    def __init__(self, decay=0.999, start_epoch=1):
        self.decay = decay
        self.start_epoch = start_epoch
        self.shadow_params = {}

    def on_train_epoch_start(self, trainer, pl_module):
        # Initialize shadow weights at the start of the specified epoch
        if trainer.current_epoch == self.start_epoch:
            for name, param in pl_module.model.named_parameters():
                self.shadow_params[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update shadow weights after each training batch
        if trainer.current_epoch >= self.start_epoch:
            for name, param in pl_module.model.named_parameters():
                self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.data

            # pl_module.ema_model = 

    def on_validation_epoch_start(self, trainer, pl_module):
        # Replace model parameters with EMA weights before validation
        if trainer.current_epoch >= self.start_epoch:
            self.backup_params = {name: param.data.clone() for name, param in pl_module.model.named_parameters()}
            for name, param in pl_module.model.named_parameters():
                param.data.copy_(self.shadow_params[name])

            for name, param in pl_module.ema_model.named_parameters():
                param.data.copy_(self.shadow_params[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        # Restore original parameters after validation
        if trainer.current_epoch >= self.start_epoch:
            for name, param in pl_module.model.named_parameters():
                param.data.copy_(self.backup_params[name])



class myPotentialLightningModule(PotentialLightningModule):
    def __init__(self, init_lr=1e-2, 
                 decay_factor = 0.9,
                 patience = 5,
                 min_lr = 1e-5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = init_lr
        self.ema_model = deepcopy(self.model)
        self.ema_model.eval()
        self.ema_model.requires_grad = False

        self.decay_factor = decay_factor
        self.patience = patience
        self.min_lr = min_lr
        
    def on_train_epoch_end(self):
        # Get the validation metric value
        val_metric = self.trainer.callback_metrics.get("val_Total_Loss")

        train_Energy_MAE = self.trainer.callback_metrics.get("train_Energy_MAE")
        train_Force_MAE = self.trainer.callback_metrics.get("train_Force_MAE")

        val_Energy_MAE = self.trainer.callback_metrics.get("val_Energy_MAE")
        val_Force_MAE = self.trainer.callback_metrics.get("val_Force_MAE")
        
        print()
        print(f"train_Energy_MAE: {train_Energy_MAE}, train_Force_MAE: {train_Force_MAE}")
        print(f"val_Energy_MAE: {val_Energy_MAE}, val_Force_MAE: {val_Force_MAE}")

        if val_metric is not None:
            # Call the scheduler with the metric value
            self.scheduler.step(val_metric)

        try:
            print("lr = ", self.scheduler.get_last_lr())
        except:
            pass

    def configure_optimizers(self):
        try:
            self.lr = self.scheduler.get_last_lr()[0]
        except:
            self.lr = self.init_lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0, amsgrad=True)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.decay_factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_Total_Loss",
            }
        }




class StopAtEpochCallback(pl.Callback):
    def __init__(self, stop_epoch):
        self.stop_epoch = stop_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        # Stop training when current epoch reaches `stop_epoch`
        if trainer.current_epoch >= self.stop_epoch:
            trainer.should_stop = True




dataset_path = '/home/zhongpc/Project/matgl_long-range/train_CHGNet/dataset/Au-MgO-Al_4.5'

########## load dataset ###########
dataset = MGLDataset(include_line_graph=True, directed_line_graph=True, save_dir=dataset_path, raw_dir= dataset_path)

########### split dataset ###########
training_set, validation_set, test_set = split_dataset(
    dataset, 
    frac_list=[0.9, 0.1, 0], 
    random_state=42, 
    shuffle=True
)
collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=False, include_magmom=False)


########### create dataloader ###########
train_loader, val_loader = MGLDataLoader(
    train_data=training_set,
    val_data=validation_set,
    collate_fn=collate_fn,
    batch_size= 4,
    num_workers=12,
)
print(len(train_loader))

########### create bond graph ###########

for (g, lat, l_g, attrs, lbs) in tqdm(dataset):
    try:
        bond_graph = ensure_line_graph_compatibility(g, l_g, threebody_cutoff=3, directed=True)
    except:
        print(lbs['matpes_id'])


for (g, lat, l_g, state_attr, e, f, s) in tqdm(train_loader):
    bond_graph = ensure_line_graph_compatibility(g, l_g, threebody_cutoff=3, directed=True)



######### fit the AtomRef ###########
train_graphs = []
energies = []
forces = []
for (g, lat, l_g, attrs, lbs) in training_set:
    train_graphs.append(g)
    energies.append(lbs["energies"])
    forces.append(lbs['forces'])
element_refs = AtomRef(torch.zeros(89))
# element_refs.fit(train_graphs, torch.hstack(energies))

print("!!!!!!!!element_refs!!!!!!!!!")
print(element_refs.property_offset)
print("!!!!!!!!!!!!!!!!!")


model = CHGNet_LR(cutoff = 4.5, num_blocks = 4)


# xavier_init(model)
# Count number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,}")




model.calc_stresses = False

path = os.getcwd()
logger = CSVLogger(save_dir=path)


checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="val_Total_Loss",
    mode="min",
    filename="{epoch:04d}-best_model",
    save_last=True,
)



# %%
### Phase 1 training ###

num_restart = 4 # the last restart is doubled, the total number of restarts is 5
epochs_F = 50

epochs_per_phase_E = 50
start_lr = 0.005

max_epochs = 1000

phase1_stop_callback = StopAtEpochCallback(stop_epoch=epochs_F *2 )

phase2_stop_callback = StopAtEpochCallback(stop_epoch=epochs_F *2 + epochs_per_phase_E * 1)

phase3_stop_callback = StopAtEpochCallback(stop_epoch=epochs_F *2 + epochs_per_phase_E * 2)

phase4_stop_callback = StopAtEpochCallback(stop_epoch=epochs_F *2 + epochs_per_phase_E * 3)

# phase5_stop_callback = StopAtEpochCallback(stop_epoch=epochs_F *2 + epochs_per_phase_E * 4)



ema_callback = EMACallback(decay=0.99, start_epoch= 5)  # Adjust decay as needed

force_checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="val_Force_RMSE",
    mode="min",
    filename="bestF_model-{epoch:04d}",
    save_last= True)



for ii in range(num_restart):

    start_lr = 0.005 * 0.9**(ii)
     
    optimizer = torch.optim.Adam(model.parameters(), lr= start_lr, weight_decay=0, amsgrad=True)

    scheduler = CosineAnnealingLR(optimizer, T_max = epochs_F * 2, eta_min= 0.0005)

    lit_model = myPotentialLightningModule(
        model=model,
        element_refs=element_refs.property_offset,
        # data_std=rms_forces,
        optimizer=optimizer,
        scheduler=scheduler,
        energy_weight = 0.1,
        force_weight = 1000.0,
        # stress_weight= 0.05,
        # magmom_weight= 0,
        loss="mse_loss",
        # loss_params={'delta':0.1},
        include_long_range = True,
        include_line_graph = True,
        per_atom_energy_loss = False,
        les_params = {'les_dl': 2.0, 
                      'les_sigma': 1.0, 
                      'les_norm_factor': 1.0,
                      'les_pbc': True},
    )


    if ii == (num_restart -1): # last restart
    
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[
                force_checkpoint_callback, 
                ema_callback,
                phase1_stop_callback
            ],
            devices=1,
            max_epochs= max_epochs,
            accelerator="gpu",
            gradient_clip_val= 0.1,
            accumulate_grad_batches=4,
            profiler="simple",
            enable_progress_bar=False
            )
    else:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[
                ema_callback,
                checkpoint_callback, 
            ],
            devices=1,
            max_epochs= epochs_F,
            accelerator="gpu",
            gradient_clip_val= 1,
            accumulate_grad_batches=4,
            profiler="simple",
            enable_progress_bar=False
            )
        
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


trainer.save_checkpoint("./phase1_last_checkpoint.ckpt")
torch.save(lit_model, "./phase1_lit_model.pth")

######################### Phase 2 training #########################
lit_model.energy_weight = 1.0
lit_model.patience = 10


trainer = pl.Trainer(
    logger=logger,
    callbacks=[
        force_checkpoint_callback, 
        ema_callback,
        phase2_stop_callback
    ],
    devices=1,
    max_epochs= max_epochs,
    accelerator="gpu",
    gradient_clip_val= 0.1,
    accumulate_grad_batches=4,
    profiler="simple",
    enable_progress_bar=False
)



# Resume from the best checkpoint
checkpoint_path = "./phase1_last_checkpoint.ckpt"  # Replace with your checkpoint path
trainer.fit(
    model=lit_model, 
    train_dataloaders=train_loader, 
    val_dataloaders=val_loader,
    ckpt_path=checkpoint_path  # Critical for resuming
)



# %%
trainer.save_checkpoint("./phase2_last_checkpoint.ckpt")
torch.save(lit_model, "./phase2_lit_model.pth")

# %%
lit_model.energy_weight = 10.0

trainer = pl.Trainer(
    logger=logger,
    callbacks=[
        force_checkpoint_callback, 
        ema_callback,
        phase3_stop_callback
    ],
    devices=1,
    max_epochs= max_epochs,
    accelerator="gpu",
    gradient_clip_val= 0.1,
    accumulate_grad_batches=4,
    profiler="simple",
    enable_progress_bar=False
)



# Resume from the best checkpoint
checkpoint_path = "./phase2_last_checkpoint.ckpt"  # Replace with your checkpoint path
trainer.fit(
    model=lit_model, 
    train_dataloaders=train_loader, 
    val_dataloaders=val_loader,
    ckpt_path=checkpoint_path  # Critical for resuming
)
trainer.save_checkpoint("./phase3_last_checkpoint.ckpt")
torch.save(lit_model, "./phase3_lit_model.pth")