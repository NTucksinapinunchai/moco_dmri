# -*- coding: utf-8 -*-
"""
Created on Mon Sep 1 2025

@author: Nontharat Tucksinapinunchai
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys
import torch
import numpy as np
import nibabel as nib
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb

from torch import optim
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from monai.data import load_decathlon_datalist
from monai.data import DataLoader,Dataset
from monai.networks.nets import VoxelMorphUNet
from monai.networks.blocks import Warp
from datetime import datetime

path = "/home/ge.polymtl.ca/p122983/moco_dmri/"
data_path = "/home/ge.polymtl.ca/p122983/moco_dmri/sourcedata/"
sys.path.insert(0,data_path)
json_path = os.path.join(data_path, 'dataset.json')

# -----------------------------
# DataGenerator
# -----------------------------
class DataGenerator(Dataset):
    def __init__(self, file_list):
        super().__init__(data=file_list)
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        # moving: nib -> (X,Y,Z,T). To torch -> (1, D, H, W, T)
        moving_np = nib.load(sample["moving"]).get_fdata()
        moving = torch.from_numpy(moving_np.astype(np.float32)).permute(2, 1, 0, 3).unsqueeze(0)

        # fixed: nib -> (X,Y,Z). To torch -> (1, D, H, W)
        fixed_np = nib.load(sample["fixed"]).get_fdata()
        fixed = torch.from_numpy(fixed_np.astype(np.float32)).permute(2, 1, 0).unsqueeze(0)

        return {"moving": moving, "fixed": fixed}

# -----------------------------
# DataModule
# -----------------------------
class DataModule(pl.LightningDataModule):
    def __init__(self, json_path, batch_size, num_workers):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load file lists from JSON
        self.train_files = load_decathlon_datalist(self.json_path, True, "training")
        self.val_files = load_decathlon_datalist(self.json_path, True, "validation")
        self.test_files = load_decathlon_datalist(self.json_path, True, "testing")

        # Build datasets
        self.train_ds = DataGenerator(self.train_files)
        self.val_ds = DataGenerator(self.val_files)
        self.test_ds = DataGenerator(self.test_files)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

# -----------------------------
# Smoothness regularization
# -----------------------------
def gradient_loss(flow_5d: torch.Tensor) -> torch.Tensor:
    # flow_5d: (B, 3, D, H, W)
    dz = torch.abs(flow_5d[:, :, 1:, :, :] - flow_5d[:, :, :-1, :, :])  # along D
    dy = torch.abs(flow_5d[:, :, :, 1:, :] - flow_5d[:, :, :, :-1, :])  # along H
    dx = torch.abs(flow_5d[:, :, :, :, 1:] - flow_5d[:, :, :, :, :-1])  # along W

    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
    return torch.mean(dx ** 2 + dy ** 2 + dz ** 2)

# -----------------------------
# L2 loss (MSE)
# -----------------------------
def image_loss(warped_all: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    # warped_all: (B, 1, D, H, W, T)
    B, C, D, H, W, T = warped_all.shape
    if fixed.shape[2:] != (D, H, W):
        fixed = F.interpolate(fixed, size=(D, H, W), mode="trilinear", align_corners=False)
    fixed_T = fixed.unsqueeze(-1).expand(-1, -1, D, H, W, T)
    return F.mse_loss(warped_all, fixed_T)

# -----------------------------
# Some helper function
# -----------------------------
def power(x, n=32):
    return ((int(x) + n - 1) // n) * n

# -----------------------------
# LightningModule
# -----------------------------
class VoxelMorphReg(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda_smooth=1.0):
        super().__init__()
        self.lr = lr
        self.lambda_smooth = lambda_smooth

        self.unet = VoxelMorphUNet(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=3,                # flow (dx, dy, dz)
            channels=(16, 32, 32, 32, 32, 32),  # keep pairs; this is OK
            final_conv_channels=(16, 16),
            kernel_size=3,
            dropout=0.5
        )
        # Use border padding to reduce black edge artifacts
        self.transformer = Warp(mode="bilinear", padding_mode="border")

    def forward(self, moving, fixed):
        """
        moving: (B, 1, D, H, W, T)
        fixed:  (B, 1, D, H, W)
        returns:
          warped_all:     (B, 1, D*, H*, W*, T)
          flows_4d:       (B, 3, D*, H*, W*, T)
          (dx, dy, dz):   each (B, 1, D*, H*, W*, T)
        """
        B, C, D, H, W, T = moving.shape

        # Make sizes UNet-friendly
        target_D = power(D, 32)
        target_H = power(H, 32)
        target_W = power(W, 32)

        # Pre-resize fixed once
        fixed_res = F.interpolate(fixed, size=(target_D, target_H, target_W),
                                  mode="trilinear", align_corners=False)

        warped_list = []
        flow_list = []

        for t in range(T):
            # (B, 1, D, H, W)
            moving_t = moving[..., t]
            moving_t_res = F.interpolate(moving_t, size=(target_D, target_H, target_W),
                                         mode="trilinear", align_corners=False)

            x = torch.cat([moving_t_res, fixed_res], dim=1)  # (B, 2, D, H, W)

            flow = self.unet(x)  # (B, 3, D*, H*, W*)
            warped = self.transformer(moving_t_res, flow)  # (B, 1, D*, H*, W*)

            warped_list.append(warped)      # each (B,1,D*,H*,W*)
            flow_list.append(flow)          # each (B,3,D*,H*,W*)

        # Stack along time dimension
        warped_all = torch.stack(warped_list, dim=-1)  # (B,1,D*,H*,W*,T)
        flows_4d   = torch.stack(flow_list,  dim=-1)   # (B,3,D*,H*,W*,T)

        dx = flows_4d[:, 0:1, ...]
        dy = flows_4d[:, 1:2, ...]
        dz = flows_4d[:, 2:3, ...]

        return warped_all, flows_4d, (dx, dy, dz)

    def training_step(self, batch, batch_idx):
        moving, fixed = batch["moving"], batch["fixed"]
        warped_all, flows_4d, _ = self(moving, fixed)

        loss_sim = image_loss(warped_all, fixed)

        B, _, D, H, W, T = flows_4d.shape
        flows_bt = flows_4d.permute(0, 5, 1, 2, 3, 4).reshape(B * T, 3, D, H, W)
        loss_smooth = gradient_loss(flows_bt)

        loss = loss_sim + self.lambda_smooth * loss_smooth

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_sim_loss", loss_sim)
        self.log("train_smooth_loss", loss_smooth)
        return loss

    def validation_step(self, batch, batch_idx):
        moving, fixed = batch["moving"], batch["fixed"]
        warped_all, flows_4d, _ = self(moving, fixed)

        loss_sim = image_loss(warped_all, fixed)

        B, _, D, H, W, T = flows_4d.shape
        flows_bt = flows_4d.permute(0, 5, 1, 2, 3, 4).reshape(B * T, 3, D, H, W)
        loss_smooth = gradient_loss(flows_bt)

        loss = loss_sim + self.lambda_smooth * loss_smooth

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_sim_loss", loss_sim, prog_bar=True)
        self.log("val_smooth_loss", loss_smooth, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Reduce LR when val_loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ckpt_dir = os.path.join(path, 'trained_weights')
os.makedirs(ckpt_dir, exist_ok=True)
pretrained_ckpt = os.path.join(ckpt_dir, f"voxelmorph-best-weighted_{timestamp}.ckpt")

# -----------------------------
# logger (Weights & Biases)
# -----------------------------
wandb_logger = WandbLogger(project="moco-dmri", name="1st_training")

num_epochs = 200
batch_size = 1
lr = 1e-4
lambda_smooth = 1.0
num_workers = 4

wandb_logger.experiment.config.update(
    dict(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_smooth=lambda_smooth,
        data_path=data_path
    )
)

# -----------------------------
# callbacks
# -----------------------------
checkpoint_cb = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=f"voxelmorph-best-weighted_{timestamp}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=25,
    verbose=True
)

dm = DataModule(json_path=json_path, batch_size=batch_size, num_workers=num_workers)
model = VoxelMorphReg(lr=lr, lambda_smooth=lambda_smooth)

# -----------------------------
# trainer
# -----------------------------
if __name__ == "__main__":
    dm = DataModule(json_path=json_path, batch_size=batch_size, num_workers=num_workers)
    model = VoxelMorphReg(lr=lr, lambda_smooth=lambda_smooth)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        accelerator="gpu",
        devices=1,
        log_every_n_steps=5,
        deterministic=False
    )

    # -----------------------------
    # fit
    # -----------------------------
    trainer.fit(model, datamodule=dm)

# -----------------------------
# Best checkpoint info
# -----------------------------
print("Best checkpoint path:", checkpoint_cb.best_model_path)
if checkpoint_cb.best_model_score is not None:
    best_val_loss = checkpoint_cb.best_model_score.item()
    print("Best val_loss:", best_val_loss)
else:
    best_val_loss = None
    print("No checkpoint was saved yet.")

wandb_logger.log_metrics({
    "best_val_loss": best_val_loss,
    "best_checkpoint_path": checkpoint_cb.best_model_path
})