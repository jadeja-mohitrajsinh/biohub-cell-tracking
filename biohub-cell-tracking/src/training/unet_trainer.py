"""Training pipeline for 3D U-Net cell detector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from detection.unet_3d import UNet3D

logger = logging.getLogger(__name__)


class CellSegmentationDataset(Dataset):
    """Dataset for cell segmentation training."""
    
    def __init__(
        self,
        volumes: list,
        masks: list,
        patch_size: int = 64,
        augment: bool = True
    ):
        self.volumes = volumes
        self.masks = masks
        self.patch_size = patch_size
        self.augment = augment
        
        assert len(volumes) == len(masks), "Volumes and masks must have same length"
    
    def __len__(self) -> int:
        return len(self.volumes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = self.volumes[idx]
        mask = self.masks[idx]
        
        # Random crop
        z, y, x = volume.shape
        if z > self.patch_size and y > self.patch_size and x > self.patch_size:
            z_start = np.random.randint(0, z - self.patch_size)
            y_start = np.random.randint(0, y - self.patch_size)
            x_start = np.random.randint(0, x - self.patch_size)
            
            volume = volume[z_start:z_start+self.patch_size,
                          y_start:y_start+self.patch_size,
                          x_start:x_start+self.patch_size]
            mask = mask[z_start:z_start+self.patch_size,
                       y_start:y_start+self.patch_size,
                       x_start:x_start+self.patch_size]
        
        # Normalize
        if volume.max() > 0:
            volume = volume / volume.max()
        
        # Augmentation
        if self.augment:
            # Random flip
            if np.random.random() > 0.5:
                volume = np.flip(volume, axis=0)
                mask = np.flip(mask, axis=0)
            if np.random.random() > 0.5:
                volume = np.flip(volume, axis=1)
                mask = np.flip(mask, axis=1)
            if np.random.random() > 0.5:
                volume = np.flip(volume, axis=2)
                mask = np.flip(mask, axis=2)
            
            # Random rotation (90 degrees)
            if np.random.random() > 0.5:
                volume = np.rot90(volume, k=np.random.randint(1, 4), axes=(1, 2))
                mask = np.rot90(mask, k=np.random.randint(1, 4), axes=(1, 2))
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return volume_tensor, mask_tensor


class UNetTrainer:
    """Trainer for 3D U-Net cell segmentation."""
    
    def __init__(
        self,
        model: UNet3D,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function (combination of Dice and BCE)
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        logger.info(
            f"UNetTrainer initialized: device={self.device}, "
            f"lr={learning_rate}, save_dir={self.save_dir}"
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for volumes, masks in pbar:
            volumes = volumes.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            logits = self.model(volumes)
            
            # Compute losses
            dice_loss = self.dice_loss(logits, masks)
            bce_loss = self.bce_loss(logits, masks)
            loss = dice_loss + bce_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                dice = self.dice_loss.compute_dice(probs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        with torch.no_grad():
            for volumes, masks in tqdm(val_loader, desc="Validation"):
                volumes = volumes.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                logits = self.model(volumes)
                
                # Compute losses
                dice_loss = self.dice_loss(logits, masks)
                bce_loss = self.bce_loss(logits, masks)
                loss = dice_loss + bce_loss
                
                # Metrics
                probs = torch.sigmoid(logits)
                dice = self.dice_loss.compute_dice(probs, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {'loss': avg_loss, 'dice': avg_dice}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_every: int = 10
    ) -> None:
        """Full training loop."""
        best_dice = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Dice={train_metrics['dice']:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val Dice={val_metrics['dice']:.4f}"
                )
                
                # Save best model
                if val_metrics['dice'] > best_dice:
                    best_dice = val_metrics['dice']
                    self.save_checkpoint(epoch, 'best')
            else:
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Dice={train_metrics['dice']:.4f}"
                )
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
            
            # Update scheduler
            self.scheduler.step()
        
        logger.info(f"Training complete. Best Dice: {best_dice:.4f}")


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return 1 - self.compute_dice(probs, targets)
    
    def compute_dice(self, probs: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Dice coefficient."""
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return dice.item()
