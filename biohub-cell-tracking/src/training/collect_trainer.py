"""Training pipeline for COLLECT contrastive learning embedding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from detection.collect_embedding import COLLECTEmbedding

logger = logging.getLogger(__name__)


class CellTrackingDataset(Dataset):
    """Dataset for contrastive learning cell tracking."""
    
    def __init__(
        self,
        volumes: List[np.ndarray],
        tracks: List[List[Tuple[int, int, int]]],  # List of tracks (list of positions)
        patch_size: int = 21,
        time_window: int = 3,
        augment: bool = True
    ):
        self.volumes = volumes
        self.tracks = tracks
        self.patch_size = patch_size
        self.time_window = time_window
        self.augment = augment
        
        # Build training pairs
        self.pairs = self._build_pairs()
        
        logger.info(f"CellTrackingDataset: {len(self.pairs)} training pairs")
    
    def _build_pairs(self) -> List[Tuple]:
        """Build positive and negative pairs for contrastive learning."""
        pairs = []
        
        for vol_idx, track in enumerate(self.tracks):
            if len(track) < 2:
                continue
            
            volume = self.volumes[vol_idx]
            
            # Positive pairs: consecutive frames in same track
            for i in range(len(track) - 1):
                pos1 = track[i]
                pos2 = track[i + 1]
                pairs.append((vol_idx, pos1, pos2, 1))  # 1 = positive
            
            # Negative pairs: random cells from different tracks
            for i in range(len(track)):
                pos1 = track[i]
                # Pick random position from different frame
                other_track_idx = np.random.randint(0, len(self.tracks))
                if other_track_idx != vol_idx and len(self.tracks[other_track_idx]) > 0:
                    other_track = self.tracks[other_track_idx]
                    pos2 = other_track[np.random.randint(0, len(other_track))]
                    pairs.append((vol_idx, pos1, pos2, 0))  # 0 = negative
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        vol_idx, pos1, pos2, label = self.pairs[idx]
        volume = self.volumes[vol_idx]
        
        # Extract patches
        patch1 = self._extract_patch(volume, pos1)
        patch2 = self._extract_patch(volume, pos2)
        
        # Normalize
        if patch1.max() > 0:
            patch1 = patch1 / patch1.max()
        if patch2.max() > 0:
            patch2 = patch2 / patch2.max()
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                patch1 = np.flip(patch1, axis=0)
            if np.random.random() > 0.5:
                patch1 = np.flip(patch1, axis=1)
            if np.random.random() > 0.5:
                patch1 = np.flip(patch1, axis=2)
        
        # Convert to tensor
        patch1_tensor = torch.from_numpy(patch1).float().unsqueeze(0)
        patch2_tensor = torch.from_numpy(patch2).float().unsqueeze(0)
        
        return patch1_tensor, patch2_tensor, label
    
    def _extract_patch(self, volume: np.ndarray, position: Tuple[int, int, int]) -> np.ndarray:
        """Extract 3D patch around position."""
        z, y, x = position
        half_size = self.patch_size // 2
        
        z_start = max(0, z - half_size)
        z_end = min(volume.shape[0], z + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(volume.shape[1], y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(volume.shape[2], x + half_size + 1)
        
        patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if patch.shape != (self.patch_size, self.patch_size, self.patch_size):
            pad_z = self.patch_size - patch.shape[0]
            pad_y = self.patch_size - patch.shape[1]
            pad_x = self.patch_size - patch.shape[2]
            patch = np.pad(
                patch,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode='constant',
                constant_values=0
            )
        
        return patch


class COLLECTTrainer:
    """Trainer for COLLECT contrastive learning."""
    
    def __init__(
        self,
        model: COLLECTEmbedding,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
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
            f"COLLECTTrainer initialized: device={self.device}, "
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
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for patch1, patch2, labels in pbar:
            patch1 = patch1.to(self.device)
            patch2 = patch2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            _, proj1 = self.model(patch1)
            _, proj2 = self.model(patch2)
            
            # Concatenate projections
            projections = torch.cat([proj1, proj2], dim=0)
            labels_cat = torch.cat([labels, labels], dim=0)
            
            # Compute contrastive loss
            loss = self.model.compute_contrastive_loss(projections, labels_cat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for patch1, patch2, labels in tqdm(val_loader, desc="Validation"):
                patch1 = patch1.to(self.device)
                patch2 = patch2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                _, proj1 = self.model(patch1)
                _, proj2 = self.model(patch2)
                
                # Concatenate
                projections = torch.cat([proj1, proj2], dim=0)
                labels_cat = torch.cat([labels, labels], dim=0)
                
                # Loss
                loss = self.model.compute_contrastive_loss(projections, labels_cat)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_every: int = 10
    ) -> None:
        """Full training loop."""
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}"
                )
                
                # Save best model
                if val_metrics['loss'] < best_loss:
                    best_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, 'best')
            else:
                logger.info(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
            
            # Update scheduler
            self.scheduler.step()
        
        logger.info(f"Training complete. Best Loss: {best_loss:.4f}")
    
    def save_checkpoint(self, epoch: int, name: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        path = self.save_dir / f"{name}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
