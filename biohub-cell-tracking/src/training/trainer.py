"""Training loop for 3D U-Net cell detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

from detection.unet_detector import CellDetectionUNet
from training.dataset import LazyZarrDataset
from training.label_generator import WeightedLabelGenerator

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """Combined Dice + Focal loss for imbalanced segmentation with deep supervision."""
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        include_background: bool = False,
        use_deep_supervision: bool = False,
        deep_supervision_weights: tuple = (0.5, 0.3, 0.2),  # aux_2, aux_4, aux_8
    ) -> None:
        """Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            focal_gamma: Focusing parameter for Focal loss
            include_background: Whether to include background in Dice loss
            use_deep_supervision: Whether to use deep supervision
            deep_supervision_weights: Weights for auxiliary outputs (aux_2, aux_4, aux_8)
        """
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.use_deep_supervision = use_deep_supervision
        self.deep_supervision_weights = deep_supervision_weights
        
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=True,
            squared_pred=False,
            batch=True,
        )
        
        self.focal_loss = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=focal_gamma,
        )
        
        logger.info(
            f"CombinedLoss initialized: dice_weight={dice_weight}, "
            f"focal_weight={focal_weight}, focal_gamma={focal_gamma}, "
            f"use_deep_supervision={use_deep_supervision}"
        )
    
    def forward(self, pred: dict, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            pred: Predictions - either dict with 'out' and aux keys, or tensor
            target: Ground truth (B, D, H, W)
        
        Returns:
            Combined loss
        """
        if self.use_deep_supervision and isinstance(pred, dict):
            # Deep supervision: compute loss at multiple scales
            main_output = pred['out']
            aux_2 = pred.get('aux_2')
            aux_4 = pred.get('aux_4')
            aux_8 = pred.get('aux_8')
            
            # Main loss
            dice_main = self.dice_loss(main_output, target)
            focal_main = self.focal_loss(main_output, target)
            main_loss = self.dice_weight * dice_main + self.focal_weight * focal_main
            
            total_loss = main_loss
            
            # Auxiliary losses (upsample target to match auxiliary output size)
            if aux_2 is not None:
                target_2 = F.interpolate(target.unsqueeze(1).float(), size=aux_2.shape[2:], mode='nearest').squeeze(1).long()
                dice_aux2 = self.dice_loss(aux_2, target_2)
                focal_aux2 = self.focal_loss(aux_2, target_2)
                total_loss += self.deep_supervision_weights[0] * (self.dice_weight * dice_aux2 + self.focal_weight * focal_aux2)
            
            if aux_4 is not None:
                target_4 = F.interpolate(target.unsqueeze(1).float(), size=aux_4.shape[2:], mode='nearest').squeeze(1).long()
                dice_aux4 = self.dice_loss(aux_4, target_4)
                focal_aux4 = self.focal_loss(aux_4, target_4)
                total_loss += self.deep_supervision_weights[1] * (self.dice_weight * dice_aux4 + self.focal_weight * focal_aux4)
            
            if aux_8 is not None:
                target_8 = F.interpolate(target.unsqueeze(1).float(), size=aux_8.shape[2:], mode='nearest').squeeze(1).long()
                dice_aux8 = self.dice_loss(aux_8, target_8)
                focal_aux8 = self.focal_loss(aux_8, target_8)
                total_loss += self.deep_supervision_weights[2] * (self.dice_weight * dice_aux8 + self.focal_weight * focal_aux8)
            
            return total_loss
        else:
            # Standard loss without deep supervision
            if isinstance(pred, dict):
                pred = pred['out']
            
            dice = self.dice_loss(pred, target)
            focal = self.focal_loss(pred, target)
            
            return self.dice_weight * dice + self.focal_weight * focal


class CellDetectionTrainer:
    """Trainer for 3D U-Net cell detection."""
    
    def __init__(
        self,
        model: CellDetectionUNet,
        train_dataset: LazyZarrDataset,
        val_dataset: LazyZarrDataset,
        config: Dict,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: 3D U-Net model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Training parameters
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 2)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.grad_clip = config.get("grad_clip", 1.0)
        
        # Loss parameters
        self.dice_weight = config.get("dice_weight", 0.5)
        self.focal_weight = config.get("focal_weight", 0.5)
        self.focal_gamma = config.get("focal_gamma", 2.0)
        self.use_deep_supervision = config.get("use_deep_supervision", False)
        self.deep_supervision_weights = config.get("deep_supervision_weights", (0.5, 0.3, 0.2))
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = config.get("save_frequency", 5)
        self.best_metric_name = config.get("best_metric", "val_loss")
        
        # Early stopping
        self.patience = config.get("patience", 15)
        self.min_delta = config.get("min_delta", 1e-4)
        
        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        
        # Setup model
        self.model.to(self.device)
        
        # Setup loss
        self.criterion = CombinedLoss(
            dice_weight=self.dice_weight,
            focal_weight=self.focal_weight,
            focal_gamma=self.focal_gamma,
            include_background=False,
            use_deep_supervision=self.use_deep_supervision,
            deep_supervision_weights=self.deep_supervision_weights,
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        
        # Setup data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf') if self.best_metric_name == "val_loss" else 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
        
        # Setup AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        logger.info(
            f"Trainer initialized: device={self.device}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, lr={self.learning_rate}"
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint["history"]
        self.best_metric = checkpoint["best_metric"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}, epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop.
        
        Returns:
            Training history
        """
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["learning_rate"].append(current_lr)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}"
            )
            
            # Check if best
            is_best = False
            if self.best_metric_name == "val_loss":
                if val_loss < self.best_metric - self.min_delta:
                    self.best_metric = val_loss
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:
                if val_loss > self.best_metric + self.min_delta:
                    self.best_metric = val_loss
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed")
        return self.history


def create_trainer(
    train_dataset: LazyZarrDataset,
    val_dataset: LazyZarrDataset,
    config: Dict,
) -> CellDetectionTrainer:
    """Create trainer with model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
    
    Returns:
        Configured trainer
    """
    # Create model
    model = CellDetectionUNet(
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 3),
        spatial_dims=config.get("spatial_dims", 3),
        channels=config.get("channels", (32, 64, 128, 256, 512)),
        strides=config.get("strides", (2, 2, 2, 2)),
        num_res_units=config.get("num_res_units", 2),
        dropout=config.get("dropout", 0.1),
        use_deep_supervision=config.get("use_deep_supervision", False),
    )
    
    # Create trainer
    trainer = CellDetectionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )
    
    return trainer
