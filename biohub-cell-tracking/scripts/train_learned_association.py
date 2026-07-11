#!/usr/bin/env python
"""Train learned association network for cell tracking."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DatasetCatalog
from tracking.learned_association import AssociationFeatureExtractor, LearnedAssociationNet
from tracking.base import TrackingResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AssociationDataset(Dataset):
    """Dataset for learned association training."""
    
    def __init__(
        self,
        sample_ids: List[str],
        data_root: Path,
        scale: Tuple[float, float, float],
        max_distance_um: float = 7.0,
        negative_ratio: int = 3,
        use_intensity: bool = True,
    ):
        """Initialize association dataset.
        
        Args:
            sample_ids: List of sample IDs
            data_root: Path to data directory
            scale: Physical scale (Z, Y, X)
            max_distance_um: Maximum distance for positive pairs
            negative_ratio: Number of negative pairs per positive
            use_intensity: Whether to use intensity features
        """
        self.sample_ids = sample_ids
        self.data_root = data_root
        self.scale = scale
        self.max_distance_um = max_distance_um
        self.negative_ratio = negative_ratio
        self.use_intensity = use_intensity
        
        self.catalog = DatasetCatalog(data_root)
        self.feature_extractor = AssociationFeatureExtractor(scale)
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        logger.info(f"AssociationDataset: {len(self.pairs)} pairs from {len(sample_ids)} samples")
    
    def _generate_pairs(self) -> List[Tuple]:
        """Generate positive and negative pairs from ground truth."""
        pairs = []
        
        for sample_id in self.sample_ids:
            try:
                sample = self.catalog.load_sample(sample_id, require_tracks=True)
                tracks = sample.tracks
                
                # Get detections by timepoint from ground truth
                nodes_by_time = {}
                for node_id, node_data in tracks.nodes.items():
                    t = node_data['t']
                    if t not in nodes_by_time:
                        nodes_by_time[t] = []
                    nodes_by_time[t].append((node_id, node_data))
                
                # Generate pairs from consecutive timepoints
                sorted_times = sorted(nodes_by_time.keys())
                for i in range(len(sorted_times) - 1):
                    t = sorted_times[i]
                    t1 = sorted_times[i + 1]
                    
                    if t not in nodes_by_time or t1 not in nodes_by_time:
                        continue
                    
                    nodes_t = nodes_by_time[t]
                    nodes_t1 = nodes_by_time[t1]
                    
                    # Get ground truth edges
                    gt_edges = set()
                    for edge in tracks.edges:
                        if edge[0] in [n[0] for n in nodes_t] and edge[1] in [n[0] for n in nodes_t1]:
                            gt_edges.add((edge[0], edge[1]))
                    
                    # Positive pairs (ground truth edges)
                    for edge in gt_edges:
                        node_a = next(n for n in nodes_t if n[0] == edge[0])
                        node_b = next(n for n in nodes_t1 if n[0] == edge[1])
                        
                        cell_a = (node_a[1]['z'], node_a[1]['y'], node_a[1]['x'], 1.0)
                        cell_b = (node_b[1]['z'], node_b[1]['y'], node_b[1]['x'], 1.0)
                        
                        # Extract features
                        feat = self.feature_extractor.extract_features(cell_a, cell_b)
                        pairs.append((feat, 1.0))  # Positive
                        
                        # Negative pairs
                        for _ in range(self.negative_ratio):
                            # Random negative
                            neg_node_b = random.choice(nodes_t1)
                            if neg_node_b[0] != edge[1]:  # Not the true match
                                cell_neg = (neg_node_b[1]['z'], neg_node_b[1]['y'], neg_node_b[1]['x'], 1.0)
                                feat_neg = self.feature_extractor.extract_features(cell_a, cell_neg)
                                pairs.append((feat_neg, 0.0))  # Negative
                
            except Exception as e:
                logger.warning(f"Failed to process sample {sample_id}: {e}")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, label = self.pairs[idx]
        return torch.from_numpy(features).float(), torch.tensor(label).float()


class AssociationTrainer:
    """Trainer for learned association network."""
    
    def __init__(self, config: Dict):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Load embryo split
        split_path = Path(config["data"]["embryo_split"])
        with open(split_path) as f:
            split = json.load(f)
        
        train_samples = split["train"]
        val_samples = split["validation"]
        
        logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
        
        # Create datasets
        scale = (1.625, 0.40625, 0.40625)
        self.train_dataset = AssociationDataset(
            sample_ids=train_samples,
            data_root=Path(config["data"]["data_root"]),
            scale=scale,
            max_distance_um=config["data"]["max_distance_um"],
            negative_ratio=config["data"]["negative_ratio"],
            use_intensity=config["data"]["use_intensity_features"],
        )
        
        self.val_dataset = AssociationDataset(
            sample_ids=val_samples,
            data_root=Path(config["data"]["data_root"]),
            scale=scale,
            max_distance_um=config["data"]["max_distance_um"],
            negative_ratio=config["data"]["negative_ratio"],
            use_intensity=config["data"]["use_intensity_features"],
        )
        
        # Create model
        self.model = LearnedAssociationNet(
            input_dim=config["model"]["input_dim"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )
        
        # Device
        self.device = torch.device(
            config["training"]["device"] if config["training"]["device"] != "auto"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        )
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        
        # Training parameters
        self.epochs = config["training"]["epochs"]
        self.grad_clip = config["training"]["grad_clip"]
        self.use_amp = config["training"]["use_amp"]
        
        # Checkpointing
        self.checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = config["checkpointing"]["save_frequency"]
        
        # Early stopping
        self.patience = config["early_stopping"]["patience"]
        self.min_delta = config["early_stopping"]["min_delta"]
        
        # State
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        logger.info(f"Trainer initialized: device={self.device}, epochs={self.epochs}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")
        
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(features).squeeze(-1)
                    loss = self.criterion(predictions, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(features).squeeze(-1)
                loss = self.criterion(predictions, labels)
                
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
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(features).squeeze(-1)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        logger.info(f"Saved checkpoint to {path}")
    
    def train(self) -> Dict:
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}"
            )
            
            # Check if best
            is_best = False
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train learned association network")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/learned_association.yaml"),
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = AssociationTrainer(config)
    
    # Train
    history = trainer.train()
    
    logger.info("Training finished")


if __name__ == "__main__":
    main()
