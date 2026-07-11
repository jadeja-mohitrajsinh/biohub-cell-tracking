#!/usr/bin/env python
"""Training script for 3D U-Net cell detection."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DatasetCatalog
from training.dataset import create_training_datasets
from training.trainer import create_trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config for easier access
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    return flat_config


def get_sample_ids(
    data_root: Path,
    val_split: float = 0.2,
    val_samples: List[str] | None = None,
) -> tuple[List[str], List[str]]:
    """Get training and validation sample IDs.
    
    Args:
        data_root: Path to data directory
        val_split: Fraction of samples for validation
        val_samples: Optional specific validation samples
    
    Returns:
        Tuple of (train_sample_ids, val_sample_ids)
    """
    catalog = DatasetCatalog(data_root)
    all_samples = catalog.list_samples()
    
    if val_samples:
        # Use specific validation samples
        train_samples = [s for s in all_samples if s not in val_samples]
        val_sample_ids = val_samples
    else:
        # Random split
        n_val = max(1, int(len(all_samples) * val_split))
        val_sample_ids = all_samples[:n_val]
        train_samples = all_samples[n_val:]
    
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_sample_ids)}")
    
    return train_samples, val_sample_ids


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 3D U-Net for cell detection")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to training configuration"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../train"),
        help="Path to training data directory"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override resume path if provided
    if args.resume:
        config["resume_from"] = str(args.resume)
    
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Data root: {args.data_root}")
    
    # Get sample IDs
    train_samples, val_samples = get_sample_ids(
        data_root=args.data_root,
        val_split=config.get("val_split", 0.2),
        val_samples=config.get("val_samples", None),
    )
    
    # Create datasets
    train_dataset, val_dataset = create_training_datasets(
        data_root=args.data_root,
        train_sample_ids=train_samples,
        val_sample_ids=val_samples,
        scale=tuple(config.get("scale", [1.625, 0.40625, 0.40625])),
        center_radius=config.get("center_radius", 2),
        periphery_radius=config.get("periphery_radius", 4),
        crop_size=tuple(config.get("crop_size", [32, 128, 128])),
        cache_size=config.get("cache_size", 4),
    )
    
    # Create trainer
    trainer = create_trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )
    
    # Resume from checkpoint if specified
    if config.get("resume_from"):
        checkpoint_path = Path(config["resume_from"])
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    # Train
    history = trainer.train()
    
    # Save final history
    history_path = Path(config.get("checkpoint_dir", "outputs/checkpoints")) / "training_history.yaml"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(history_path, "w") as f:
        yaml.dump(history, f)
    
    logger.info(f"Training history saved to {history_path}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
