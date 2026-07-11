#!/usr/bin/env python
"""Training script using 4-stage curriculum on all embryos."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DatasetCatalog
from detection.unet_3d import UNet3D

# Import curriculum trainer directly from file to avoid monai dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "curriculum_trainer",
    Path(__file__).parent.parent / "src" / "training" / "curriculum_trainer.py"
)
curriculum_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curriculum_trainer)
SequenceTrainer = curriculum_trainer.SequenceTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with curriculum on all embryos")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_curriculum.yaml"),
        help="Path to training configuration"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../train"),
        help="Path to training data directory"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Train specific stage only (1-4), or None for all stages"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Data root: {args.data_root}")
    
    # Get all available samples
    catalog = DatasetCatalog(args.data_root)
    all_samples = catalog.list_samples()
    
    logger.info(f"Found {len(all_samples)} total samples")
    
    # Split into train/validation
    val_split = config.get("val_split", 0.2)
    n_val = max(1, int(len(all_samples) * val_split))
    
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    
    # Initialize model
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        bilinear=config.get("bilinear", False),
        features=config.get("features", (32, 64, 128, 256)),
    )
    
    logger.info("Model initialized")
    
    # Create trainer
    trainer = SequenceTrainer(
        model=model,
        data_root=args.data_root,
        train_sample_ids=train_samples,
        val_sample_ids=val_samples,
        device=config.get("device", "cuda"),
        save_dir=config.get("save_dir", "checkpoints"),
        stage_epochs=tuple(config.get("stage_epochs", [20, 20, 20, 40])),
        learning_rates=tuple(config.get("learning_rates", [1e-4, 5e-5, 2e-5, 1e-5])),
        batch_size=config.get("batch_size", 2),
    )
    
    # Train
    if args.stage is not None:
        # Train specific stage only
        logger.info(f"Training stage {args.stage} only")
        history = trainer._train_stage(
            stage=args.stage,
            epochs=trainer.stage_epochs[args.stage - 1],
            lr=trainer.learning_rates[args.stage - 1],
        )
    else:
        # Train all stages
        logger.info("Training all 4 stages")
        history = trainer.train()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
