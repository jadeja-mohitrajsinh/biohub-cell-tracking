#!/usr/bin/env python
"""Create embryo-disjoint train/validation split."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DatasetCatalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_embryo_disjoint_split(
    data_root: Path,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    min_val_samples: int = 2,
) -> Tuple[List[str], List[str]]:
    """Create embryo-disjoint train/validation split.
    
    Args:
        data_root: Path to data directory
        val_ratio: Fraction of samples for validation
        random_seed: Random seed for reproducibility
        min_val_samples: Minimum number of validation samples
        
    Returns:
        Tuple of (train_sample_ids, val_sample_ids)
    """
    # Load all available samples
    catalog = DatasetCatalog(data_root)
    all_samples = catalog.list_samples()
    
    logger.info(f"Found {len(all_samples)} total samples")
    
    # Filter samples with ground truth
    samples_with_gt = []
    for sample_id in all_samples:
        try:
            sample = catalog.load_sample(sample_id, require_tracks=True)
            samples_with_gt.append(sample_id)
        except Exception as e:
            logger.warning(f"Sample {sample_id} has no ground truth: {e}")
    
    logger.info(f"Found {len(samples_with_gt)} samples with ground truth")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle samples
    shuffled_samples = np.random.permutation(samples_with_gt).tolist()
    
    # Calculate validation size
    num_val = max(min_val_samples, int(len(shuffled_samples) * val_ratio))
    num_train = len(shuffled_samples) - num_val
    
    logger.info(f"Split: {num_train} train, {num_val} validation")
    
    # Split
    train_samples = shuffled_samples[:num_train]
    val_samples = shuffled_samples[num_train:]
    
    logger.info(f"Train samples: {train_samples}")
    logger.info(f"Validation samples: {val_samples}")
    
    return train_samples, val_samples


def save_split(
    train_samples: List[str],
    val_samples: List[str],
    output_path: Path,
) -> None:
    """Save split to JSON file.
    
    Args:
        train_samples: List of training sample IDs
        val_samples: List of validation sample IDs
        output_path: Path to save JSON file
    """
    split_data = {
        "train": train_samples,
        "validation": val_samples,
        "num_train": len(train_samples),
        "num_validation": len(val_samples),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Saved split to {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create embryo-disjoint train/validation split")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../train"),
        help="Path to data directory"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for validation"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=2,
        help="Minimum number of validation samples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/embryo_split.json"),
        help="Path to save split JSON file"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Creating embryo-disjoint split")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Validation ratio: {args.val_ratio}")
    logger.info(f"Random seed: {args.random_seed}")
    
    # Create split
    train_samples, val_samples = create_embryo_disjoint_split(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        min_val_samples=args.min_val_samples,
    )
    
    # Save split
    save_split(train_samples, val_samples, args.output)


if __name__ == "__main__":
    main()
