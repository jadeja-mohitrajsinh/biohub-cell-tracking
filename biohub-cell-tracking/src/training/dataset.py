"""PyTorch dataset for 3D cell detection training with lazy Zarr loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import DatasetCatalog, EmbryoDataset, EmbryoSample
from data.models import VoxelScale
from training.augmentation import AnisotropicAugmentation, TrainingAugmentation, ValidationAugmentation
from training.label_generator import LabelGenerator, WeightedLabelGenerator

logger = logging.getLogger(__name__)


class CellDetectionDataset(Dataset):
    """PyTorch dataset for cell detection with lazy Zarr loading."""
    
    def __init__(
        self,
        sample: EmbryoSample,
        label_generator: LabelGenerator,
        augmentation: Optional[Callable] = None,
        normalize: bool = True,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        timepoint_indices: Optional[List[int]] = None,
    ) -> None:
        """Initialize cell detection dataset.
        
        Args:
            sample: EmbryoSample with image data and ground truth tracks
            label_generator: LabelGenerator for creating training labels
            augmentation: Optional augmentation function
            normalize: Whether to normalize intensities
            quantile_low: Lower quantile for normalization
            quantile_high: Upper quantile for normalization
            timepoint_indices: Optional list of timepoint indices to use
        """
        self.sample = sample
        self.label_generator = label_generator
        self.augmentation = augmentation
        self.normalize = normalize
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        
        # Create embryo dataset for loading
        self.embryo_dataset = EmbryoDataset(
            sample,
            normalize=normalize,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )
        
        # Determine which timepoints to use
        total_timepoints = sample.image_metadata.shape[0]
        if timepoint_indices is None:
            self.timepoint_indices = list(range(total_timepoints))
        else:
            self.timepoint_indices = [t for t in timepoint_indices if 0 <= t < total_timepoints]
        
        logger.info(
            f"CellDetectionDataset initialized: {len(self.timepoint_indices)} timepoints, "
            f"augmentation={augmentation is not None}"
        )
    
    def __len__(self) -> int:
        """Return number of timepoints in dataset."""
        return len(self.timepoint_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.
        
        Args:
            idx: Index of timepoint
        
        Returns:
            Dictionary with 'image' and 'label' tensors
        """
        timepoint = self.timepoint_indices[idx]
        
        # Load volume
        volume = self.embryo_dataset.get_timepoint(timepoint)
        
        # Generate labels from ground truth
        volume_shape = volume.shape
        labels = self.label_generator.generate_labels_from_graph(
            volume_shape,
            self.sample.tracks,
            timepoint,
        )
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            volume, labels = self.augmentation(volume, labels)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # Add channel dim
        label_tensor = torch.from_numpy(labels).long()
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'timepoint': timepoint,
        }


class MultiSampleDataset(Dataset):
    """Dataset that combines multiple embryo samples."""
    
    def __init__(
        self,
        samples: List[EmbryoSample],
        label_generator: LabelGenerator,
        augmentation: Optional[Callable] = None,
        normalize: bool = True,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        samples_per_epoch: Optional[int] = None,
    ) -> None:
        """Initialize multi-sample dataset.
        
        Args:
            samples: List of EmbryoSample objects
            label_generator: LabelGenerator for creating training labels
            augmentation: Optional augmentation function
            normalize: Whether to normalize intensities
            quantile_low: Lower quantile for normalization
            quantile_high: Upper quantile for normalization
            samples_per_epoch: Number of samples per epoch (None = all timepoints)
        """
        self.samples = samples
        self.label_generator = label_generator
        self.augmentation = augmentation
        self.normalize = normalize
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        
        # Create individual datasets
        self.datasets = []
        self.sample_indices = []
        self.timepoint_indices = []
        
        for sample_idx, sample in enumerate(samples):
            dataset = CellDetectionDataset(
                sample,
                label_generator,
                augmentation=None,  # Apply augmentation at multi-sample level
                normalize=normalize,
                quantile_low=quantile_low,
                quantile_high=quantile_high,
            )
            self.datasets.append(dataset)
            
            for t in range(len(dataset)):
                self.sample_indices.append(sample_idx)
                self.timepoint_indices.append(t)
        
        # Determine samples per epoch
        if samples_per_epoch is None:
            self.samples_per_epoch = len(self.sample_indices)
        else:
            self.samples_per_epoch = min(samples_per_epoch, len(self.sample_indices))
        
        logger.info(
            f"MultiSampleDataset initialized: {len(samples)} samples, "
            f"{len(self.sample_indices)} total timepoints, "
            f"samples_per_epoch={self.samples_per_epoch}"
        )
    
    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.
        
        Args:
            idx: Index (will be wrapped if >= len)
        
        Returns:
            Dictionary with 'image' and 'label' tensors
        """
        # Wrap index for random sampling
        idx = idx % len(self.sample_indices)
        
        sample_idx = self.sample_indices[idx]
        timepoint_idx = self.timepoint_indices[idx]
        
        # Get sample from individual dataset
        data = self.datasets[sample_idx][timepoint_idx]
        
        # Apply augmentation at this level
        if self.augmentation is not None:
            volume = data['image'].squeeze(0).numpy()
            labels = data['label'].numpy()
            volume, labels = self.augmentation(volume, labels)
            
            data['image'] = torch.from_numpy(volume).float().unsqueeze(0)
            data['label'] = torch.from_numpy(labels).long()
        
        # Add sample index
        data['sample_idx'] = sample_idx
        
        return data


class LazyZarrDataset(Dataset):
    """Memory-efficient dataset with lazy Zarr loading and on-the-fly augmentation."""
    
    def __init__(
        self,
        data_root: Path,
        sample_ids: List[str],
        label_generator: LabelGenerator,
        augmentation: Optional[Callable] = None,
        normalize: bool = True,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        cache_size: int = 4,
    ) -> None:
        """Initialize lazy Zarr dataset.
        
        Args:
            data_root: Path to data directory
            sample_ids: List of sample IDs to use
            label_generator: LabelGenerator for creating training labels
            augmentation: Optional augmentation function
            normalize: Whether to normalize intensities
            quantile_low: Lower quantile for normalization
            quantile_high: Upper quantile for normalization
            cache_size: Number of samples to keep in memory
        """
        self.data_root = data_root
        self.sample_ids = sample_ids
        self.label_generator = label_generator
        self.augmentation = augmentation
        self.normalize = normalize
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.cache_size = cache_size
        
        # Create dataset catalog
        self.catalog = DatasetCatalog(data_root)
        
        # Pre-compute timepoint counts for each sample
        self.sample_timepoint_counts = {}
        self.cumulative_counts = [0]
        
        for sample_id in sample_ids:
            sample = self.catalog.load_sample(sample_id, require_tracks=True)
            num_timepoints = sample.image_metadata.shape[0]
            self.sample_timepoint_counts[sample_id] = num_timepoints
            self.cumulative_counts.append(self.cumulative_counts[-1] + num_timepoints)
        
        # Simple LRU cache for samples
        self.sample_cache: Dict[str, EmbryoSample] = {}
        self.cache_order: List[str] = []
        
        logger.info(
            f"LazyZarrDataset initialized: {len(sample_ids)} samples, "
            f"{self.cumulative_counts[-1]} total timepoints, cache_size={cache_size}"
        )
    
    def __len__(self) -> int:
        """Return total number of timepoints across all samples."""
        return self.cumulative_counts[-1]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample with lazy loading.
        
        Args:
            idx: Global timepoint index
        
        Returns:
            Dictionary with 'image' and 'label' tensors
        """
        # Find which sample this index belongs to
        sample_idx = self._find_sample_idx(idx)
        sample_id = self.sample_ids[sample_idx]
        
        # Get local timepoint index within sample
        local_idx = idx - self.cumulative_counts[sample_idx]
        
        # Load sample (with caching)
        sample = self._get_cached_sample(sample_id)
        
        # Load volume
        embryo_dataset = EmbryoDataset(
            sample,
            normalize=self.normalize,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
        )
        volume = embryo_dataset.get_timepoint(local_idx)
        
        # Generate labels
        volume_shape = volume.shape
        labels = self.label_generator.generate_labels_from_graph(
            volume_shape,
            sample.tracks,
            local_idx,
        )
        
        # Apply augmentation
        if self.augmentation is not None:
            volume, labels = self.augmentation(volume, labels)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(volume).float().unsqueeze(0)
        label_tensor = torch.from_numpy(labels).long()
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'sample_id': sample_id,
            'timepoint': local_idx,
        }
    
    def _find_sample_idx(self, idx: int) -> int:
        """Find which sample index contains the given global index."""
        for i in range(len(self.cumulative_counts) - 1):
            if self.cumulative_counts[i] <= idx < self.cumulative_counts[i + 1]:
                return i
        raise IndexError(f"Index {idx} out of range")
    
    def _get_cached_sample(self, sample_id: str) -> EmbryoSample:
        """Get sample from cache or load it."""
        if sample_id in self.sample_cache:
            # Move to end of cache order (most recently used)
            self.cache_order.remove(sample_id)
            self.cache_order.append(sample_id)
            return self.sample_cache[sample_id]
        
        # Load sample
        sample = self.catalog.load_sample(sample_id, require_tracks=True)
        
        # Add to cache
        self.sample_cache[sample_id] = sample
        self.cache_order.append(sample_id)
        
        # Evict if over cache size
        while len(self.sample_cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.sample_cache[oldest]
        
        return sample


def create_training_datasets(
    data_root: Path,
    train_sample_ids: List[str],
    val_sample_ids: List[str],
    scale: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    center_radius: int = 2,
    periphery_radius: int = 4,
    crop_size: Optional[Tuple[int, int, int]] = (32, 128, 128),
    cache_size: int = 4,
) -> Tuple[LazyZarrDataset, LazyZarrDataset]:
    """Create training and validation datasets.
    
    Args:
        data_root: Path to data directory
        train_sample_ids: List of training sample IDs
        val_sample_ids: List of validation sample IDs
        scale: Physical scale (Z, Y, X) for augmentation
        center_radius: Radius for center labels
        periphery_radius: Radius for periphery labels
        crop_size: Optional crop size for training
        cache_size: Cache size for lazy loading
    
    Returns:
        Training and validation datasets
    """
    # Create label generator
    label_generator = WeightedLabelGenerator(
        center_radius=center_radius,
        periphery_radius=periphery_radius,
        center_weight=10.0,
        periphery_weight=5.0,
        background_weight=1.0,
    )
    
    # Create training augmentation
    train_augmentation = TrainingAugmentation(
        scale=scale,
        xy_rotation_range=15.0,
        z_flip_prob=0.5,
        xy_flip_prob=0.5,
        elastic_alpha=1.0,
        elastic_sigma=5.0,
        intensity_contrast_range=(0.8, 1.2),
        intensity_brightness_range=(-0.1, 0.1),
        noise_std=0.02,
        blur_sigma_range=(0.5, 1.5),
        z_scale_range=(0.9, 1.1),
        crop_size=crop_size,
    )
    
    # Create validation augmentation (minimal)
    val_augmentation = ValidationAugmentation()
    
    # Create datasets
    train_dataset = LazyZarrDataset(
        data_root=data_root,
        sample_ids=train_sample_ids,
        label_generator=label_generator,
        augmentation=train_augmentation,
        normalize=True,
        quantile_low=0.01,
        quantile_high=0.99,
        cache_size=cache_size,
    )
    
    val_dataset = LazyZarrDataset(
        data_root=data_root,
        sample_ids=val_sample_ids,
        label_generator=label_generator,
        augmentation=val_augmentation,
        normalize=True,
        quantile_low=0.01,
        quantile_high=0.99,
        cache_size=cache_size,
    )
    
    logger.info(
        f"Created datasets: train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples"
    )
    
    return train_dataset, val_dataset
