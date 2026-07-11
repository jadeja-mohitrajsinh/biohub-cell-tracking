"""Temporal window dataset using existing data infrastructure."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from training.multi_embryo_dataset import MultiEmbryoDataset

logger = logging.getLogger(__name__)


class TemporalWindowDataset(Dataset):
    """Dataset that samples temporal windows from embryos.
    
    Each sample is a clip of consecutive frames (e.g., 8-16 frames)
    suitable for learning temporal consistency and tracking.
    """
    
    def __init__(
        self,
        multi_embryo_dataset: MultiEmbryoDataset,
        window_size: int = 8,
        stride: int = 4,
        min_window_size: int = 4,
        max_window_size: int = 16,
        randomize_window_size: bool = True,
        include_ground_truth: bool = False,
    ):
        """Initialize temporal window dataset.
        
        Args:
            multi_embryo_dataset: Multi-embryo dataset to sample from
            window_size: Default window size in frames
            stride: Stride between consecutive windows
            min_window_size: Minimum window size for randomization
            max_window_size: Maximum window size for randomization
            randomize_window_size: Whether to randomize window sizes
            include_ground_truth: Whether to include ground truth tracks
        """
        self.multi_embryo_dataset = multi_embryo_dataset
        self.window_size = window_size
        self.stride = stride
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.randomize_window_size = randomize_window_size
        self.include_ground_truth = include_ground_truth
        
        # Build index of all possible windows
        self.window_index = self._build_window_index()
        
        logger.info(
            f"TemporalWindowDataset: {len(self.window_index)} windows, "
            f"window_size={window_size}, stride={stride}"
        )
    
    def _build_window_index(self) -> List[Tuple[int, int, int]]:
        """Build index of all possible temporal windows.
        
        Returns:
            List of (sample_idx, t_start, t_end) tuples
        """
        window_index = []
        
        for sample_idx in range(len(self.multi_embryo_dataset)):
            sample_id = self.multi_embryo_dataset.sample_ids[sample_idx]
            sample_info = self.multi_embryo_dataset.get_sample_info(sample_id)
            num_frames = sample_info['shape'][0]
            
            # Generate windows
            t_start = 0
            while t_start + self.min_window_size <= num_frames:
                # Determine window size
                if self.randomize_window_size:
                    window_size = np.random.randint(
                        self.min_window_size,
                        min(self.max_window_size, num_frames - t_start) + 1
                    )
                else:
                    window_size = min(self.window_size, num_frames - t_start)
                
                t_end = min(t_start + window_size, num_frames)
                window_index.append((sample_idx, t_start, t_end))
                
                t_start += self.stride
        
        return window_index
    
    def __len__(self) -> int:
        return len(self.window_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a temporal window sample.
        
        Args:
            idx: Index of window
        
        Returns:
            Dictionary with volume clip and metadata
        """
        sample_idx, t_start, t_end = self.window_index[idx]
        sample_id = self.multi_embryo_dataset.sample_ids[sample_idx]
        
        # Load sample
        sample = self.multi_embryo_dataset.load_sample(sample_id)
        
        # Extract temporal window from volume
        volume = sample.volume
        volume_clip = volume[t_start:t_end]
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume_clip).float()
        
        # Add channel dimension if needed
        if volume_tensor.ndim == 4:
            volume_tensor = volume_tensor.unsqueeze(1)  # (T, 1, Z, Y, X)
        
        result = {
            'volume': volume_tensor,
            'sample_id': sample_id,
            't_start': t_start,
            't_end': t_end,
            'scale': sample.image_metadata.scale,
        }
        
        # Include ground truth if available
        if self.include_ground_truth and hasattr(sample, 'ground_truth'):
            result['ground_truth'] = sample.ground_truth
        
        return result


class PatchTemporalWindowDataset(TemporalWindowDataset):
    """Dataset that samples both temporal windows and spatial patches.
    
    This is memory-efficient for large 3D volumes by cropping to
    smaller spatial patches (e.g., 96x96x96) while keeping temporal context.
    """
    
    def __init__(
        self,
        multi_embryo_dataset: MultiEmbryoDataset,
        patch_size: int = 96,
        overlap: int = 32,
        window_size: int = 8,
        stride: int = 4,
        randomize_window_size: bool = True,
        include_ground_truth: bool = False,
    ):
        """Initialize patch temporal window dataset.
        
        Args:
            multi_embryo_dataset: Multi-embryo dataset
            patch_size: Size of spatial patches (cubic)
            overlap: Overlap between patches
            window_size: Default temporal window size
            stride: Stride between windows
            randomize_window_size: Whether to randomize window sizes
            include_ground_truth: Whether to include ground truth
        """
        self.patch_size = patch_size
        self.overlap = overlap
        
        super().__init__(
            multi_embryo_dataset=multi_embryo_dataset,
            window_size=window_size,
            stride=stride,
            randomize_window_size=randomize_window_size,
            include_ground_truth=include_ground_truth,
        )
        
        # Rebuild index with patches
        self.window_index = self._build_patch_window_index()
        
        logger.info(
            f"PatchTemporalWindowDataset: {len(self.window_index)} patch-windows, "
            f"patch_size={patch_size}, overlap={overlap}"
        )
    
    def _build_patch_window_index(self) -> List[Tuple[int, int, int, Tuple[int, int, int], Tuple[int, int, int]]]:
        """Build index of patch-window combinations.
        
        Returns:
            List of (sample_idx, t_start, t_end, z_slice, yx_slice) tuples
        """
        window_index = []
        
        for sample_idx in range(len(self.multi_embryo_dataset)):
            sample_id = self.multi_embryo_dataset.sample_ids[sample_idx]
            sample_info = self.multi_embryo_dataset.get_sample_info(sample_id)
            num_frames = sample_info['shape'][0]
            z_size, y_size, x_size = sample_info['shape'][1:]
            
            # Generate temporal windows
            t_start = 0
            while t_start + self.min_window_size <= num_frames:
                window_size = min(self.window_size, num_frames - t_start)
                t_end = t_start + window_size
                
                # Generate spatial patches
                z_stride = self.patch_size - self.overlap
                y_stride = self.patch_size - self.overlap
                x_stride = self.patch_size - self.overlap
                
                z_start = 0
                while z_start + self.patch_size <= z_size:
                    z_end = min(z_start + self.patch_size, z_size)
                    
                    y_start = 0
                    while y_start + self.patch_size <= y_size:
                        y_end = min(y_start + self.patch_size, y_size)
                        
                        x_start = 0
                        while x_start + self.patch_size <= x_size:
                            x_end = min(x_start + self.patch_size, x_size)
                            
                            window_index.append((
                                sample_idx,
                                t_start,
                                t_end,
                                (z_start, z_end),
                                (y_start, y_end, x_start, x_end),
                            ))
                            
                            x_start += x_stride
                        y_start += y_stride
                    z_start += z_stride
                
                t_start += self.stride
        
        return window_index
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a patch-window sample.
        
        Args:
            idx: Index of patch-window
        
        Returns:
            Dictionary with volume patch and metadata
        """
        sample_idx, t_start, t_end, z_slice, yx_slice = self.window_index[idx]
        z_start, z_end = z_slice
        y_start, y_end, x_start, x_end = yx_slice
        
        sample_id = self.multi_embryo_dataset.sample_ids[sample_idx]
        sample = self.multi_embryo_dataset.load_sample(sample_id)
        
        # Load temporal window with spatial crop
        volume = sample.volume
        volume_clip = volume[t_start:t_end, z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume_clip).float()
        
        # Add channel dimension
        if volume_tensor.ndim == 4:
            volume_tensor = volume_tensor.unsqueeze(1)
        
        result = {
            'volume': volume_tensor,
            'sample_id': sample_id,
            't_start': t_start,
            't_end': t_end,
            'z_slice': z_slice,
            'yx_slice': yx_slice,
            'scale': sample.image_metadata.scale,
        }
        
        if self.include_ground_truth and hasattr(sample, 'ground_truth'):
            result['ground_truth'] = sample.ground_truth
        
        return result
