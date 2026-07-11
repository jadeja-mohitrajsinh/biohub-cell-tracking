"""Balanced batch sampler for curriculum training."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class EmbryoDifficultySampler(Sampler):
    """Sampler that balances samples by embryo difficulty.
    
    Embryos are categorized by difficulty:
    - Stage 1: Sparse, low motion, no divisions
    - Stage 2: Medium density, moderate motion
    - Stage 3: Dense, high motion, divisions
    - Stage 4: Mixed (all stages)
    """
    
    def __init__(
        self,
        dataset,
        difficulty_map: Dict[str, int],
        stage: int = 1,
        batch_size: int = 4,
        samples_per_stage: int = 1000,
    ):
        """Initialize difficulty-based sampler.
        
        Args:
            dataset: Dataset to sample from
            difficulty_map: Mapping from sample_id to difficulty stage (1-4)
            stage: Current training stage (1-4)
            batch_size: Batch size
            samples_per_stage: Number of samples to draw per stage
        """
        self.dataset = dataset
        self.difficulty_map = difficulty_map
        self.stage = stage
        self.batch_size = batch_size
        self.samples_per_stage = samples_per_stage
        
        # Group samples by difficulty
        self.samples_by_stage = self._group_by_difficulty()
        
        # Build sampling schedule
        self.sampling_schedule = self._build_schedule()
        
        logger.info(
            f"EmbryoDifficultySampler: stage={stage}, "
            f"batch_size={batch_size}, "
            f"total_batches={len(self.sampling_schedule)}"
        )
    
    def _group_by_difficulty(self) -> Dict[int, List[int]]:
        """Group dataset indices by difficulty stage."""
        samples_by_stage = {1: [], 2: [], 3: [], 4: []}
        
        for idx in range(len(self.dataset)):
            sample_id = self.dataset.sample_ids[idx]
            difficulty = self.difficulty_map.get(sample_id, 2)  # Default to medium
            samples_by_stage[difficulty].append(idx)
        
        for stage, indices in samples_by_stage.items():
            logger.info(f"Stage {stage}: {len(indices)} samples")
        
        return samples_by_stage
    
    def _build_schedule(self) -> List[List[int]]:
        """Build sampling schedule based on current stage."""
        schedule = []
        
        if self.stage == 1:
            # Only stage 1 samples
            indices = self.samples_by_stage[1]
            if not indices:
                indices = self.samples_by_stage[2]  # Fall back to medium
            schedule = self._create_batches(indices)
        
        elif self.stage == 2:
            # Mix stage 1 and 2 (70% stage 2, 30% stage 1)
            stage1 = self.samples_by_stage[1]
            stage2 = self.samples_by_stage[2]
            mixed = self._mix_stages([stage2, stage1], [0.7, 0.3])
            schedule = self._create_batches(mixed)
        
        elif self.stage == 3:
            # Mix stage 2 and 3 (60% stage 3, 40% stage 2)
            stage2 = self.samples_by_stage[2]
            stage3 = self.samples_by_stage[3]
            mixed = self._mix_stages([stage3, stage2], [0.6, 0.4])
            schedule = self._create_batches(mixed)
        
        elif self.stage == 4:
            # Mix all stages (40% stage 3, 30% stage 2, 20% stage 1, 10% stage 4)
            stage1 = self.samples_by_stage[1]
            stage2 = self.samples_by_stage[2]
            stage3 = self.samples_by_stage[3]
            stage4 = self.samples_by_stage[4]
            mixed = self._mix_stages([stage3, stage2, stage1, stage4], [0.4, 0.3, 0.2, 0.1])
            schedule = self._create_batches(mixed)
        
        return schedule
    
    def _mix_stages(self, stage_indices: List[List[int]], weights: List[float]) -> List[int]:
        """Mix samples from different stages with given weights."""
        mixed = []
        total_samples = self.samples_per_stage
        
        for indices, weight in zip(stage_indices, weights):
            n_samples = int(total_samples * weight)
            if indices:
                selected = np.random.choice(indices, min(n_samples, len(indices)), replace=True)
                mixed.extend(selected.tolist())
        
        np.random.shuffle(mixed)
        return mixed
    
    def _create_batches(self, indices: List[int]) -> List[List[int]]:
        """Create batches from indices."""
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
            else:
                # Pad last batch if needed
                padding = np.random.choice(indices, self.batch_size - len(batch), replace=True)
                batch.extend(padding.tolist())
                batches.append(batch)
        
        return batches
    
    def __iter__(self):
        """Iterate over batches."""
        for batch in self.sampling_schedule:
            yield batch
    
    def __len__(self) -> int:
        return len(self.sampling_schedule)


class TemporalBatchSampler(Sampler):
    """Sampler that creates batches of temporal clips from different embryos.
    
    Each batch contains clips from multiple embryos to improve generalization.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 4,
        clips_per_batch: int = 8,
        shuffle: bool = True,
    ):
        """Initialize temporal batch sampler.
        
        Args:
            dataset: TemporalWindowDataset
            batch_size: Number of clips per batch
            clips_per_batch: Number of clips to sample per embryo in batch
            shuffle: Whether to shuffle samples
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.clips_per_batch = clips_per_batch
        self.shuffle = shuffle
        
        # Group indices by sample_id
        self.indices_by_sample = self._group_by_sample()
        
        # Build batches
        self.batches = self._build_batches()
        
        logger.info(
            f"TemporalBatchSampler: {len(self.batches)} batches, "
            f"batch_size={batch_size}, clips_per_batch={clips_per_batch}"
        )
    
    def _group_by_sample(self) -> Dict[str, List[int]]:
        """Group dataset indices by sample_id."""
        indices_by_sample = {}
        
        for idx in range(len(self.dataset)):
            sample_id = self.dataset.window_index[idx][0]
            if sample_id not in indices_by_sample:
                indices_by_sample[sample_id] = []
            indices_by_sample[sample_id].append(idx)
        
        return indices_by_sample
    
    def _build_batches(self) -> List[List[int]]:
        """Build batches of temporal clips."""
        batches = []
        sample_ids = list(self.indices_by_sample.keys())
        
        if self.shuffle:
            np.random.shuffle(sample_ids)
        
        # Create batches by sampling clips from random embryos
        num_batches = (len(self.dataset) // self.batch_size) * 2  # Oversample
        
        for _ in range(num_batches):
            # Select random embryos for this batch
            selected_samples = np.random.choice(
                sample_ids,
                min(self.batch_size, len(sample_ids)),
                replace=False
            )
            
            batch = []
            for sample_id in selected_samples:
                sample_indices = self.indices_by_sample[sample_id]
                # Sample clips from this embryo
                clips = np.random.choice(
                    sample_indices,
                    min(self.clips_per_batch, len(sample_indices)),
                    replace=True
                )
                batch.extend(clips.tolist())
            
            if len(batch) >= self.batch_size:
                batches.append(batch[:self.batch_size])
        
        return batches
    
    def __iter__(self):
        """Iterate over batches."""
        for batch in self.batches:
            yield batch
    
    def __len__(self) -> int:
        return len(self.batches)


def create_difficulty_map(data_root, sample_ids: Optional[List[str]] = None) -> Dict[str, int]:
    """Create difficulty map for embryos based on statistics.
    
    Args:
        data_root: Path to data directory
        sample_ids: Optional list of sample IDs
    
    Returns:
        Mapping from sample_id to difficulty stage (1-4)
    """
    from data.dataset import DatasetCatalog
    
    catalog = DatasetCatalog(data_root)
    all_samples = catalog.list_samples()
    
    if sample_ids:
        all_samples = [s for s in all_samples if s in sample_ids]
    
    difficulty_map = {}
    
    for sample_id in all_samples:
        # For now, assign random difficulty (should be based on actual stats)
        # In production, this would analyze:
        # - Cell density
        # - Motion statistics
        # - Division frequency
        # - Image quality
        
        import random
        difficulty = random.choices([1, 2, 3, 4], weights=[0.2, 0.4, 0.3, 0.1])[0]
        difficulty_map[sample_id] = difficulty
    
    return difficulty_map
