"""Multi-embryo dataset using existing data infrastructure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator

import numpy as np

from data.dataset import DatasetCatalog, EmbryoDataset
from data.models import EmbryoSample

logger = logging.getLogger(__name__)


class MultiEmbryoDataset:
    """Dataset that loads multiple embryos using existing infrastructure.
    
    This wraps the existing EmbryoDataset to provide a simple interface
    for training on multiple embryos.
    """
    
    def __init__(
        self,
        data_root: Path,
        sample_ids: Optional[List[str]] = None,
        normalize: bool = True,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
    ):
        """Initialize multi-embryo dataset.
        
        Args:
            data_root: Path to data directory
            sample_ids: Optional list of specific sample IDs to use
            normalize: Whether to normalize volumes
            quantile_low: Lower quantile for normalization
            quantile_high: Upper quantile for normalization
        """
        self.data_root = data_root
        self.normalize = normalize
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        
        # Get available samples
        catalog = DatasetCatalog(data_root)
        all_samples = catalog.list_samples()
        
        if sample_ids:
            self.sample_ids = [s for s in sample_ids if s in all_samples]
        else:
            self.sample_ids = all_samples
        
        logger.info(f"MultiEmbryoDataset: {len(self.sample_ids)} embryos")
        
        # Cache for sample metadata
        self._metadata_cache: Dict[str, Dict] = {}
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> EmbryoSample:
        """Load a single embryo sample.
        
        Args:
            idx: Index of sample
        
        Returns:
            EmbryoSample with loaded volume
        """
        sample_id = self.sample_ids[idx]
        return self.load_sample(sample_id)
    
    def load_sample(self, sample_id: str) -> EmbryoSample:
        """Load a specific embryo sample using existing infrastructure.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            EmbryoSample with loaded volume
        """
        if sample_id in self._metadata_cache:
            return self._metadata_cache[sample_id]['sample']
        
        # Use DatasetCatalog to load the sample
        catalog = DatasetCatalog(self.data_root)
        sample = catalog.load_sample(sample_id, load_volume=True)
        
        # Cache the sample
        self._metadata_cache[sample_id] = {
            'sample': sample,
            'shape': sample.image_metadata.shape,
            'scale': sample.image_metadata.scale,
        }
        
        return sample
    
    def get_sample_info(self, sample_id: str) -> Dict:
        """Get metadata for a sample."""
        if sample_id not in self._metadata_cache:
            self.load_sample(sample_id)
        return self._metadata_cache[sample_id]
    
    def iter_samples(self) -> Iterator[EmbryoSample]:
        """Iterate over all samples."""
        for sample_id in self.sample_ids:
            yield self.load_sample(sample_id)
