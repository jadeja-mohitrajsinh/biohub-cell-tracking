"""GT-guided detector for debugging and understanding intensity distribution."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter

from data.dataset import EmbryoDataset
from data.models import CellNode, TrackingGraph
from detection.base import Detector, DetectionResult

logger = logging.getLogger(__name__)


class GTGuidedDetector(Detector):
    """Detector that learns from GT positions to understand intensity distribution."""
    
    def __init__(self, config: dict) -> None:
        """Initialize GT-guided detector.
        
        Args:
            config: Configuration dictionary with keys:
                - sample_radius: Radius around GT to sample (voxels)
                - threshold_percentile: Percentile of GT intensities to use as threshold
                - sigma: Gaussian smoothing sigma
                - min_size: Minimum component size
                - max_size: Maximum component size
        """
        super().__init__(config)
        self.sample_radius = config.get("sample_radius", 5)
        self.threshold_percentile = config.get("threshold_percentile", 10)
        self.sigma = config.get("sigma", 1.5)
        self.min_size = config.get("min_size", 3)
        self.max_size = config.get("max_size", 1000)
        
        # Will be learned from GT
        self.learned_threshold = None
        
        logger.info(
            f"GTGuidedDetector initialized: sample_radius={self.sample_radius}, "
            f"threshold_percentile={self.threshold_percentile}"
        )
    
    def learn_from_gt(
        self,
        sample,
        dataset: EmbryoDataset,
        num_timepoints: int = 10
    ) -> None:
        """Learn intensity distribution from GT positions.
        
        Args:
            sample: EmbryoSample with GT tracks
            dataset: EmbryoDataset for loading volumes
            num_timepoints: Number of timepoints to sample
        """
        logger.info("Learning intensity distribution from GT...")
        
        intensities = []
        
        # Sample intensities around GT positions
        for t in range(min(num_timepoints, sample.image_metadata.shape[0])):
            volume = dataset.get_timepoint(t)
            
            # Find GT nodes at this timepoint
            gt_nodes = [node for node in sample.tracks.iter_nodes() if node.t == t]
            
            for node in gt_nodes:
                # Sample intensity in neighborhood
                z_start = max(0, node.z - self.sample_radius)
                z_end = min(volume.shape[0], node.z + self.sample_radius + 1)
                y_start = max(0, node.y - self.sample_radius)
                y_end = min(volume.shape[1], node.y + self.sample_radius + 1)
                x_start = max(0, node.x - self.sample_radius)
                x_end = min(volume.shape[2], node.x + self.sample_radius + 1)
                
                neighborhood = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                intensities.extend(neighborhood.flatten())
        
        if not intensities:
            logger.warning("No intensities sampled from GT, using default threshold")
            self.learned_threshold = 0.1
            return
        
        intensities = np.array(intensities)
        
        # Compute threshold based on percentile
        self.learned_threshold = np.percentile(intensities, self.threshold_percentile)
        
        logger.info(
            f"Learned threshold: {self.learned_threshold:.4f} "
            f"(from {len(intensities)} samples, "
            f"intensity range: [{intensities.min():.4f}, {intensities.max():.4f}], "
            f"mean: {intensities.mean():.4f}, "
            f"median: {np.median(intensities):.4f})"
        )
    
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells using learned threshold with local maxima.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with centroids
        """
        if self.learned_threshold is None:
            logger.warning("No learned threshold, using default 0.1")
            threshold = 0.1
        else:
            threshold = self.learned_threshold
        
        logger.debug(f"Running GT-guided detection with threshold={threshold:.4f}")
        
        # Normalize volume to 0-1
        if volume.max() > 0:
            volume_norm = volume / volume.max()
        else:
            volume_norm = volume
        
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(volume_norm, sigma=self.sigma)
        
        # Use local maxima detection instead of connected components
        from scipy.ndimage import maximum_filter
        
        # Find local maxima in 3D
        neighborhood_size = 3  # 3x3x3 neighborhood
        local_max = maximum_filter(smoothed, size=neighborhood_size) == smoothed
        
        # Apply threshold
        binary = local_max & (smoothed > threshold)
        
        logger.debug(f"Local maxima found: {binary.sum()}")
        
        # Get coordinates of local maxima
        centroids = []
        confidences = []
        
        z_coords, y_coords, x_coords = np.where(binary)
        
        for z, y, x in zip(z_coords, y_coords, x_coords):
            centroids.append((int(z), int(y), int(x)))
            confidences.append(float(smoothed[z, y, x]))
        
        logger.info(f"GT-guided detection: {len(centroids)} cells detected")
        
        return DetectionResult(
            centroids=centroids,
            confidence=confidences,
            metadata={
                "method": "gt_guided_local_max",
                "threshold": threshold,
                "learned": self.learned_threshold is not None,
            }
        )
