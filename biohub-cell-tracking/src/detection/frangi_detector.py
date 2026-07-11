"""Simple threshold-based cell detection (fallback for missing scikit-image)."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter

from detection.base import Detector, DetectionResult

logger = logging.getLogger(__name__)


class FrangiDetector(Detector):
    """Cell detection using simple threshold + Gaussian smoothing (fallback)."""
    
    def __init__(self, config: dict) -> None:
        """Initialize detector.
        
        Args:
            config: Configuration dictionary with keys:
                - threshold: Detection threshold
                - sigma: Gaussian smoothing sigma
                - min_size: Minimum component size
                - max_size: Maximum component size
        """
        super().__init__(config)
        self.threshold = config.get("threshold", 0.15)
        self.sigma = config.get("sigma", 2.0)
        self.min_size = config.get("min_size", 5)
        self.max_size = config.get("max_size", 500)
        
        logger.info(
            f"FrangiDetector (threshold-based) initialized: "
            f"threshold={self.threshold}, sigma={self.sigma}, "
            f"min_size={self.min_size}, max_size={self.max_size}"
        )
    
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells using threshold + Gaussian smoothing.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with centroids
        """
        logger.debug(f"Running detection on volume shape {volume.shape}")
        
        # Normalize volume to 0-1
        if volume.max() > 0:
            volume_norm = volume / volume.max()
        else:
            volume_norm = volume
        
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(volume_norm, sigma=self.sigma)
        
        # Threshold
        binary = smoothed > self.threshold
        
        logger.debug(f"Threshold applied, binary pixels: {binary.sum()}")
        
        # Find connected components
        labeled, num_features = label(binary)
        
        logger.debug(f"Found {num_features} components before size filtering")
        
        # Filter by size and compute centroids
        centroids = []
        confidences = []
        
        for label_id in range(1, num_features + 1):
            mask = labeled == label_id
            size = mask.sum()
            
            if self.min_size <= size <= self.max_size:
                # Compute centroid
                center = center_of_mass(mask)
                z, y, x = [int(round(c)) for c in center]
                
                # Ensure within bounds
                z = max(0, min(z, volume.shape[0] - 1))
                y = max(0, min(y, volume.shape[1] - 1))
                x = max(0, min(x, volume.shape[2] - 1))
                
                centroids.append((z, y, x))
                # Use mean smoothed intensity as confidence
                confidences.append(float(smoothed[mask].mean()))
        
        logger.info(f"Detection: {len(centroids)} cells detected")
        
        return DetectionResult(
            centroids=centroids,
            confidence=confidences,
            metadata={
                "method": "threshold_gaussian",
                "num_components": num_features,
                "threshold": self.threshold,
            }
        )
