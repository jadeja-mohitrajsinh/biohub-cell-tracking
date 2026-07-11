"""Base detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DetectionResult:
    """Result of cell detection."""
    
    centroids: List[Tuple[int, int, int]]  # List of (z, y, x) centroids
    confidence: List[float]  # Confidence scores for each detection
    metadata: dict  # Additional metadata
    
    def __len__(self) -> int:
        return len(self.centroids)


class Detector(ABC):
    """Base class for cell detectors."""
    
    def __init__(self, config: dict) -> None:
        """Initialize detector with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells in a 3D volume.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with centroids and metadata
        """
        pass
    
    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess volume before detection.
        
        Args:
            volume: Input volume
        
        Returns:
            Preprocessed volume
        """
        # Default: no preprocessing
        return volume
    
    def postprocess(self, result: DetectionResult) -> DetectionResult:
        """Postprocess detection results.
        
        Args:
            result: DetectionResult to postprocess
        
        Returns:
            Postprocessed DetectionResult
        """
        # Default: no postprocessing
        return result
