"""Base pipeline interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from data.models import EmbryoSample, TrackingGraph


class Pipeline(ABC):
    """Base class for detection + tracking pipelines."""
    
    def __init__(self, config: dict) -> None:
        """Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def process(self, sample: EmbryoSample) -> TrackingGraph:
        """Process a sample to generate tracking graph.
        
        Args:
            sample: EmbryoSample with image data
        
        Returns:
            TrackingGraph with predicted tracks
        """
        pass
