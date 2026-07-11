"""Base tracker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from data.models import TrackingEdge, VoxelScale


@dataclass
class TrackingResult:
    """Result of cell tracking."""
    
    edges: List[TrackingEdge]
    node_ids: Dict[int, Tuple[int, int, int, int]]  # node_id -> (t, z, y, x)
    metadata: dict
    
    def __len__(self) -> int:
        return len(self.edges)


class Tracker(ABC):
    """Base class for cell trackers."""
    
    def __init__(self, config: dict, scale: VoxelScale) -> None:
        """Initialize tracker with configuration.
        
        Args:
            config: Configuration dictionary
            scale: VoxelScale for physical distance calculation
        """
        self.config = config
        self.scale = scale
    
    @abstractmethod
    def track(self, detections_by_time: Dict[int, List[Tuple[int, int, int]]]) -> TrackingResult:
        """Link detections across time.
        
        Args:
            detections_by_time: Dict mapping time -> list of (z, y, x) detections
        
        Returns:
            TrackingResult with edges and node IDs
        """
        pass
    
    def physical_distance(
        self,
        pos1: Tuple[int, int, int, int],
        pos2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate physical distance in micrometers.
        
        Args:
            pos1: (t, z, y, x) position
            pos2: (t, z, y, x) position
        
        Returns:
            Distance in micrometers
        """
        t1, z1, y1, x1 = pos1
        t2, z2, y2, x2 = pos2
        
        dt = (t2 - t1) * self.scale.t
        dz = (z2 - z1) * self.scale.z
        dy = (y2 - y1) * self.scale.y
        dx = (x2 - x1) * self.scale.x
        
        return float(np.sqrt(dt**2 + dz**2 + dy**2 + dx**2))
