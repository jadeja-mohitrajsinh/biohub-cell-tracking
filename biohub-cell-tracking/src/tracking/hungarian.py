"""Hungarian algorithm-based cell tracking."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.models import TrackingEdge, VoxelScale
from tracking.base import Tracker, TrackingResult

logger = logging.getLogger(__name__)


class HungarianTracker(Tracker):
    """Cell tracking using Hungarian optimal assignment."""
    
    def __init__(self, config: dict, scale: VoxelScale) -> None:
        """Initialize Hungarian tracker.
        
        Args:
            config: Configuration dictionary with keys:
                - max_distance_um: Maximum matching distance in micrometers
            scale: VoxelScale for physical distance calculation
        """
        super().__init__(config, scale)
        self.max_distance_um = config.get("max_distance_um", 7.0)
        
        logger.info(f"HungarianTracker initialized: max_distance_um={self.max_distance_um}")
    
    def track(self, detections_by_time: Dict[int, List[Tuple[int, int, int]]]) -> TrackingResult:
        """Link detections across time using Hungarian matching.
        
        Args:
            detections_by_time: Dict mapping time -> list of (z, y, x) detections
        
        Returns:
            TrackingResult with edges and node IDs
        """
        logger.debug(f"Running Hungarian tracking on {len(detections_by_time)} timepoints")
        
        # Assign node IDs
        node_ids = {}
        self._detection_to_node_id = {}
        node_id_counter = 0
        
        for t in sorted(detections_by_time.keys()):
            for idx, detection in enumerate(detections_by_time[t]):
                node_id = t * 10000 + node_id_counter  # Unique ID
                node_ids[node_id] = (t, *detection)
                self._detection_to_node_id[(t, idx)] = node_id
                node_id_counter += 1
        
        # Build edges using Hungarian matching
        edges = []
        sorted_times = sorted(detections_by_time.keys())
        
        for i in range(len(sorted_times) - 1):
            t_current = sorted_times[i]
            t_next = sorted_times[i + 1]
            
            if t_current not in detections_by_time or t_next not in detections_by_time:
                continue
            
            current_detections = detections_by_time[t_current]
            next_detections = detections_by_time[t_next]
            
            # Build cost matrix
            cost_matrix = self._build_cost_matrix(
                t_current, current_detections,
                t_next, next_detections
            )
            
            # Solve assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Create edges for valid assignments
            for row, col in zip(row_indices, col_indices):
                cost = cost_matrix[row, col]
                if cost < self.max_distance_um:
                    # Find corresponding node IDs
                    source_idx = row
                    target_idx = col
                    
                    # Get node IDs (need to map back)
                    source_id = self._get_node_id(t_current, source_idx, node_ids)
                    target_id = self._get_node_id(t_next, target_idx, node_ids)
                    
                    if source_id is not None and target_id is not None:
                        edges.append(TrackingEdge(source_id=source_id, target_id=target_id))
        
        logger.info(f"Hungarian tracking: {len(edges)} edges created")
        
        # Clear mapping cache after tracking is done
        self._detection_to_node_id = {}
        
        return TrackingResult(
            edges=edges,
            node_ids=node_ids,
            metadata={
                "method": "hungarian",
                "max_distance_um": self.max_distance_um,
                "num_timepoints": len(detections_by_time),
            }
        )
    
    def _build_cost_matrix(
        self,
        t_current: int,
        current_detections: List[Tuple[int, int, int]],
        t_next: int,
        next_detections: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Build cost matrix for Hungarian matching.
        
        Args:
            t_current: Current timepoint
            current_detections: List of (z, y, x) at current time
            t_next: Next timepoint
            next_detections: List of (z, y, x) at next time
        
        Returns:
            Cost matrix (N_current × N_next)
        """
        n_current = len(current_detections)
        n_next = len(next_detections)
        
        # Initialize with large cost (10× max distance)
        cost_matrix = np.full((n_current, n_next), self.max_distance_um * 10.0, dtype=np.float64)
        
        for i, (z_curr, y_curr, x_curr) in enumerate(current_detections):
            for j, (z_next, y_next, x_next) in enumerate(next_detections):
                # Calculate physical distance
                pos1 = (t_current, z_curr, y_curr, x_curr)
                pos2 = (t_next, z_next, y_next, x_next)
                distance = self.physical_distance(pos1, pos2)
                
                if distance <= self.max_distance_um:
                    cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def _get_node_id(
        self,
        t: int,
        idx: int,
        node_ids: Dict[int, Tuple[int, int, int, int]]
    ) -> int | None:
        """Get node ID for a given timepoint and detection index.
        
        Args:
            t: Timepoint
            idx: Detection index at that timepoint
            node_ids: Dict of node_id -> (t, z, y, x)
        
        Returns:
            Node ID or None if not found
        """
        # Fast lookup
        if hasattr(self, "_detection_to_node_id") and self._detection_to_node_id:
            return self._detection_to_node_id.get((t, idx))
            
        # Fallback to linear search
        count = 0
        for node_id, (node_t, _, _, _) in node_ids.items():
            if node_t == t:
                if count == idx:
                    return node_id
                count += 1
        return None
