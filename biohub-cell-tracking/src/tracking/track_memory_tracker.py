"""Tracker with persistent track memory and Kalman motion prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.models import TrackingEdge, VoxelScale
from tracking.base import Tracker, TrackingResult
from tracking.kalman_filter import KalmanFilter3D

logger = logging.getLogger(__name__)


@dataclass
class TrackMemory:
    """Memory for a single track with persistent ID."""
    track_id: int
    position: np.ndarray  # (z, y, x)
    velocity: np.ndarray  # (vz, vy, vx)
    age: int
    confidence: float
    kalman_filter: KalmanFilter3D
    last_seen_time: int


class TrackMemoryTracker(Tracker):
    """Tracker with persistent track memory and Kalman motion prediction."""
    
    def __init__(
        self,
        config: dict,
        scale: VoxelScale,
    ) -> None:
        super().__init__(config, scale)
        
        # Track memory: track_id -> TrackMemory
        self.track_memory: Dict[int, TrackMemory] = {}
        
        # Mapping: node_id -> track_id
        self.node_to_track: Dict[int, int] = {}
        
        # Track ID counter
        self.track_id_counter = 0
        
        # Kalman filter parameters
        self.process_noise = config.get("process_noise", 0.1)
        self.measurement_noise = config.get("measurement_noise", 0.5)
        self.initial_velocity_std = config.get("initial_velocity_std", 1.0)
        
        # Motion gating parameters
        self.max_prediction_distance_um = config.get("max_prediction_distance_um", 10.0)
        self.max_velocity_um_per_frame = config.get("max_velocity_um_per_frame", 5.0)
        
        # Hungarian matching parameters
        self.max_distance_um = config.get("max_distance_um", 7.0)
        
        logger.info(
            f"TrackMemoryTracker initialized: max_distance={self.max_distance_um}, "
            f"motion_gate={self.max_prediction_distance_um}"
        )
    
    def track(
        self,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        confidences_by_time: Optional[Dict[int, List[float]]] = None,
        volumes_by_time: Optional[Dict[int, np.ndarray]] = None,
    ) -> TrackingResult:
        """Link detections across time using track memory and Kalman prediction.
        
        Args:
            detections_by_time: Dict mapping time -> list of (z, y, x) detections
            confidences_by_time: Optional dict mapping time -> confidence scores
            volumes_by_time: Optional dict mapping time -> volume arrays
        
        Returns:
            TrackingResult with edges and node IDs
        """
        logger.debug(f"Running track memory tracking on {len(detections_by_time)} timepoints")
        
        # Assign node IDs
        node_ids = {}
        self._detection_to_node_id = {}
        node_id_counter = 0
        
        for t in sorted(detections_by_time.keys()):
            for idx, detection in enumerate(detections_by_time[t]):
                node_id = t * 10000 + node_id_counter
                self._detection_to_node_id[(t, idx)] = node_id
                node_ids[node_id] = (t, *detection)
                node_id_counter += 1
        
        # Build edges using track memory
        edges = []
        sorted_times = sorted(detections_by_time.keys())
        
        for i in range(len(sorted_times) - 1):
            t_current = sorted_times[i]
            t_next = sorted_times[i + 1]
            
            if t_current not in detections_by_time or t_next not in detections_by_time:
                continue
            
            current_detections = detections_by_time[t_current]
            next_detections = detections_by_time[t_next]
            
            current_confidences = confidences_by_time.get(t_current, [1.0] * len(current_detections))
            next_confidences = confidences_by_time.get(t_next, [1.0] * len(next_detections))
            
            # Build cost matrix with motion prediction
            cost_matrix = self._build_cost_matrix_with_motion(
                current_detections,
                next_detections,
                current_confidences,
                next_confidences,
                t_current,
                t_next,
            )
            
            # Solve assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Create edges and update track memory
            for row, col in zip(row_indices, col_indices):
                cost = cost_matrix[row, col]
                
                if cost < self.max_distance_um:
                    source_id = self._get_node_id(t_current, row, node_ids)
                    target_id = self._get_node_id(t_next, col, node_ids)
                    
                    if source_id is not None and target_id is not None:
                        edges.append(TrackingEdge(source_id=source_id, target_id=target_id))
                        
                        # Update track memory
                        self._update_track_memory(
                            source_id,
                            target_id,
                            current_detections[row],
                            next_detections[col],
                            current_confidences[row],
                            next_confidences[col],
                            t_next
                        )
            
            # Handle new tracks (unmatched detections at t_next)
            self._initialize_new_tracks(
                t_next,
                next_detections,
                next_confidences,
                row_indices,
                col_indices,
                node_ids
            )
        
        logger.info(f"Track memory tracking: {len(edges)} edges created, {len(self.track_memory)} tracks")
        
        # Clear mapping cache
        self._detection_to_node_id = {}
        
        return TrackingResult(
            edges=edges,
            node_ids=node_ids,
            metadata={
                "method": "track_memory",
                "num_tracks": len(self.track_memory),
                "num_timepoints": len(detections_by_time),
            }
        )
    
    def _build_cost_matrix_with_motion(
        self,
        detections_t: List[Tuple[int, int, int]],
        detections_t1: List[Tuple[int, int, int]],
        confidences_t: List[float],
        confidences_t1: List[float],
        t_current: int,
        t_next: int,
    ) -> np.ndarray:
        """Build cost matrix with Kalman motion prediction and gating."""
        n_t = len(detections_t)
        n_t1 = len(detections_t1)
        
        # Initialize with large cost
        cost_matrix = np.full((n_t, n_t1), self.max_distance_um * 10.0, dtype=np.float64)
        
        for i, (z_curr, y_curr, x_curr) in enumerate(detections_t):
            source_id = self._get_node_id(t_current, i, self._detection_to_node_id)
            
            if source_id is None:
                continue
            
            # Get track memory if exists
            track_id = self.node_to_track.get(source_id)
            
            if track_id is not None and track_id in self.track_memory:
                track = self.track_memory[track_id]
                
                # Predict next position using Kalman filter
                predicted_state = track.kalman_filter.predict()
                predicted_position = predicted_state.position
                
                # Calculate cost for each candidate
                for j, (z_next, y_next, x_next) in enumerate(detections_t1):
                    candidate_position = np.array([z_next, y_next, x_next])
                    
                    # Physical distance from prediction
                    pred_distance = self._physical_distance_3d(
                        predicted_position, candidate_position
                    )
                    
                    # Motion gating: reject if too far from prediction
                    if pred_distance > self.max_prediction_distance_um:
                        cost_matrix[i, j] = 999.0  # Reject
                    else:
                        # Cost = distance from prediction (lower is better)
                        cost_matrix[i, j] = pred_distance
            else:
                # No track memory: use simple distance
                for j, (z_next, y_next, x_next) in enumerate(detections_t1):
                    distance = self._physical_distance(
                        (t_current, z_curr, y_curr, x_curr),
                        (t_next, z_next, y_next, x_next)
                    )
                    if distance <= self.max_distance_um:
                        cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def _update_track_memory(
        self,
        source_id: int,
        target_id: int,
        source_position: Tuple[int, int, int],
        target_position: Tuple[int, int, int],
        source_confidence: float,
        target_confidence: float,
        t_next: int
    ) -> None:
        """Update track memory after successful assignment."""
        # Get or create track ID
        if source_id in self.node_to_track:
            track_id = self.node_to_track[source_id]
        else:
            # New track
            track_id = self.track_id_counter
            self.track_id_counter += 1
            
            # Initialize Kalman filter
            kalman = KalmanFilter3D(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                initial_velocity_std=self.initial_velocity_std
            )
            kalman.initialize(np.array(source_position, dtype=float))
            
            # Create track memory
            self.track_memory[track_id] = TrackMemory(
                track_id=track_id,
                position=np.array(source_position, dtype=float),
                velocity=np.zeros(3),
                age=0,
                confidence=source_confidence,
                kalman_filter=kalman,
                last_seen_time=t_next - 1
            )
        
        # Map target node to same track
        self.node_to_track[target_id] = track_id
        
        # Update track memory with new measurement
        track = self.track_memory[track_id]
        track.kalman_filter.update(np.array(target_position, dtype=float))
        
        # Update track state
        track.position = np.array(target_position, dtype=float)
        track.velocity = track.kalman_filter.get_velocity()
        track.age += 1
        track.confidence = target_confidence
        track.last_seen_time = t_next
        
        self.track_memory[track_id] = track
    
    def _initialize_new_tracks(
        self,
        t_next: int,
        detections: List[Tuple[int, int, int]],
        confidences: List[float],
        assigned_rows: List[int],
        assigned_cols: List[int],
        node_ids: Dict[int, Tuple[int, int, int, int]]
    ) -> None:
        """Initialize new tracks for unmatched detections."""
        assigned_cols_set = set(assigned_cols)
        
        for j, (z, y, x) in enumerate(detections):
            if j not in assigned_cols_set:
                # Unmatched detection: start new track
                target_id = self._get_node_id(t_next, j, node_ids)
                
                if target_id is not None:
                    track_id = self.track_id_counter
                    self.track_id_counter += 1
                    
                    # Initialize Kalman filter
                    kalman = KalmanFilter3D(
                        process_noise=self.process_noise,
                        measurement_noise=self.measurement_noise,
                        initial_velocity_std=self.initial_velocity_std
                    )
                    kalman.initialize(np.array([z, y, x], dtype=float))
                    
                    # Create track memory
                    self.track_memory[track_id] = TrackMemory(
                        track_id=track_id,
                        position=np.array([z, y, x], dtype=float),
                        velocity=np.zeros(3),
                        age=0,
                        confidence=confidences[j],
                        kalman_filter=kalman,
                        last_seen_time=t_next
                    )
                    
                    # Map node to track
                    self.node_to_track[target_id] = track_id
    
    def _get_node_id(
        self,
        t: int,
        idx: int,
        node_ids: Dict[int, Tuple[int, int, int, int]]
    ) -> Optional[int]:
        """Get node ID for a given timepoint and detection index."""
        return self._detection_to_node_id.get((t, idx))
    
    def _physical_distance_3d(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray
    ) -> float:
        """Calculate physical 3D distance."""
        dz = (pos2[0] - pos1[0]) * self.scale.z
        dy = (pos2[1] - pos1[1]) * self.scale.y
        dx = (pos2[2] - pos1[2]) * self.scale.x
        return float(np.sqrt(dz**2 + dy**2 + dx**2))
