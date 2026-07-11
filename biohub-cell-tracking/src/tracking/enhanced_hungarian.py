"""Enhanced Hungarian tracker with multi-feature cost function."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.dataset import EmbryoDataset
from data.models import TrackingEdge, VoxelScale, EmbryoSample
from tracking.base import Tracker, TrackingResult
from tracking.enhanced_cost import (
    EnhancedCostFunction,
    AdaptiveCostFunction,
    DetectionFeatures,
    CostFeatures,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State of a tracked cell with memory."""
    node_id: int
    position: np.ndarray  # (z, y, x)
    velocity: np.ndarray  # (vz, vy, vx)
    age: int
    confidence: float
    appearance: Optional[np.ndarray] = None  # Appearance embedding


class EnhancedHungarianTracker(Tracker):
    """Hungarian tracker with enhanced cost function using multiple features."""
    
    def __init__(
        self,
        config: dict,
        scale: VoxelScale,
        sample: Optional[EmbryoSample] = None,
    ) -> None:
        """Initialize enhanced Hungarian tracker.
        
        Args:
            config: Configuration dictionary
            scale: VoxelScale for physical distance calculation
            sample: Optional EmbryoSample for volume access (needed for features)
        """
        super().__init__(config, scale)
        
        self.sample = sample
        self.use_adaptive = config.get("use_adaptive", False)
        
        # Initialize cost function
        cost_config = config.get("cost_function", {})
        self.cost_function: EnhancedCostFunction = (
            AdaptiveCostFunction(scale, cost_config)
            if self.use_adaptive
            else EnhancedCostFunction(scale, cost_config)
        )
        
        # Feature extraction parameters
        self.neighborhood_radius = config.get("neighborhood_radius", 5)
        
        # Motion statistics for adaptive tracking
        self.displacements: List[float] = []
        
        # Track memory for persistent track IDs
        self.track_memory: Dict[int, TrackState] = {}
        self.node_to_track: Dict[int, int] = {}
        self.track_id_counter = 0
        
        # Motion gating parameters
        self.motion_radius_um = config.get("motion_radius_um", 8.0)
        
        logger.info(
            f"EnhancedHungarianTracker initialized: adaptive={self.use_adaptive}, "
            f"neighborhood_radius={self.neighborhood_radius}, track_memory_enabled=True"
        )
    
    def track(
        self,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        confidences_by_time: Optional[Dict[int, List[float]]] = None,
        volumes_by_time: Optional[Dict[int, np.ndarray]] = None,
    ) -> TrackingResult:
        """Link detections across time using enhanced Hungarian matching.
        
        Args:
            detections_by_time: Dict mapping time -> list of (z, y, x) detections
            confidences_by_time: Optional dict mapping time -> confidence scores
            volumes_by_time: Optional dict mapping time -> volume arrays
        
        Returns:
            TrackingResult with edges and node IDs
        """
        logger.debug(f"Running enhanced Hungarian tracking on {len(detections_by_time)} timepoints")
        
        # Assign node IDs (sequential)
        node_ids = {}
        self._detection_to_node_id = {}
        node_id_counter = 0
        
        for t in sorted(detections_by_time.keys()):
            for idx, detection in enumerate(detections_by_time[t]):
                node_id = node_id_counter
                node_ids[node_id] = (t, *detection)
                self._detection_to_node_id[(t, idx)] = node_id
                node_id_counter += 1
        
        # Build edges using enhanced Hungarian matching
        edges = []
        sorted_times = sorted(detections_by_time.keys())
        
        # Extract features for all timepoints if volumes provided
        import time
        t_features_start = time.perf_counter()
        features_by_time = {}
        if volumes_by_time is not None and confidences_by_time is not None:
            for t in sorted_times:
                if t in detections_by_time and t in volumes_by_time:
                    features_by_time[t] = self.cost_function.extract_detection_features(
                        volumes_by_time[t],
                        detections_by_time[t],
                        confidences_by_time.get(t, [1.0] * len(detections_by_time[t])),
                        t,
                        self.neighborhood_radius,
                    )
        t_features_end = time.perf_counter()
        logger.info(f"Feature extraction completed: {t_features_end - t_features_start:.2f}s for {len(features_by_time)} timepoints")
        
        for i in range(len(sorted_times) - 1):
            t_current = sorted_times[i]
            t_next = sorted_times[i + 1]
            
            # Progress logging
            if i % 10 == 0 or i == len(sorted_times) - 2:
                logger.info(f"Processing frame {t_current} -> {t_next} ({i+1}/{len(sorted_times)-1})")
            
            # Timing for profiling
            import time
            t_start = time.perf_counter()
            
            if t_current not in detections_by_time or t_next not in detections_by_time:
                continue
            
            current_detections = detections_by_time[t_current]
            next_detections = detections_by_time[t_next]
            
            t_before_cost = time.perf_counter()
            
            # Build cost matrix with track memory enabled
            if t_current in features_by_time and t_next in features_by_time:
                cost_matrix = self._build_cost_matrix_with_memory(
                    features_by_time[t_current],
                    features_by_time[t_next],
                    t_current,
                    t_next,
                )
            else:
                # Fall back to simple distance-based cost
                cost_matrix = self._build_simple_cost_matrix(
                    t_current, current_detections,
                    t_next, next_detections
                )
            
            t_after_cost = time.perf_counter()
            
            # Solve assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            t_after_hungarian = time.perf_counter()
            
            # Create edges for valid assignments and update track memory
            for row, col in zip(row_indices, col_indices):
                cost = cost_matrix[row, col]
                max_cost = self.cost_function.get_adaptive_max_distance() if self.use_adaptive else self.cost_function.max_distance_um
                
                if cost < max_cost:
                    source_id = self._get_node_id(t_current, row, node_ids)
                    target_id = self._get_node_id(t_next, col, node_ids)
                    
                    if source_id is not None and target_id is not None:
                        edges.append(TrackingEdge(source_id=source_id, target_id=target_id))
                        
                        # Propagate persistent track IDs
                        if source_id in self.node_to_track:
                            track_id = self.node_to_track[source_id]
                        else:
                            # New track
                            track_id = self.track_id_counter
                            self.track_id_counter += 1
                            
                            # Initialize track state
                            if t_current in features_by_time:
                                feat = features_by_time[t_current][row]
                                position = np.array(feat.position)
                                self.track_memory[track_id] = TrackState(
                                    node_id=source_id,
                                    position=position,
                                    velocity=np.zeros(3),
                                    age=0,
                                    confidence=feat.confidence,
                                )
                        
                        # Map target node to same track ID
                        self.node_to_track[target_id] = track_id
                        
                        # Update track memory with new position and velocity
                        if track_id in self.track_memory and t_current in features_by_time and t_next in features_by_time:
                            current_state = self.track_memory[track_id]
                            new_position = np.array(next_detections[col])
                            new_velocity = new_position - current_state.position
                            
                            self.track_memory[track_id] = TrackState(
                                node_id=target_id,
                                position=new_position,
                                velocity=new_velocity,
                                age=current_state.age + 1,
                                confidence=current_state.confidence,
                                appearance=current_state.appearance
                            )
                        
                        # Record displacement for adaptive learning
                        if self.use_adaptive and t_current in features_by_time and t_next in features_by_time:
                            feat_current = features_by_time[t_current][row]
                            feat_next = features_by_time[t_next][col]
                            cost_features = self.cost_function.compute_cost_features(feat_current, feat_next)
                            self.displacements.append(cost_features.distance_um)
            
            t_after_edges = time.perf_counter()
            
            # Update motion statistics if adaptive
            if self.use_adaptive and len(self.displacements) > 10:
                self.cost_function.update_motion_statistics(self.displacements[-100:])
            
            t_after_motion = time.perf_counter()
            
            # Detect divisions (one-to-two assignments) if enabled
            if self.config.get("enable_division", False):
                division_edges = self._detect_divisions(
                    t_current, t_next,
                    current_detections, next_detections,
                    features_by_time,
                    node_ids, edges
                )
                if division_edges:
                    logger.info(f"Detected {len(division_edges)//2} divisions at t={t_current}->{t_next}")
                edges.extend(division_edges)
            
            t_end = time.perf_counter()
            
            # Log timing breakdown every 10 frames
            if i % 10 == 0:
                logger.info(
                    f"Frame {t_current}->{t_next} timing: "
                    f"cost_matrix={t_after_cost-t_before_cost:.2f}s, "
                    f"hungarian={t_after_hungarian-t_after_cost:.2f}s, "
                    f"edges={t_after_edges-t_after_hungarian:.2f}s, "
                    f"motion={t_after_motion-t_after_edges:.2f}s, "
                    f"division={t_end-t_after_motion:.2f}s, "
                    f"total={t_end-t_start:.2f}s"
                )
        
        logger.info(f"Enhanced Hungarian tracking: {len(edges)} edges created")
        
        # Clear mapping cache after tracking is done
        self._detection_to_node_id = {}
        
        return TrackingResult(
            edges=edges,
            node_ids=node_ids,
            metadata={
                "method": "enhanced_hungarian",
                "adaptive": self.use_adaptive,
                "num_timepoints": len(detections_by_time),
                "final_max_distance": self.cost_function.get_adaptive_max_distance() if self.use_adaptive else self.cost_function.max_distance_um,
            }
        )
    
    def _build_enhanced_cost_matrix(
        self,
        features_t: List[DetectionFeatures],
        features_t1: List[DetectionFeatures],
    ) -> np.ndarray:
        """Build cost matrix using enhanced cost function (track memory disabled).
        
        Args:
            features_t: Features at time t
            features_t1: Features at time t+1
        
        Returns:
            Cost matrix (N_t × N_t+1)
        """
        return self.cost_function.build_cost_matrix(features_t, features_t1)
    
    def _build_cost_matrix_with_memory(
        self,
        features_t: List[DetectionFeatures],
        features_t1: List[DetectionFeatures],
        t_current: int,
        t_next: int,
    ) -> np.ndarray:
        """Build cost matrix using track memory and velocity prediction.
        
        Args:
            features_t: Features at time t
            features_t1: Features at time t+1
            t_current: Current timepoint
            t_next: Next timepoint
        
        Returns:
            Cost matrix (N_t × N_t+1)
        """
        base_cost_matrix = self.cost_function.build_cost_matrix(features_t, features_t1)
        
        # Apply track memory modifications
        for i, feat in enumerate(features_t):
            node_id = self._get_node_id_from_detection(t_current, i)
            
            # Look up track ID from node ID
            track_id = self.node_to_track.get(node_id)
            
            if track_id is not None and track_id in self.track_memory:
                track_state = self.track_memory[track_id]
                
                # 1. Velocity prediction: predict where this cell should be at t_next
                predicted_position = track_state.position + track_state.velocity
                
                # 2. Calculate distance from predicted position to each candidate
                for j, feat_next in enumerate(features_t1):
                    candidate_position = np.array(feat_next.position)
                    
                    # Physical distance from prediction
                    pred_distance = self._physical_distance_3d(
                        predicted_position, candidate_position
                    )
                    
                    # Motion gating: reject candidates outside motion radius
                    if pred_distance > self.motion_radius_um:
                        base_cost_matrix[i, j] = 999.0  # Reject impossible matches
                    else:
                        # Bonus for close predictions (within motion radius)
                        prediction_bonus = -1.0 * (1.0 - pred_distance / self.motion_radius_um)
                        base_cost_matrix[i, j] += prediction_bonus
        
        return base_cost_matrix
    
    def _physical_distance_3d(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
    ) -> float:
        """Calculate physical 3D distance between two positions.
        
        Args:
            pos1: (z, y, x) position
            pos2: (z, y, x) position
        
        Returns:
            Physical distance in micrometers
        """
        dz = (pos2[0] - pos1[0]) * self.scale.z
        dy = (pos2[1] - pos1[1]) * self.scale.y
        dx = (pos2[2] - pos1[2]) * self.scale.x
        
        return float(np.sqrt(dz**2 + dy**2 + dx**2))
    
    def _get_node_id_from_detection(self, timepoint: int, detection_idx: int) -> Optional[int]:
        """Get node ID for a detection at a given timepoint and index.
        
        Args:
            timepoint: Timepoint
            detection_idx: Detection index at that timepoint
        
        Returns:
            Node ID or None
        """
        return self._detection_to_node_id.get((timepoint, detection_idx))
    
    def _build_simple_cost_matrix(
        self,
        t_current: int,
        current_detections: List[Tuple[int, int, int]],
        t_next: int,
        next_detections: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Build simple distance-based cost matrix (fallback).
        
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
        
        max_distance = self.cost_function.get_adaptive_max_distance() if self.use_adaptive else self.cost_function.max_distance_um
        cost_matrix = np.full((n_current, n_next), max_distance * 10.0, dtype=np.float64)
        
        for i, (z_curr, y_curr, x_curr) in enumerate(current_detections):
            for j, (z_next, y_next, x_next) in enumerate(next_detections):
                pos1 = (t_current, z_curr, y_curr, x_curr)
                pos2 = (t_next, z_next, y_next, x_next)
                distance = self.physical_distance(pos1, pos2)
                
                if distance <= max_distance:
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
    
    def _detect_divisions(
        self,
        t_current: int,
        t_next: int,
        current_detections: List[Tuple[int, int, int]],
        next_detections: List[Tuple[int, int, int]],
        features_by_time: Dict[int, List[DetectionFeatures]],
        node_ids: Dict[int, Tuple[int, int, int, int]],
        existing_edges: List[TrackingEdge],
    ) -> List[TrackingEdge]:
        """Detect division events (one parent -> two children) with multi-feature scoring.
        
        Args:
            t_current: Current timepoint
            t_next: Next timepoint
            current_detections: Detections at t_current
            next_detections: Detections at t_next
            features_by_time: Features for each timepoint
            node_ids: Mapping of node_id -> (t, z, y, x)
            existing_edges: Already created edges (to avoid conflicts)
        
        Returns:
            List of division edges (parent -> child1, parent -> child2)
        """
        division_edges = []
        
        if t_current not in features_by_time or t_next not in features_by_time:
            return division_edges
        
        current_features = features_by_time[t_current]
        next_features = features_by_time[t_next]
        
        # Get division config
        div_config = self.config.get("division", {})
        max_parent_child_um = div_config.get("max_parent_child_distance_um", 5.0)
        max_child_child_um = div_config.get("max_child_child_distance_um", 4.0)
        min_angle_deg = div_config.get("min_division_angle_deg", 30.0)
        max_angle_deg = div_config.get("max_division_angle_deg", 150.0)
        
        # Build set of existing targets to avoid conflicts
        existing_targets = {edge.target_id for edge in existing_edges}
        existing_sources = {edge.source_id for edge in existing_edges}
        
        # For each potential parent (must have 0 outgoing edges from Hungarian)
        for parent_idx, parent_feat in enumerate(current_features):
            parent_id = self._get_node_id(t_current, parent_idx, node_ids)
            if parent_id is None:
                continue
            
            # Parent must not have any outgoing edges from Hungarian matching
            if parent_id in existing_sources:
                continue  # Skip - parent already has an outgoing edge
            
            parent_pos = np.array(parent_feat.position)
            
            # Find nearby children candidates
            child_candidates = []
            for child_idx, child_feat in enumerate(next_features):
                child_pos = np.array(child_feat.position)
                dist_um = np.linalg.norm((child_pos - parent_pos) * np.array([self.scale.z, self.scale.y, self.scale.x]))
                
                if dist_um < max_parent_child_um:
                    child_id = self._get_node_id(t_next, child_idx, node_ids)
                    if child_id is not None and child_id not in existing_targets:
                        child_candidates.append((child_idx, child_pos, dist_um, child_feat))
            
            # If we have at least 2 candidates, evaluate division pairs
            if len(child_candidates) >= 2:
                best_pair = None
                best_score = -1
                
                # Evaluate all pairs of children
                for i in range(len(child_candidates)):
                    for j in range(i + 1, len(child_candidates)):
                        child1_idx, child1_pos, dist1, feat1 = child_candidates[i]
                        child2_idx, child2_pos, dist2, feat2 = child_candidates[j]
                        
                        # Compute child-child distance
                        child_child_dist_um = np.linalg.norm(
                            (child2_pos - child1_pos) * np.array([self.scale.z, self.scale.y, self.scale.x])
                        )
                        
                        if child_child_dist_um > max_child_child_um:
                            continue
                        
                        # Compute division angle
                        v1 = child1_pos - parent_pos
                        v2 = child2_pos - parent_pos
                        angle_rad = np.arccos(
                            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                        )
                        angle_deg = np.degrees(angle_rad)
                        
                        if angle_deg < min_angle_deg or angle_deg > max_angle_deg:
                            continue
                        
                        # Multi-feature scoring
                        score = self._compute_division_score(
                            parent_feat, feat1, feat2,
                            dist1, dist2, child_child_dist_um, angle_deg
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_pair = (child1_idx, child2_idx)
                
                # If best pair passes threshold, create division edges
                if best_pair is not None and best_score > 0.3:  # Lowered threshold from 0.5
                    child1_idx, child2_idx = best_pair
                    parent_id = self._get_node_id(t_current, parent_idx, node_ids)
                    child1_id = self._get_node_id(t_next, child1_idx, node_ids)
                    child2_id = self._get_node_id(t_next, child2_idx, node_ids)
                    
                    if parent_id is not None and child1_id is not None and child2_id is not None:
                        logger.debug(
                            f"Division detected: parent={parent_id}, children=({child1_id}, {child2_id}), "
                            f"score={best_score:.3f}, angle={angle_deg:.1f}°"
                        )
                        division_edges.append(TrackingEdge(source_id=parent_id, target_id=child1_id))
                        division_edges.append(TrackingEdge(source_id=parent_id, target_id=child2_id))
                        
                        # Mark as assigned to avoid conflicts
                        existing_targets.add(child1_id)
                        existing_targets.add(child2_id)
        
        return division_edges
    
    def _compute_division_score(
        self,
        parent_feat: DetectionFeatures,
        child1_feat: DetectionFeatures,
        child2_feat: DetectionFeatures,
        dist1_um: float,
        dist2_um: float,
        child_child_dist_um: float,
        angle_deg: float,
    ) -> float:
        """Compute division score using multiple biological features.
        
        Args:
            parent_feat: Parent detection features
            child1_feat: First child features
            child2_feat: Second child features
            dist1_um: Distance from parent to child1
            dist2_um: Distance from parent to child2
            child_child_dist_um: Distance between children
            angle_deg: Division angle in degrees
        
        Returns:
            Division score (higher is better, range [0, 1])
        """
        # Distance score (closer is better)
        max_dist = 5.0
        dist_score = 1.0 - (dist1_um + dist2_um) / (2 * max_dist)
        dist_score = max(0.0, dist_score)
        
        # Child-child separation score (moderate separation is good)
        ideal_separation = 2.0
        separation_score = 1.0 - abs(child_child_dist_um - ideal_separation) / ideal_separation
        separation_score = max(0.0, separation_score)
        
        # Angle score (ideal around 90 degrees)
        ideal_angle = 90.0
        angle_score = 1.0 - abs(angle_deg - ideal_angle) / 60.0
        angle_score = max(0.0, angle_score)
        
        # Intensity conservation (parent intensity ≈ child1 + child2)
        parent_intensity = parent_feat.intensity
        child_sum = child1_feat.intensity + child2_feat.intensity
        if parent_intensity > 0:
            intensity_ratio = child_sum / parent_intensity
            intensity_score = 1.0 - abs(intensity_ratio - 1.0) / 0.5  # Allow 50% deviation
            intensity_score = max(0.0, min(1.0, intensity_score))
        else:
            intensity_score = 0.5
        
        # Size conservation (parent size ≈ child1 + child2)
        parent_size = parent_feat.size
        child_size_sum = child1_feat.size + child2_feat.size
        if parent_size > 0:
            size_ratio = child_size_sum / parent_size
            size_score = 1.0 - abs(size_ratio - 1.0) / 0.5
            size_score = max(0.0, min(1.0, size_score))
        else:
            size_score = 0.5
        
        # Daughter symmetry (similar intensities and sizes)
        intensity_similarity = 1.0 - abs(child1_feat.intensity - child2_feat.intensity) / (
            child1_feat.intensity + child2_feat.intensity + 1e-6
        )
        size_similarity = 1.0 - abs(child1_feat.size - child2_feat.size) / (
            child1_feat.size + child2_feat.size + 1e-6
        )
        symmetry_score = (intensity_similarity + size_similarity) / 2.0
        
        # Weighted combination
        total_score = (
            0.3 * dist_score +
            0.2 * separation_score +
            0.2 * angle_score +
            0.15 * intensity_score +
            0.1 * size_score +
            0.05 * symmetry_score
        )
        
        return total_score


class EnhancedHungarianTrackerWithSample(EnhancedHungarianTracker):
    """Enhanced Hungarian tracker that automatically loads volumes from sample."""
    
    def __init__(
        self,
        config: dict,
        scale: VoxelScale,
        sample: EmbryoSample,
    ) -> None:
        """Initialize enhanced Hungarian tracker with sample.
        
        Args:
            config: Configuration dictionary
            scale: VoxelScale for physical distance calculation
            sample: EmbryoSample for volume access
        """
        super().__init__(config, scale, sample)
        
        self.sample = sample
        self.dataset = EmbryoDataset(sample, normalize=True, quantile_low=0.01, quantile_high=0.99)
        
        logger.info("EnhancedHungarianTrackerWithSample initialized with automatic volume loading")
    
    def track(
        self,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        confidences_by_time: Optional[Dict[int, List[float]]] = None,
        volumes_by_time: Optional[Dict[int, np.ndarray]] = None,
    ) -> TrackingResult:
        """Link detections with automatic volume loading.
        
        Args:
            detections_by_time: Dict mapping time -> list of (z, y, x) detections
            confidences_by_time: Optional dict mapping time -> confidence scores
            volumes_by_time: Optional dict mapping time -> volume arrays
        
        Returns:
            TrackingResult with edges and node IDs
        """
        # Load volumes for all timepoints if not provided
        if volumes_by_time is None:
            volumes_by_time = {}
            for t in detections_by_time.keys():
                volumes_by_time[t] = self.dataset.get_timepoint(t)
        
        return super().track(detections_by_time, confidences_by_time, volumes_by_time)
