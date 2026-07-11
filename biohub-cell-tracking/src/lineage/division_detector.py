"""Division detection and handling for cell tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

from data.models import TrackingGraph, CellNode, TrackingEdge, VoxelScale

logger = logging.getLogger(__name__)


@dataclass
class DivisionCandidate:
    """Candidate division event."""
    parent_id: int
    child1_id: int
    child2_id: int
    parent_time: int
    confidence: float
    distance_between_children: float
    angle: float


class DivisionDetector:
    """Detect and validate cell division events with motion analysis."""
    
    def __init__(
        self,
        scale: VoxelScale,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize division detector.
        
        Args:
            scale: VoxelScale for physical distance calculation
            config: Configuration dictionary
        """
        self.scale = scale
        
        if config is None:
            config = {}
        
        # Division parameters
        self.max_parent_child_distance_um = config.get("max_parent_child_distance_um", 5.0)
        self.max_child_child_distance_um = config.get("max_child_child_distance_um", 4.0)
        self.min_division_angle_deg = config.get("min_division_angle_deg", 30.0)
        self.max_division_angle_deg = config.get("max_division_angle_deg", 150.0)
        self.require_temporal_consistency = config.get("require_temporal_consistency", True)
        
        # Motion-based parameters
        self.use_motion_analysis = config.get("use_motion_analysis", True)
        self.max_parent_velocity_um_per_frame = config.get("max_parent_velocity_um_per_frame", 3.0)
        self.min_child_separation_velocity = config.get("min_child_separation_velocity", 0.5)
        
        # Intensity/volume conservation parameters
        self.use_intensity_conservation = config.get("use_intensity_conservation", True)
        self.intensity_tolerance_ratio = config.get("intensity_tolerance_ratio", 0.3)
        self.min_parent_intensity = config.get("min_parent_intensity", 0.1)
        
        logger.info(
            f"DivisionDetector initialized: max_parent_child={self.max_parent_child_distance_um}, "
            f"max_child_child={self.max_child_child_distance_um}, use_motion={self.use_motion_analysis}, "
            f"use_intensity={self.use_intensity_conservation}"
        )
    
    def detect_divisions(
        self,
        graph: TrackingGraph,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        volume: Optional[np.ndarray] = None,
    ) -> List[DivisionCandidate]:
        """Detect division events from tracking graph.
        
        Args:
            graph: Tracking graph with predicted tracks
            detections_by_time: Dict mapping time -> detections
            volume: Optional volume array for intensity analysis
        
        Returns:
            List of division candidates
        """
        candidates = []
        
        # Find nodes with two outgoing edges (potential divisions)
        for node_id, node in graph.nodes.items():
            outgoing = graph.outgoing(node_id)
            
            if len(outgoing) == 2:
                # This is a potential division
                child1_id = outgoing[0].target_id
                child2_id = outgoing[1].target_id
                
                if child1_id not in graph.nodes or child2_id not in graph.nodes:
                    continue
                
                child1 = graph.nodes[child1_id]
                child2 = graph.nodes[child2_id]
                
                # Validate division
                candidate = self._validate_division(
                    node, child1, child2, detections_by_time, volume
                )
                
                if candidate is not None:
                    candidates.append(candidate)
        
        logger.info(f"Detected {len(candidates)} division candidates")
        return candidates
    
    def _validate_division(
        self,
        parent: CellNode,
        child1: CellNode,
        child2: CellNode,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        volume: Optional[np.ndarray] = None,
    ) -> Optional[DivisionCandidate]:
        """Validate a potential division event with motion and intensity analysis.
        
        Args:
            parent: Parent cell node
            child1: First child cell node
            child2: Second child cell node
            detections_by_time: Detections by timepoint
            volume: Optional volume array for intensity analysis
        
        Returns:
            DivisionCandidate if valid, None otherwise
        """
        # Check temporal consistency
        if self.require_temporal_consistency:
            if child1.t != parent.t + 1 or child2.t != parent.t + 1:
                return None
        
        # Calculate distances
        parent_pos = np.array([parent.z, parent.y, parent.x])
        child1_pos = np.array([child1.z, child1.y, child1.x])
        child2_pos = np.array([child2.z, child2.y, child2.x])
        
        # Physical distances
        parent_child1_dist = self._physical_distance(parent_pos, child1_pos)
        parent_child2_dist = self._physical_distance(parent_pos, child2_pos)
        child_child_dist = self._physical_distance(child1_pos, child2_pos)
        
        # Check distance constraints
        if (parent_child1_dist > self.max_parent_child_distance_um or
            parent_child2_dist > self.max_parent_child_distance_um):
            return None
        
        if child_child_dist > self.max_child_child_distance_um:
            return None
        
        # Motion analysis: check parent velocity
        if self.use_motion_analysis and parent.t > 0:
            parent_velocity = self._calculate_parent_velocity(parent, graph=None)
            if parent_velocity > self.max_parent_velocity_um_per_frame:
                return None
        
        # Motion analysis: check child separation velocity
        if self.use_motion_analysis:
            separation_velocity = self._calculate_separation_velocity(
                parent_pos, child1_pos, child2_pos
            )
            if separation_velocity < self.min_child_separation_velocity:
                return None
        
        # Intensity conservation: check parent-child intensity relationship
        if self.use_intensity_conservation and volume is not None:
            if not self._validate_intensity_conservation(
                parent, child1, child2, volume
            ):
                return None
        
        # Calculate division angle
        vec1 = child1_pos - parent_pos
        vec2 = child2_pos - parent_pos
        angle = self._calculate_angle(vec1, vec2)
        
        # Check angle constraints
        if angle < self.min_division_angle_deg or angle > self.max_division_angle_deg:
            return None
        
        # Calculate confidence based on geometric consistency
        confidence = self._calculate_division_confidence(
            parent_child1_dist, parent_child2_dist, child_child_dist, angle
        )
        
        return DivisionCandidate(
            parent_id=parent.node_id,
            child1_id=child1.node_id,
            child2_id=child2.node_id,
            parent_time=parent.t,
            confidence=confidence,
            distance_between_children=child_child_dist,
            angle=angle,
        )
    
    def _calculate_parent_velocity(
        self,
        parent: CellNode,
        graph: Optional[TrackingGraph] = None,
    ) -> float:
        """Calculate parent cell velocity before division.
        
        Args:
            parent: Parent cell node
            graph: Tracking graph (to find parent's parent)
        
        Returns:
            Velocity in um/frame
        """
        if graph is None:
            return 0.0
        
        # Find parent's incoming edge (its parent)
        incoming = graph.incoming(parent.node_id)
        if not incoming:
            return 0.0
        
        grandparent_id = incoming[0].source_id
        if grandparent_id not in graph.nodes:
            return 0.0
        
        grandparent = graph.nodes[grandparent_id]
        
        # Calculate velocity
        parent_pos = np.array([parent.z, parent.y, parent.x])
        grandparent_pos = np.array([grandparent.z, grandparent.y, grandparent.x])
        
        velocity = self._physical_distance(grandparent_pos, parent_pos)
        return velocity
    
    def _calculate_separation_velocity(
        self,
        parent_pos: np.ndarray,
        child1_pos: np.ndarray,
        child2_pos: np.ndarray,
    ) -> float:
        """Calculate separation velocity of children after division.
        
        Args:
            parent_pos: Parent position
            child1_pos: First child position
            child2_pos: Second child position
        
        Returns:
            Separation velocity (um/frame)
        """
        # Calculate how far children have moved from parent
        child1_displacement = self._physical_distance(parent_pos, child1_pos)
        child2_displacement = self._physical_distance(parent_pos, child2_pos)
        
        # Average displacement as separation velocity
        return (child1_displacement + child2_displacement) / 2.0
    
    def _physical_distance(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
    ) -> float:
        """Calculate physical distance between two positions.
        
        Args:
            pos1: Position (z, y, x)
            pos2: Position (z, y, x)
        
        Returns:
            Physical distance in micrometers
        """
        dz = (pos2[0] - pos1[0]) * self.scale.z
        dy = (pos2[1] - pos1[1]) * self.scale.y
        dx = (pos2[2] - pos1[2]) * self.scale.x
        
        return np.sqrt(dz**2 + dy**2 + dx**2)
    
    def _calculate_angle(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Calculate angle between two vectors in degrees.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Angle in degrees
        """
        # Scale to physical coordinates
        vec1_scaled = np.array([vec1[0] * self.scale.z, vec1[1] * self.scale.y, vec1[2] * self.scale.x])
        vec2_scaled = np.array([vec2[0] * self.scale.z, vec2[1] * self.scale.y, vec2[2] * self.scale.x])
        
        # Calculate angle
        dot_product = np.dot(vec1_scaled, vec2_scaled)
        norm1 = np.linalg.norm(vec1_scaled)
        norm2 = np.linalg.norm(vec2_scaled)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _calculate_division_confidence(
        self,
        parent_child1_dist: float,
        parent_child2_dist: float,
        child_child_dist: float,
        angle: float,
    ) -> float:
        """Calculate confidence score for division.
        
        Args:
            parent_child1_dist: Distance from parent to child1
            parent_child2_dist: Distance from parent to child2
            child_child_dist: Distance between children
            angle: Division angle
        
        Returns:
            Confidence score (0-1)
        """
        # Symmetric distances are better
        distance_symmetry = 1.0 - abs(parent_child1_dist - parent_child2_dist) / (parent_child1_dist + parent_child2_dist + 1e-6)
        
        # Optimal angle around 90 degrees
        angle_score = 1.0 - abs(angle - 90.0) / 90.0
        
        # Children should be reasonably close
        distance_score = 1.0 - min(child_child_dist / self.max_child_child_distance_um, 1.0)
        
        # Combine scores
        confidence = 0.4 * distance_symmetry + 0.4 * angle_score + 0.2 * distance_score
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _validate_intensity_conservation(
        self,
        parent: CellNode,
        child1: CellNode,
        child2: CellNode,
        volume: np.ndarray,
    ) -> bool:
        """Validate intensity conservation for division.
        
        When a cell divides, the total intensity/volume should be approximately
        conserved (parent intensity ≈ child1 intensity + child2 intensity).
        
        Args:
            parent: Parent cell node
            child1: First child cell node
            child2: Second child cell node
            volume: Volume array to extract intensities from
        
        Returns:
            True if intensity conservation is satisfied
        """
        # Extract local intensities around each cell
        parent_intensity = self._extract_local_intensity(parent, volume)
        child1_intensity = self._extract_local_intensity(child1, volume)
        child2_intensity = self._extract_local_intensity(child2, volume)
        
        # Check minimum parent intensity
        if parent_intensity < self.min_parent_intensity:
            return False
        
        # Expected child intensity sum (parent should split into two)
        expected_child_sum = parent_intensity
        
        # Actual child intensity sum
        actual_child_sum = child1_intensity + child2_intensity
        
        # Check conservation within tolerance
        intensity_ratio = actual_child_sum / (expected_child_sum + 1e-6)
        tolerance_low = 1.0 - self.intensity_tolerance_ratio
        tolerance_high = 1.0 + self.intensity_tolerance_ratio
        
        return tolerance_low <= intensity_ratio <= tolerance_high
    
    def _extract_local_intensity(
        self,
        node: CellNode,
        volume: np.ndarray,
        radius: int = 3,
    ) -> float:
        """Extract local intensity around a cell position.
        
        Args:
            node: Cell node with position
            volume: Volume array (can be 3D or 4D)
            radius: Radius around position to sample
        
        Returns:
            Mean intensity in local region
        """
        t, z, y, x = node.t, node.z, node.y, node.x
        
        # Handle both 3D (Z, Y, X) and 4D (T, Z, Y, X) volumes
        if volume.ndim == 3:
            z_size, y_size, x_size = volume.shape
            # For 3D volume, ignore time dimension
            if z < z_size:
                z_start = max(0, z - radius)
                z_end = min(z_size, z + radius + 1)
                y_start = max(0, y - radius)
                y_end = min(y_size, y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(x_size, x + radius + 1)
                
                local_region = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                if local_region.size > 0:
                    return float(np.mean(local_region))
        elif volume.ndim == 4:
            t_size, z_size, y_size, x_size = volume.shape
            # Calculate slice bounds
            z_start = max(0, z - radius)
            z_end = min(z_size, z + radius + 1)
            y_start = max(0, y - radius)
            y_end = min(y_size, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(x_size, x + radius + 1)
            
            # Extract local region
            if t < t_size:
                local_region = volume[t, z_start:z_end, y_start:y_end, x_start:x_end]
                if local_region.size > 0:
                    return float(np.mean(local_region))
        
        return 0.0


class DivisionAwareTracker:
    """Tracker that explicitly handles division events."""
    
    def __init__(
        self,
        scale: VoxelScale,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize division-aware tracker.
        
        Args:
            scale: VoxelScale for physical distance calculation
            config: Configuration dictionary
        """
        self.scale = scale
        self.division_detector = DivisionDetector(scale, config)
        
        self.allow_one_to_two = config.get("allow_one_to_two", True) if config else True
        self.division_search_radius_um = config.get("division_search_radius_um", 3.0) if config else 3.0
        
        logger.info(
            f"DivisionAwareTracker initialized: allow_one_to_two={self.allow_one_to_two}"
        )
    
    def post_process_divisions(
        self,
        graph: TrackingGraph,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        volume: Optional[np.ndarray] = None,
    ) -> TrackingGraph:
        """Post-process tracking graph to add missing divisions.
        
        Args:
            graph: Tracking graph from basic tracker
            detections_by_time: Detections by timepoint
            volume: Optional volume array for intensity analysis
        
        Returns:
            Updated tracking graph with divisions
        """
        if not self.allow_one_to_two:
            return graph
        
        # First, detect existing divisions (nodes with 2 outgoing edges)
        candidates = self.division_detector.detect_divisions(graph, detections_by_time, volume)
        
        # Filter by confidence
        high_confidence_candidates = [
            c for c in candidates if c.confidence > 0.5
        ]
        
        logger.info(f"High-confidence divisions from existing edges: {len(high_confidence_candidates)}")
        
        # Find missing divisions (nodes with no outgoing edges that should divide)
        missing_divisions = self.find_missing_divisions(graph, detections_by_time, volume)
        logger.info(f"Potential missing divisions found: {len(missing_divisions)}")
        
        # Add missing division edges
        edges_added = 0
        for parent_id, child1_id, child2_id in missing_divisions:
            # Check if parent already has outgoing edges
            if len(graph.outgoing(parent_id)) == 0:
                # Add division edges
                graph.edges.append(TrackingEdge(source_id=parent_id, target_id=child1_id))
                graph.edges.append(TrackingEdge(source_id=parent_id, target_id=child2_id))
                edges_added += 2
        
        logger.info(f"Added {edges_added} division edges")
        
        return graph
    
    def find_missing_divisions(
        self,
        graph: TrackingGraph,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        volume: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int, int]]:
        """Find potential missing divisions (unassigned parents with nearby children).
        
        Args:
            graph: Tracking graph
            detections_by_time: Detections by timepoint
            volume: Optional volume array for intensity analysis
        
        Returns:
            List of (parent_id, child1_id, child2_id) tuples
        """
        missing_divisions = []
        
        # Find nodes with 0 or 1 outgoing edges (potential missed or partial divisions)
        for node_id, node in graph.nodes.items():
            outgoing = graph.outgoing(node_id)
            
            if len(outgoing) <= 1 and node.t + 1 in detections_by_time:
                # This node has 0 or 1 children - check if it should divide
                parent_pos = np.array([node.z, node.y, node.x])
                
                # Find nearby detections in next frame
                next_detections = detections_by_time[node.t + 1]
                nearby_children = []
                
                for child_idx, (z, y, x) in enumerate(next_detections):
                    child_pos = np.array([z, y, x])
                    dist = self._physical_distance(parent_pos, child_pos)
                    
                    if dist < self.division_search_radius_um:
                        # Find node ID for this detection
                        child_id = self._find_detection_node_id(
                            graph, node.t + 1, child_idx
                        )
                        if child_id is not None:
                            # Check if this child has no incoming edges (unassigned)
                            # or is the existing child (if parent has 1 outgoing)
                            child_incoming = graph.incoming(child_id)
                            is_existing_child = len(outgoing) == 1 and outgoing[0] == child_id
                            
                            if len(child_incoming) == 0 or is_existing_child:
                                nearby_children.append((child_id, dist, child_pos, is_existing_child))
                
                # If we found 2 nearby children (at least 1 unassigned), this might be a missed division
                if len(nearby_children) >= 2:
                    # Sort by distance and take closest 2
                    nearby_children.sort(key=lambda x: x[1])
                    child1_id, _, child1_pos, _ = nearby_children[0]
                    child2_id, _, child2_pos, _ = nearby_children[1]
                    
                    # Validate geometrically
                    if self._validate_potential_division(parent_pos, child1_pos, child2_pos):
                        # Optionally validate intensity conservation if volume is available
                        if volume is not None and self.division_detector.use_intensity_conservation:
                            if child1_id in graph.nodes and child2_id in graph.nodes:
                                child1_node = graph.nodes[child1_id]
                                child2_node = graph.nodes[child2_id]
                                if not self.division_detector._validate_intensity_conservation(
                                    node, child1_node, child2_node, volume
                                ):
                                    continue
                        missing_divisions.append((node_id, child1_id, child2_id))
        
        logger.info(f"Found {len(missing_divisions)} potential missing divisions")
        return missing_divisions
    
    def _physical_distance(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
    ) -> float:
        """Calculate physical distance."""
        dz = (pos2[0] - pos1[0]) * self.scale.z
        dy = (pos2[1] - pos1[1]) * self.scale.y
        dx = (pos2[2] - pos1[2]) * self.scale.x
        return np.sqrt(dz**2 + dy**2 + dx**2)
    
    def _find_detection_node_id(
        self,
        graph: TrackingGraph,
        timepoint: int,
        detection_idx: int,
    ) -> Optional[int]:
        """Find node ID for a detection at a given timepoint.
        
        Args:
            graph: Tracking graph
            timepoint: Timepoint
            detection_idx: Detection index at that timepoint
        
        Returns:
            Node ID or None
        """
        count = 0
        for node_id, node in graph.nodes.items():
            if node.t == timepoint:
                if count == detection_idx:
                    return node_id
                count += 1
        return None
    
    def _validate_potential_division(
        self,
        parent_pos: np.ndarray,
        child1_pos: np.ndarray,
        child2_pos: np.ndarray,
    ) -> bool:
        """Validate potential division geometrically.
        
        Args:
            parent_pos: Parent position
            child1_pos: First child position
            child2_pos: Second child position
        
        Returns:
            True if valid division geometry
        """
        # Calculate angle
        vec1 = child1_pos - parent_pos
        vec2 = child2_pos - parent_pos
        
        vec1_scaled = np.array([vec1[0] * self.scale.z, vec1[1] * self.scale.y, vec1[2] * self.scale.x])
        vec2_scaled = np.array([vec2[0] * self.scale.z, vec2[1] * self.scale.y, vec2[2] * self.scale.x])
        
        dot_product = np.dot(vec1_scaled, vec2_scaled)
        norm1 = np.linalg.norm(vec1_scaled)
        norm2 = np.linalg.norm(vec2_scaled)
        
        if norm1 == 0 or norm2 == 0:
            return False
        
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        # Angle should be reasonable (30-150 degrees)
        return 30.0 <= angle_deg <= 150.0
