"""Gap closing and track continuity for robust cell tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.models import TrackingGraph, CellNode, TrackingEdge, VoxelScale

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    """A gap in a track (missing link between consecutive timepoints)."""
    track_id: int
    start_node_id: int
    end_node_id: int
    gap_length: int  # Number of missing timepoints
    start_position: Tuple[int, int, int]
    end_position: Tuple[int, int, int]
    displacement_um: float
    start_time: int  # Actual frame index of gap start
    end_time: int  # Actual frame index of gap end


class GapCloser:
    """Close gaps in tracks using motion interpolation and linking."""
    
    def __init__(
        self,
        scale: VoxelScale,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize gap closer.
        
        Args:
            scale: VoxelScale for physical distance calculation
            config: Configuration dictionary
        """
        self.scale = scale
        
        if config is None:
            config = {}
        
        self.max_gap_length = config.get("max_gap_length", 3)
        self.max_gap_distance_um = config.get("max_gap_distance_um", 15.0)
        self.max_gap_velocity_um_per_frame = config.get("max_gap_velocity_um_per_frame", 5.0)
        self.use_linear_interpolation = config.get("use_linear_interpolation", True)
        
        logger.info(
            f"GapCloser initialized: max_gap_length={self.max_gap_length}, "
            f"max_gap_distance={self.max_gap_distance_um}"
        )
    
    def find_gaps(
        self,
        graph: TrackingGraph,
    ) -> List[Gap]:
        """Find gaps in tracking graph, including cross-track gaps.
        
        Args:
            graph: Tracking graph
        
        Returns:
            List of gaps
        """
        gaps = []
        
        # Build tracks (connected components)
        tracks = self._extract_tracks(graph)
        
        # 1. Find intra-track gaps (within same track)
        for track_id, nodes in tracks.items():
            # Sort nodes by time
            sorted_nodes = sorted(nodes, key=lambda n: n.t)
            
            # Find gaps
            for i in range(len(sorted_nodes) - 1):
                current = sorted_nodes[i]
                next_node = sorted_nodes[i + 1]
                
                gap_length = next_node.t - current.t
                
                if gap_length > 1 and gap_length <= self.max_gap_length:
                    # Check if there's already an edge
                    has_edge = any(
                        edge.source_id == current.node_id and edge.target_id == next_node.node_id
                        for edge in graph.edges
                    )
                    
                    if not has_edge:
                        # Calculate displacement
                        displacement_um = self._calculate_distance(current, next_node)
                        
                        # Check if gap is reasonable
                        if displacement_um <= self.max_gap_distance_um:
                            gaps.append(Gap(
                                track_id=track_id,
                                start_node_id=current.node_id,
                                end_node_id=next_node.node_id,
                                gap_length=gap_length,
                                start_position=(current.z, current.y, current.x),
                                end_position=(next_node.z, next_node.y, next_node.x),
                                displacement_um=displacement_um,
                                start_time=current.t,
                                end_time=next_node.t,
                            ))
        
        # 2. Find cross-track gaps (connect different track fragments)
        # This is the key fix for reconnecting fragmented tracks
        track_endpoints = []
        for track_id, nodes in tracks.items():
            sorted_nodes = sorted(nodes, key=lambda n: n.t)
            if len(sorted_nodes) > 0:
                track_endpoints.append({
                    'track_id': track_id,
                    'start_node': sorted_nodes[0],
                    'end_node': sorted_nodes[-1],
                })
        
        # Compare all pairs of track endpoints using Hungarian matching
        if len(track_endpoints) > 1:
            from scipy.optimize import linear_sum_assignment
            
            # Build cost matrix for track endpoint connections
            n_tracks = len(track_endpoints)
            cost_matrix = np.full((n_tracks, n_tracks), 999.0)
            
            for i, track_a in enumerate(track_endpoints):
                for j, track_b in enumerate(track_endpoints):
                    if i == j:
                        continue
                    
                    end_a = track_a['end_node']
                    start_b = track_b['start_node']
                    
                    time_gap = start_b.t - end_a.t
                    
                    if time_gap > 0 and time_gap <= self.max_gap_length:
                        displacement_um = self._calculate_distance(end_a, start_b)
                        
                        if displacement_um <= self.max_gap_distance_um:
                            # Check velocity constraint
                            velocity = displacement_um / time_gap
                            if velocity <= self.max_gap_velocity_um_per_frame:
                                cost_matrix[i, j] = displacement_um
            
            # Solve assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Add gaps for valid assignments
            for row, col in zip(row_indices, col_indices):
                cost = cost_matrix[row, col]
                if cost < 999.0:
                    track_a = track_endpoints[row]
                    track_b = track_endpoints[col]
                    
                    gaps.append(Gap(
                        track_id=track_a['track_id'],
                        start_node_id=track_a['end_node'].node_id,
                        end_node_id=track_b['start_node'].node_id,
                        gap_length=track_b['start_node'].t - track_a['end_node'].t,
                        start_position=(track_a['end_node'].z, track_a['end_node'].y, track_a['end_node'].x),
                        end_position=(track_b['start_node'].z, track_b['start_node'].y, track_b['start_node'].x),
                        displacement_um=cost,
                        start_time=track_a['end_node'].t,
                        end_time=track_b['start_node'].t,
                    ))
        
        logger.info(f"Found {len(gaps)} gaps (including cross-track)")
        return gaps
    
    def _extract_tracks(
        self,
        graph: TrackingGraph,
        max_nodes: int = 50000,
    ) -> Dict[int, List[CellNode]]:
        """Extract tracks (connected components) from graph.
        
        Args:
            graph: Tracking graph
            max_nodes: Maximum nodes to process (skip if larger)
        
        Returns:
            Dict mapping track_id to list of nodes
        """
        # Skip gap closing for very large graphs to avoid timeout
        if len(graph.nodes) > max_nodes:
            logger.warning(
                f"Graph has {len(graph.nodes)} nodes > {max_nodes}, "
                "skipping gap closing to avoid timeout"
            )
            return {}
        
        tracks = {}
        visited = set()
        track_id = 0
        
        for node_id in graph.nodes:
            if node_id not in visited:
                # Get connected component
                component = graph.connected_component(node_id)
                
                # Track nodes
                track_nodes = [graph.nodes[nid] for nid in component]
                tracks[track_id] = track_nodes
                
                # Mark as visited
                visited.update(component)
                track_id += 1
        
        return tracks
    
    def _calculate_distance(
        self,
        node1: CellNode,
        node2: CellNode,
    ) -> float:
        """Calculate physical distance between two nodes.
        
        Args:
            node1: First cell node
            node2: Second cell node
        
        Returns:
            Physical distance in micrometers
        """
        dz = (node2.z - node1.z) * self.scale.z
        dy = (node2.y - node1.y) * self.scale.y
        dx = (node2.x - node1.x) * self.scale.x
        
        return np.sqrt(dz**2 + dy**2 + dx**2)
    
    def close_gaps(
        self,
        graph: TrackingGraph,
        gaps: List[Gap],
        detections_by_time: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    ) -> TrackingGraph:
        """Close gaps in tracking graph.
        
        Args:
            graph: Tracking graph
            gaps: List of gaps to close
            detections_by_time: Optional detections for gap timepoints
        
        Returns:
            Updated tracking graph with closed gaps
        """
        if not gaps:
            return graph
        
        # Create new edges list
        new_edges = list(graph.edges)
        nodes_closed = 0
        
        for gap in gaps:
            # Check if gap can be closed
            if self._can_close_gap(gap):
                # Add direct edge
                new_edges.append(TrackingEdge(
                    source_id=gap.start_node_id,
                    target_id=gap.end_node_id,
                ))
                nodes_closed += gap.gap_length - 1
        
        logger.info(f"Closed {len(gaps)} gaps, {nodes_closed} missing nodes bridged")
        
        return TrackingGraph(
            nodes=graph.nodes,
            edges=new_edges,
            metadata=graph.metadata,
        )
    
    def _can_close_gap(
        self,
        gap: Gap,
    ) -> bool:
        """Check if a gap can be closed.
        
        Args:
            gap: Gap to check
        
        Returns:
            True if gap can be closed
        """
        # Check velocity constraint
        velocity = gap.displacement_um / gap.gap_length
        if velocity > self.max_gap_velocity_um_per_frame:
            return False
        
        return True
    
    def interpolate_gap_nodes(
        self,
        gap: Gap,
    ) -> List[Tuple[int, int, int, int]]:
        """Interpolate missing nodes for a gap.
        
        Args:
            gap: Gap to interpolate
        
        Returns:
            List of (t, z, y, x) for interpolated nodes
        """
        if not self.use_linear_interpolation:
            return []
        
        interpolated = []
        
        start_z, start_y, start_x = gap.start_position
        end_z, end_y, end_x = gap.end_position
        
        # Linear interpolation using actual frame indices
        for i in range(1, gap.gap_length):
            alpha = i / gap.gap_length
            z = int(start_z + alpha * (end_z - start_z))
            y = int(start_y + alpha * (end_y - start_y))
            x = int(start_x + alpha * (end_x - start_x))
            t = gap.start_time + i  # Correct: use actual start time
            
            interpolated.append((t, z, y, x))
        
        return interpolated


class TrackContinuityOptimizer:
    """Optimize track continuity by resolving conflicts and improving linking."""
    
    def __init__(
        self,
        scale: VoxelScale,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize track continuity optimizer.
        
        Args:
            scale: VoxelScale for physical distance calculation
            config: Configuration dictionary
        """
        self.scale = scale
        
        if config is None:
            config = {}
        
        self.resolve_conflicts = config.get("resolve_conflicts", True)
        self.merge_short_tracks = config.get("merge_short_tracks", True)
        self.min_track_length = config.get("min_track_length", 3)
        
        logger.info("TrackContinuityOptimizer initialized")
    
    def optimize(
        self,
        graph: TrackingGraph,
    ) -> TrackingGraph:
        """Optimize track continuity.
        
        Args:
            graph: Tracking graph
        
        Returns:
            Optimized tracking graph
        """
        # Resolve conflicts (nodes with multiple incoming/outgoing edges)
        if self.resolve_conflicts:
            graph = self._resolve_conflicts(graph)
        
        # Merge short tracks
        if self.merge_short_tracks:
            graph = self._merge_short_tracks(graph)
        
        return graph
    
    def _resolve_conflicts(
        self,
        graph: TrackingGraph,
    ) -> TrackingGraph:
        """Resolve linking conflicts.
        
        Args:
            graph: Tracking graph
        
        Returns:
            Graph with resolved conflicts
        """
        # Find nodes with multiple incoming edges (conflicts)
        conflicts = []
        
        for node_id, node in graph.nodes.items():
            incoming = graph.incoming(node_id)
            if len(incoming) > 1:
                conflicts.append((node_id, incoming))
        
        logger.info(f"Found {len(conflicts)} incoming edge conflicts")
        
        # For each conflict, keep the best edge based on distance
        new_edges = list(graph.edges)
        
        for node_id, incoming_edges in conflicts:
            # Calculate distance for each incoming edge
            edge_distances = []
            for source_id in incoming_edges:
                source = graph.nodes.get(source_id)
                target = graph.nodes.get(node_id)
                if source and target:
                    distance = self._calculate_distance(source, target)
                    edge_distances.append((source_id, distance))
            
            # Sort by distance and keep the best
            if edge_distances:
                edge_distances.sort(key=lambda x: x[1])
                best_source_id = edge_distances[0][0]
                
                # Remove edges from other sources
                for source_id, _ in edge_distances[1:]:
                    # Find and remove the edge from this source to node_id
                    edge_to_remove = None
                    for edge in new_edges:
                        if edge.source_id == source_id and edge.target_id == node_id:
                            edge_to_remove = edge
                            break
                    if edge_to_remove:
                        new_edges.remove(edge_to_remove)
        
        return TrackingGraph(
            nodes=graph.nodes,
            edges=new_edges,
            metadata=graph.metadata,
        )
    
    def _merge_short_tracks(
        self,
        graph: TrackingGraph,
    ) -> TrackingGraph:
        """Merge short tracks that are likely fragments.
        
        Args:
            graph: Tracking graph
        
        Returns:
            Graph with merged tracks
        """
        # Extract tracks
        tracks = self._extract_tracks(graph)
        
        # Find short tracks
        short_tracks = []
        for track_id, nodes in tracks.items():
            if len(nodes) < self.min_track_length:
                short_tracks.append(track_id)
        
        logger.info(f"Found {len(short_tracks)} short tracks")
        
        # Try to merge short tracks with nearby tracks
        # This is a simplified version - full implementation would use
        # spatial/temporal proximity and appearance similarity
        
        return graph
    
    def _extract_tracks(
        self,
        graph: TrackingGraph,
    ) -> Dict[int, List[CellNode]]:
        """Extract tracks from graph using O(V+E) single-pass BFS."""
        tracks = {}
        visited = set()
        track_id = 0
        
        for node_id in graph.nodes:
            if node_id not in visited:
                # Single BFS to extract this connected component
                component = []
                queue = [node_id]
                visited.add(node_id)
                
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    
                    # Add outgoing neighbors
                    for neighbor in graph.outgoing(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                    
                    # Add incoming neighbors
                    for neighbor in graph.incoming(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                track_nodes = [graph.nodes[nid] for nid in component]
                tracks[track_id] = track_nodes
                track_id += 1
        
        return tracks
    
    def _calculate_distance(
        self,
        node1: CellNode,
        node2: CellNode,
    ) -> float:
        """Calculate physical distance."""
        dz = (node2.z - node1.z) * self.scale.z
        dy = (node2.y - node1.y) * self.scale.y
        dx = (node2.x - node1.x) * self.scale.x
        return np.sqrt(dz**2 + dy**2 + dx**2)


class PostProcessingPipeline:
    """Complete post-processing pipeline for tracking results."""
    
    def __init__(
        self,
        scale: VoxelScale,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize post-processing pipeline.
        
        Args:
            scale: VoxelScale
            config: Configuration dictionary
        """
        self.scale = scale
        
        if config is None:
            config = {}
        
        # Initialize components
        gap_config = config.get("gap_closing", {})
        self.gap_closer = GapCloser(scale, gap_config)
        
        continuity_config = config.get("track_continuity", {})
        self.continuity_optimizer = TrackContinuityOptimizer(scale, continuity_config)
        
        logger.info("PostProcessingPipeline initialized")
    
    def process(
        self,
        graph: TrackingGraph,
        detections_by_time: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
        max_nodes: int = 50000,
    ) -> TrackingGraph:
        """Apply complete post-processing pipeline.
        
        Args:
            graph: Tracking graph
            detections_by_time: Optional detections for gap closing
            max_nodes: Maximum nodes before skipping post-processing
        
        Returns:
            Processed tracking graph
        """
        # Skip expensive post-processing for very large graphs
        if len(graph.nodes) > max_nodes:
            logger.warning(
                f"Large graph detected ({len(graph.nodes)} nodes > {max_nodes}). "
                "Skipping expensive post-processing (gap closing, track merging)."
            )
            return graph
        
        # Find and close gaps
        gaps = self.gap_closer.find_gaps(graph)
        graph = self.gap_closer.close_gaps(graph, gaps, detections_by_time)
        
        # Optimize track continuity
        graph = self.continuity_optimizer.optimize(graph)
        
        logger.info("Post-processing complete")
        
        return graph
