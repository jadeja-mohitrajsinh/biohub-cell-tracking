"""Core typed data structures for volumes and tracking graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VoxelScale:
    """Physical voxel spacing in micrometers per axis."""

    t: float = 1.0
    z: float = 1.625
    y: float = 0.40625
    x: float = 0.40625

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return scale as ``(t, z, y, x)``."""
        return (self.t, self.z, self.y, self.x)

    def physical_coords(self, t: int, z: int, y: int, x: int) -> tuple[float, float, float, float]:
        """Convert index coordinates to physical micrometer coordinates."""
        return (t * self.t, z * self.z, y * self.y, x * self.x)


@dataclass(frozen=True, slots=True)
class AxisMetadata:
    """Metadata for one GEFF or OME-Zarr axis."""

    name: str
    axis_type: str
    unit: str | None
    min_value: float | None = None
    max_value: float | None = None
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class ImageVolumeMetadata:
    """Metadata describing a 4D microscopy volume."""

    shape: tuple[int, int, int, int]
    dtype: str
    scale: VoxelScale
    axes: tuple[str, ...] = ("T", "Z", "Y", "X")
    quantiles: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeffMetadata:
    """Parsed GEFF graph metadata."""

    geff_version: str
    directed: bool
    axes: tuple[AxisMetadata, ...]
    estimated_number_of_nodes: int | None = None


@dataclass(frozen=True, slots=True)
class CellNode:
    """One cell detection in index coordinates."""

    node_id: int
    t: int
    z: int
    y: int
    x: int

    def physical_coords(self, scale: VoxelScale) -> tuple[float, float, float, float]:
        """Return physical coordinates in micrometers."""
        return scale.physical_coords(self.t, self.z, self.y, self.x)


@dataclass(frozen=True, slots=True)
class TrackingEdge:
    """Directed edge linking two detections across time."""

    source_id: int
    target_id: int


@dataclass
class TrackingGraph:
    """In-memory tracking graph backed by node and edge tables."""

    nodes: dict[int, CellNode]
    edges: list[TrackingEdge]
    metadata: GeffMetadata | None = None

    @property
    def num_nodes(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Return number of edges."""
        return len(self.edges)

    def iter_nodes(self) -> Iterator[CellNode]:
        """Iterate nodes in insertion order."""
        yield from self.nodes.values()

    def __post_init__(self) -> None:
        self._outgoing_map = {}
        self._incoming_map = {}
        for edge in self.edges:
            self._outgoing_map.setdefault(edge.source_id, []).append(edge.target_id)
            self._incoming_map.setdefault(edge.target_id, []).append(edge.source_id)

    def outgoing(self, node_id: int) -> list[int]:
        """Return target node IDs for outgoing edges."""
        if not hasattr(self, "_outgoing_map"):
            self.__post_init__()
        return self._outgoing_map.get(node_id, [])

    def incoming(self, node_id: int) -> list[int]:
        """Return source node IDs for incoming edges."""
        if not hasattr(self, "_incoming_map"):
            self.__post_init__()
        return self._incoming_map.get(node_id, [])

    def division_nodes(self) -> list[int]:
        """Return node IDs with exactly two outgoing edges."""
        return [node_id for node_id in self.nodes if len(self.outgoing(node_id)) == 2]

    def adjacency_matrix(self, node_ids: list[int]) -> NDArray[np.bool_]:
        """Build adjacency matrix for the provided node subset."""
        index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        matrix = np.zeros((len(node_ids), len(node_ids)), dtype=bool)
        for edge in self.edges:
            if edge.source_id in index and edge.target_id in index:
                matrix[index[edge.source_id], index[edge.target_id]] = True
        return matrix

    def connected_component(self, seed_id: int) -> set[int]:
        """Return undirected connected component containing ``seed_id``."""
        # Build adjacency list lazily during BFS
        neighbors: dict[int, set[int]] = {}
        
        def get_neighbors(node_id: int) -> set[int]:
            """Get neighbors for a node, building adjacency list as needed."""
            if node_id not in neighbors:
                neighbors[node_id] = set()
            return neighbors[node_id]
        
        # Build adjacency from edges (only for nodes we encounter)
        for edge in self.edges:
            get_neighbors(edge.source_id).add(edge.target_id)
            get_neighbors(edge.target_id).add(edge.source_id)

        component: set[int] = set()
        stack = [seed_id]
        
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            
            # Add neighbors if they exist
            if current in neighbors:
                stack.extend(neighbors[current] - component)
        
        return component


@dataclass(frozen=True, slots=True)
class EmbryoSample:
    """One competition sample pairing image volume metadata with optional tracks."""

    sample_id: str
    image_path: str
    tracks_path: str | None
    image_metadata: ImageVolumeMetadata
    tracks: TrackingGraph | None = None

    @property
    def has_ground_truth(self) -> bool:
        """Return whether ground-truth tracks are available."""
        return self.tracks is not None
