"""Validate data models and graph helpers."""

from data.models import CellNode, TrackingEdge, TrackingGraph, VoxelScale


def test_tracking_graph_division_nodes() -> None:
    graph = TrackingGraph(
        nodes={
            1: CellNode(1, 0, 1, 2, 3),
            2: CellNode(2, 1, 1, 2, 3),
            3: CellNode(3, 1, 1, 2, 4),
        },
        edges=[
            TrackingEdge(1, 2),
            TrackingEdge(1, 3),
        ],
    )
    assert graph.division_nodes() == [1]


def test_physical_coords() -> None:
    scale = VoxelScale()
    node = CellNode(1, 2, 3, 4, 5)
    assert node.physical_coords(scale) == (2.0, 4.875, 1.625, 2.03125)
