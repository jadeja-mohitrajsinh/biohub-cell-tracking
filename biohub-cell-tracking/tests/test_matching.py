"""Validate node matching behavior."""

from data.models import CellNode, TrackingGraph, VoxelScale
from metrics.matching import match_nodes


def test_match_nodes_pairs_close_centroids() -> None:
    scale = VoxelScale()
    predicted = TrackingGraph(
        nodes={1: CellNode(1, 0, 10, 20, 30)},
        edges=[],
    )
    ground_truth = TrackingGraph(
        nodes={10: CellNode(10, 0, 10, 20, 30)},
        edges=[],
    )
    matching = match_nodes(predicted, ground_truth, scale, max_distance_um=7.0)
    assert matching.pred_to_gt[1] == 10


def test_match_nodes_rejects_distant_centroids() -> None:
    scale = VoxelScale()
    predicted = TrackingGraph(nodes={1: CellNode(1, 0, 0, 0, 0)}, edges=[])
    ground_truth = TrackingGraph(nodes={10: CellNode(10, 0, 50, 50, 50)}, edges=[])
    matching = match_nodes(predicted, ground_truth, scale, max_distance_um=7.0)
    assert matching.pred_to_gt == {}
