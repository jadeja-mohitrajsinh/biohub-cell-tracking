"""Validate edge metric counting."""

from data.models import CellNode, GeffMetadata, TrackingEdge, TrackingGraph, VoxelScale
from metrics.edge_jaccard import edge_counts, evaluate_single_graph
from metrics.matching import match_nodes


def _simple_graphs() -> tuple[TrackingGraph, TrackingGraph]:
    gt = TrackingGraph(
        nodes={
            1: CellNode(1, 0, 0, 0, 0),
            2: CellNode(2, 1, 0, 0, 0),
        },
        edges=[TrackingEdge(1, 2)],
        metadata=GeffMetadata("1.1", True, (), estimated_number_of_nodes=2),
    )
    pred = TrackingGraph(
        nodes={
            10: CellNode(10, 0, 0, 0, 0),
            11: CellNode(11, 1, 0, 0, 0),
        },
        edges=[TrackingEdge(10, 11)],
    )
    return pred, gt


def test_perfect_edge_match() -> None:
    predicted, ground_truth = _simple_graphs()
    matching = match_nodes(predicted, ground_truth, VoxelScale())
    counts = edge_counts(predicted, ground_truth, matching)
    assert counts.tp == 1
    assert counts.fp == 0
    assert counts.fn == 0


def test_identity_graph_scores_perfectly() -> None:
    _, ground_truth = _simple_graphs()
    result = evaluate_single_graph(ground_truth, ground_truth, VoxelScale())
    assert result["edge_jaccard"] == 1.0
    assert result["adjusted_edge_jaccard"] == 1.0


def test_adjusted_jaccard_penalizes_excess_nodes() -> None:
    _, ground_truth = _simple_graphs()
    extra = TrackingGraph(
        nodes={
            **ground_truth.nodes,
            99: CellNode(99, 0, 10, 10, 10),
        },
        edges=list(ground_truth.edges),
        metadata=GeffMetadata("1.1", True, (), estimated_number_of_nodes=2),
    )
    result = evaluate_single_graph(extra, ground_truth, VoxelScale())
    assert result["adjusted_edge_jaccard"] < result["edge_jaccard"]
