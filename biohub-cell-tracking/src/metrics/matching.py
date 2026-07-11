"""Node matching utilities for sparse ground-truth evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.models import CellNode, TrackingGraph, VoxelScale


@dataclass(frozen=True, slots=True)
class NodeMatchingResult:
    """Result of bipartite node matching."""

    pred_to_gt: dict[int, int]
    gt_to_pred: dict[int, int]
    distances_um: dict[int, float]


def physical_distance_um(left: CellNode, right: CellNode, scale: VoxelScale) -> float:
    """Compute centroid distance in micrometers."""
    lt, lz, ly, lx = left.physical_coords(scale)
    rt, rz, ry, rx = right.physical_coords(scale)
    return float(np.sqrt((lt - rt) ** 2 + (lz - rz) ** 2 + (ly - ry) ** 2 + (lx - rx) ** 2))


def match_nodes(
    predicted: TrackingGraph,
    ground_truth: TrackingGraph,
    scale: VoxelScale,
    max_distance_um: float = 7.0,
) -> NodeMatchingResult:
    """Match predicted nodes to ground-truth nodes using Hungarian assignment."""
    pred_nodes = list(predicted.iter_nodes())
    gt_nodes = list(ground_truth.iter_nodes())
    if not pred_nodes or not gt_nodes:
        return NodeMatchingResult(pred_to_gt={}, gt_to_pred={}, distances_um={})

    cost = np.full((len(pred_nodes), len(gt_nodes)), fill_value=max_distance_um * 10.0, dtype=np.float64)
    for pred_index, pred_node in enumerate(pred_nodes):
        for gt_index, gt_node in enumerate(gt_nodes):
            distance = physical_distance_um(pred_node, gt_node, scale)
            if distance <= max_distance_um:
                cost[pred_index, gt_index] = distance

    row_indices, col_indices = linear_sum_assignment(cost)
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    distances_um: dict[int, float] = {}

    for row, col in zip(row_indices, col_indices, strict=True):
        distance = cost[row, col]
        if distance > max_distance_um:
            continue
        pred_node = pred_nodes[row]
        gt_node = gt_nodes[col]
        pred_to_gt[pred_node.node_id] = gt_node.node_id
        gt_to_pred[gt_node.node_id] = pred_node.node_id
        distances_um[pred_node.node_id] = float(distance)

    return NodeMatchingResult(pred_to_gt=pred_to_gt, gt_to_pred=gt_to_pred, distances_um=distances_um)
