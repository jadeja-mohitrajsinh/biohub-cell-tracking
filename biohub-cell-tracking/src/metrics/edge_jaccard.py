"""Edge and division Jaccard metrics."""

from __future__ import annotations

from dataclasses import dataclass

from data.models import TrackingGraph, VoxelScale
from metrics.matching import NodeMatchingResult, match_nodes


@dataclass(frozen=True, slots=True)
class CountResult:
    """Raw TP/FP/FN counts."""

    tp: int
    fp: int
    fn: int

    @property
    def jaccard(self) -> float:
        """Compute Jaccard index."""
        denominator = self.tp + self.fp + self.fn
        if denominator == 0:
            return 1.0
        return self.tp / denominator


def _gt_edge_set(graph: TrackingGraph) -> set[tuple[int, int]]:
    return {(edge.source_id, edge.target_id) for edge in graph.edges}


def edge_counts(
    predicted: TrackingGraph,
    ground_truth: TrackingGraph,
    matching: NodeMatchingResult,
) -> CountResult:
    """Compute edge TP/FP/FN according to competition rules."""
    gt_edges = _gt_edge_set(ground_truth)
    gt_outgoing = {node_id: ground_truth.outgoing(node_id) for node_id in ground_truth.nodes}
    gt_incoming = {node_id: ground_truth.incoming(node_id) for node_id in ground_truth.nodes}

    matched_gt_edges: set[tuple[int, int]] = set()
    fp = 0

    for edge in predicted.edges:
        gt_source = matching.pred_to_gt.get(edge.source_id)
        gt_target = matching.pred_to_gt.get(edge.target_id)
        if gt_source is None or gt_target is None:
            continue

        gt_pair = (gt_source, gt_target)
        if gt_pair in gt_edges:
            matched_gt_edges.add(gt_pair)
            continue

        target_conflict = gt_target in gt_incoming and len(gt_incoming[gt_target]) > 0
        source_conflict = gt_source in gt_outgoing and len(gt_outgoing[gt_source]) > 0
        if target_conflict or source_conflict:
            fp += 1

    tp = len(matched_gt_edges)
    fn = len(gt_edges - matched_gt_edges)
    return CountResult(tp=tp, fp=fp, fn=fn)


def adjusted_edge_jaccard(
    edge_result: CountResult,
    predicted_nodes: int,
    estimated_true_nodes: int,
    penalty_weight: float = 0.1,
) -> float:
    """Apply node-count penalty to edge Jaccard."""
    base = edge_result.jaccard
    if estimated_true_nodes <= 0:
        return base
    excess_nodes = max(0, predicted_nodes - estimated_true_nodes)
    penalty = penalty_weight * excess_nodes / estimated_true_nodes
    return max(0.0, min(1.0, base * (1.0 - penalty)))


def _division_subgraph_nodes(graph: TrackingGraph, divider_id: int) -> tuple[set[int], set[int], set[int]]:
    """Return parent-era nodes and two daughter lineages for a GT division."""
    parent_sources = graph.incoming(divider_id)
    parent_era = set(parent_sources + [divider_id])
    children = graph.outgoing(divider_id)
    if len(children) != 2:
        return parent_era, set(), set()

    lineage_a = graph.connected_component(children[0])
    lineage_b = graph.connected_component(children[1])
    return parent_era, lineage_a, lineage_b


def division_counts(
    predicted: TrackingGraph,
    ground_truth: TrackingGraph,
    matching: NodeMatchingResult,
) -> CountResult:
    """Compute division TP/FP/FN using competition criteria."""
    gt_divisions = ground_truth.division_nodes()
    matched_pred_divisions = set()
    tp = 0
    fn = 0

    for divider_id in gt_divisions:
        parent_era, lineage_a, lineage_b = _division_subgraph_nodes(ground_truth, divider_id)
        matched_parent = {
            matching.gt_to_pred[node_id]
            for node_id in parent_era
            if node_id in matching.gt_to_pred
        }
        matched_a = {
            matching.gt_to_pred[node_id]
            for node_id in lineage_a
            if node_id in matching.gt_to_pred
        }
        matched_b = {
            matching.gt_to_pred[node_id]
            for node_id in lineage_b
            if node_id in matching.gt_to_pred
        }

        candidate_components: list[set[int]] = []
        for seed in matched_parent | matched_a | matched_b:
            component = predicted.connected_component(seed)
            if component & (matched_parent | matched_a | matched_b):
                candidate_components.append(component)

        success = False
        for component in candidate_components:
            has_parent = bool(component & matched_parent)
            has_a = bool(component & matched_a)
            has_b = bool(component & matched_b)
            has_fork = any(len(predicted.outgoing(node_id)) == 2 for node_id in component)
            if has_parent and has_a and has_b and has_fork:
                success = True
                for node_id in component:
                    if len(predicted.outgoing(node_id)) == 2:
                        matched_pred_divisions.add(node_id)
                break
        if success:
            tp += 1
        else:
            fn += 1

    fp = 0
    for divider_id in predicted.division_nodes():
        if divider_id in matched_pred_divisions:
            continue
        gt_match = matching.pred_to_gt.get(divider_id)
        if gt_match is None:
            continue
        if ground_truth.outgoing(gt_match):
            fp += 1

    return CountResult(tp=tp, fp=fp, fn=fn)


def evaluate_single_graph(
    predicted: TrackingGraph,
    ground_truth: TrackingGraph,
    scale: VoxelScale,
    *,
    max_distance_um: float = 7.0,
    penalty_weight: float = 0.1,
    division_weight: float = 0.1,
) -> dict[str, float | int]:
    """Evaluate one predicted graph against sparse ground truth."""
    matching = match_nodes(predicted, ground_truth, scale, max_distance_um=max_distance_um)
    edge_result = edge_counts(predicted, ground_truth, matching)
    division_result = division_counts(predicted, ground_truth, matching)

    estimated_true_nodes = (
        ground_truth.metadata.estimated_number_of_nodes
        if ground_truth.metadata and ground_truth.metadata.estimated_number_of_nodes
        else ground_truth.num_nodes
    )
    adjusted = adjusted_edge_jaccard(
        edge_result,
        predicted_nodes=predicted.num_nodes,
        estimated_true_nodes=int(estimated_true_nodes),
        penalty_weight=penalty_weight,
    )
    score = adjusted + division_weight * division_result.jaccard

    return {
        "edge_tp": edge_result.tp,
        "edge_fp": edge_result.fp,
        "edge_fn": edge_result.fn,
        "edge_jaccard": edge_result.jaccard,
        "adjusted_edge_jaccard": adjusted,
        "division_tp": division_result.tp,
        "division_fp": division_result.fp,
        "division_fn": division_result.fn,
        "division_jaccard": division_result.jaccard,
        "score": score,
        "matched_nodes": len(matching.pred_to_gt),
    }
