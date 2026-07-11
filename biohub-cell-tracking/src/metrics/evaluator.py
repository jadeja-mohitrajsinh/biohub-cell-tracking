"""High-level evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from data.models import EmbryoSample, TrackingGraph, VoxelScale
from metrics.edge_jaccard import CountResult, evaluate_single_graph


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Micro-averaged evaluation summary."""

    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    edge_jaccard: float
    adjusted_edge_jaccard: float
    division_jaccard: float
    score: float
    num_samples: int

    @classmethod
    def from_counts(
        cls,
        edge: CountResult,
        division: CountResult,
        *,
        adjusted_edge_jaccard: float,
        division_weight: float,
        num_samples: int,
    ) -> "EvaluationResult":
        """Build a summary from aggregated counts."""
        return cls(
            edge_tp=edge.tp,
            edge_fp=edge.fp,
            edge_fn=edge.fn,
            division_tp=division.tp,
            division_fp=division.fp,
            division_fn=division.fn,
            edge_jaccard=edge.jaccard,
            adjusted_edge_jaccard=adjusted_edge_jaccard,
            division_jaccard=division.jaccard,
            score=adjusted_edge_jaccard + division_weight * division.jaccard,
            num_samples=num_samples,
        )


def evaluate_graph(
    predicted: TrackingGraph,
    ground_truth: TrackingGraph,
    scale: VoxelScale,
    *,
    max_distance_um: float = 7.0,
    penalty_weight: float = 0.1,
    division_weight: float = 0.1,
) -> dict[str, float | int]:
    """Evaluate one graph pair."""
    return evaluate_single_graph(
        predicted,
        ground_truth,
        scale,
        max_distance_um=max_distance_um,
        penalty_weight=penalty_weight,
        division_weight=division_weight,
    )


def evaluate_samples(
    predictions: dict[str, TrackingGraph],
    samples: list[EmbryoSample],
    *,
    max_distance_um: float = 7.0,
    penalty_weight: float = 0.1,
    division_weight: float = 0.1,
) -> EvaluationResult:
    """Micro-average evaluation across multiple samples."""
    edge = CountResult(tp=0, fp=0, fn=0)
    division = CountResult(tp=0, fp=0, fn=0)
    adjusted_numerator = 0.0
    adjusted_denominator = 0.0
    evaluated = 0

    sample_map = {sample.sample_id: sample for sample in samples}
    for sample_id, predicted in predictions.items():
        sample = sample_map.get(sample_id)
        if sample is None or sample.tracks is None:
            continue
        result = evaluate_single_graph(
            predicted,
            sample.tracks,
            sample.image_metadata.scale,
            max_distance_um=max_distance_um,
            penalty_weight=penalty_weight,
            division_weight=division_weight,
        )
        edge = CountResult(
            tp=edge.tp + int(result["edge_tp"]),
            fp=edge.fp + int(result["edge_fp"]),
            fn=edge.fn + int(result["edge_fn"]),
        )
        division = CountResult(
            tp=division.tp + int(result["division_tp"]),
            fp=division.fp + int(result["division_fp"]),
            fn=division.fn + int(result["division_fn"]),
        )
        sample_weight = int(result["edge_tp"]) + int(result["edge_fp"]) + int(result["edge_fn"])
        adjusted_numerator += float(result["adjusted_edge_jaccard"]) * sample_weight
        adjusted_denominator += sample_weight
        evaluated += 1

    adjusted_edge = adjusted_numerator / adjusted_denominator if adjusted_denominator else edge.jaccard
    return EvaluationResult.from_counts(
        edge,
        division,
        adjusted_edge_jaccard=adjusted_edge,
        division_weight=division_weight,
        num_samples=evaluated,
    )
