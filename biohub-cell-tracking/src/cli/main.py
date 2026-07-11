"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset import DatasetCatalog, EmbryoDataset
from data.models import CellNode, TrackingEdge, TrackingGraph
from evaluation.submission import SubmissionGenerator
from metrics.evaluator import evaluate_graph
from utils.config import load_config, resolve_path
from utils.logging import console, setup_logging


def _build_identity_graph(sample_id: str, dataset: EmbryoDataset) -> TrackingGraph:
    """Return ground-truth graph as a sanity-check baseline."""
    if dataset.sample.tracks is None:
        raise ValueError(f"Sample {sample_id} has no ground truth tracks")
    return TrackingGraph(
        nodes=dict(dataset.sample.tracks.nodes),
        edges=list(dataset.sample.tracks.edges),
        metadata=dataset.sample.tracks.metadata,
    )


def _build_nearest_neighbor_graph(dataset: EmbryoDataset, max_distance_um: float) -> TrackingGraph:
    """Create a simple nearest-neighbor baseline tracker."""
    if dataset.sample.tracks is None:
        raise ValueError("Nearest-neighbor baseline requires ground-truth node locations")

    scale = dataset.sample.image_metadata.scale
    nodes_by_time: dict[int, list[CellNode]] = {}
    for node in dataset.sample.tracks.iter_nodes():
        nodes_by_time.setdefault(node.t, []).append(node)

    new_nodes: dict[int, CellNode] = {}
    next_id = 1
    id_map: dict[int, int] = {}
    for node in dataset.sample.tracks.iter_nodes():
        new_id = next_id
        next_id += 1
        id_map[node.node_id] = new_id
        new_nodes[new_id] = CellNode(new_id, node.t, node.z, node.y, node.x)

    edges: list[TrackingEdge] = []
    times = sorted(nodes_by_time)
    for left_time, right_time in zip(times, times[1:], strict=False):
        if right_time - left_time != 1:
            continue
        for left_node in nodes_by_time[left_time]:
            best_target: CellNode | None = None
            best_distance = max_distance_um
            for right_node in nodes_by_time[right_time]:
                distance = left_node.physical_coords(scale)[1:]  # spatial only
                target_distance = right_node.physical_coords(scale)[1:]
                dist = sum((a - b) ** 2 for a, b in zip(distance, target_distance, strict=True)) ** 0.5
                if dist <= best_distance:
                    best_distance = dist
                    best_target = right_node
            if best_target is not None:
                edges.append(
                    TrackingEdge(
                        source_id=id_map[left_node.node_id],
                        target_id=id_map[best_target.node_id],
                    )
                )

    return TrackingGraph(nodes=new_nodes, edges=edges, metadata=dataset.sample.tracks.metadata)


def cmd_list(args: argparse.Namespace) -> None:
    catalog = DatasetCatalog(args.data_root)
    samples = catalog.list_samples()
    console.print(f"Found {len(samples)} samples under {args.data_root}")
    for sample_id in samples[: args.limit]:
        console.print(f"- {sample_id}")


def cmd_inspect(args: argparse.Namespace) -> None:
    catalog = DatasetCatalog(args.data_root)
    sample = catalog.load_sample(args.sample_id, require_tracks=not args.no_tracks)
    console.print(f"Sample: {sample.sample_id}")
    console.print(f"Image shape: {sample.image_metadata.shape}")
    console.print(f"Scale (TZYX um): {sample.image_metadata.scale.as_tuple()}")
    if sample.tracks:
        console.print(f"Tracks: {sample.tracks.num_nodes} nodes, {sample.tracks.num_edges} edges")


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_root = args.data_root or resolve_path(config, "paths.data_root")
    catalog = DatasetCatalog(data_root)
    sample = catalog.load_sample(args.sample_id, require_tracks=True)
    dataset = EmbryoDataset(sample)

    if args.baseline == "identity":
        predicted = _build_identity_graph(args.sample_id, dataset)
    else:
        predicted = _build_nearest_neighbor_graph(
            dataset,
            max_distance_um=float(config.metrics.max_match_distance_um) * 2,
        )

    result = evaluate_graph(
        predicted,
        sample.tracks,
        sample.image_metadata.scale,
        max_distance_um=float(config.metrics.max_match_distance_um),
        penalty_weight=float(config.metrics.node_penalty_weight),
        division_weight=float(config.metrics.division_score_weight),
    )
    console.print(result)


def cmd_submit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_root = args.data_root or resolve_path(config, "paths.data_root")
    output_dir = args.output_dir or resolve_path(config, "paths.submissions_dir")
    catalog = DatasetCatalog(data_root)
    sample = catalog.load_sample(args.sample_id, require_tracks=True)
    dataset = EmbryoDataset(sample)
    predicted = _build_identity_graph(args.sample_id, dataset)

    generator = SubmissionGenerator(
        output_dir,
        scale=sample.image_metadata.scale,
        overwrite=bool(config.submission.overwrite),
        zarr_format=int(config.submission.zarr_format),
    )
    path = generator.write_sample(args.sample_id, predicted)
    console.print(f"Submission written to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Biohub cell tracking CLI")
    parser.add_argument("--config", default="default", help="Config name under configs/")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root directory")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available samples")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--data-root", type=Path, default=None)
    list_parser.set_defaults(func=cmd_list)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one sample")
    inspect_parser.add_argument("sample_id")
    inspect_parser.add_argument("--no-tracks", action="store_true")
    inspect_parser.add_argument("--data-root", type=Path, default=None)
    inspect_parser.set_defaults(func=cmd_inspect)

    evaluate_parser = subparsers.add_parser("evaluate", help="Offline evaluation for one sample")
    evaluate_parser.add_argument("sample_id")
    evaluate_parser.add_argument("--baseline", choices=["identity", "nearest_neighbor"], default="identity")
    evaluate_parser.add_argument("--data-root", type=Path, default=None)
    evaluate_parser.set_defaults(func=cmd_evaluate)

    submit_parser = subparsers.add_parser("submit", help="Write a GEFF submission for one sample")
    submit_parser.add_argument("sample_id")
    submit_parser.add_argument("--output-dir", type=Path, default=None)
    submit_parser.add_argument("--data-root", type=Path, default=None)
    submit_parser.set_defaults(func=cmd_submit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
