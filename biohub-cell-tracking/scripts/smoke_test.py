#!/usr/bin/env python3
"""Quick smoke test against local competition data."""

from pathlib import Path

from data.dataset import DatasetCatalog
from metrics.evaluator import evaluate_graph
from utils.logging import console


def main() -> None:
    data_root = Path(__file__).resolve().parents[1].parent / "train"
    catalog = DatasetCatalog(data_root)
    sample_id = catalog.list_samples()[0]
    sample = catalog.load_sample(sample_id, require_tracks=True)
    assert sample.tracks is not None
    result = evaluate_graph(sample.tracks, sample.tracks, sample.image_metadata.scale)
    console.print(f"Smoke test passed for {sample_id}: {result}")


if __name__ == "__main__":
    main()
