"""Validate GEFF round-trip IO."""

from pathlib import Path

import pytest

from data.geff_reader import GeffReader, GeffWriter
from data.models import CellNode, GeffMetadata, TrackingEdge, TrackingGraph, VoxelScale

DATA_ROOT = Path(__file__).resolve().parents[1].parent / "train"


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Local train data not available")
def test_geff_round_trip(tmp_path: Path) -> None:
    source = next(DATA_ROOT.glob("*.geff"))
    original = GeffReader(source).read_graph()
    target = tmp_path / "roundtrip.geff"
    GeffWriter(target, scale=VoxelScale(), overwrite=True).write(original)
    restored = GeffReader(target).read_graph()
    assert restored.num_nodes == original.num_nodes
    assert restored.num_edges == original.num_edges


def test_write_minimal_graph(tmp_path: Path) -> None:
    graph = TrackingGraph(
        nodes={
            1: CellNode(1, 0, 1, 2, 3),
            2: CellNode(2, 1, 1, 2, 4),
        },
        edges=[TrackingEdge(1, 2)],
        metadata=GeffMetadata("1.1", True, (), estimated_number_of_nodes=2),
    )
    path = GeffWriter(tmp_path / "mini.geff").write(graph)
    restored = GeffReader(path).read_graph()
    assert restored.num_nodes == 2
    assert restored.num_edges == 1
