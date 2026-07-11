"""Integration tests against local competition data."""

from pathlib import Path

import pytest

from data.dataset import DatasetCatalog
from data.zarr_reader import ZarrVolumeReader
from metrics.evaluator import evaluate_graph

DATA_ROOT = Path(__file__).resolve().parents[1].parent / "train"


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Local train data not available")
def test_catalog_discovers_samples() -> None:
    catalog = DatasetCatalog(DATA_ROOT)
    samples = catalog.list_samples()
    assert len(samples) > 0


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Local train data not available")
def test_load_sample_metadata_and_tracks() -> None:
    catalog = DatasetCatalog(DATA_ROOT)
    sample_id = catalog.list_samples()[0]
    sample = catalog.load_sample(sample_id, require_tracks=True)
    assert sample.image_metadata.shape[0] > 0
    assert sample.tracks is not None
    assert sample.tracks.num_nodes > 0


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Local train data not available")
def test_read_single_timepoint() -> None:
    catalog = DatasetCatalog(DATA_ROOT)
    sample_id = catalog.list_samples()[0]
    image_path, _ = catalog.get_paths(sample_id)
    reader = ZarrVolumeReader(image_path)
    frame = reader.read_timepoint(0)
    assert frame.ndim == 3


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Local train data not available")
def test_identity_evaluation_on_first_sample() -> None:
    catalog = DatasetCatalog(DATA_ROOT)
    sample_id = catalog.list_samples()[0]
    sample = catalog.load_sample(sample_id, require_tracks=True)
    assert sample.tracks is not None
    result = evaluate_graph(sample.tracks, sample.tracks, sample.image_metadata.scale)
    assert result["edge_jaccard"] == 1.0
