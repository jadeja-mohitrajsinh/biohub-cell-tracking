# Biohub Cell Tracking During Development

Production-quality, modular Python codebase for the [Biohub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Kaggle competition.

## Competition Goal

Track cells across 3D+time microscopy volumes of developing zebrafish embryos. Predictions are sparse tracking graphs: cell detections `(t, z, y, x)` linked by directed temporal edges, with divisions encoded as one parent node with two outgoing edges.

## Dataset

Each sample contains:

- `{sample_id}.zarr` — OME-Zarr volume `(T, Z, Y, X)` with physical scale `(1.0, 1.625, 0.40625, 0.40625)` µm per voxel axis
- `{sample_id}.geff` — sparse GEFF ground-truth graph (train split only)

Local data in this workspace lives at `../train` relative to the project root.

## Installation

```bash
cd biohub-cell-tracking
python -m pip install -e ".[dev]"
```

Optional visualization extras:

```bash
python -m pip install -e ".[viz]"
```

## Quick Start

```bash
# List available samples
biohub list --data-root ../train

# Inspect one sample
biohub inspect 6bba_fe670320 --data-root ../train

# Offline evaluation (identity baseline on ground truth)
biohub evaluate 6bba_fe670320 --data-root ../train

# Write a GEFF submission file
biohub submit 6bba_fe670320 --data-root ../train
```

Python API example:

```python
from pathlib import Path
from data.dataset import DatasetCatalog, EmbryoDataset
from metrics.evaluator import evaluate_graph

catalog = DatasetCatalog(Path("../train"))
sample = catalog.load_sample("6bba_fe670320", require_tracks=True)
dataset = EmbryoDataset(sample)
frame = dataset.get_timepoint(0)

result = evaluate_graph(sample.tracks, sample.tracks, sample.image_metadata.scale)
print(result)
```

## Folder Structure

See `ARCHITECTURE.md` and `docs/architecture/` for module-level design. Phase 1 implements:

- Configuration (`configs/`, `src/utils/config.py`)
- Dataset reader (`src/data/`)
- Logging (`src/utils/logging.py`)
- Visualization (`src/visualization/`)
- Offline metrics (`src/metrics/`)
- Submission generator (`src/evaluation/`)

Segmentation, deep learning tracking, and lineage modules are scaffolded for later phases.

## Roadmap

See `ROADMAP.md`.

## References

- [Competition page](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- [Official metric description](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- [GEFF specification](https://liveimagetrackingtools.org/geff/latest/specification/)
