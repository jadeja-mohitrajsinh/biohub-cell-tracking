# Components

See also per-module README files under `src/*/README.md`.

## Data Loader

**Purpose:** Discover and load competition samples.

**Input:** Data root path.

**Output:** `EmbryoSample`, `TrackingGraph`, volume metadata.

**Classes:** `DatasetCatalog`, `EmbryoDataset`, `ZarrVolumeReader`, `GeffReader`, `GeffWriter`

**Functions:** `list_samples`, `load_sample`, `read_timepoint`, `read_graph`

**Dependencies:** `zarr`, `numpy`

**Future Improvements:** PyTorch Dataset, sharded loading

## Metrics

**Purpose:** Offline competition evaluation.

**Input:** Predicted and GT `TrackingGraph`, `VoxelScale`.

**Output:** TP/FP/FN counts, Jaccard scores.

**Classes:** `EvaluationResult`, `NodeMatchingResult`, `CountResult`

**Functions:** `match_nodes`, `evaluate_graph`, `evaluate_samples`

**Dependencies:** `scipy`, `numpy`

**Future Improvements:** Reference test vectors from official repo

## Submission Generator

**Purpose:** Export GEFF predictions.

**Input:** `TrackingGraph` per sample.

**Output:** `{sample_id}.geff` directory.

**Classes:** `SubmissionGenerator`, `GeffWriter`

**Dependencies:** `zarr`

**Future Improvements:** Batch zip packaging for Kaggle

## Visualization

**Purpose:** Static and interactive inspection.

**Input:** Volumes and graphs.

**Output:** PNG figures, Napari viewer.

**Functions:** `plot_mip`, `plot_tracks_2d`, `launch_napari`

**Dependencies:** `matplotlib`, optional `napari`

**Future Improvements:** Metric-aware edge coloring
