# System Architecture

High-level pipeline for the Biohub cell tracking competition.

```text
Dataset (Zarr + GEFF)
        ↓
   Data Loader
        ↓
  Preprocessing
        ↓
 Visualization
        ↓
 Segmentation
        ↓
 Cell Detection
        ↓
    Tracking
        ↓
Division Detection
        ↓
 Lineage Graph
        ↓
   Evaluation
        ↓
Submission Generator
```

## Dataset

**Purpose:** Discover samples and load OME-Zarr volumes with optional GEFF ground truth.

**Inputs:** Competition directory containing `{id}.zarr` and `{id}.geff` pairs.

**Outputs:** `EmbryoSample`, lazy `ZarrVolumeReader`, in-memory `TrackingGraph`.

**Dependencies:** `zarr`, `numpy`.

**Future improvements:** Streaming batch iterators, normalization cache, multi-process prefetch.

## Preprocessing

**Purpose:** Normalize, denoise, and prepare volumes for downstream models.

**Inputs:** Raw Zarr timepoints.

**Outputs:** Processed arrays or cached tensors.

**Dependencies:** `numpy`, `scipy`, `opencv`.

**Future improvements:** GPU normalization, adaptive histogram equalization.

## Visualization

**Purpose:** Inspect volumes and tracks for debugging and publication figures.

**Inputs:** Volume slices, tracking graphs.

**Outputs:** MIP PNGs, scatter plots, optional Napari sessions.

**Dependencies:** `matplotlib`, optional `napari`.

**Future improvements:** Edge TP/FP/FN overlays, division highlighting.

## Segmentation

**Purpose:** Produce cell instance masks or centroids.

**Inputs:** Preprocessed volumes.

**Outputs:** Label masks or detection points.

**Dependencies:** Planned `cellpose`, `stardist`.

**Future improvements:** Transformer-based segmenters, uncertainty maps.

## Tracking

**Purpose:** Link detections across time and infer divisions.

**Inputs:** Detections, optional appearance features.

**Outputs:** Predicted `TrackingGraph`.

**Dependencies:** Planned `laptrack`, `ultrack`, `trackastra`.

**Future improvements:** Learned linking transformers, global optimization.

## Evaluation

**Purpose:** Offline validation and submission packaging.

**Inputs:** Predicted and ground-truth graphs.

**Outputs:** Edge/division Jaccard metrics, GEFF submission folders.

**Dependencies:** `scipy`, internal metrics module.

**Future improvements:** Match official `tracking_cellmot` metric tests exactly.

## Submission Generator

**Purpose:** Serialize predictions to competition GEFF format.

**Inputs:** Predicted graphs per sample.

**Outputs:** `{sample_id}.geff` Zarr stores.

**Dependencies:** `zarr`, GEFF writer.

**Future improvements:** Validation against GEFF JSON schema.
