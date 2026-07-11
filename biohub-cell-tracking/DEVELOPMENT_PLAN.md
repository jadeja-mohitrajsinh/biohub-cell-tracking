# Development Plan

Each module follows the same lifecycle template.

## Data Loader

**Goal:** Load competition samples with lazy Zarr access and GEFF parsing.

**Requirements:** Zarr v3, typed models, streaming timepoints.

**Implementation:** `DatasetCatalog`, `ZarrVolumeReader`, `GeffReader`.

**Testing:** Integration tests against `../train`.

**Benchmark:** Time to load first timepoint vs full volume.

**Future Work:** Dataloader for PyTorch, cache manifests.

## Metrics

**Goal:** Reproduce competition edge and division Jaccard offline.

**Requirements:** 7 µm matching threshold, sparse GT rules, micro-averaging.

**Implementation:** `metrics/matching.py`, `metrics/edge_jaccard.py`, `metrics/evaluator.py`.

**Testing:** Synthetic graphs + identity evaluation on real samples.

**Benchmark:** Compare against official `tracking_cellmot` when integrated.

**Future Work:** Division matching parity tests with reference assets.

## Submission Generator

**Goal:** Write valid GEFF submissions for the test split.

**Requirements:** Zarr v3 GEFF metadata, node/edge arrays.

**Implementation:** `GeffWriter`, `SubmissionGenerator`.

**Testing:** Round-trip read/write tests.

**Benchmark:** File size and write latency per sample.

**Future Work:** Schema validation, batch packaging for Kaggle.

## Visualization

**Goal:** Quick inspection without notebook-only code.

**Requirements:** MIP export, 2D track overlays, optional Napari.

**Implementation:** `visualization/viewer.py`.

**Testing:** Smoke tests with saved PNG outputs.

**Benchmark:** Render time per timepoint.

**Future Work:** TP/FP/FN edge coloring like official baseline repo.
