# Architecture Decisions

## Modular `src/` layout

Each pipeline stage is an independent Python package with its own README and tests. This supports parallel research threads (segmentation vs tracking) without notebook-only code.

## YAML + OmegaConf configuration

Hydra-compatible defaults inheritance without requiring Hydra in Phase 1. Deep learning phases can migrate configs to Hydra when training modules land.

## Native GEFF IO

GEFF is read/written directly with Zarr v3 rather than depending on `tracksdata` in Phase 1. This keeps the foundation lightweight while remaining compatible with the competition format.

## Lazy volume loading

Volumes can exceed available RAM. `read_timepoint` loads one `(Z, Y, X)` chunk at a time.

## Metric implementation from specification

Edge and division Jaccard follow the public `metrics.md` document. Future work will add parity tests against `tracking_cellmot`.

## CLI-first workflows

`biohub` commands cover listing, inspection, evaluation, and submission without notebooks.
