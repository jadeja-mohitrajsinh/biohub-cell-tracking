# Data Module

## Purpose

Load OME-Zarr volumes and GEFF tracking graphs for the Biohub competition.

## API

| Symbol | Description |
|--------|-------------|
| `DatasetCatalog` | Discover and load samples |
| `EmbryoDataset` | Lazy per-sample accessor |
| `ZarrVolumeReader` | Stream `(T,Z,Y,X)` data |
| `GeffReader` / `GeffWriter` | GEFF graph IO |
| `TrackingGraph` | In-memory graph model |

## Public Classes

- `EmbryoSample`, `CellNode`, `TrackingEdge`, `TrackingGraph`, `VoxelScale`

## Roadmap

- PyTorch dataset wrapper
- Normalization cache
- Test-split loader without GT
