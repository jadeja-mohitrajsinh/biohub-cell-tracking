# Dataset

## Structure

Each sample is a pair of stores in the same directory:

```text
train/
├── {sample_id}.zarr     # 4D image volume
└── {sample_id}.geff     # sparse tracking graph (train only)
```

## Folder Hierarchy

### OME-Zarr (`*.zarr`)

- Root group with `multiscales` metadata
- Array path `0` with shape `(T, Z, Y, X)` and dtype `uint16`
- Chunked by time (`chunk_shape = [1, Z, Y, X]`)

### GEFF (`*.geff`)

- `nodes/ids` — node identifiers
- `nodes/props/{t,z,y,x}/values` — detection coordinates
- `edges/ids` — `(source, target)` pairs
- Root attributes under `geff` key

## Zarr Explanation

Volumes use OME-NGFF multiscale metadata embedded in Zarr v3 sidecars (`zarr.json`). The reader accesses one timepoint at a time for memory efficiency.

## GEFF Explanation

GEFF (Graph Exchange File Format) stores spatial graphs in Zarr. See the [GEFF specification](https://liveimagetrackingtools.org/geff/latest/specification/).

## Coordinate System

- Index coordinates in GEFF are integer voxel indices `(t, z, y, x)`
- Physical coordinates (µm) = index × scale per axis

## Physical Scaling

Default competition scale (µm per voxel):

| Axis | Scale |
|------|-------|
| T | 1.0 |
| Z | 1.625 |
| Y | 0.40625 |
| X | 0.40625 |

Example volume shape: `(100, 64, 256, 256)`.

## Embryo Split

- **Train:** image + GEFF ground truth
- **Test:** image only (submit GEFF predictions)

## Visualization Examples

```bash
biohub inspect <sample_id> --data-root ../train
python -c "
from pathlib import Path
from data.dataset import DatasetCatalog, EmbryoDataset
from visualization.viewer import plot_mip

catalog = DatasetCatalog(Path('../train'))
sample = catalog.load_sample(catalog.list_samples()[0], require_tracks=True)
dataset = EmbryoDataset(sample)
plot_mip(dataset.get_timepoint(0), output_path='outputs/mip_t0.png')
"
```
