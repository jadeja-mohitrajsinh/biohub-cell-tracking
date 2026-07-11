# Requirements

## Python Version

- Python **3.12+** recommended (project tested on 3.12/3.14)

## Core Dependencies (installed by default)

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| scipy | Hungarian matching |
| zarr | OME-Zarr and GEFF IO |
| pyyaml / omegaconf | Configuration |
| pandas | Tabular experiment logs |
| opencv-python-headless | Image preprocessing |
| rich | Logging and CLI output |
| matplotlib | Static visualization |
| tqdm | Progress bars |

## Optional — Visualization

| Package | Purpose |
|---------|---------|
| napari | Interactive 4D viewer |

## Optional — Deep Learning Phase

| Package | Purpose |
|---------|---------|
| torch | Model training |
| hydra-core | Experiment configuration |
| cellpose | Segmentation |
| stardist | Star-convex segmentation |
| laptrack | Linear assignment tracking |
| ultrack | Large-scale tracking |
| trackastra | Transformer tracking |

## Hardware

- **CPU:** Required for data pipeline and classical baselines
- **CUDA:** Optional for deep learning phases
- **RAM:** 16 GB minimum recommended; volumes stream one timepoint at a time

## Development

```bash
pip install -e ".[dev]"
pytest
```
