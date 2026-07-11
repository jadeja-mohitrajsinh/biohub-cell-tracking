# Visualization Module

## Purpose

Render volume MIPs and 2D track overlays; optional Napari viewer.

## API

- `maximum_intensity_projection(volume, axis=1)`
- `plot_mip(volume, output_path=...)`
- `plot_tracks_2d(graph, scale, time_index=...)`
- `launch_napari(volume, graph=None)`

## Roadmap

- Metric-aware edge coloring
- Division node highlighting
