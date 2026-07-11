"""Visualization helpers for volumes and tracks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from data.models import TrackingGraph, VoxelScale


def maximum_intensity_projection(
    volume: NDArray[np.uint16 | np.float32],
    axis: int = 1,
) -> NDArray[np.float32]:
    """Compute a maximum intensity projection along ``axis``."""
    return np.max(volume, axis=axis).astype(np.float32)


def plot_mip(
    volume: NDArray[np.uint16 | np.float32],
    *,
    axis: int = 1,
    title: str | None = None,
    output_path: str | Path | None = None,
    cmap: str = "gray",
    dpi: int = 120,
) -> Path | None:
    """Save a MIP figure to disk or display inline."""
    mip = maximum_intensity_projection(volume, axis=axis)
    figure, axis_handle = plt.subplots(figsize=(6, 6))
    axis_handle.imshow(mip, cmap=cmap)
    axis_handle.set_axis_off()
    if title:
        axis_handle.set_title(title)
    figure.tight_layout()

    if output_path is None:
        plt.show()
        plt.close(figure)
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_tracks_2d(
    graph: TrackingGraph,
    scale: VoxelScale,
    *,
    time_index: int,
    projection: str = "yx",
    output_path: str | Path | None = None,
    max_points: int = 5000,
) -> Path | None:
    """Plot matched detections for one timepoint in 2D."""
    nodes = [node for node in graph.iter_nodes() if node.t == time_index][:max_points]
    if not nodes:
        raise ValueError(f"No nodes found at time_index={time_index}")

    figure, axis_handle = plt.subplots(figsize=(7, 7))
    xs: list[float] = []
    ys: list[float] = []
    for node in nodes:
        _, z_um, y_um, x_um = node.physical_coords(scale)
        if projection == "yx":
            xs.append(x_um)
            ys.append(y_um)
        elif projection == "zx":
            xs.append(x_um)
            ys.append(z_um)
        else:
            raise ValueError(f"Unsupported projection: {projection}")

    axis_handle.scatter(xs, ys, s=8, c="cyan", alpha=0.8)
    for edge in graph.edges:
        source = graph.nodes.get(edge.source_id)
        target = graph.nodes.get(edge.target_id)
        if source is None or target is None:
            continue
        if source.t != time_index or target.t != time_index:
            continue
        if projection == "yx":
            axis_handle.plot(
                [source.x * scale.x, target.x * scale.x],
                [source.y * scale.y, target.y * scale.y],
                color="yellow",
                linewidth=0.5,
                alpha=0.6,
            )
    axis_handle.set_title(f"Tracks t={time_index} ({projection})")
    axis_handle.set_aspect("equal")
    figure.tight_layout()

    if output_path is None:
        plt.show()
        plt.close(figure)
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def launch_napari(volume: NDArray[np.uint16], graph: TrackingGraph | None = None) -> None:
    """Optional Napari viewer for interactive inspection."""
    try:
        import napari
    except ImportError as exc:
        raise ImportError("Install visualization extras: pip install -e '.[viz]'") from exc

    viewer = napari.Viewer()
    viewer.add_image(volume)
    if graph is not None:
        points = np.array([[node.t, node.z, node.y, node.x] for node in graph.iter_nodes()])
        if len(points):
            viewer.add_points(points, size=2, name="tracks")
    napari.run()
