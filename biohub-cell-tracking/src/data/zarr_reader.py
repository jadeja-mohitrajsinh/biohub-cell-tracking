"""Memory-efficient OME-Zarr volume reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import zarr
from numpy.typing import NDArray

from data.models import ImageVolumeMetadata, VoxelScale


class ZarrVolumeReader:
    """Read 4D microscopy volumes stored as OME-Zarr (Zarr v3)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Zarr store not found: {self.path}")

        self._root = zarr.open_group(str(self.path), mode="r")
        self._array = self._root["0"]
        self.metadata = self._parse_metadata()

    def _parse_metadata(self) -> ImageVolumeMetadata:
        """Parse multiscale metadata from the store root."""
        attrs = dict(self._root.attrs)
        multiscales = attrs.get("multiscales", [{}])[0]
        axes = tuple(axis["name"] for axis in multiscales.get("axes", []))
        scale_values = multiscales.get("datasets", [{}])[0].get(
            "coordinateTransformations", [{}]
        )[0].get("scale", [1.0, 1.625, 0.40625, 0.40625])
        quantiles = attrs.get("image_statistics", {}).get("quantiles", {})

        scale = VoxelScale(
            t=float(scale_values[0]),
            z=float(scale_values[1]),
            y=float(scale_values[2]),
            x=float(scale_values[3]),
        )
        return ImageVolumeMetadata(
            shape=tuple(int(v) for v in self._array.shape),
            dtype=str(self._array.dtype),
            scale=scale,
            axes=axes if axes else ("T", "Z", "Y", "X"),
            quantiles={str(k): float(v) for k, v in quantiles.items()},
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return volume shape ``(T, Z, Y, X)``."""
        return self.metadata.shape

    @property
    def scale(self) -> VoxelScale:
        """Return physical voxel scale."""
        return self.metadata.scale

    def read_timepoint(self, time_index: int) -> NDArray[np.uint16]:
        """Load one timepoint without reading the full volume."""
        if time_index < 0 or time_index >= self.shape[0]:
            raise IndexError(f"time_index {time_index} out of range for shape {self.shape}")
        return np.asarray(self._array[time_index])

    def read_slice(
        self,
        time_slice: slice | int,
        z_slice: slice | int | None = None,
        y_slice: slice | int | None = None,
        x_slice: slice | int | None = None,
    ) -> NDArray[np.uint16]:
        """Load an arbitrary sub-volume."""
        keys: list[slice | int] = [time_slice]
        keys.extend(
            value if value is not None else slice(None)
            for value in (z_slice, y_slice, x_slice)
        )
        return np.asarray(self._array[tuple(keys)])

    def iter_timepoints(self) -> Iterator[NDArray[np.uint16]]:
        """Stream timepoints one at a time."""
        for time_index in range(self.shape[0]):
            yield self.read_timepoint(time_index)

    @staticmethod
    def normalize_volume(
        volume: NDArray[np.uint16],
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
    ) -> NDArray[np.float32]:
        """Quantile-normalize a volume to ``[0, 1]``."""
        low = np.quantile(volume, quantile_low)
        high = np.quantile(volume, quantile_high)
        if high <= low:
            return np.zeros(volume.shape, dtype=np.float32)
        normalized = (volume.astype(np.float32) - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0)

    @classmethod
    def from_json_metadata(cls, path: str | Path) -> ImageVolumeMetadata:
        """Parse metadata without opening the full array."""
        root_path = Path(path)
        metadata_path = root_path / "zarr.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        attrs = payload.get("attributes", {})
        multiscales = attrs.get("multiscales", [{}])[0]
        scale_values = multiscales.get("datasets", [{}])[0].get(
            "coordinateTransformations", [{}]
        )[0].get("scale", [1.0, 1.625, 0.40625, 0.40625])
        array_meta_path = root_path / "0" / "zarr.json"
        with array_meta_path.open("r", encoding="utf-8") as handle:
            array_payload = json.load(handle)
        return ImageVolumeMetadata(
            shape=tuple(int(v) for v in array_payload["shape"]),
            dtype=array_payload["data_type"],
            scale=VoxelScale(
                t=float(scale_values[0]),
                z=float(scale_values[1]),
                y=float(scale_values[2]),
                x=float(scale_values[3]),
            ),
            quantiles={
                str(k): float(v)
                for k, v in attrs.get("image_statistics", {}).get("quantiles", {}).items()
            },
        )
