"""Dataset discovery and sample loading."""

from __future__ import annotations

from pathlib import Path

from data.geff_reader import GeffReader
from data.models import EmbryoSample
from data.zarr_reader import ZarrVolumeReader


class DatasetCatalog:
    """Discover competition samples in a directory."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

    def list_samples(self) -> list[str]:
        """Return sorted sample IDs discovered under the data root."""
        sample_ids = sorted(path.name.removesuffix(".zarr") for path in self.data_root.glob("*.zarr"))
        return sample_ids

    def get_paths(self, sample_id: str) -> tuple[Path, Path | None]:
        """Return image and optional GEFF paths for a sample."""
        image_path = self.data_root / f"{sample_id}.zarr"
        tracks_path = self.data_root / f"{sample_id}.geff"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image store for sample {sample_id}: {image_path}")
        return image_path, tracks_path if tracks_path.exists() else None

    def load_sample(
        self,
        sample_id: str,
        *,
        require_tracks: bool = False,
        load_volume: bool = False,
    ) -> EmbryoSample:
        """Load metadata and optionally tracks for one sample."""
        image_path, tracks_path = self.get_paths(sample_id)
        if require_tracks and tracks_path is None:
            raise FileNotFoundError(f"Ground-truth tracks missing for sample {sample_id}")

        if load_volume:
            volume = ZarrVolumeReader(image_path)
            image_metadata = volume.metadata
        else:
            image_metadata = ZarrVolumeReader.from_json_metadata(image_path)

        tracks = None
        if tracks_path is not None:
            tracks = GeffReader(tracks_path).read_graph()

        return EmbryoSample(
            sample_id=sample_id,
            image_path=str(image_path),
            tracks_path=str(tracks_path) if tracks_path else None,
            image_metadata=image_metadata,
            tracks=tracks,
        )

    def load_all(
        self,
        *,
        require_tracks: bool = False,
        load_volume: bool = False,
    ) -> list[EmbryoSample]:
        """Load all discovered samples."""
        return [
            self.load_sample(
                sample_id,
                require_tracks=require_tracks,
                load_volume=load_volume,
            )
            for sample_id in self.list_samples()
        ]


class EmbryoDataset:
    """High-level dataset accessor with lazy volume loading."""

    def __init__(
        self,
        sample: EmbryoSample,
        normalize: bool = False,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
    ) -> None:
        self.sample = sample
        self.normalize = normalize
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self._reader: ZarrVolumeReader | None = None

    @property
    def reader(self) -> ZarrVolumeReader:
        """Lazily open the underlying Zarr reader."""
        if self._reader is None:
            self._reader = ZarrVolumeReader(self.sample.image_path)
        return self._reader

    def get_timepoint(self, time_index: int):
        """Load one timepoint, optionally normalized."""
        import numpy as np

        volume = self.reader.read_timepoint(time_index)
        if self.normalize:
            return ZarrVolumeReader.normalize_volume(
                volume,
                quantile_low=self.quantile_low,
                quantile_high=self.quantile_high,
            )
        return volume

    @property
    def tracks(self):
        """Return ground-truth tracks when available."""
        return self.sample.tracks
