"""Submission generation utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

from data.geff_reader import GeffWriter
from data.models import TrackingGraph, VoxelScale
from utils.logging import get_logger

logger = get_logger(__name__)


class SubmissionGenerator:
    """Write competition GEFF submissions for one or more samples."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        scale: VoxelScale | None = None,
        overwrite: bool = True,
        zarr_format: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.scale = scale or VoxelScale()
        self.overwrite = overwrite
        self.zarr_format = zarr_format
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_sample(self, sample_id: str, graph: TrackingGraph) -> Path:
        """Write one GEFF submission for a sample."""
        target = self.output_dir / f"{sample_id}.geff"
        writer = GeffWriter(
            target,
            scale=self.scale,
            overwrite=self.overwrite,
            zarr_format=self.zarr_format,
        )
        path = writer.write(graph)
        logger.info("Wrote submission for %s to %s", sample_id, path)
        return path

    def write_batch(self, predictions: dict[str, TrackingGraph]) -> list[Path]:
        """Write multiple sample submissions."""
        return [self.write_sample(sample_id, graph) for sample_id, graph in predictions.items()]

    def package_directory(self, destination: str | Path, *, clean: bool = True) -> Path:
        """Copy the submission directory to a destination folder."""
        destination_path = Path(destination)
        if destination_path.exists() and clean:
            shutil.rmtree(destination_path)
        destination_path.mkdir(parents=True, exist_ok=True)
        for geff_path in self.output_dir.glob("*.geff"):
            target = destination_path / geff_path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(geff_path, target)
        logger.info("Packaged submissions into %s", destination_path)
        return destination_path
