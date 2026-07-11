"""Per-embryo normalization and analysis for robust tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from data.dataset import EmbryoDataset, EmbryoSample
from data.models import VoxelScale

logger = logging.getLogger(__name__)


@dataclass
class EmbryoStatistics:
    """Statistics for a single embryo."""
    sample_id: str
    mean_intensity: float
    std_intensity: float
    median_intensity: float
    estimated_cell_count: int
    mean_displacement_um: float
    std_displacement_um: float
    cell_density_per_um3: float
    num_divisions: int
    volume_shape: Tuple[int, int, int, int]


class EmbryoAnalyzer:
    """Analyze embryo statistics for adaptive tracking."""
    
    def __init__(self, scale: VoxelScale) -> None:
        """Initialize embryo analyzer.
        
        Args:
            scale: VoxelScale for physical calculations
        """
        self.scale = scale
        logger.info("EmbryoAnalyzer initialized")
    
    def analyze_sample(
        self,
        sample: EmbryoSample,
        num_timepoints_to_sample: int = 10,
    ) -> EmbryoStatistics:
        """Analyze a single embryo sample.
        
        Args:
            sample: EmbryoSample to analyze
            num_timepoints_to_sample: Number of timepoints to sample for statistics
        
        Returns:
            EmbryoStatistics
        """
        dataset = EmbryoDataset(sample, normalize=False)
        
        total_timepoints = sample.image_metadata.shape[0]
        sample_indices = np.linspace(0, total_timepoints - 1, num_timepoints_to_sample, dtype.int)
        
        intensities = []
        displacements = []
        
        # Sample timepoints for intensity statistics
        for t in sample_indices:
            volume = dataset.get_timepoint(t)
            intensities.extend(volume.flatten())
        
        # Calculate intensity statistics
        intensities = np.array(intensities)
        mean_intensity = float(np.mean(intensities))
        std_intensity = float(np.std(intensities))
        median_intensity = float(np.median(intensities))
        
        # Estimate cell count from ground truth if available
        if sample.tracks:
            estimated_cell_count = sample.tracks.num_nodes
            num_divisions = len(sample.tracks.division_nodes())
        else:
            estimated_cell_count = 0
            num_divisions = 0
        
        # Calculate displacement statistics from ground truth
        if sample.tracks:
            displacements = self._calculate_displacements_from_tracks(sample.tracks)
            if displacements:
                mean_displacement_um = float(np.mean(displacements))
                std_displacement_um = float(np.std(displacements))
            else:
                mean_displacement_um = 2.0  # Default
                std_displacement_um = 1.0
        else:
            mean_displacement_um = 2.0  # Default
            std_displacement_um = 1.0
        
        # Calculate cell density
        volume_shape = sample.image_metadata.shape
        physical_volume = (
            volume_shape[0] * self.scale.t *  # Time (not used for density)
            volume_shape[1] * self.scale.z *
            volume_shape[2] * self.scale.y *
            volume_shape[3] * self.scale.x
        )
        # Use spatial volume only (Z, Y, X)
        spatial_volume = (
            volume_shape[1] * self.scale.z *
            volume_shape[2] * self.scale.y *
            volume_shape[3] * self.scale.x
        )
        
        if estimated_cell_count > 0 and spatial_volume > 0:
            cell_density_per_um3 = estimated_cell_count / spatial_volume
        else:
            cell_density_per_um3 = 0.0
        
        return EmbryoStatistics(
            sample_id=sample.sample_id,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            median_intensity=median_intensity,
            estimated_cell_count=estimated_cell_count,
            mean_displacement_um=mean_displacement_um,
            std_displacement_um=std_displacement_um,
            cell_density_per_um3=cell_density_per_um3,
            num_divisions=num_divisions,
            volume_shape=volume_shape,
        )
    
    def _calculate_displacements_from_tracks(
        self,
        tracks,
    ) -> List[float]:
        """Calculate displacements from tracking graph.
        
        Args:
            tracks: TrackingGraph
        
        Returns:
            List of displacements in micrometers
        """
        displacements = []
        
        for edge in tracks.edges:
            source = tracks.nodes.get(edge.source_id)
            target = tracks.nodes.get(edge.target_id)
            
            if source and target:
                dz = (target.z - source.z) * self.scale.z
                dy = (target.y - source.y) * self.scale.y
                dx = (target.x - source.x) * self.scale.x
                displacement = np.sqrt(dz**2 + dy**2 + dx**2)
                displacements.append(displacement)
        
        return displacements
    
    def analyze_multiple_samples(
        self,
        samples: List[EmbryoSample],
    ) -> Dict[str, EmbryoStatistics]:
        """Analyze multiple embryo samples.
        
        Args:
            samples: List of EmbryoSample objects
        
        Returns:
            Dict mapping sample_id to EmbryoStatistics
        """
        statistics = {}
        
        for sample in samples:
            stats = self.analyze_sample(sample)
            statistics[sample.sample_id] = stats
            logger.info(
                f"Analyzed {sample.sample_id}: "
                f"mean_intensity={stats.mean_intensity:.2f}, "
                f"estimated_cells={stats.estimated_cell_count}, "
                f"mean_displacement={stats.mean_displacement_um:.3f}"
            )
        
        return statistics


class AdaptiveNormalizer:
    """Per-embryo adaptive normalization for robust tracking."""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize adaptive normalizer.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        
        self.method = config.get("method", "percentile")
        self.percentile_low = config.get("percentile_low", 1.0)
        self.percentile_high = config.get("percentile_high", 99.0)
        self.clip_outliers = config.get("clip_outliers", True)
        self.use_robust_scale = config.get("use_robust_scale", True)
        
        logger.info(
            f"AdaptiveNormalizer initialized: method={self.method}, "
            f"percentile_range=({self.percentile_low}, {self.percentile_high})"
        )
    
    def normalize_volume(
        self,
        volume: np.ndarray,
        stats: Optional[EmbryoStatistics] = None,
    ) -> np.ndarray:
        """Normalize a single volume adaptively.
        
        Args:
            volume: 3D volume (Z, Y, X)
            stats: Optional embryo statistics for adaptive normalization
        
        Returns:
            Normalized volume
        """
        if self.method == "percentile":
            return self._percentile_normalize(volume)
        elif self.method == "zscore":
            return self._zscore_normalize(volume, stats)
        elif self.method == "minmax":
            return self._minmax_normalize(volume)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _percentile_normalize(
        self,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Normalize using percentile clipping.
        
        Args:
            volume: 3D volume
        
        Returns:
            Normalized volume
        """
        if volume.max() == 0:
            return volume
        
        # Calculate percentiles
        p_low = np.percentile(volume, self.percentile_low)
        p_high = np.percentile(volume, self.percentile_high)
        
        # Clip
        if self.clip_outliers:
            volume = np.clip(volume, p_low, p_high)
        
        # Normalize to [0, 1]
        if p_high > p_low:
            volume = (volume - p_low) / (p_high - p_low)
        else:
            volume = volume - p_low
        
        return volume
    
    def _zscore_normalize(
        self,
        volume: np.ndarray,
        stats: Optional[EmbryoStatistics] = None,
    ) -> np.ndarray:
        """Normalize using z-score.
        
        Args:
            volume: 3D volume
            stats: Optional embryo statistics
        
        Returns:
            Normalized volume
        """
        if stats and self.use_robust_scale:
            # Use robust statistics from embryo analysis
            mean = stats.median_intensity
            std = stats.std_intensity
        else:
            mean = np.mean(volume)
            std = np.std(volume)
        
        if std == 0:
            return volume - mean
        
        volume = (volume - mean) / std
        
        # Clip to reasonable range
        volume = np.clip(volume, -3, 3)
        
        # Rescale to [0, 1]
        volume = (volume + 3) / 6
        
        return volume
    
    def _minmax_normalize(
        self,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Normalize using min-max scaling.
        
        Args:
            volume: 3D volume
        
        Returns:
            Normalized volume
        """
        if volume.max() == volume.min():
            return np.zeros_like(volume)
        
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume


class PerEmbryoConfigurator:
    """Configure tracking parameters per embryo based on analysis."""
    
    def __init__(self, scale: VoxelScale) -> None:
        """Initialize per-embryo configurator.
        
        Args:
            scale: VoxelScale
        """
        self.scale = scale
        self.analyzer = EmbryoAnalyzer(scale)
        logger.info("PerEmbryoConfigurator initialized")
    
    def configure_for_sample(
        self,
        sample: EmbryoSample,
        base_config: Dict,
    ) -> Dict:
        """Generate adaptive configuration for a specific embryo.
        
        Args:
            sample: EmbryoSample
            base_config: Base configuration dictionary
        
        Returns:
            Adapted configuration for this embryo
        """
        # Analyze embryo
        stats = self.analyzer.analyze_sample(sample)
        
        # Create adapted config
        adapted_config = base_config.copy()
        
        # Adapt max distance based on displacement statistics
        if "tracking" in adapted_config:
            tracking_config = adapted_config["tracking"].copy()
            
            # Adaptive max distance: mean + 2*std
            adaptive_max_distance = stats.mean_displacement_um + 2 * stats.std_displacement_um
            tracking_config["max_distance_um"] = min(
                max(adaptive_max_distance, 3.0),  # Minimum 3um
                10.0  # Maximum 10um
            )
            
            adapted_config["tracking"] = tracking_config
        
        # Adapt detection threshold based on intensity statistics
        if "detection" in adapted_config:
            detection_config = adapted_config["detection"].copy()
            
            # Higher intensity embryos may need higher thresholds
            if stats.mean_intensity > 1000:
                detection_config["base_threshold"] = detection_config.get("base_threshold", 0.05) * 1.2
            
            adapted_config["detection"] = detection_config
        
        # Add statistics to config for reference
        adapted_config["embryo_statistics"] = {
            "mean_intensity": stats.mean_intensity,
            "estimated_cell_count": stats.estimated_cell_count,
            "mean_displacement_um": stats.mean_displacement_um,
            "cell_density": stats.cell_density_per_um3,
        }
        
        logger.info(
            f"Adapted config for {sample.sample_id}: "
            f"max_distance={adapted_config['tracking']['max_distance_um']:.2f}, "
            f"cells={stats.estimated_cell_count}"
        )
        
        return adapted_config
    
    def configure_for_multiple_samples(
        self,
        samples: List[EmbryoSample],
        base_config: Dict,
    ) -> Dict[str, Dict]:
        """Generate adaptive configurations for multiple embryos.
        
        Args:
            samples: List of EmbryoSample
            base_config: Base configuration
        
        Returns:
            Dict mapping sample_id to adapted configuration
        """
        configs = {}
        
        for sample in samples:
            configs[sample.sample_id] = self.configure_for_sample(sample, base_config)
        
        return configs
