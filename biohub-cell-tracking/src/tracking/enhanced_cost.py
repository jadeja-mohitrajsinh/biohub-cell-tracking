"""Enhanced cost function for cell tracking with multiple features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass

from data.models import VoxelScale

logger = logging.getLogger(__name__)


@dataclass
class DetectionFeatures:
    """Features for a single detection."""
    position: Tuple[int, int, int]  # (z, y, x)
    confidence: float
    intensity: float
    size: float
    neighborhood_density: float
    timepoint: int
    # Shape features
    eccentricity: float
    elongation: float
    orientation: float  # in radians


@dataclass
class CostFeatures:
    """Features for computing linking cost between two detections."""
    distance_um: float
    confidence_product: float
    confidence_ratio: float
    intensity_similarity: float
    size_similarity: float
    displacement_z: float
    displacement_xy: float
    neighborhood_similarity: float
    temporal_gap: int
    # Shape similarity features
    eccentricity_similarity: float
    elongation_similarity: float
    orientation_change: float


class EnhancedCostFunction:
    """Enhanced cost function using multiple features for cell linking."""
    
    def __init__(
        self,
        scale: VoxelScale,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize enhanced cost function.
        
        Args:
            scale: VoxelScale for physical distance calculation
            config: Configuration dictionary with feature weights
        """
        self.scale = scale
        
        if config is None:
            config = {}
        
        # Feature weights
        self.distance_weight = config.get("distance_weight", 1.0)
        self.confidence_weight = config.get("confidence_weight", 0.3)
        self.intensity_weight = config.get("intensity_weight", 0.2)
        self.size_weight = config.get("size_weight", 0.1)
        self.neighborhood_weight = config.get("neighborhood_weight", 0.1)
        self.displacement_weight = config.get("displacement_weight", 0.2)
        self.shape_weight = config.get("shape_weight", 0.15)
        
        # Normalization parameters
        self.max_distance_um = config.get("max_distance_um", 7.0)
        self.max_temporal_gap = config.get("max_temporal_gap", 2)
        
        logger.info(
            f"EnhancedCostFunction initialized: distance_weight={self.distance_weight}, "
            f"confidence_weight={self.confidence_weight}, "
            f"intensity_weight={self.intensity_weight}"
        )
    
    def extract_detection_features(
        self,
        volume: np.ndarray,
        detections: List[Tuple[int, int, int]],
        confidences: List[float],
        timepoint: int,
        neighborhood_radius: int = 5,
    ) -> List[DetectionFeatures]:
        """Extract features for detections.
        
        Args:
            volume: 3D volume (Z, Y, X)
            detections: List of (z, y, x) centroids
            confidences: List of confidence scores
            timepoint: Timepoint index
            neighborhood_radius: Radius for neighborhood density calculation
        
        Returns:
            List of DetectionFeatures
        """
        features = []
        
        for (z, y, x), conf in zip(detections, confidences):
            # Extract local intensity
            intensity = self._extract_local_intensity(volume, (z, y, x), radius=3)
            
            # Estimate cell size
            size = self._estimate_cell_size(volume, (z, y, x))
            
            # Compute neighborhood density
            density = self._compute_neighborhood_density(
                detections, (z, y, x), neighborhood_radius
            )
            
            # Extract shape features
            eccentricity, elongation, orientation = self._extract_shape_features(
                volume, (z, y, x)
            )
            
            features.append(DetectionFeatures(
                position=(z, y, x),
                confidence=conf,
                intensity=intensity,
                size=size,
                neighborhood_density=density,
                timepoint=timepoint,
                eccentricity=eccentricity,
                elongation=elongation,
                orientation=orientation,
            ))
        
        return features
    
    def _extract_local_intensity(
        self,
        volume: np.ndarray,
        position: Tuple[int, int, int],
        radius: int = 3,
    ) -> float:
        """Extract local intensity around position.
        
        Args:
            volume: 3D volume
            position: (z, y, x) position
            radius: Extraction radius
        
        Returns:
            Mean intensity in local region
        """
        z, y, x = position
        z_shape, y_shape, x_shape = volume.shape
        
        z_start = max(0, z - radius)
        z_end = min(z_shape, z + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(y_shape, y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(x_shape, x + radius + 1)
        
        local_region = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if local_region.size == 0:
            return 0.0
        
        return float(local_region.mean())
    
    def _estimate_cell_size(
        self,
        volume: np.ndarray,
        position: Tuple[int, int, int],
        threshold_factor: float = 0.5,
    ) -> float:
        """Estimate cell size from local region.
        
        Args:
            volume: 3D volume
            position: (z, y, x) position
            threshold_factor: Threshold for size estimation
        
        Returns:
            Estimated cell size (voxels)
        """
        z, y, x = position
        intensity = volume[z, y, x]
        
        if intensity == 0:
            return 1.0
        
        threshold = intensity * threshold_factor
        radius = 5
        
        z_shape, y_shape, x_shape = volume.shape
        z_start = max(0, z - radius)
        z_end = min(z_shape, z + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(y_shape, y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(x_shape, x + radius + 1)
        
        local_region = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        size = (local_region > threshold).sum()
        
        return float(max(1.0, size))
    
    def _compute_neighborhood_density(
        self,
        detections: List[Tuple[int, int, int]],
        position: Tuple[int, int, int],
        radius: int,
    ) -> float:
        """Compute neighborhood density of detections.
        
        Args:
            detections: List of all detections
            position: Reference position
            radius: Neighborhood radius
        
        Returns:
            Density (detections per unit volume)
        """
        z0, y0, x0 = position
        
        count = 0
        for z, y, x in detections:
            dist = np.sqrt((z - z0)**2 + (y - y0)**2 + (x - x0)**2)
            if dist <= radius:
                count += 1
        
        # Density per unit volume (sphere volume)
        volume = (4/3) * np.pi * (radius ** 3)
        density = count / volume if volume > 0 else 0.0
        
        return density
    
    def _extract_shape_features(
        self,
        volume: np.ndarray,
        position: Tuple[int, int, int],
        radius: int = 5,
    ) -> Tuple[float, float, float]:
        """Extract shape features (eccentricity, elongation, orientation).
        
        Args:
            volume: 3D volume
            position: (z, y, x) position
            radius: Extraction radius
        
        Returns:
            Tuple of (eccentricity, elongation, orientation in radians)
        """
        z, y, x = position
        z_shape, y_shape, x_shape = volume.shape
        
        z_start = max(0, z - radius)
        z_end = min(z_shape, z + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(y_shape, y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(x_shape, x + radius + 1)
        
        local_region = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if local_region.size == 0:
            return 0.0, 1.0, 0.0
        
        # Threshold to get binary mask
        threshold = local_region.max() * 0.5
        binary = local_region > threshold
        
        if binary.sum() < 3:
            return 0.0, 1.0, 0.0
        
        # Get coordinates of above-threshold voxels
        coords = np.argwhere(binary)
        
        if len(coords) < 3:
            return 0.0, 1.0, 0.0
        
        # Compute covariance matrix
        coords_centered = coords - coords.mean(axis=0)
        cov = np.cov(coords_centered.T)
        
        # Eigenvalues indicate shape
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Eccentricity: ratio of largest to smallest eigenvalue
        if eigenvalues[-1] > 1e-6:
            eccentricity = eigenvalues[0] / eigenvalues[-1]
        else:
            eccentricity = 1.0
        
        # Elongation: ratio of largest to middle eigenvalue
        if eigenvalues[1] > 1e-6:
            elongation = eigenvalues[0] / eigenvalues[1]
        else:
            elongation = 1.0
        
        # Orientation: angle of principal eigenvector in XY plane
        principal_vec = eigenvectors[:, 0]
        orientation = np.arctan2(principal_vec[1], principal_vec[2])  # Y, X components
        
        return float(eccentricity), float(elongation), float(orientation)
    
    def compute_cost_features(
        self,
        feat1: DetectionFeatures,
        feat2: DetectionFeatures,
    ) -> CostFeatures:
        """Compute cost features between two detections.
        
        Args:
            feat1: Features of first detection
            feat2: Features of second detection
        
        Returns:
            CostFeatures
        """
        # Physical distance
        pos1 = (feat1.timepoint, *feat1.position)
        pos2 = (feat2.timepoint, *feat2.position)
        distance_um = self._physical_distance(pos1, pos2)
        
        # Confidence features
        confidence_product = feat1.confidence * feat2.confidence
        confidence_ratio = min(feat1.confidence, feat2.confidence) / (max(feat1.confidence, feat2.confidence) + 1e-6)
        
        # Intensity similarity
        intensity_similarity = 1.0 - abs(feat1.intensity - feat2.intensity) / (feat1.intensity + feat2.intensity + 1e-6)
        
        # Size similarity
        size_similarity = 1.0 - abs(feat1.size - feat2.size) / (feat1.size + feat2.size + 1e-6)
        
        # Displacement components
        dz = feat2.position[0] - feat1.position[0]
        dy = feat2.position[1] - feat1.position[1]
        dx = feat2.position[2] - feat1.position[2]
        
        displacement_z = abs(dz * self.scale.z)
        displacement_xy = np.sqrt((dy * self.scale.y)**2 + (dx * self.scale.x)**2)
        
        # Neighborhood similarity
        neighborhood_similarity = 1.0 - abs(feat1.neighborhood_density - feat2.neighborhood_density) / (
            feat1.neighborhood_density + feat2.neighborhood_density + 1e-6
        )
        
        # Temporal gap
        temporal_gap = feat2.timepoint - feat1.timepoint
        
        # Shape similarity features
        eccentricity_similarity = 1.0 - abs(feat1.eccentricity - feat2.eccentricity) / (
            feat1.eccentricity + feat2.eccentricity + 1e-6
        )
        elongation_similarity = 1.0 - abs(feat1.elongation - feat2.elongation) / (
            feat1.elongation + feat2.elongation + 1e-6
        )
        
        # Orientation change (handle circular nature of angles)
        orientation_diff = abs(feat2.orientation - feat1.orientation)
        orientation_diff = min(orientation_diff, 2 * np.pi - orientation_diff)
        orientation_change = orientation_diff / np.pi  # Normalize to [0, 1]
        
        return CostFeatures(
            distance_um=distance_um,
            confidence_product=confidence_product,
            confidence_ratio=confidence_ratio,
            intensity_similarity=intensity_similarity,
            size_similarity=size_similarity,
            displacement_z=displacement_z,
            displacement_xy=displacement_xy,
            neighborhood_similarity=neighborhood_similarity,
            temporal_gap=temporal_gap,
            eccentricity_similarity=eccentricity_similarity,
            elongation_similarity=elongation_similarity,
            orientation_change=orientation_change,
        )
    
    def _physical_distance(
        self,
        pos1: Tuple[int, int, int, int],
        pos2: Tuple[int, int, int, int],
    ) -> float:
        """Compute physical distance between two positions.
        
        Args:
            pos1: (t, z, y, x)
            pos2: (t, z, y, x)
        
        Returns:
            Physical distance in micrometers
        """
        t1, z1, y1, x1 = pos1
        t2, z2, y2, x2 = pos2
        
        dz = (z2 - z1) * self.scale.z
        dy = (y2 - y1) * self.scale.y
        dx = (x2 - x1) * self.scale.x
        
        return np.sqrt(dz**2 + dy**2 + dx**2)
    
    def compute_cost(
        self,
        features: CostFeatures,
    ) -> float:
        """Compute final cost from features.
        
        Args:
            features: CostFeatures
        
        Returns:
            Final cost (lower is better)
        """
        # Distance cost (normalized)
        distance_cost = features.distance_um / self.max_distance_um
        
        # Confidence cost (higher is better, so invert)
        confidence_cost = 1.0 - features.confidence_product
        
        # Intensity cost (higher similarity is better, so invert)
        intensity_cost = 1.0 - features.intensity_similarity
        
        # Size cost (higher similarity is better, so invert)
        size_cost = 1.0 - features.size_similarity
        
        # Displacement cost (penalize large Z displacement more)
        displacement_cost = (features.displacement_z / (self.scale.z * 5.0) + 
                           features.displacement_xy / (self.scale.y * 10.0))
        
        # Neighborhood cost (higher similarity is better, so invert)
        neighborhood_cost = 1.0 - features.neighborhood_similarity
        
        # Temporal gap cost
        temporal_cost = features.temporal_gap / self.max_temporal_gap
        
        # Shape similarity costs (higher similarity is better, so invert)
        eccentricity_cost = 1.0 - features.eccentricity_similarity
        elongation_cost = 1.0 - features.elongation_similarity
        orientation_cost = features.orientation_change  # Lower change is better
        
        # Combine with weights
        total_cost = (
            self.distance_weight * distance_cost +
            self.confidence_weight * confidence_cost +
            self.intensity_weight * intensity_cost +
            self.size_weight * size_cost +
            self.displacement_weight * displacement_cost +
            self.neighborhood_weight * neighborhood_cost +
            0.1 * temporal_cost +
            self.shape_weight * (eccentricity_cost + elongation_cost + orientation_cost) / 3.0
        )
        
        return total_cost
    
    def build_cost_matrix(
        self,
        features_t: List[DetectionFeatures],
        features_t1: List[DetectionFeatures],
    ) -> np.ndarray:
        """Build cost matrix for Hungarian matching.
        
        Args:
            features_t: Features at time t
            features_t1: Features at time t+1
        
        Returns:
            Cost matrix (N_t × N_t+1)
        """
        n_t = len(features_t)
        n_t1 = len(features_t1)
        
        # Initialize with large cost
        cost_matrix = np.full((n_t, n_t1), self.max_distance_um * 10.0, dtype=np.float64)
        
        for i, feat_t in enumerate(features_t):
            for j, feat_t1 in enumerate(features_t1):
                # Compute cost features
                cost_features = self.compute_cost_features(feat_t, feat_t1)
                
                # Check temporal gap
                if cost_features.temporal_gap > self.max_temporal_gap:
                    continue
                
                # Check distance
                if cost_features.distance_um > self.max_distance_um:
                    continue
                
                # Compute cost
                cost = self.compute_cost(cost_features)
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def get_adaptive_max_distance(self) -> float:
        """Get adaptive maximum distance. For base class, this is static max_distance_um."""
        return self.max_distance_um
        
    def update_motion_statistics(self, displacements: List[float]) -> None:
        """Update motion statistics. No-op for base class."""
        pass


class AdaptiveCostFunction(EnhancedCostFunction):
    """Cost function with adaptive parameters based on motion statistics."""
    
    def __init__(
        self,
        scale: VoxelScale,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize adaptive cost function.
        
        Args:
            scale: VoxelScale for physical distance calculation
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        super().__init__(scale, config)
        
        # Motion statistics (learned from data)
        self.mean_displacement_um = config.get("mean_displacement_um", 2.0)
        self.std_displacement_um = config.get("std_displacement_um", 1.0)
        
        # Adaptive radius
        self.adaptive_radius = config.get("adaptive_radius", True)
        self.radius_multiplier = config.get("radius_multiplier", 2.0)
        
        logger.info(
            f"AdaptiveCostFunction initialized: "
            f"mean_displacement={self.mean_displacement_um}, "
            f"adaptive_radius={self.adaptive_radius}"
        )
    
    def update_motion_statistics(
        self,
        displacements: List[float],
    ) -> None:
        """Update motion statistics from observed displacements.
        
        Args:
            displacements: List of observed displacements (µm)
        """
        if displacements:
            self.mean_displacement_um = np.mean(displacements)
            self.std_displacement_um = np.std(displacements)
            logger.info(
                f"Updated motion statistics: mean={self.mean_displacement_um:.3f}, "
                f"std={self.std_displacement_um:.3f}"
            )
    
    def get_adaptive_max_distance(self) -> float:
        """Get adaptive maximum distance based on motion statistics.
        
        Returns:
            Adaptive maximum distance
        """
        if self.adaptive_radius:
            return self.mean_displacement_um + self.radius_multiplier * self.std_displacement_um
        else:
            return self.max_distance_um
    
    def compute_cost(
        self,
        features: CostFeatures,
    ) -> float:
        """Compute final cost with adaptive normalization.
        
        Args:
            features: CostFeatures
        
        Returns:
            Final cost
        """
        # Use adaptive max distance
        adaptive_max_distance = self.get_adaptive_max_distance()
        
        # Distance cost (normalized adaptively, with safety check)
        if adaptive_max_distance > 1e-6:
            distance_cost = features.distance_um / adaptive_max_distance
        else:
            distance_cost = features.distance_um / self.max_distance_um
        
        # Confidence cost
        confidence_cost = 1.0 - features.confidence_product
        
        # Intensity cost
        intensity_cost = 1.0 - features.intensity_similarity
        
        # Size cost
        size_cost = 1.0 - features.size_similarity
        
        # Displacement cost (normalized by learned statistics)
        displacement_cost = (features.displacement_z / (self.mean_displacement_um + 1e-6) + 
                           features.displacement_xy / (self.mean_displacement_um + 1e-6)) / 2.0
        
        # Neighborhood cost
        neighborhood_cost = 1.0 - features.neighborhood_similarity
        
        # Temporal cost
        temporal_cost = features.temporal_gap / self.max_temporal_gap
        
        # Combine with weights
        total_cost = (
            self.distance_weight * distance_cost +
            self.confidence_weight * confidence_cost +
            self.intensity_weight * intensity_cost +
            self.size_weight * size_cost +
            self.displacement_weight * displacement_cost +
            self.neighborhood_weight * neighborhood_cost +
            0.1 * temporal_cost
        )
        
        return total_cost
