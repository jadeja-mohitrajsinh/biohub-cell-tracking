"""Label generation from GEFF centroids for training."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from data.models import TrackingGraph, CellNode

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate training labels from GEFF ground truth centroids."""
    
    def __init__(
        self,
        center_radius: int = 2,
        periphery_radius: int = 4,
        background_threshold: float = 0.5,
    ) -> None:
        """Initialize label generator.
        
        Args:
            center_radius: Radius for center sphere (voxels)
            periphery_radius: Radius for periphery sphere (voxels)
            background_threshold: Distance threshold for background (voxels)
        """
        self.center_radius = center_radius
        self.periphery_radius = periphery_radius
        self.background_threshold = background_threshold
        
        logger.info(
            f"LabelGenerator initialized: center_radius={center_radius}, "
            f"periphery_radius={periphery_radius}"
        )
    
    def generate_labels(
        self,
        volume_shape: Tuple[int, int, int],
        centroids: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate label volume from centroids.
        
        Args:
            volume_shape: Shape of volume (Z, Y, X)
            centroids: List of (z, y, x) centroids
        
        Returns:
            Label array with values: 0=background, 1=center, 2=periphery
        """
        labels = np.zeros(volume_shape, dtype=np.int32)
        
        if not centroids:
            return labels
        
        # Create distance map from centroids
        centroid_mask = np.zeros(volume_shape, dtype=bool)
        for z, y, x in centroids:
            if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
                centroid_mask[z, y, x] = True
        
        # Compute distance transform
        distances = distance_transform_edt(~centroid_mask)
        
        # Assign labels based on distance
        for z, y, x in centroids:
            if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
                # Center sphere
                self._assign_sphere(labels, (z, y, x), self.center_radius, 1)
                
                # Periphery sphere (excluding center)
                self._assign_sphere(labels, (z, y, x), self.periphery_radius, 2, exclude_center=True)
        
        return labels
    
    def _assign_sphere(
        self,
        labels: np.ndarray,
        center: Tuple[int, int, int],
        radius: int,
        label_value: int,
        exclude_center: bool = False,
    ) -> None:
        """Assign label value to spherical region.
        
        Args:
            labels: Label array to modify
            center: Center of sphere (z, y, x)
            radius: Radius of sphere
            label_value: Label value to assign
            exclude_center: If True, don't assign to center region
        """
        z0, y0, x0 = center
        z_shape, y_shape, x_shape = labels.shape
        
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dz**2 + dy**2 + dx**2 <= radius**2:
                        z, y, x = z0 + dz, y0 + dy, x0 + dx
                        
                        # Check bounds
                        if 0 <= z < z_shape and 0 <= y < y_shape and 0 <= x < x_shape:
                            # Exclude center if requested
                            if exclude_center and dz**2 + dy**2 + dx**2 <= self.center_radius**2:
                                continue
                            
                            labels[z, y, x] = label_value
    
    def generate_labels_from_graph(
        self,
        volume_shape: Tuple[int, int, int],
        graph: TrackingGraph,
        timepoint: int,
    ) -> np.ndarray:
        """Generate labels from tracking graph for specific timepoint.
        
        Args:
            volume_shape: Shape of volume (Z, Y, X)
            graph: Tracking graph with ground truth
            timepoint: Timepoint to extract labels for
        
        Returns:
            Label array with values: 0=background, 1=center, 2=periphery
        """
        # Extract centroids for this timepoint
        centroids = []
        for node_id, node in graph.nodes.items():
            if node.t == timepoint:
                centroids.append((node.z, node.y, node.x))
        
        return self.generate_labels(volume_shape, centroids)
    
    def generate_soft_labels(
        self,
        volume_shape: Tuple[int, int, int],
        centroids: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate soft probability labels for training.
        
        Args:
            volume_shape: Shape of volume (Z, Y, X)
            centroids: List of (z, y, x) centroids
        
        Returns:
            Soft label array of shape (3, Z, Y, X) with probabilities
        """
        # Generate hard labels first
        hard_labels = self.generate_labels(volume_shape, centroids)
        
        # Convert to soft probabilities
        soft_labels = np.zeros((3, *volume_shape), dtype=np.float32)
        
        # Background channel (0)
        soft_labels[0] = (hard_labels == 0).astype(np.float32)
        
        # Center channel (1)
        soft_labels[1] = (hard_labels == 1).astype(np.float32)
        
        # Periphery channel (2)
        soft_labels[2] = (hard_labels == 2).astype(np.float32)
        
        # Apply Gaussian smoothing for soft boundaries
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            soft_labels[c] = gaussian_filter(soft_labels[c], sigma=1.0)
        
        # Normalize
        row_sums = soft_labels.sum(axis=0)
        row_sums[row_sums == 0] = 1.0
        soft_labels = soft_labels / row_sums
        
        return soft_labels


class WeightedLabelGenerator(LabelGenerator):
    """Label generator with class weighting for imbalanced data."""
    
    def __init__(
        self,
        center_radius: int = 2,
        periphery_radius: int = 4,
        background_threshold: float = 0.5,
        center_weight: float = 10.0,
        periphery_weight: float = 5.0,
        background_weight: float = 1.0,
    ) -> None:
        """Initialize weighted label generator.
        
        Args:
            center_radius: Radius for center sphere (voxels)
            periphery_radius: Radius for periphery sphere (voxels)
            background_threshold: Distance threshold for background (voxels)
            center_weight: Weight for center class
            periphery_weight: Weight for periphery class
            background_weight: Weight for background class
        """
        super().__init__(center_radius, periphery_radius, background_threshold)
        
        self.center_weight = center_weight
        self.periphery_weight = periphery_weight
        self.background_weight = background_weight
        
        logger.info(
            f"WeightedLabelGenerator initialized: "
            f"center_weight={center_weight}, periphery_weight={periphery_weight}"
        )
    
    def compute_class_weights(
        self,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Compute class weights from label distribution.
        
        Args:
            labels: Label array (Z, Y, X) with values 0, 1, 2
        
        Returns:
            Class weights array of shape (3,)
        """
        # Count pixels in each class
        class_counts = np.bincount(labels.flatten(), minlength=3)
        
        # Compute inverse frequency weights
        total_pixels = labels.size
        class_weights = total_pixels / (3 * class_counts + 1e-6)
        
        # Normalize
        class_weights = class_weights / class_weights.sum() * 3
        
        # Apply manual scaling
        class_weights[0] *= self.background_weight
        class_weights[1] *= self.center_weight
        class_weights[2] *= self.periphery_weight
        
        return class_weights
    
    def generate_weight_map(
        self,
        labels: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate weight map for weighted loss.
        
        Args:
            labels: Label array (Z, Y, X)
            class_weights: Optional class weights (3,)
        
        Returns:
            Weight map (Z, Y, X)
        """
        if class_weights is None:
            class_weights = self.compute_class_weights(labels)
        
        weight_map = np.zeros_like(labels, dtype=np.float32)
        
        # Assign class weights
        weight_map[labels == 0] = class_weights[0]
        weight_map[labels == 1] = class_weights[1]
        weight_map[labels == 2] = class_weights[2]
        
        return weight_map
