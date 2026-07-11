"""Anisotropic-aware augmentation pipeline for 3D microscopy data."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.ndimage import map_coordinates, gaussian_filter

logger = logging.getLogger(__name__)


class AnisotropicAugmentation:
    """Augmentation pipeline that respects anisotropic voxel spacing."""
    
    def __init__(
        self,
        scale: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
        xy_rotation_range: float = 15.0,
        z_flip_prob: float = 0.5,
        xy_flip_prob: float = 0.5,
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 5.0,
        intensity_contrast_range: Tuple[float, float] = (0.8, 1.2),
        intensity_brightness_range: Tuple[float, float] = (-0.1, 0.1),
        noise_std: float = 0.02,
        blur_sigma_range: Tuple[float, float] = (0.5, 1.5),
        z_scale_range: Tuple[float, float] = (0.9, 1.1),
    ) -> None:
        """Initialize anisotropic augmentation pipeline.
        
        Args:
            scale: Physical scale (Z, Y, X) in micrometers per voxel
            xy_rotation_range: Maximum rotation angle in XY plane (degrees)
            z_flip_prob: Probability of flipping along Z axis
            xy_flip_prob: Probability of flipping along Y or X axis
            elastic_alpha: Alpha parameter for elastic deformation
            elastic_sigma: Sigma parameter for elastic deformation
            intensity_contrast_range: Range for contrast multiplication
            intensity_brightness_range: Range for brightness addition
            noise_std: Standard deviation for Gaussian noise
            blur_sigma_range: Range for Gaussian blur sigma
            z_scale_range: Range for Z-axis scaling (simulating focus variation)
        """
        self.scale = scale
        self.xy_rotation_range = xy_rotation_range
        self.z_flip_prob = z_flip_prob
        self.xy_flip_prob = xy_flip_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.intensity_contrast_range = intensity_contrast_range
        self.intensity_brightness_range = intensity_brightness_range
        self.noise_std = noise_std
        self.blur_sigma_range = blur_sigma_range
        self.z_scale_range = z_scale_range
        
        logger.info(
            f"AnisotropicAugmentation initialized: scale={scale}, "
            f"xy_rotation_range={xy_rotation_range}"
        )
    
    def __call__(
        self,
        volume: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation to volume and optional labels.
        
        Args:
            volume: 3D volume (Z, Y, X)
            labels: Optional 3D labels (Z, Y, X)
        
        Returns:
            Augmented volume and labels
        """
        # Normalize volume to 0-1 if needed
        if volume.max() > 1.0:
            volume = volume / volume.max()
        
        # Apply augmentations
        volume = self._apply_intensity_augmentation(volume)
        
        if labels is None:
            # Only geometric augmentations if no labels
            volume = self._apply_geometric_augmentation(volume, None)
            return volume, None
        else:
            # Apply same geometric transforms to both
            volume, labels = self._apply_geometric_augmentation(volume, labels)
            return volume, labels
    
    def _apply_intensity_augmentation(self, volume: np.ndarray) -> np.ndarray:
        """Apply intensity-based augmentations.
        
        Args:
            volume: 3D volume (Z, Y, X)
        
        Returns:
            Augmented volume
        """
        # Contrast adjustment
        contrast = np.random.uniform(*self.intensity_contrast_range)
        volume = volume * contrast
        
        # Brightness adjustment
        brightness = np.random.uniform(*self.intensity_brightness_range)
        volume = volume + brightness
        
        # Clip to valid range
        volume = np.clip(volume, 0.0, 1.0)
        
        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0.0, 1.0)
        
        # Gaussian blur (anisotropic: more blur in Z)
        if self.blur_sigma_range[0] > 0:
            sigma_z = np.random.uniform(*self.blur_sigma_range) * (self.scale[0] / self.scale[1])
            sigma_xy = np.random.uniform(*self.blur_sigma_range)
            volume = gaussian_filter(volume, sigma=[sigma_z, sigma_xy, sigma_xy])
        
        return volume
    
    def _apply_geometric_augmentation(
        self,
        volume: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply geometric augmentations respecting anisotropy.
        
        Args:
            volume: 3D volume (Z, Y, X)
            labels: Optional 3D labels (Z, Y, X)
        
        Returns:
            Augmented volume and labels
        """
        # Z-axis flip (preserve anisotropy)
        if np.random.random() < self.z_flip_prob:
            volume = volume[::-1, :, :]
            if labels is not None:
                labels = labels[::-1, :, :]
        
        # Y-axis flip
        if np.random.random() < self.xy_flip_prob:
            volume = volume[:, ::-1, :]
            if labels is not None:
                labels = labels[:, ::-1, :]
        
        # X-axis flip
        if np.random.random() < self.xy_flip_prob:
            volume = volume[:, :, ::-1]
            if labels is not None:
                labels = labels[:, :, ::-1]
        
        # XY plane rotation (preserve Z anisotropy)
        if self.xy_rotation_range > 0:
            angle = np.random.uniform(-self.xy_rotation_range, self.xy_rotation_range)
            volume = self._rotate_xy(volume, angle)
            if labels is not None:
                labels = self._rotate_xy(labels, angle, order=0)  # Nearest neighbor for labels
        
        # Elastic deformation (anisotropic: less deformation in Z)
        if self.elastic_alpha > 0:
            volume, labels = self._elastic_deformation(volume, labels)
        
        # Z-axis scaling (simulate focus variation)
        if self.z_scale_range[0] != 1.0 or self.z_scale_range[1] != 1.0:
            scale = np.random.uniform(*self.z_scale_range)
            volume, labels = self._scale_z(volume, labels, scale)
        
        return volume, labels
    
    def _rotate_xy(
        self,
        array: np.ndarray,
        angle: float,
        order: int = 1,
    ) -> np.ndarray:
        """Rotate array in XY plane.
        
        Args:
            array: 3D array (Z, Y, X)
            angle: Rotation angle in degrees
            order: Interpolation order
        
        Returns:
            Rotated array
        """
        from scipy.ndimage import rotate
        
        # Rotate each Z slice independently
        rotated = np.zeros_like(array)
        for z in range(array.shape[0]):
            rotated[z] = rotate(array[z], angle, reshape=False, order=order, mode='reflect')
        
        return rotated
    
    def _elastic_deformation(
        self,
        volume: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply elastic deformation with anisotropic scaling.
        
        Args:
            volume: 3D volume (Z, Y, X)
            labels: Optional 3D labels
        
        Returns:
            Deformed volume and labels
        """
        shape = volume.shape
        
        # Generate displacement field (anisotropic: less displacement in Z)
        dz = gaussian_filter(
            np.random.randn(*shape) * self.elastic_alpha,
            sigma=self.elastic_sigma * (self.scale[0] / self.scale[1])
        )
        dy = gaussian_filter(
            np.random.randn(*shape) * self.elastic_alpha,
            sigma=self.elastic_sigma
        )
        dx = gaussian_filter(
            np.random.randn(*shape) * self.elastic_alpha,
            sigma=self.elastic_sigma
        )
        
        # Create coordinate grids
        z_coords, y_coords, x_coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
        
        # Apply displacement
        z_coords_def = z_coords + dz
        y_coords_def = y_coords + dy
        x_coords_def = x_coords + dx
        
        # Interpolate
        volume_def = map_coordinates(
            volume,
            [z_coords_def, y_coords_def, x_coords_def],
            order=1,
            mode='reflect'
        )
        
        if labels is not None:
            labels_def = map_coordinates(
                labels,
                [z_coords_def, y_coords_def, x_coords_def],
                order=0,  # Nearest neighbor for labels
                mode='nearest'
            )
        else:
            labels_def = None
        
        return volume_def, labels_def
    
    def _scale_z(
        self,
        volume: np.ndarray,
        labels: Optional[np.ndarray],
        scale: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Scale along Z axis.
        
        Args:
            volume: 3D volume (Z, Y, X)
            labels: Optional 3D labels
            scale: Scaling factor for Z axis
        
        Returns:
            Scaled volume and labels
        """
        from scipy.ndimage import zoom
        
        # Scale only Z dimension
        zoom_factors = (scale, 1.0, 1.0)
        
        volume_scaled = zoom(volume, zoom_factors, order=1, mode='reflect')
        
        if labels is not None:
            labels_scaled = zoom(labels, zoom_factors, order=0, mode='nearest')
        else:
            labels_scaled = None
        
        return volume_scaled, labels_scaled


class TrainingAugmentation(AnisotropicAugmentation):
    """Augmentation pipeline specifically for training with labels."""
    
    def __init__(
        self,
        scale: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
        xy_rotation_range: float = 15.0,
        z_flip_prob: float = 0.5,
        xy_flip_prob: float = 0.5,
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 5.0,
        intensity_contrast_range: Tuple[float, float] = (0.8, 1.2),
        intensity_brightness_range: Tuple[float, float] = (-0.1, 0.1),
        noise_std: float = 0.02,
        blur_sigma_range: Tuple[float, float] = (0.5, 1.5),
        z_scale_range: Tuple[float, float] = (0.9, 1.1),
        crop_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Initialize training augmentation pipeline.
        
        Args:
            scale: Physical scale (Z, Y, X) in micrometers per voxel
            xy_rotation_range: Maximum rotation angle in XY plane (degrees)
            z_flip_prob: Probability of flipping along Z axis
            xy_flip_prob: Probability of flipping along Y or X axis
            elastic_alpha: Alpha parameter for elastic deformation
            elastic_sigma: Sigma parameter for elastic deformation
            intensity_contrast_range: Range for contrast multiplication
            intensity_brightness_range: Range for brightness addition
            noise_std: Standard deviation for Gaussian noise
            blur_sigma_range: Range for Gaussian blur sigma
            z_scale_range: Range for Z-axis scaling
            crop_size: Optional crop size (Z, Y, X) for random cropping
        """
        super().__init__(
            scale=scale,
            xy_rotation_range=xy_rotation_range,
            z_flip_prob=z_flip_prob,
            xy_flip_prob=xy_flip_prob,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            intensity_contrast_range=intensity_contrast_range,
            intensity_brightness_range=intensity_brightness_range,
            noise_std=noise_std,
            blur_sigma_range=blur_sigma_range,
            z_scale_range=z_scale_range,
        )
        
        self.crop_size = crop_size
        
        logger.info(f"TrainingAugmentation initialized with crop_size={crop_size}")
    
    def __call__(
        self,
        volume: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to volume and labels for training.
        
        Args:
            volume: 3D volume (Z, Y, X)
            labels: 3D labels (Z, Y, X)
        
        Returns:
            Augmented volume and labels
        """
        # Apply parent augmentations
        volume, labels = super().__call__(volume, labels)
        
        # Random crop if specified
        if self.crop_size is not None:
            volume, labels = self._random_crop(volume, labels, self.crop_size)
        
        return volume, labels
    
    def _random_crop(
        self,
        volume: np.ndarray,
        labels: np.ndarray,
        crop_size: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random crop to volume and labels.
        
        Args:
            volume: 3D volume (Z, Y, X)
            labels: 3D labels (Z, Y, X)
            crop_size: Crop size (Z, Y, X)
        
        Returns:
            Cropped volume and labels
        """
        z_size, y_size, x_size = crop_size
        
        # Ensure crop size is smaller than input
        z_max = max(1, volume.shape[0] - z_size + 1)
        y_max = max(1, volume.shape[1] - y_size + 1)
        x_max = max(1, volume.shape[2] - x_size + 1)
        
        # Random crop position
        z_start = np.random.randint(0, z_max)
        y_start = np.random.randint(0, y_max)
        x_start = np.random.randint(0, x_max)
        
        # Crop
        volume_crop = volume[
            z_start:z_start + z_size,
            y_start:y_start + y_size,
            x_start:x_start + x_size
        ]
        labels_crop = labels[
            z_start:z_start + z_size,
            y_start:y_start + y_size,
            x_start:x_start + x_size
        ]
        
        return volume_crop, labels_crop


class ValidationAugmentation:
    """Minimal augmentation for validation (normalization only)."""
    
    def __call__(
        self,
        volume: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply minimal augmentation for validation.
        
        Args:
            volume: 3D volume (Z, Y, X)
            labels: Optional 3D labels (Z, Y, X)
        
        Returns:
            Normalized volume and labels
        """
        # Only normalize
        if volume.max() > 1.0:
            volume = volume / volume.max()
        
        return volume, labels
