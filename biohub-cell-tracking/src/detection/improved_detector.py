"""Improved classical detector with adaptive thresholding and multi-scale analysis."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter, maximum_filter

from detection.base import Detector, DetectionResult

logger = logging.getLogger(__name__)


class ImprovedDetector(Detector):
    """Improved detector with adaptive thresholding and multi-scale blob detection."""
    
    def __init__(self, config: dict) -> None:
        """Initialize improved detector.
        
        Args:
            config: Configuration dictionary with keys:
                - base_threshold: Base threshold (will be adapted locally)
                - sigma: Gaussian smoothing sigma
                - min_size: Minimum component size (voxels)
                - max_size: Maximum component size (voxels)
                - use_adaptive: Use local adaptive thresholding
                - adaptive_block_size: Block size for adaptive thresholding
                - multi_scale: Use multi-scale detection
                - scales: List of scales for multi-scale detection
        """
        super().__init__(config)
        self.base_threshold = config.get("base_threshold", 0.1)
        self.sigma = config.get("sigma", 1.5)
        self.min_size = config.get("min_size", 3)
        self.max_size = config.get("max_size", 1000)
        self.use_adaptive = config.get("use_adaptive", True)
        self.adaptive_block_size = config.get("adaptive_block_size", 15)
        self.multi_scale = config.get("multi_scale", True)
        self.scales = config.get("scales", [1.0, 1.5, 2.0])
        self.use_nms = config.get("use_nms", True)
        self.nms_radius = config.get("nms_radius", 5)  # voxels
        self.threshold_percentile = config.get("threshold_percentile", 95)  # Use 95th percentile
        self.use_intensity_calibration = config.get("use_intensity_calibration", True)
        self.local_max_radius = config.get("local_max_radius", 3)  # voxels
        
        logger.info(
            f"ImprovedDetector initialized: base_threshold={self.base_threshold}, "
            f"sigma={self.sigma}, adaptive={self.use_adaptive}, "
            f"multi_scale={self.multi_scale}, nms={self.use_nms}, "
            f"intensity_calib={self.use_intensity_calibration}"
        )
    
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells using improved classical methods with local maxima.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with centroids
        """
        logger.debug(f"Running improved detection on volume shape {volume.shape}")
        
        # Normalize volume to 0-1
        if volume.max() > 0:
            volume_norm = volume / volume.max()
        else:
            volume_norm = volume
        
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(volume_norm, sigma=self.sigma)
        
        # Compute threshold using intensity calibration if enabled
        if self.use_intensity_calibration:
            threshold = self._compute_percentile_threshold(smoothed, self.threshold_percentile)
        elif self.use_adaptive:
            threshold = self._compute_otsu_threshold(smoothed)
        else:
            threshold = self.base_threshold
        
        logger.debug(f"Using threshold: {threshold:.4f}")
        
        # Use local maxima detection
        from scipy.ndimage import maximum_filter
        
        # Find local maxima in 3D
        local_max = maximum_filter(smoothed, size=self.local_max_radius) == smoothed
        
        # Apply threshold
        binary = local_max & (smoothed > threshold)
        
        logger.debug(f"Local maxima found: {binary.sum()}")
        
        # Multi-scale detection if enabled
        if self.multi_scale:
            binary = self._multi_scale_refinement(volume_norm, binary, smoothed)
        
        # Get coordinates of local maxima
        centroids = []
        confidences = []
        
        z_coords, y_coords, x_coords = np.where(binary)
        
        for z, y, x in zip(z_coords, y_coords, x_coords):
            centroids.append((int(z), int(y), int(x)))
            confidences.append(float(smoothed[z, y, x]))
        
        # Apply non-maximum suppression if enabled
        if self.use_nms and centroids:
            centroids, confidences = self._non_maximum_suppression(centroids, confidences)
        
        logger.info(f"Improved detection: {len(centroids)} cells detected")
        
        return DetectionResult(
            centroids=centroids,
            confidence=confidences,
            metadata={
                "method": "improved_classical_local_max",
                "threshold": threshold,
                "adaptive": self.use_adaptive,
                "multi_scale": self.multi_scale,
            }
        )
    
    def _adaptive_threshold_3d(self, volume: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to 3D volume.
        
        Args:
            volume: 3D numpy array
        
        Returns:
            Binary mask
        """
        binary = np.zeros_like(volume, dtype=bool)
        
        # Process each Z slice independently
        for z in range(volume.shape[0]):
            slice_2d = volume[z]
            
            # Compute local mean
            local_mean = self._local_mean_2d(slice_2d, self.adaptive_block_size)
            
            # Adaptive threshold: pixel > local_mean + base_threshold
            threshold_slice = slice_2d > (local_mean + self.base_threshold)
            binary[z] = threshold_slice
        
        return binary
    
    def _local_mean_2d(self, image: np.ndarray, block_size: int) -> np.ndarray:
        """Compute local mean using uniform filter.
        
        Args:
            image: 2D numpy array
            block_size: Size of local neighborhood
        
        Returns:
            Local mean map
        """
        from scipy.ndimage import uniform_filter
        return uniform_filter(image, size=block_size, mode='reflect')
    
    def _compute_otsu_threshold(self, volume: np.ndarray) -> float:
        """Compute Otsu threshold for the volume.
        
        Args:
            volume: 3D numpy array
        
        Returns:
            Threshold value
        """
        # Flatten and compute histogram
        values = volume.flatten()
        
        # Compute histogram
        hist, bin_edges = np.histogram(values, bins=256, range=(0, 1))
        
        # Normalize histogram
        hist = hist.astype(float) / hist.sum()
        
        # Compute cumulative sums
        cumsum = np.cumsum(hist)
        cumsum_mean = np.cumsum(hist * bin_edges[:-1])
        
        # Compute between-class variance
        mean_total = cumsum_mean[-1]
        between_class_variance = np.zeros(256)
        
        for i in range(256):
            w0 = cumsum[i]
            w1 = 1 - w0
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = cumsum_mean[i] / w0
            mu1 = (mean_total - cumsum_mean[i]) / w1
            
            between_class_variance[i] = w0 * w1 * (mu0 - mu1) ** 2
        
        # Find threshold that maximizes between-class variance
        threshold_idx = np.argmax(between_class_variance)
        threshold = bin_edges[threshold_idx]
        
        return float(threshold)
    
    def _compute_percentile_threshold(self, volume: np.ndarray, percentile: float) -> float:
        """Compute threshold based on intensity percentile.
        
        Args:
            volume: 3D numpy array
            percentile: Percentile to use (e.g., 95 for 95th percentile)
        
        Returns:
            Threshold value
        """
        values = volume.flatten()
        threshold = np.percentile(values, percentile)
        return float(threshold)
    
    def _multi_scale_refinement(self, original: np.ndarray, binary: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
        """Refine binary mask using multi-scale analysis.
        
        Args:
            original: Original normalized volume
            binary: Initial binary mask
            smoothed: Smoothed volume
        
        Returns:
            Refined binary mask
        """
        refined = binary.copy()
        
        for scale in self.scales:
            # Smooth at different scale
            scaled_smooth = gaussian_filter(original, sigma=self.sigma * scale)
            
            # Find local maxima at this scale
            local_max = maximum_filter(scaled_smooth, size=self.local_max_radius) == scaled_smooth
            
            # Use same threshold method as base detection
            if self.use_intensity_calibration:
                threshold = self._compute_percentile_threshold(scaled_smooth, self.threshold_percentile)
            elif self.use_adaptive:
                threshold = self._compute_otsu_threshold(scaled_smooth)
            else:
                threshold = self.base_threshold
            scaled_binary = local_max & (scaled_smooth > threshold)
            
            # Union with current refined mask
            refined = refined | scaled_binary
        
        return refined
    
    def _non_maximum_suppression(
        self, 
        centroids: List[Tuple[int, int, int]], 
        confidences: List[float]
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Apply non-maximum suppression to remove nearby detections.
        
        Args:
            centroids: List of (z, y, x) coordinates
            confidences: List of confidence scores
        
        Returns:
            Filtered centroids and confidences
        """
        if not centroids:
            return [], []
            
        from scipy.spatial import KDTree
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_centroids = [centroids[i] for i in sorted_indices]
        sorted_confidences = [confidences[i] for i in sorted_indices]
        
        # Build KDTree for efficient neighbor search
        tree = KDTree(sorted_centroids)
        
        # NMS
        keep = []
        suppressed = np.zeros(len(sorted_centroids), dtype=bool)
        
        for i in range(len(sorted_centroids)):
            if suppressed[i]:
                continue
            
            keep.append(i)
            
            # Find all neighbors within nms_radius
            neighbors = tree.query_ball_point(sorted_centroids[i], self.nms_radius)
            
            # Suppress neighbors
            for j in neighbors:
                suppressed[j] = True
        
        # Return filtered results
        filtered_centroids = [sorted_centroids[i] for i in keep]
        filtered_confidences = [sorted_confidences[i] for i in keep]
        
        logger.debug(f"NMS: {len(centroids)} -> {len(filtered_centroids)} detections")
        
        return filtered_centroids, filtered_confidences
