"""Test-time augmentation for cell detection."""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import numpy as np

from detection.base import Detector, DetectionResult

logger = logging.getLogger(__name__)


class TTADetector(Detector):
    """Detector with test-time augmentation."""
    
    def __init__(self, base_detector: Detector, config: dict) -> None:
        """Initialize TTA detector.
        
        Args:
            base_detector: Base detector to augment
            config: Configuration dictionary with TTA parameters
        """
        super().__init__(config)
        
        self.base_detector = base_detector
        
        # TTA parameters
        self.use_xy_flip = config.get("use_xy_flip", True)
        self.use_intensity_augment = config.get("use_intensity_augment", False)
        self.intensity_range = config.get("intensity_range", [0.9, 1.1])
        self.ensemble_method = config.get("ensemble_method", "average")  # average, nms, voting
        
        logger.info(
            f"TTADetector initialized: xy_flip={self.use_xy_flip}, "
            f"intensity={self.use_intensity_augment}"
        )
    
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells with test-time augmentation.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with ensembled centroids
        """
        all_detections = []
        all_confidences = []
        
        # Original detection
        result = self.base_detector.detect(volume)
        all_detections.append(result.centroids)
        all_confidences.append(result.confidence if result.confidence else [1.0] * len(result.centroids))
        
        # XY flip augmentation
        if self.use_xy_flip:
            # Flip Y axis
            volume_y_flip = np.flip(volume, axis=1)
            result_y = self.base_detector.detect(volume_y_flip)
            # Flip back centroids
            centroids_y_flip = [(z, volume.shape[1] - 1 - y, x) for z, y, x in result_y.centroids]
            all_detections.append(centroids_y_flip)
            all_confidences.append(result_y.confidence if result_y.confidence else [1.0] * len(centroids_y_flip))
            
            # Flip X axis
            volume_x_flip = np.flip(volume, axis=2)
            result_x = self.base_detector.detect(volume_x_flip)
            # Flip back centroids
            centroids_x_flip = [(z, y, volume.shape[2] - 1 - x) for z, y, x in result_x.centroids]
            all_detections.append(centroids_x_flip)
            all_confidences.append(result_x.confidence if result_x.confidence else [1.0] * len(centroids_x_flip))
            
            # Flip both axes
            volume_xy_flip = np.flip(np.flip(volume, axis=1), axis=2)
            result_xy = self.base_detector.detect(volume_xy_flip)
            # Flip back centroids
            centroids_xy_flip = [
                (z, volume.shape[1] - 1 - y, volume.shape[2] - 1 - x)
                for z, y, x in result_xy.centroids
            ]
            all_detections.append(centroids_xy_flip)
            all_confidences.append(result_xy.confidence if result_xy.confidence else [1.0] * len(centroids_xy_flip))
        
        # Intensity augmentation
        if self.use_intensity_augment:
            for factor in self.intensity_range:
                volume_intensity = volume * factor
                volume_intensity = np.clip(volume_intensity, 0, 1)
                result_i = self.base_detector.detect(volume_intensity)
                all_detections.append(result_i.centroids)
                all_confidences.append(result_i.confidence if result_i.confidence else [1.0] * len(result_i.centroids))
        
        # Ensemble detections
        if self.ensemble_method == "average":
            final_centroids, final_confidences = self._ensemble_average(
                all_detections, all_confidences
            )
        elif self.ensemble_method == "nms":
            final_centroids, final_confidences = self._ensemble_nms(
                all_detections, all_confidences
            )
        else:
            # Default to average
            final_centroids, final_confidences = self._ensemble_average(
                all_detections, all_confidences
            )
        
        logger.info(f"TTA detection: {len(final_centroids)} cells detected (from {len(all_detections)} augmentations)")
        
        return DetectionResult(
            centroids=final_centroids,
            confidence=final_confidences,
            metadata={
                "method": "tta",
                "num_augmentations": len(all_detections),
                "ensemble_method": self.ensemble_method,
            }
        )
    
    def _ensemble_average(
        self,
        all_detections: List[List[Tuple[int, int, int]]],
        all_confidences: List[List[float]],
        distance_threshold: float = 2.0,
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Ensemble detections by averaging nearby detections.
        
        Args:
            all_detections: List of detection lists from augmentations
            all_confidences: List of confidence lists
            distance_threshold: Distance threshold for clustering (voxels)
        
        Returns:
            Tuple of (final_centroids, final_confidences)
        """
        # Flatten all detections
        flat_detections = []
        flat_confidences = []
        
        for detections, confidences in zip(all_detections, all_confidences):
            for det, conf in zip(detections, confidences):
                flat_detections.append(det)
                flat_confidences.append(conf)
        
        if not flat_detections:
            return [], []
        
        # Cluster detections by distance
        clusters = self._cluster_detections(flat_detections, distance_threshold)
        
        # Average each cluster
        final_centroids = []
        final_confidences = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            # Average positions
            z_coords = [d[0] for d in cluster]
            y_coords = [d[1] for d in cluster]
            x_coords = [d[2] for d in cluster]
            
            avg_z = int(round(np.mean(z_coords)))
            avg_y = int(round(np.mean(y_coords)))
            avg_x = int(round(np.mean(x_coords)))
            
            # Average confidence
            avg_conf = np.mean([flat_confidences[i] for i, d in enumerate(flat_detections) if d in cluster])
            
            final_centroids.append((avg_z, avg_y, avg_x))
            final_confidences.append(float(avg_conf))
        
        return final_centroids, final_confidences
    
    def _cluster_detections(
        self,
        detections: List[Tuple[int, int, int]],
        distance_threshold: float,
    ) -> List[List[Tuple[int, int, int]]]:
        """Cluster detections by distance.
        
        Args:
            detections: List of detections
            distance_threshold: Distance threshold
        
        Returns:
            List of clusters
        """
        if not detections:
            return []
        
        # Simple greedy clustering
        clusters = []
        used = [False] * len(detections)
        
        for i, det in enumerate(detections):
            if used[i]:
                continue
            
            # Start new cluster
            cluster = [det]
            used[i] = True
            
            # Find nearby detections
            for j, other_det in enumerate(detections):
                if used[j]:
                    continue
                
                dist = np.sqrt(
                    (det[0] - other_det[0])**2 +
                    (det[1] - other_det[1])**2 +
                    (det[2] - other_det[2])**2
                )
                
                if dist < distance_threshold:
                    cluster.append(other_det)
                    used[j] = True
            
            clusters.append(cluster)
        
        return clusters
    
    def _ensemble_nms(
        self,
        all_detections: List[List[Tuple[int, int, int]]],
        all_confidences: List[List[float]],
        distance_threshold: float = 2.0,
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Ensemble detections using non-maximum suppression.
        
        Args:
            all_detections: List of detection lists from augmentations
            all_confidences: List of confidence lists
            distance_threshold: Distance threshold for NMS
        
        Returns:
            Tuple of (final_centroids, final_confidences)
        """
        # Flatten all detections with their confidences
        flat_detections = []
        flat_confidences = []
        
        for detections, confidences in zip(all_detections, all_confidences):
            for det, conf in zip(detections, confidences):
                flat_detections.append(det)
                flat_confidences.append(conf)
        
        if not flat_detections:
            return [], []
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(flat_confidences)[::-1]
        
        final_centroids = []
        final_confidences = []
        used = [False] * len(flat_detections)
        
        for idx in sorted_indices:
            if used[idx]:
                continue
            
            det = flat_detections[idx]
            conf = flat_confidences[idx]
            
            final_centroids.append(det)
            final_confidences.append(conf)
            used[idx] = True
            
            # Suppress nearby detections
            for j, other_det in enumerate(flat_detections):
                if used[j]:
                    continue
                
                dist = np.sqrt(
                    (det[0] - other_det[0])**2 +
                    (det[1] - other_det[1])**2 +
                    (det[2] - other_det[2])**2
                )
                
                if dist < distance_threshold:
                    used[j] = True
        
        return final_centroids, final_confidences
