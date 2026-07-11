"""Model ensemble for cell detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

from detection.base import Detector, DetectionResult

logger = logging.getLogger(__name__)


class EnsembleDetector(Detector):
    """Ensemble of multiple detectors for robust detection."""
    
    def __init__(self, detectors: List[Detector], config: dict) -> None:
        """Initialize ensemble detector.
        
        Args:
            detectors: List of detectors to ensemble
            config: Configuration dictionary with ensemble parameters
        """
        super().__init__(config)
        
        self.detectors = detectors
        
        # Ensemble parameters
        self.ensemble_method = config.get("ensemble_method", "average")  # average, weighted, voting
        self.weights = config.get("weights", None)  # Optional weights for each detector
        self.distance_threshold = config.get("distance_threshold", 2.0)  # For clustering
        
        # Validate weights if provided
        if self.weights is not None:
            if len(self.weights) != len(self.detectors):
                raise ValueError("Number of weights must match number of detectors")
            # Normalize weights
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        
        logger.info(
            f"EnsembleDetector initialized: {len(detectors)} detectors, "
            f"method={self.ensemble_method}"
        )
    
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells using ensemble of detectors.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with ensembled centroids
        """
        all_detections = []
        all_confidences = []
        
        # Run all detectors
        for i, detector in enumerate(self.detectors):
            try:
                result = detector.detect(volume)
                all_detections.append(result.centroids)
                
                # Apply weights if provided
                if self.weights is not None:
                    conf = result.confidence if result.confidence else [1.0] * len(result.centroids)
                    weighted_conf = [c * self.weights[i] for c in conf]
                    all_confidences.append(weighted_conf)
                else:
                    all_confidences.append(result.confidence if result.confidence else [1.0] * len(result.centroids))
            except Exception as e:
                logger.warning(f"Detector {i} failed: {e}")
                all_detections.append([])
                all_confidences.append([])
        
        # Ensemble detections
        if self.ensemble_method == "average":
            final_centroids, final_confidences = self._ensemble_average(
                all_detections, all_confidences
            )
        elif self.ensemble_method == "weighted":
            final_centroids, final_confidences = self._ensemble_weighted(
                all_detections, all_confidences
            )
        elif self.ensemble_method == "voting":
            final_centroids, final_confidences = self._ensemble_voting(
                all_detections, all_confidences
            )
        else:
            # Default to average
            final_centroids, final_confidences = self._ensemble_average(
                all_detections, all_confidences
            )
        
        logger.info(f"Ensemble detection: {len(final_centroids)} cells detected")
        
        return DetectionResult(
            centroids=final_centroids,
            confidence=final_confidences,
            metadata={
                "method": "ensemble",
                "num_detectors": len(self.detectors),
                "ensemble_method": self.ensemble_method,
            }
        )
    
    def _ensemble_average(
        self,
        all_detections: List[List[Tuple[int, int, int]]],
        all_confidences: List[List[float]],
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Ensemble by averaging nearby detections.
        
        Args:
            all_detections: List of detection lists
            all_confidences: List of confidence lists
        
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
        clusters = self._cluster_detections(flat_detections, self.distance_threshold)
        
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
            cluster_confidences = [flat_confidences[i] for i, d in enumerate(flat_detections) if d in cluster]
            avg_conf = np.mean(cluster_confidences)
            
            final_centroids.append((avg_z, avg_y, avg_x))
            final_confidences.append(float(avg_conf))
        
        return final_centroids, final_confidences
    
    def _ensemble_weighted(
        self,
        all_detections: List[List[Tuple[int, int, int]]],
        all_confidences: List[List[float]],
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Ensemble with weighted averaging.
        
        Args:
            all_detections: List of detection lists
            all_confidences: List of confidence lists (already weighted)
        
        Returns:
            Tuple of (final_centroids, final_confidences)
        """
        # Similar to average but confidences are already weighted
        return self._ensemble_average(all_detections, all_confidences)
    
    def _ensemble_voting(
        self,
        all_detections: List[List[Tuple[int, int, int]]],
        all_confidences: List[List[float]],
        min_votes: int = 2,
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Ensemble by voting (detections must appear in multiple detectors).
        
        Args:
            all_detections: List of detection lists
            all_confidences: List of confidence lists
            min_votes: Minimum number of detectors that must agree
        
        Returns:
            Tuple of (final_centroids, final_confidences)
        """
        # Count votes for each detection
        vote_counts: Dict[Tuple[int, int, int], List[float]] = {}
        
        for detections, confidences in zip(all_detections, all_confidences):
            for det, conf in zip(detections, confidences):
                # Round to nearest integer to handle small variations
                rounded_det = (int(round(det[0])), int(round(det[1])), int(round(det[2])))
                
                if rounded_det not in vote_counts:
                    vote_counts[rounded_det] = []
                vote_counts[rounded_det].append(conf)
        
        # Keep only detections with enough votes
        final_centroids = []
        final_confidences = []
        
        for det, confs in vote_counts.items():
            if len(confs) >= min_votes:
                final_centroids.append(det)
                final_confidences.append(np.mean(confs))
        
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


def create_ensemble_from_checkpoints(
    detector_class,
    checkpoint_paths: List[Path],
    config: dict,
    base_config: dict,
) -> EnsembleDetector:
    """Create ensemble from multiple checkpoint files.
    
    Args:
        detector_class: Detector class to instantiate
        checkpoint_paths: List of checkpoint file paths
        config: Ensemble configuration
        base_config: Base detector configuration
    
    Returns:
        EnsembleDetector with loaded models
    """
    detectors = []
    
    for checkpoint_path in checkpoint_paths:
        # Create detector with this checkpoint
        detector_config = base_config.copy()
        detector_config["model_path"] = str(checkpoint_path)
        
        detector = detector_class(detector_config)
        detectors.append(detector)
    
    return EnsembleDetector(detectors, config)
