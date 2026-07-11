"""Learned association network for predicting cell matching probability."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AssociationFeatureExtractor:
    """Extract features for cell pair association."""
    
    def __init__(self, scale: Tuple[float, float, float] = (1.625, 0.40625, 0.40625)):
        """Initialize feature extractor.
        
        Args:
            scale: Physical scale (Z, Y, X) in micrometers per voxel
        """
        self.scale = scale  # (z_scale, y_scale, x_scale)
    
    def extract_features(
        self,
        cell_a: Tuple[float, float, float, float],  # (z, y, x, confidence)
        cell_b: Tuple[float, float, float, float],  # (z, y, x, confidence)
        volume_a: Optional[np.ndarray] = None,
        volume_b: Optional[np.ndarray] = None,
        position_a: Optional[Tuple[int, int, int]] = None,
        position_b: Optional[Tuple[int, int, int]] = None,
        motion_stats: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Extract features for cell pair.
        
        Args:
            cell_a: Cell A (z, y, x, confidence)
            cell_b: Cell B (z, y, x, confidence)
            volume_a: Volume at time t (optional, for intensity features)
            volume_b: Volume at time t+1 (optional)
            position_a: Position in volume_a (z, y, x)
            position_b: Position in volume_b (z, y, x)
            motion_stats: Motion statistics (mean_displacement_um, std_displacement_um)
        
        Returns:
            Feature vector (numpy array)
        """
        z_a, y_a, x_a, conf_a = cell_a
        z_b, y_b, x_b, conf_b = cell_b
        
        # Physical distance
        dz_um = (z_b - z_a) * self.scale[0]
        dy_um = (y_b - y_a) * self.scale[1]
        dx_um = (x_b - x_a) * self.scale[2]
        distance_um = np.sqrt(dz_um**2 + dy_um**2 + dx_um**2)
        
        # Euclidean distance in voxel space
        distance_voxel = np.sqrt((z_b - z_a)**2 + (y_b - y_a)**2 + (x_b - x_a)**2)
        
        # Confidence features
        conf_mean = (conf_a + conf_b) / 2
        conf_diff = abs(conf_a - conf_b)
        
        # Motion features
        if motion_stats is not None:
            mean_disp = motion_stats.get('mean_displacement_um', 2.0)
            std_disp = motion_stats.get('std_displacement_um', 1.0)
            normalized_distance = (distance_um - mean_disp) / (std_disp + 1e-6)
        else:
            normalized_distance = distance_um / 2.0  # Default normalization
        
        # Intensity features (if volumes provided)
        intensity_mean = 0.0
        intensity_std = 0.0
        intensity_corr = 0.0
        
        if volume_a is not None and volume_b is not None and position_a is not None and position_b is not None:
            # Extract local patches
            patch_size = 5
            z_start_a = max(0, position_a[0] - patch_size // 2)
            z_end_a = min(volume_a.shape[0], position_a[0] + patch_size // 2 + 1)
            y_start_a = max(0, position_a[1] - patch_size // 2)
            y_end_a = min(volume_a.shape[1], position_a[1] + patch_size // 2 + 1)
            x_start_a = max(0, position_a[2] - patch_size // 2)
            x_end_a = min(volume_a.shape[2], position_a[2] + patch_size // 2 + 1)
            
            z_start_b = max(0, position_b[0] - patch_size // 2)
            z_end_b = min(volume_b.shape[0], position_b[0] + patch_size // 2 + 1)
            y_start_b = max(0, position_b[1] - patch_size // 2)
            y_end_b = min(volume_b.shape[1], position_b[1] + patch_size // 2 + 1)
            x_start_b = max(0, position_b[2] - patch_size // 2)
            x_end_b = min(volume_b.shape[2], position_b[2] + patch_size // 2 + 1)
            
            patch_a = volume_a[z_start_a:z_end_a, y_start_a:y_end_a, x_start_a:x_end_a]
            patch_b = volume_b[z_start_b:z_end_b, y_start_b:y_end_b, x_start_b:x_end_b]
            
            if patch_a.size > 0 and patch_b.size > 0:
                # Resize patches to same size
                min_z = min(patch_a.shape[0], patch_b.shape[0])
                min_y = min(patch_a.shape[1], patch_b.shape[1])
                min_x = min(patch_a.shape[2], patch_b.shape[2])
                
                patch_a = patch_a[:min_z, :min_y, :min_x]
                patch_b = patch_b[:min_z, :min_y, :min_x]
                
                intensity_mean = (patch_a.mean() + patch_b.mean()) / 2
                intensity_std = (patch_a.std() + patch_b.std()) / 2
                
                # Correlation
                if patch_a.size > 1:
                    flat_a = patch_a.flatten()
                    flat_b = patch_b.flatten()
                    if flat_a.std() > 1e-6 and flat_b.std() > 1e-6:
                        intensity_corr = np.corrcoef(flat_a, flat_b)[0, 1]
                    else:
                        intensity_corr = 0.0
        
        # Assemble feature vector
        features = np.array([
            distance_um,
            distance_voxel,
            dz_um,
            dy_um,
            dx_um,
            conf_mean,
            conf_diff,
            normalized_distance,
            intensity_mean,
            intensity_std,
            intensity_corr,
        ], dtype=np.float32)
        
        return features


class LearnedAssociationNet(nn.Module):
    """Neural network for predicting cell matching probability."""
    
    def __init__(
        self,
        input_dim: int = 11,  # Number of features
        hidden_dims: List[int] = [64, 32, 16],
        dropout: float = 0.2,
    ):
        """Initialize learned association network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (probability)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        logger.info(
            f"LearnedAssociationNet initialized: input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features (B, input_dim)
        
        Returns:
            Matching probability (B, 1)
        """
        return self.network(x)


class LearnedAssociationTracker:
    """Tracker using learned association network."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scale: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
        config: Optional[Dict] = None,
    ):
        """Initialize learned association tracker.
        
        Args:
            model_path: Path to trained model weights
            scale: Physical scale (Z, Y, X) in micrometers per voxel
            config: Configuration dictionary
        """
        self.scale = scale
        self.config = config or {}
        
        # Feature extractor
        self.feature_extractor = AssociationFeatureExtractor(scale)
        
        # Model
        self.model = LearnedAssociationNet(
            input_dim=self.config.get("input_dim", 11),
            hidden_dims=self.config.get("hidden_dims", [64, 32, 16]),
            dropout=self.config.get("dropout", 0.2),
        )
        
        # Device
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Load weights if provided
        if model_path:
            self._load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Threshold for matching
        self.match_threshold = self.config.get("match_threshold", 0.5)
        
        logger.info(
            f"LearnedAssociationTracker initialized: device={self.device}, "
            f"match_threshold={self.match_threshold}"
        )
    
    def _load_weights(self, model_path: str) -> None:
        """Load model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
    
    def compute_cost_matrix(
        self,
        detections_t: List[Tuple[float, float, float, float]],
        detections_t1: List[Tuple[float, float, float, float]],
        volume_t: Optional[np.ndarray] = None,
        volume_t1: Optional[np.ndarray] = None,
        positions_t: Optional[List[Tuple[int, int, int]]] = None,
        positions_t1: Optional[List[Tuple[int, int, int]]] = None,
        motion_stats: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute cost matrix using learned association.
        
        Args:
            detections_t: Detections at time t [(z, y, x, conf), ...]
            detections_t1: Detections at time t+1 [(z, y, x, conf), ...]
            volume_t: Volume at time t (optional)
            volume_t1: Volume at time t+1 (optional)
            positions_t: Positions in volume_t (optional)
            positions_t1: Positions in volume_t1 (optional)
            motion_stats: Motion statistics (optional)
        
        Returns:
            Cost matrix (N_t, N_t1) where cost = 1 - P(match)
        """
        n_t = len(detections_t)
        n_t1 = len(detections_t1)
        
        if n_t == 0 or n_t1 == 0:
            return np.zeros((n_t, n_t1), dtype=np.float32)
        
        # Extract features for all pairs
        features = []
        for i, cell_a in enumerate(detections_t):
            for j, cell_b in enumerate(detections_t1):
                pos_a = positions_t[i] if positions_t else None
                pos_b = positions_t1[j] if positions_t1 else None
                
                feat = self.feature_extractor.extract_features(
                    cell_a, cell_b,
                    volume_t, volume_t1,
                    pos_a, pos_b,
                    motion_stats,
                )
                features.append(feat)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(np.array(features)).float().to(self.device)
        
        # Predict probabilities
        with torch.no_grad():
            probs = self.model(features_tensor).squeeze(-1)  # (N_t * N_t1,)
        
        # Reshape to matrix
        prob_matrix = probs.cpu().numpy().reshape(n_t, n_t1)
        
        # Convert to cost (1 - probability)
        cost_matrix = 1.0 - prob_matrix
        
        return cost_matrix
