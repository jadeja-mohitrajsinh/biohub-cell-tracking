"""3D U-Net detector for cell center detection."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.losses import DiceLoss, FocalLoss

from detection.base import Detector, DetectionResult
from detection.custom_unet import CustomUNetWithDeepSupervision

logger = logging.getLogger(__name__)


class CellDetectionUNet(nn.Module):
    """3D U-Net for cell center detection with deep supervision."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,  # background, center, periphery
        spatial_dims: int = 3,
        channels: tuple = (32, 64, 128, 256, 512),
        strides: tuple = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1,
        use_deep_supervision: bool = True,
    ) -> None:
        """Initialize 3D U-Net for cell detection with deep supervision.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels (3 for background/center/periphery)
            spatial_dims: Spatial dimensions (3 for 3D)
            channels: Channel counts at each level
            strides: Strides for downsampling
            num_res_units: Number of residual units
            dropout: Dropout rate
            use_deep_supervision: Whether to use deep supervision with auxiliary outputs
        """
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.out_channels = out_channels
        
        # Use custom U-Net with deep supervision or MONAI U-Net
        if use_deep_supervision:
            self.unet = CustomUNetWithDeepSupervision(
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                dropout=dropout,
                use_deep_supervision=True,
            )
        else:
            self.unet = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                num_res_units=num_res_units,
                dropout=dropout,
                norm="instance",
                act="relu",
            )
            logger.info(
                f"CellDetectionUNet initialized (MONAI): in_channels={in_channels}, "
                f"out_channels={out_channels}, channels={channels}"
            )
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass with deep supervision.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            Dictionary with:
                - 'out': Main output (B, 3, D, H, W) with softmax probabilities
                - 'aux_2': Auxiliary output at 1/2 resolution (if deep supervision)
                - 'aux_4': Auxiliary output at 1/4 resolution (if deep supervision)
                - 'aux_8': Auxiliary output at 1/8 resolution (if deep supervision)
        """
        if self.use_deep_supervision:
            # Custom U-Net returns dictionary with auxiliary outputs
            return self.unet(x)
        else:
            # MONAI U-Net returns logits directly
            logits = self.unet(x)
            return {'out': F.softmax(logits, dim=1)}


class UNetDetector(Detector):
    """Detector using 3D U-Net for cell center detection."""
    
    def __init__(self, config: dict) -> None:
        """Initialize U-Net detector.
        
        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to trained model weights
                - device: Device to run inference on ('cuda', 'cpu', or 'auto')
                - center_threshold: Threshold for center probability
                - periphery_threshold: Threshold for periphery probability
                - min_distance: Minimum distance between detections (voxels)
                - use_refinement: Use sub-voxel centroid refinement
        """
        super().__init__(config)
        
        self.model_path = config.get("model_path")
        self.device = config.get("device", "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.center_threshold = config.get("center_threshold", 0.5)
        self.periphery_threshold = config.get("periphery_threshold", 0.3)
        self.min_distance = config.get("min_distance", 3)
        self.use_refinement = config.get("use_refinement", True)
        
        # Initialize model
        self.model = CellDetectionUNet(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 3),
            spatial_dims=config.get("spatial_dims", 3),
            channels=config.get("channels", (32, 64, 128, 256, 512)),
            strides=config.get("strides", (2, 2, 2, 2)),
            num_res_units=config.get("num_res_units", 2),
            dropout=config.get("dropout", 0.1),
            use_deep_supervision=config.get("use_deep_supervision", True),
        )
        
        # Load weights if provided
        if self.model_path:
            self._load_weights()
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            f"UNetDetector initialized: device={self.device}, "
            f"center_threshold={self.center_threshold}, "
            f"use_refinement={self.use_refinement}"
        )
    
    def _load_weights(self) -> None:
        """Load model weights from checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
    
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells using 3D U-Net.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with centroids
        """
        logger.debug(f"Running U-Net detection on volume shape {volume.shape}")
        
        # Normalize volume
        if volume.max() > 0:
            volume_norm = volume / volume.max()
        else:
            volume_norm = volume
        
        # Add batch and channel dimensions
        volume_tensor = torch.from_numpy(volume_norm).float()
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
        volume_tensor = volume_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output_dict = self.model(volume_tensor)  # Dictionary with 'out' key
            probabilities = output_dict['out']  # (1, 3, Z, Y, X)
        
        # Move to CPU and convert to numpy
        probabilities = probabilities.cpu().numpy()[0]  # (3, Z, Y, X)
        
        # Extract center and periphery probabilities
        center_prob = probabilities[1]  # Center channel
        periphery_prob = probabilities[2]  # Periphery channel
        
        # Find local maxima in center probability
        from scipy.ndimage import maximum_filter
        
        neighborhood_size = 3
        local_max = maximum_filter(center_prob, size=neighborhood_size) == center_prob
        
        # Apply thresholds
        center_mask = (center_prob > self.center_threshold) & local_max
        periphery_mask = periphery_prob > self.periphery_threshold
        
        # Get coordinates of center detections
        centroids = []
        confidences = []
        
        z_coords, y_coords, x_coords = np.where(center_mask)
        
        for z, y, x in zip(z_coords, y_coords, x_coords):
            # Apply minimum distance constraint
            if self._is_too_close((z, y, x), centroids, self.min_distance):
                continue
            
            # Sub-voxel refinement if enabled
            if self.use_refinement:
                refined = self._refine_centroid(
                    volume_norm, center_prob, periphery_prob, (z, y, x)
                )
                centroids.append(refined)
            else:
                centroids.append((int(z), int(y), int(x)))
            
            confidences.append(float(center_prob[z, y, x]))
        
        logger.info(f"U-Net detection: {len(centroids)} cells detected")
        
        return DetectionResult(
            centroids=centroids,
            confidence=confidences,
            metadata={
                "method": "unet_3d",
                "center_threshold": self.center_threshold,
                "periphery_threshold": self.periphery_threshold,
                "use_refinement": self.use_refinement,
            }
        )
    
    def _is_too_close(
        self,
        point: Tuple[int, int, int],
        existing: List[Tuple[int, int, int]],
        min_distance: float
    ) -> bool:
        """Check if point is too close to existing detections.
        
        Args:
            point: New point (z, y, x)
            existing: List of existing points
            min_distance: Minimum distance
        
        Returns:
            True if too close
        """
        z1, y1, x1 = point
        for z2, y2, x2 in existing:
            dist = np.sqrt((z1 - z2)**2 + (y1 - y2)**2 + (x1 - x2)**2)
            if dist < min_distance:
                return True
        return False
    
    def _refine_centroid(
        self,
        volume: np.ndarray,
        center_prob: np.ndarray,
        periphery_prob: np.ndarray,
        centroid: Tuple[int, int, int],
        window_size: int = 5
    ) -> Tuple[int, int, int]:
        """Refine centroid using weighted centroid of probability.
        
        Args:
            volume: Original volume
            center_prob: Center probability map
            periphery_prob: Periphery probability map
            centroid: Initial centroid (z, y, x)
            window_size: Size of local window for refinement
        
        Returns:
            Refined centroid (z, y, x)
        """
        z, y, x = centroid
        
        # Extract local window
        z_start = max(0, z - window_size // 2)
        z_end = min(center_prob.shape[0], z + window_size // 2 + 1)
        y_start = max(0, y - window_size // 2)
        y_end = min(center_prob.shape[1], y + window_size // 2 + 1)
        x_start = max(0, x - window_size // 2)
        x_end = min(center_prob.shape[2], x + window_size // 2 + 1)
        
        local_prob = center_prob[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Compute weighted centroid
        zz, yy, xx = np.meshgrid(
            np.arange(z_start, z_end),
            np.arange(y_start, y_end),
            np.arange(x_start, x_end),
            indexing='ij'
        )
        
        total_weight = local_prob.sum()
        if total_weight > 0:
            refined_z = int((zz * local_prob).sum() / total_weight)
            refined_y = int((yy * local_prob).sum() / total_weight)
            refined_x = int((xx * local_prob).sum() / total_weight)
        else:
            refined_z, refined_y, refined_x = centroid
        
        return (refined_z, refined_y, refined_x)
