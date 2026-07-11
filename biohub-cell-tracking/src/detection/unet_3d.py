"""3D U-Net for cell detection and segmentation."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.base import Detector, DetectionResult

logger = logging.getLogger(__name__)


class DoubleConv3D(nn.Module):
    """Double convolution block for 3D U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for cell segmentation."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        bilinear: bool = False,
        features: Tuple[int, ...] = (64, 128, 256, 512)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        self.inc = DoubleConv3D(in_channels, features[0])
        self.down1 = Down3D(features[0], features[1])
        self.down2 = Down3D(features[1], features[2])
        self.down3 = Down3D(features[2], features[3])
        
        factor = 2 if bilinear else 1
        self.down4 = Down3D(features[3], features[4] // factor)
        
        self.up1 = Up3D(features[4], features[3] // factor, bilinear)
        self.up2 = Up3D(features[3], features[2] // factor, bilinear)
        self.up3 = Up3D(features[2], features[1] // factor, bilinear)
        self.up4 = Up3D(features[1], features[0], bilinear)
        
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class UNet3DDetector(Detector):
    """Cell detector using 3D U-Net."""
    
    def __init__(
        self,
        config: dict,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__(config)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.in_channels = config.get("in_channels", 1)
        self.out_channels = config.get("out_channels", 1)
        self.bilinear = config.get("bilinear", False)
        self.features = config.get("features", (32, 64, 128, 256))
        
        # Detection parameters
        self.threshold = config.get("threshold", 0.5)
        self.min_size = config.get("min_size", 3)
        self.nms_radius = config.get("nms_radius", 5)
        
        # Initialize model
        self.model = UNet3D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            bilinear=self.bilinear,
            features=self.features
        )
        
        # Load weights if provided
        if model_path:
            self.load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            f"UNet3DDetector initialized: device={self.device}, "
            f"threshold={self.threshold}, nms_radius={self.nms_radius}"
        )
    
    def load_weights(self, model_path: str) -> None:
        """Load model weights from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
    
    def detect(self, volume: np.ndarray) -> DetectionResult:
        """Detect cells using 3D U-Net.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
        
        Returns:
            DetectionResult with centroids
        """
        logger.debug(f"Running UNet3D detection on volume shape {volume.shape}")
        
        # Normalize volume
        if volume.max() > 0:
            volume_norm = volume / volume.max()
        else:
            volume_norm = volume
        
        # Add channel and batch dimensions
        volume_tensor = torch.from_numpy(volume_norm).float().unsqueeze(0).unsqueeze(0)
        volume_tensor = volume_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(volume_tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Threshold
        binary = prob > self.threshold
        
        # Find local maxima
        from scipy.ndimage import maximum_filter
        neighborhood_size = 3
        local_max = maximum_filter(prob, size=neighborhood_size) == prob
        binary = binary & local_max
        
        # Get coordinates
        z_coords, y_coords, x_coords = np.where(binary)
        
        centroids = []
        confidences = []
        
        for z, y, x in zip(z_coords, y_coords, x_coords):
            centroids.append((int(z), int(y), int(x)))
            confidences.append(float(prob[z, y, x]))
        
        # Apply NMS
        if self.nms_radius > 0 and centroids:
            centroids, confidences = self._non_maximum_suppression(
                centroids, confidences, self.nms_radius
            )
        
        logger.info(f"UNet3D detection: {len(centroids)} cells detected")
        
        return DetectionResult(
            centroids=centroids,
            confidence=confidences,
            metadata={
                "method": "unet_3d",
                "threshold": self.threshold,
                "device": str(self.device),
            }
        )
    
    def _non_maximum_suppression(
        self,
        centroids: list,
        confidences: list,
        radius: int
    ) -> tuple:
        """Apply non-maximum suppression to detections."""
        if len(centroids) == 0:
            return [], []
        
        # Sort by confidence
        indices = np.argsort(confidences)[::-1]
        
        keep = []
        for idx in indices:
            z, y, x = centroids[idx]
            should_keep = True
            
            for kept_idx in keep:
                kz, ky, kx = centroids[kept_idx]
                dist = np.sqrt((z - kz)**2 + (y - ky)**2 + (x - kx)**2)
                if dist < radius:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(idx)
        
        keep = sorted(keep)
        return [centroids[i] for i in keep], [confidences[i] for i in keep]
