"""Contrastive learning embedding model for cell tracking (CELLECT-style)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Encoder3D(nn.Module):
    """3D CNN encoder for cell patches."""
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        
        self.fc = nn.Linear(base_channels * 8, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class COLLECTEmbedding(nn.Module):
    """CELLECT-style contrastive learning embedding for cell tracking."""
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.encoder = Encoder3D(in_channels, base_channels, embedding_dim)
        self.projection = ProjectionHead(embedding_dim, embedding_dim * 2, projection_dim)
        self.temperature = temperature
        
        logger.info(
            f"COLLECTEmbedding initialized: embedding_dim={embedding_dim}, "
            f"projection_dim={projection_dim}, temperature={temperature}"
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning embeddings and projections."""
        embeddings = self.encoder(x)
        projections = self.projection(embeddings)
        return embeddings, projections
    
    def compute_contrastive_loss(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised contrastive loss."""
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(projections, projections.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(mask.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log-softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class CellEmbeddingExtractor:
    """Extract embeddings for cell patches using trained COLLECT model."""
    
    def __init__(
        self,
        model_path: str,
        patch_size: int = 21,
        device: str = "cuda",
        in_channels: int = 1
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        
        # Load model
        self.model = COLLECTEmbedding(in_channels=in_channels)
        self.load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            f"CellEmbeddingExtractor initialized: device={self.device}, "
            f"patch_size={patch_size}"
        )
    
    def load_weights(self, model_path: str) -> None:
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
    
    def extract_patch(
        self,
        volume: np.ndarray,
        position: Tuple[int, int, int]
    ) -> np.ndarray:
        """Extract a 3D patch around a cell position."""
        z, y, x = position
        half_size = self.patch_size // 2
        
        # Compute bounds
        z_start = max(0, z - half_size)
        z_end = min(volume.shape[0], z + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(volume.shape[1], y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(volume.shape[2], x + half_size + 1)
        
        # Extract patch
        patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if patch.shape != (self.patch_size, self.patch_size, self.patch_size):
            pad_z = self.patch_size - patch.shape[0]
            pad_y = self.patch_size - patch.shape[1]
            pad_x = self.patch_size - patch.shape[2]
            patch = np.pad(
                patch,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode='constant',
                constant_values=0
            )
        
        return patch
    
    def extract_embeddings(
        self,
        volume: np.ndarray,
        positions: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Extract embeddings for multiple cell positions."""
        embeddings = []
        
        for position in positions:
            patch = self.extract_patch(volume, position)
            
            # Normalize
            if patch.max() > 0:
                patch = patch / patch.max()
            
            # Convert to tensor
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            patch_tensor = patch_tensor.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding, _ = self.model(patch_tensor)
                embeddings.append(embedding.cpu().numpy().squeeze())
        
        return np.array(embeddings)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        embedding1 = torch.from_numpy(embedding1).float()
        embedding2 = torch.from_numpy(embedding2).float()
        
        similarity = F.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        )
        
        return similarity.item()
