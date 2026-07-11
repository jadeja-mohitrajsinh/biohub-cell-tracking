"""GNN-based tracking for edge prediction (Ben-Haim & Raviv ECCV'22 style)."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from data.models import TrackingEdge, VoxelScale
from tracking.base import Tracker, TrackingResult

logger = logging.getLogger(__name__)


class EdgeFeatureExtractor:
    """Extract features for cell pairs for GNN edge prediction."""
    
    def __init__(self, scale: VoxelScale):
        self.scale = scale
    
    def extract_pair_features(
        self,
        cell1: Tuple[int, int, int, float],  # (z, y, x, confidence)
        cell2: Tuple[int, int, int, float],
        time_gap: int = 1,
        volume1: Optional[np.ndarray] = None,
        volume2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract features for a cell pair."""
        z1, y1, x1, conf1 = cell1
        z2, y2, x2, conf2 = cell2
        
        # Physical distance
        dz_um = (z2 - z1) * self.scale.z
        dy_um = (y2 - y1) * self.scale.y
        dx_um = (x2 - x1) * self.scale.x
        distance_um = np.sqrt(dz_um**2 + dy_um**2 + dx_um**2)
        
        # Euclidean distance in voxel space
        distance_voxel = np.sqrt((z2 - z1)**2 + (y2 - y1)**2 + (x2 - x1)**2)
        
        # Confidence features
        conf_mean = (conf1 + conf2) / 2
        conf_diff = abs(conf1 - conf2)
        
        # Direction features
        if distance_voxel > 0:
            dir_z = dz_um / distance_um
            dir_y = dy_um / distance_um
            dir_x = dx_um / distance_um
        else:
            dir_z, dir_y, dir_x = 0, 0, 0
        
        # Assemble feature vector
        features = np.array([
            distance_um,
            distance_voxel,
            dz_um,
            dy_um,
            dx_um,
            dir_z,
            dir_y,
            dir_x,
            conf_mean,
            conf_diff,
            time_gap,
        ], dtype=np.float32)
        
        return features


class GNNLinkPredictor(nn.Module):
    """GNN for predicting cell association edges."""
    
    def __init__(
        self,
        node_feature_dim: int = 4,  # (z, y, x, confidence)
        edge_feature_dim: int = 11,  # from EdgeFeatureExtractor
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(
            f"GNNLinkPredictor initialized: node_dim={node_feature_dim}, "
            f"edge_dim={edge_feature_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}"
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for edge prediction.
        
        Args:
            node_features: (N, node_feature_dim)
            edge_index: (2, E) - source and target node indices
            edge_features: (E, edge_feature_dim)
        
        Returns:
            Edge probabilities (E, 1)
        """
        # Encode nodes
        node_embeddings = self.node_encoder(node_features)
        
        # Encode edges
        edge_embeddings = self.edge_encoder(edge_features)
        
        # Apply GNN layers
        for conv in self.convs:
            node_embeddings = conv(node_embeddings, edge_index)
            node_embeddings = F.relu(node_embeddings)
        
        # Get source and target node embeddings for each edge
        source_idx = edge_index[0]
        target_idx = edge_index[1]
        
        source_embeddings = node_embeddings[source_idx]
        target_embeddings = node_embeddings[target_idx]
        
        # Concatenate for edge classification
        edge_input = torch.cat([
            source_embeddings,
            target_embeddings,
            edge_embeddings
        ], dim=1)
        
        # Classify edges
        edge_probs = self.edge_classifier(edge_input)
        
        return edge_probs


class GNNTracker(Tracker):
    """Tracker using GNN for edge prediction."""
    
    def __init__(
        self,
        config: dict,
        scale: VoxelScale,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__(config, scale)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Feature extractor
        self.feature_extractor = EdgeFeatureExtractor(scale)
        
        # Model parameters
        self.node_feature_dim = config.get("node_feature_dim", 4)
        self.edge_feature_dim = config.get("edge_feature_dim", 11)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.num_layers = config.get("num_layers", 3)
        
        # Threshold for edge prediction
        self.edge_threshold = config.get("edge_threshold", 0.5)
        
        # Initialize model
        self.model = GNNLinkPredictor(
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Load weights if provided
        if model_path:
            self.load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            f"GNNTracker initialized: device={self.device}, "
            f"edge_threshold={self.edge_threshold}"
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
    
    def track(
        self,
        detections_by_time: Dict[int, List[Tuple[int, int, int]]],
        confidences_by_time: Optional[Dict[int, List[float]]] = None,
        volumes_by_time: Optional[Dict[int, np.ndarray]] = None,
    ) -> TrackingResult:
        """Link detections across time using GNN edge prediction.
        
        Args:
            detections_by_time: Dict mapping time -> list of (z, y, x) detections
            confidences_by_time: Optional dict mapping time -> confidence scores
            volumes_by_time: Optional dict mapping time -> volume arrays
        
        Returns:
            TrackingResult with edges and node IDs
        """
        logger.debug(f"Running GNN tracking on {len(detections_by_time)} timepoints")
        
        # Assign node IDs
        node_ids = {}
        self._detection_to_node_id = {}
        node_id_counter = 0
        
        for t in sorted(detections_by_time.keys()):
            for idx, detection in enumerate(detections_by_time[t]):
                node_id = t * 10000 + node_id_counter
                self._detection_to_node_id[(t, idx)] = node_id
                node_ids[node_id] = (t, *detection)
                node_id_counter += 1
        
        # Build edges using GNN prediction
        edges = []
        sorted_times = sorted(detections_by_time.keys())
        
        for i in range(len(sorted_times) - 1):
            t_current = sorted_times[i]
            t_next = sorted_times[i + 1]
            
            if t_current not in detections_by_time or t_next not in detections_by_time:
                continue
            
            current_detections = detections_by_time[t_current]
            next_detections = detections_by_time[t_next]
            
            current_confidences = confidences_by_time.get(t_current, [1.0] * len(current_detections))
            next_confidences = confidences_by_time.get(t_next, [1.0] * len(next_detections))
            
            # Build graph for this frame pair
            frame_edges = self._predict_frame_edges(
                current_detections,
                next_detections,
                current_confidences,
                next_confidences,
                t_current,
                t_next,
                node_ids
            )
            
            edges.extend(frame_edges)
        
        logger.info(f"GNN tracking: {len(edges)} edges created")
        
        # Clear mapping cache
        self._detection_to_node_id = {}
        
        return TrackingResult(
            edges=edges,
            node_ids=node_ids,
            metadata={
                "method": "gnn",
                "edge_threshold": self.edge_threshold,
                "num_timepoints": len(detections_by_time),
            }
        )
    
    def _predict_frame_edges(
        self,
        detections_t: List[Tuple[int, int, int]],
        detections_t1: List[Tuple[int, int, int]],
        confidences_t: List[float],
        confidences_t1: List[float],
        t_current: int,
        t_next: int,
        node_ids: Dict[int, Tuple[int, int, int, int]]
    ) -> List[TrackingEdge]:
        """Predict edges between two consecutive frames using GNN."""
        if len(detections_t) == 0 or len(detections_t1) == 0:
            return []
        
        # Build node features
        node_features = []
        node_indices = []
        
        for i, (z, y, x) in enumerate(detections_t):
            conf = confidences_t[i]
            node_features.append([z, y, x, conf])
            node_indices.append((t_current, i))
        
        for i, (z, y, x) in enumerate(detections_t1):
            conf = confidences_t1[i]
            node_features.append([z, y, x, conf])
            node_indices.append((t_next, i))
        
        node_features = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        
        # Build edge features and edge indices
        edge_features = []
        edge_index = []
        edge_pairs = []
        
        n_t = len(detections_t)
        
        for i in range(n_t):
            for j in range(len(detections_t1)):
                cell1 = (*detections_t[i], confidences_t[i])
                cell2 = (*detections_t1[j], confidences_t1[j])
                
                feat = self.feature_extractor.extract_pair_features(cell1, cell2, time_gap=1)
                edge_features.append(feat)
                edge_index.append([i, n_t + j])
                edge_pairs.append((i, j))
        
        edge_features = torch.tensor(edge_features, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        
        # Predict edge probabilities
        with torch.no_grad():
            edge_probs = self.model(node_features, edge_index, edge_features)
        
        # Create edges for high-probability predictions
        edges = []
        for k, (i, j) in enumerate(edge_pairs):
            prob = edge_probs[k].item()
            if prob >= self.edge_threshold:
                source_id = self._get_node_id(t_current, i, node_ids)
                target_id = self._get_node_id(t_next, j, node_ids)
                
                if source_id is not None and target_id is not None:
                    edges.append(TrackingEdge(source_id=source_id, target_id=target_id))
        
        return edges
    
    def _get_node_id(
        self,
        t: int,
        idx: int,
        node_ids: Dict[int, Tuple[int, int, int, int]]
    ) -> Optional[int]:
        """Get node ID for a given timepoint and detection index."""
        return self._detection_to_node_id.get((t, idx))
