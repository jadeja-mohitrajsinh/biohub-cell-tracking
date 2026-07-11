"""Classical pipeline: Frangi + Watershed + Hungarian."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from data.dataset import EmbryoDataset
from data.models import CellNode, EmbryoSample, TrackingGraph, TrackingEdge, VoxelScale
from detection.base import Detector
from detection.frangi_detector import FrangiDetector
from detection.watershed import WatershedSegmenter
from detection.improved_detector import ImprovedDetector
from detection.gt_guided_detector import GTGuidedDetector

# Optional torch-based detectors
try:
    from detection.unet_detector import UNetDetector
except ImportError:
    UNetDetector = None

try:
    from detection.unet_3d import UNet3DDetector
except ImportError:
    UNet3DDetector = None

from pipeline.base import Pipeline
from tracking.base import Tracker
from tracking.hungarian import HungarianTracker
from tracking.enhanced_hungarian import EnhancedHungarianTracker, EnhancedHungarianTrackerWithSample

# Optional deep learning trackers
try:
    from tracking.gnn_tracker import GNNTracker
except ImportError:
    GNNTracker = None

# Track memory tracker with Kalman prediction
try:
    from tracking.track_memory_tracker import TrackMemoryTracker
except ImportError:
    TrackMemoryTracker = None

from lineage.division_detector import DivisionAwareTracker
from tracking.gap_closing import PostProcessingPipeline

logger = logging.getLogger(__name__)


class ClassicalPipeline(Pipeline):
    """Classical detection + tracking pipeline."""
    
    def __init__(self, config: dict) -> None:
        """Initialize classical pipeline.
        
        Args:
            config: Configuration dictionary with sections:
                - detection: Detection configuration
                - tracking: Tracking configuration
                - pipeline: Pipeline configuration
        """
        super().__init__(config)
        
        self.detection_config = config.get("detection", {})
        self.tracking_config = config.get("tracking", {})
        self.pipeline_config = config.get("pipeline", {})
        
        # Initialize detector
        self.detector: Detector
        detection_method = self.detection_config.get("method", "improved")
        if detection_method == "frangi":
            self.detector = FrangiDetector(self.detection_config.get("frangi", {}))
        elif detection_method == "watershed":
            self.detector = WatershedSegmenter(self.detection_config.get("watershed", {}))
        elif detection_method == "improved":
            self.detector = ImprovedDetector(self.detection_config.get("improved", {}))
        elif detection_method == "gt_guided":
            self.detector = GTGuidedDetector(self.detection_config.get("gt_guided", {}))
        elif detection_method == "unet":
            if UNetDetector is None:
                raise ValueError("UNetDetector requires torch. Install with: pip install -e '.[dl]'")
            self.detector = UNetDetector(self.detection_config.get("unet", {}))
        elif detection_method == "unet_3d":
            if UNet3DDetector is None:
                raise ValueError("UNet3DDetector requires torch. Install with: pip install -e '.[dl]'")
            self.detector = UNet3DDetector(
                config=self.detection_config.get("unet_3d", {}),
                model_path=self.detection_config.get("unet_3d", {}).get("model_path"),
                device=self.detection_config.get("unet_3d", {}).get("device", "cuda")
            )
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")
        
        # Initialize tracker (will be set when scale is known)
        self.tracker: Tracker | None = None
        
        # Post-processing options
        self.enable_division_detection = self.tracking_config.get("enable_division", False)
        self.enable_post_processing = self.tracking_config.get("enable_post_processing", False)
        
        logger.info(f"ClassicalPipeline initialized with detection={detection_method}")
    
    def process(self, sample: EmbryoSample) -> TrackingGraph:
        """Process sample with classical pipeline.
        
        Args:
            sample: EmbryoSample with image data
        
        Returns:
            TrackingGraph with predicted tracks
        """
        logger.info(f"Processing sample {sample.sample_id}")
        
        # Create dataset for loading
        normalize = self.pipeline_config.get("normalize", True)
        quantile_low = self.pipeline_config.get("quantile_low", 0.01)
        quantile_high = self.pipeline_config.get("quantile_high", 0.99)
        
        dataset = EmbryoDataset(
            sample,
            normalize=normalize,
            quantile_low=quantile_low,
            quantile_high=quantile_high
        )
        
        # Initialize tracker with scale
        if self.tracker is None:
            tracking_method = self.tracking_config.get("method", "hungarian")
            
            if tracking_method == "enhanced_hungarian":
                # Use enhanced Hungarian with sample for volume access
                self.tracker = EnhancedHungarianTrackerWithSample(
                    config=self.tracking_config,
                    scale=sample.image_metadata.scale,
                    sample=sample,
                )
            elif tracking_method == "track_memory":
                # Use track memory tracker with Kalman prediction
                if TrackMemoryTracker is None:
                    raise ValueError("TrackMemoryTracker requires dependencies")
                self.tracker = TrackMemoryTracker(
                    config=self.tracking_config,
                    scale=sample.image_metadata.scale,
                )
            else:
                # Use standard Hungarian
                self.tracker = HungarianTracker(self.tracking_config, sample.image_metadata.scale)
        
        # Run detection on all timepoints
        detections_by_time: Dict[int, List[tuple]] = {}
        confidences_by_time: Dict[int, List[float]] = {}
        total_timepoints = sample.image_metadata.shape[0]
        
        logger.info(f"Running detection on {total_timepoints} timepoints")
        
        for t in range(total_timepoints):
            volume = dataset.get_timepoint(t)
            result = self.detector.detect(volume)
            detections_by_time[t] = result.centroids
            confidences_by_time[t] = result.confidence if result.confidence else [1.0] * len(result.centroids)
            
            if t % 10 == 0:
                logger.debug(f"Timepoint {t}: {len(result.centroids)} detections")
        
        total_detections = sum(len(dets) for dets in detections_by_time.values())
        logger.info(f"Detection complete: {total_detections} total detections")
        
        # Run tracking (pass confidences if using enhanced tracker)
        if isinstance(self.tracker, EnhancedHungarianTracker):
            tracking_result = self.tracker.track(detections_by_time, confidences_by_time)
        else:
            tracking_result = self.tracker.track(detections_by_time)
        
        # Build TrackingGraph
        nodes = {}
        for node_id, (t, z, y, x) in tracking_result.node_ids.items():
            nodes[node_id] = CellNode(node_id=node_id, t=t, z=z, y=y, x=x)
        
        graph = TrackingGraph(
            nodes=nodes,
            edges=tracking_result.edges,
            metadata=None
        )
        
        logger.info(f"Tracking complete: {len(nodes)} nodes, {len(tracking_result.edges)} edges")
        
        # Division detection is now done during tracking if enabled
        # Skip post-processing division detection to avoid conflicts
        # if self.enable_division_detection:
        #     logger.info("Running division detection")
        #     division_tracker = DivisionAwareTracker(
        #         scale=sample.image_metadata.scale,
        #         config=self.tracking_config.get("division", {})
        #     )
        #     graph = division_tracker.post_process_divisions(graph, detections_by_time, volume)
        #     logger.info(f"Division detection complete: {len(graph.edges)} edges")
        
        # Post-processing (gap closing, track continuity)
        if self.enable_post_processing:
            logger.info("Running post-processing")
            post_processor = PostProcessingPipeline(
                scale=sample.image_metadata.scale,
                config=self.tracking_config.get("post_processing", {})
            )
            graph = post_processor.process(graph, detections_by_time)
            logger.info(f"Post-processing complete: {len(graph.edges)} edges")
        
        logger.info(f"Pipeline complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        return graph
