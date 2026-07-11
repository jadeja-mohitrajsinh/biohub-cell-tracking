"""Test GT-guided detector to understand intensity distribution."""

from pathlib import Path
import logging
import yaml

from data.dataset import DatasetCatalog, EmbryoDataset
from metrics.evaluator import evaluate_graph
from detection.gt_guided_detector import GTGuidedDetector
from tracking.hungarian import HungarianTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('outputs/gt_guided_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Test GT-guided detector."""
    data_root = "../train"
    config_path = "configs/classical_baseline.yaml"
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    # Initialize dataset
    catalog = DatasetCatalog(data_root)
    sample_id = catalog.list_samples()[0]  # First sample
    
    logger.info(f"\nTesting GT-guided detector on sample: {sample_id}")
    
    # Load sample with GT
    sample = catalog.load_sample(sample_id, require_tracks=True, load_volume=False)
    logger.info(f"Ground truth: {sample.tracks.num_nodes} nodes, {sample.tracks.num_edges} edges")
    
    # Create dataset for loading
    dataset = EmbryoDataset(sample, normalize=True)
    
    # Initialize GT-guided detector
    detector = GTGuidedDetector(config["detection"]["gt_guided"])
    
    # Learn from GT
    detector.learn_from_gt(sample, dataset, num_timepoints=20)
    
    # Debug: Check GT positions vs volume bounds
    logger.info(f"\nDebug: Checking GT positions vs volume bounds")
    logger.info(f"Volume shape: {sample.image_metadata.shape}")
    
    gt_nodes_list = list(sample.tracks.iter_nodes())
    logger.info(f"GT nodes: {len(gt_nodes_list)}")
    
    # Check first few GT nodes
    for i, node in enumerate(gt_nodes_list[:5]):
        logger.info(f"  GT node {i}: t={node.t}, z={node.z}, y={node.y}, x={node.x}")
        logger.info(f"    In bounds: t<{sample.image_metadata.shape[0]}, z<{sample.image_metadata.shape[1]}, y<{sample.image_metadata.shape[2]}, x<{sample.image_metadata.shape[3]}")
    
    # Run detection on all timepoints
    detections_by_time = {}
    total_timepoints = sample.image_metadata.shape[0]
    
    logger.info(f"\nRunning detection on {total_timepoints} timepoints...")
    
    for t in range(total_timepoints):
        volume = dataset.get_timepoint(t)
        result = detector.detect(volume)
        detections_by_time[t] = result.centroids
        
        if t % 10 == 0:
            logger.debug(f"Timepoint {t}: {len(result.centroids)} detections")
    
    total_detections = sum(len(dets) for dets in detections_by_time.values())
    logger.info(f"\nDetection complete: {total_detections} total detections")
    
    # Run tracking
    tracker = HungarianTracker(config["tracking"], sample.image_metadata.scale)
    tracking_result = tracker.track(detections_by_time)
    
    logger.info(f"Tracking complete: {len(tracking_result.edges)} edges")
    
    # Build TrackingGraph
    from data.models import CellNode, TrackingGraph
    nodes = {}
    for node_id, (t, z, y, x) in tracking_result.node_ids.items():
        nodes[node_id] = CellNode(node_id=node_id, t=t, z=z, y=y, x=x)
    
    predicted_graph = TrackingGraph(
        nodes=nodes,
        edges=tracking_result.edges,
        metadata=None
    )
    
    # Evaluate
    result = evaluate_graph(predicted_graph, sample.tracks, sample.image_metadata.scale)
    
    logger.info(f"\nResults:")
    logger.info(f"  Edge TP: {result['edge_tp']}")
    logger.info(f"  Edge FP: {result['edge_fp']}")
    logger.info(f"  Edge FN: {result['edge_fn']}")
    logger.info(f"  Edge Jaccard: {result['edge_jaccard']:.4f}")
    logger.info(f"  Adjusted Edge Jaccard: {result['adjusted_edge_jaccard']:.4f}")
    logger.info(f"  Division Jaccard: {result['division_jaccard']:.4f}")
    logger.info(f"  Total Score: {result['score']:.4f}")
    logger.info(f"  Matched nodes: {result['matched_nodes']}")
    
    logger.info(f"\nTest complete. Results saved to outputs/gt_guided_test.log")
    
    # Debug: Check why 0 matches
    logger.info(f"\nDebug: Checking why 0 matched nodes")
    logger.info(f"Learned threshold: {detector.learned_threshold}")
    
    # Sample a few GT positions and check intensity
    logger.info(f"\nDebug: Sampling intensity at GT positions")
    for i, node in enumerate(gt_nodes_list[:5]):
        volume = dataset.get_timepoint(node.t)
        intensity = volume[node.z, node.y, node.x]
        logger.info(f"  GT node {i}: intensity={intensity:.4f} at ({node.t}, {node.z}, {node.y}, {node.x})")


if __name__ == "__main__":
    main()
