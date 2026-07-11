"""Simple baseline cell detection and tracking for quick testing."""

from pathlib import Path
import numpy as np
from scipy.ndimage import label, center_of_mass
from data.dataset import DatasetCatalog, EmbryoDataset
from data.models import CellNode, TrackingGraph, TrackingEdge, VoxelScale
from metrics.evaluator import evaluate_graph


def simple_cell_detection(volume: np.ndarray, threshold: float = 0.8) -> list[tuple[int, int, int, int]]:
    """
    Simple threshold-based cell detection.
    
    Args:
        volume: 3D array (Z, Y, X) for one timepoint
        threshold: Intensity threshold (0-1 after normalization)
    
    Returns:
        List of (t, z, y, x) detections
    """
    # Normalize volume to 0-1
    if volume.max() > 0:
        volume_norm = volume / volume.max()
    else:
        volume_norm = volume
    
    # Threshold and find connected components
    binary = volume_norm > threshold
    labeled, num_features = label(binary)
    
    detections = []
    for label_id in range(1, num_features + 1):
        # Get center of mass for each component
        mask = labeled == label_id
        if mask.sum() > 5:  # Filter very small components
            center = center_of_mass(mask)
            z, y, x = [int(round(c)) for c in center]
            # Ensure coordinates are within bounds
            z = max(0, min(z, volume.shape[0] - 1))
            y = max(0, min(y, volume.shape[1] - 1))
            x = max(0, min(x, volume.shape[2] - 1))
            detections.append((z, y, x))
    
    return detections


def simple_tracking(detections_by_time: dict[int, list[tuple[int, int, int]]], 
                    max_distance: float = 5.0) -> list[TrackingEdge]:
    """
    Simple nearest-neighbor tracking across consecutive timepoints.
    
    Args:
        detections_by_time: Dict mapping time -> list of (z, y, x) detections
        max_distance: Maximum physical distance (micrometers) for linking
    
    Returns:
        List of TrackingEdge objects
    """
    edges = []
    scale = VoxelScale()
    
    sorted_times = sorted(detections_by_time.keys())
    
    for i in range(len(sorted_times) - 1):
        t_current = sorted_times[i]
        t_next = sorted_times[i + 1]
        
        if t_current not in detections_by_time or t_next not in detections_by_time:
            continue
            
        current_detections = detections_by_time[t_current]
        next_detections = detections_by_time[t_next]
        
        # Simple greedy matching: each current detection links to nearest next detection
        for idx_curr, (z_curr, y_curr, x_curr) in enumerate(current_detections):
            best_dist = max_distance
            best_idx = None
            
            for idx_next, (z_next, y_next, x_next) in enumerate(next_detections):
                # Calculate physical distance
                dz = (z_next - z_curr) * scale.z
                dy = (y_next - y_curr) * scale.y
                dx = (x_next - x_curr) * scale.x
                dist = np.sqrt(dz**2 + dy**2 + dx**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx_next
            
            if best_idx is not None:
                # Create edge using node_id = time_index * 1000 + detection_index
                source_id = t_current * 1000 + idx_curr
                target_id = t_next * 1000 + best_idx
                edges.append(TrackingEdge(source_id=source_id, target_id=target_id))
    
    return edges


def run_baseline_on_sample(sample_id: str, data_root: str, num_timepoints: int = 10):
    """
    Run simple baseline detection and tracking on one sample.
    
    Args:
        sample_id: Sample identifier
        data_root: Path to data directory
        num_timepoints: Number of timepoints to process (for quick testing)
    """
    print(f"\n{'='*60}")
    print(f"Processing sample: {sample_id}")
    print(f"{'='*60}")
    
    # Load sample
    catalog = DatasetCatalog(data_root)
    sample = catalog.load_sample(sample_id, require_tracks=True, load_volume=False)
    dataset = EmbryoDataset(sample, normalize=True)
    
    print(f"Ground truth: {sample.tracks.num_nodes} nodes, {sample.tracks.num_edges} edges")
    
    # Run detection on limited timepoints
    detections_by_time = {}
    node_id_counter = 0
    nodes = {}
    
    max_t = min(num_timepoints, sample.image_metadata.shape[0])
    print(f"\nRunning detection on timepoints 0-{max_t-1}...")
    
    for t in range(max_t):
        volume = dataset.get_timepoint(t)
        detections = simple_cell_detection(volume, threshold=0.7)
        
        print(f"  Time {t}: {len(detections)} detections")
        
        # Store detections with node IDs
        time_nodes = []
        for z, y, x in detections:
            node_id = t * 1000 + len(time_nodes)
            nodes[node_id] = CellNode(node_id=node_id, t=t, z=z, y=y, x=x)
            time_nodes.append((z, y, x))
        
        detections_by_time[t] = time_nodes
    
    # Run tracking
    print(f"\nRunning tracking...")
    edges = simple_tracking(detections_by_time, max_distance=3.0)
    print(f"  Generated {len(edges)} edges")
    
    # Create predicted graph
    predicted_graph = TrackingGraph(nodes=nodes, edges=edges)
    
    # Evaluate
    print(f"\nEvaluating predictions...")
    result = evaluate_graph(predicted_graph, sample.tracks, sample.image_metadata.scale)
    
    print(f"\nResults:")
    print(f"  Edge Jaccard: {result['edge_jaccard']:.4f}")
    print(f"  Division Jaccard: {result['division_jaccard']:.4f}")
    print(f"  Total Score: {result['score']:.4f}")
    print(f"  Matched nodes: {result['matched_nodes']}")
    
    return result


def main():
    """Run baseline on a small subset of training data."""
    data_root = "../train"
    
    # Use 3 samples for quick testing
    test_samples = ["6bba_fe670320", "44b6_0113de3b", "44b6_0b24845f"]
    
    print("="*60)
    print("SIMPLE BASELINE - Quick Test Pipeline")
    print("="*60)
    
    results = []
    for sample_id in test_samples:
        try:
            result = run_baseline_on_sample(sample_id, data_root, num_timepoints=10)
            results.append((sample_id, result))
        except Exception as e:
            print(f"\nError processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for sample_id, result in results:
        print(f"{sample_id}: Score={result['score']:.4f}, EdgeJ={result['edge_jaccard']:.4f}, DivJ={result['division_jaccard']:.4f}")
    
    avg_score = np.mean([r['score'] for _, r in results])
    avg_edge_j = np.mean([r['edge_jaccard'] for _, r in results])
    avg_div_j = np.mean([r['division_jaccard'] for _, r in results])
    
    print(f"\nAverage: Score={avg_score:.4f}, EdgeJ={avg_edge_j:.4f}, DivJ={avg_div_j:.4f}")


if __name__ == "__main__":
    main()
