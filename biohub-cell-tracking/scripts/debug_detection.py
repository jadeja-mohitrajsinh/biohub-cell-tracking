"""Debug script to check if detections are near GT positions."""

import numpy as np
from data.dataset import DatasetCatalog, EmbryoDataset
from detection.gt_guided_detector import GTGuidedDetector
import yaml

# Load config
with open("configs/classical_baseline.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load sample
catalog = DatasetCatalog("../train")
sample_id = catalog.list_samples()[0]
sample = catalog.load_sample(sample_id, require_tracks=True, load_volume=False)
dataset = EmbryoDataset(sample, normalize=True)

# Initialize detector
detector = GTGuidedDetector(config["detection"]["gt_guided"])
detector.learn_from_gt(sample, dataset, num_timepoints=20)

# Run detection on a single timepoint with GT nodes
t = 0
volume = dataset.get_timepoint(t)
result = detector.detect(volume)

print(f"Timepoint {t}:")
print(f"  Detections: {len(result.centroids)}")

# Get GT nodes at this timepoint
gt_nodes = [node for node in sample.tracks.iter_nodes() if node.t == t]
print(f"  GT nodes: {len(gt_nodes)}")

# Check if any detection is within 5 voxels of any GT node
if result.centroids and gt_nodes:
    for i, det in enumerate(result.centroids[:5]):
        print(f"\n  Detection {i}: z={det[0]}, y={det[1]}, x={det[2]}")
        for j, gt in enumerate(gt_nodes[:3]):
            dist = np.sqrt((det[0]-gt.z)**2 + (det[1]-gt.y)**2 + (det[2]-gt.x)**2)
            print(f"    Distance to GT {j}: {dist:.2f} voxels")
    
    # Check if any GT node is within 5 voxels of any detection
    print(f"\n  Checking if GT nodes are near detections:")
    for i, gt in enumerate(gt_nodes[:5]):
        min_dist = float('inf')
        for det in result.centroids:
            dist = np.sqrt((det[0]-gt.z)**2 + (det[1]-gt.y)**2 + (det[2]-gt.x)**2)
            if dist < min_dist:
                min_dist = dist
        print(f"    GT {i}: nearest detection at {min_dist:.2f} voxels")
else:
    print("  No detections or no GT nodes at this timepoint")

# Check the binary mask
from scipy.ndimage import gaussian_filter, label
volume_norm = volume / volume.max()
smoothed = gaussian_filter(volume_norm, sigma=detector.sigma)
binary = smoothed > detector.learned_threshold

print(f"\nBinary mask stats:")
print(f"  Threshold: {detector.learned_threshold:.4f}")
print(f"  Binary pixels: {binary.sum()}")
print(f"  Total pixels: {binary.size}")
print(f"  Binary ratio: {binary.sum() / binary.size:.6f}")

# Check if GT positions are in binary mask
print(f"\n  GT positions in binary mask:")
for gt in gt_nodes[:5]:
    in_mask = binary[gt.z, gt.y, gt.x]
    print(f"    GT z={gt.z}, y={gt.y}, x={gt.x}: in_mask={in_mask}, intensity={volume[gt.z, gt.y, gt.x]:.4f}")
