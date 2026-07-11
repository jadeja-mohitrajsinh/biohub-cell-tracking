"""Debug script to check GT coordinates vs volume bounds."""

from data.dataset import DatasetCatalog, EmbryoDataset

# Load sample
catalog = DatasetCatalog("../train")
sample_id = catalog.list_samples()[0]
sample = catalog.load_sample(sample_id, require_tracks=True, load_volume=False)

print(f"Sample: {sample_id}")
print(f"Volume shape: {sample.image_metadata.shape}")
print(f"GT nodes: {sample.tracks.num_nodes}")

# Check GT coordinates
gt_nodes = list(sample.tracks.iter_nodes())
print(f"\nFirst 10 GT nodes:")
for i, node in enumerate(gt_nodes[:10]):
    in_bounds = (
        0 <= node.t < sample.image_metadata.shape[0] and
        0 <= node.z < sample.image_metadata.shape[1] and
        0 <= node.y < sample.image_metadata.shape[2] and
        0 <= node.x < sample.image_metadata.shape[3]
    )
    print(f"  {i}: t={node.t}, z={node.z}, y={node.y}, x={node.x} - in_bounds={in_bounds}")

# Check if any nodes are out of bounds
out_of_bounds = []
for node in gt_nodes:
    if not (
        0 <= node.t < sample.image_metadata.shape[0] and
        0 <= node.z < sample.image_metadata.shape[1] and
        0 <= node.y < sample.image_metadata.shape[2] and
        0 <= node.x < sample.image_metadata.shape[3]
    ):
        out_of_bounds.append(node)

print(f"\nOut of bounds nodes: {len(out_of_bounds)}")
if out_of_bounds:
    for node in out_of_bounds[:5]:
        print(f"  t={node.t}, z={node.z}, y={node.y}, x={node.x}")

# Load a volume and check intensity at GT positions
dataset = EmbryoDataset(sample, normalize=True)
print(f"\nIntensity at first 5 GT positions:")
for i, node in enumerate(gt_nodes[:5]):
    try:
        volume = dataset.get_timepoint(node.t)
        intensity = volume[node.z, node.y, node.x]
        print(f"  {i}: intensity={intensity:.4f}")
    except IndexError as e:
        print(f"  {i}: IndexError - {e}")
