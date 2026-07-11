"""Comprehensive dataset analysis to understand data distribution and patterns."""

from pathlib import Path
from collections import defaultdict
import numpy as np
from data.dataset import DatasetCatalog, EmbryoDataset
from data.models import VoxelScale
import json


def analyze_sample(sample_id: str, catalog: DatasetCatalog, max_timepoints: int = None) -> dict:
    """
    Perform comprehensive analysis on a single sample.
    
    Returns dictionary with all statistics.
    """
    print(f"\nAnalyzing {sample_id}...")
    
    sample = catalog.load_sample(sample_id, require_tracks=True, load_volume=False)
    dataset = EmbryoDataset(sample, normalize=True)
    
    scale = sample.image_metadata.scale
    total_timepoints = sample.image_metadata.shape[0]
    
    if max_timepoints:
        analyze_timepoints = min(max_timepoints, total_timepoints)
    else:
        analyze_timepoints = total_timepoints
    
    stats = {
        'sample_id': sample_id,
        'total_timepoints': total_timepoints,
        'analyzed_timepoints': analyze_timepoints,
        'shape': sample.image_metadata.shape,
        'scale': scale.as_tuple(),
        'total_nodes': sample.tracks.num_nodes,
        'total_edges': sample.tracks.num_edges,
        'total_divisions': len(sample.tracks.division_nodes()),
    }
    
    # Analyze node distribution over time
    nodes_by_time = defaultdict(list)
    for node in sample.tracks.iter_nodes():
        nodes_by_time[node.t].append(node)
    
    # Temporal statistics
    nodes_per_time = [len(nodes_by_time[t]) for t in range(analyze_timepoints)]
    stats['nodes_per_time'] = {
        'mean': np.mean(nodes_per_time),
        'std': np.std(nodes_per_time),
        'min': np.min(nodes_per_time),
        'max': np.max(nodes_per_time),
        'values': nodes_per_time[:20],  # First 20 timepoints
    }
    
    # Spatial distribution
    all_z = []
    all_y = []
    all_x = []
    for node in sample.tracks.iter_nodes():
        if node.t < analyze_timepoints:
            all_z.append(node.z)
            all_y.append(node.y)
            all_x.append(node.x)
    
    stats['spatial_distribution'] = {
        'z_range': (min(all_z), max(all_z)),
        'y_range': (min(all_y), max(all_y)),
        'x_range': (min(all_x), max(all_x)),
        'z_mean': np.mean(all_z),
        'y_mean': np.mean(all_y),
        'x_mean': np.mean(all_x),
    }
    
    # Cell displacement analysis
    displacements = []
    for t in range(analyze_timepoints - 1):
        current_nodes = {node.node_id: node for node in nodes_by_time[t]}
        next_nodes = {node.node_id: node for node in nodes_by_time[t + 1]}
        
        for edge in sample.tracks.edges:
            if edge.source_id in current_nodes and edge.target_id in next_nodes:
                source = current_nodes[edge.source_id]
                target = next_nodes[edge.target_id]
                
                # Physical displacement in micrometers
                dz = (target.z - source.z) * scale.z
                dy = (target.y - source.y) * scale.y
                dx = (target.x - source.x) * scale.x
                displacement = np.sqrt(dz**2 + dy**2 + dx**2)
                displacements.append(displacement)
    
    if displacements:
        stats['displacement'] = {
            'mean_um': np.mean(displacements),
            'std_um': np.std(displacements),
            'min_um': np.min(displacements),
            'max_um': np.max(displacements),
            'median_um': np.median(displacements),
            'p95_um': np.percentile(displacements, 95),
        }
    else:
        stats['displacement'] = None
    
    # Division analysis
    divisions_by_time = defaultdict(int)
    for div_id in sample.tracks.division_nodes():
        div_node = sample.tracks.nodes[div_id]
        if div_node.t < analyze_timepoints:
            divisions_by_time[div_node.t] += 1
    
    stats['divisions_per_time'] = {
        'total': sum(divisions_by_time.values()),
        'by_time': dict(divisions_by_time),
    }
    
    # Intensity analysis (sample a few timepoints)
    intensity_stats = []
    sample_times = [0, analyze_timepoints // 2, analyze_timepoints - 1]
    for t in sample_times:
        if t < total_timepoints:
            volume = dataset.get_timepoint(t)
            intensity_stats.append({
                'timepoint': t,
                'mean': float(volume.mean()),
                'std': float(volume.std()),
                'min': float(volume.min()),
                'max': float(volume.max()),
            })
    
    stats['intensity'] = intensity_stats
    
    # Density analysis (cells per unit volume)
    physical_volume = (
        (stats['spatial_distribution']['z_range'][1] - stats['spatial_distribution']['z_range'][0]) * scale.z *
        (stats['spatial_distribution']['y_range'][1] - stats['spatial_distribution']['y_range'][0]) * scale.y *
        (stats['spatial_distribution']['x_range'][1] - stats['spatial_distribution']['x_range'][0]) * scale.x
    )
    if physical_volume > 0:
        stats['density'] = {
            'cells_per_cubic_micrometer': stats['total_nodes'] / physical_volume,
            'total_volume_um3': physical_volume,
        }
    else:
        stats['density'] = None
    
    return stats


def analyze_multiple_samples(catalog: DatasetCatalog, num_samples: int = 10, max_timepoints: int = 20) -> dict:
    """Analyze multiple samples and aggregate statistics."""
    sample_ids = catalog.list_samples()[:num_samples]
    
    all_stats = []
    for sample_id in sample_ids:
        try:
            stats = analyze_sample(sample_id, catalog, max_timepoints)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error analyzing {sample_id}: {e}")
    
    # Aggregate statistics
    aggregated = {
        'num_samples_analyzed': len(all_stats),
        'samples': all_stats,
    }
    
    # Cross-sample statistics
    total_nodes = [s['total_nodes'] for s in all_stats]
    total_edges = [s['total_edges'] for s in all_stats]
    total_divisions = [s['total_divisions'] for s in all_stats]
    
    aggregated['cross_sample'] = {
        'nodes': {
            'mean': np.mean(total_nodes),
            'std': np.std(total_nodes),
            'min': np.min(total_nodes),
            'max': np.max(total_nodes),
        },
        'edges': {
            'mean': np.mean(total_edges),
            'std': np.std(total_edges),
            'min': np.min(total_edges),
            'max': np.max(total_edges),
        },
        'divisions': {
            'mean': np.mean(total_divisions),
            'std': np.std(total_divisions),
            'min': np.min(total_divisions),
            'max': np.max(total_divisions),
        },
    }
    
    # Displacement aggregation
    displacements = [s['displacement']['mean_um'] for s in all_stats if s['displacement']]
    if displacements:
        aggregated['cross_sample']['displacement'] = {
            'mean_um': np.mean(displacements),
            'std_um': np.std(displacements),
            'min_um': np.min(displacements),
            'max_um': np.max(displacements),
        }
    
    return aggregated


def print_summary(aggregated: dict):
    """Print human-readable summary of analysis."""
    print("\n" + "="*70)
    print("DATASET ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nSamples analyzed: {aggregated['num_samples_analyzed']}")
    
    cs = aggregated['cross_sample']
    print(f"\n--- Cross-Sample Statistics ---")
    print(f"Nodes per sample: {cs['nodes']['mean']:.1f} ± {cs['nodes']['std']:.1f} (range: {cs['nodes']['min']}-{cs['nodes']['max']})")
    print(f"Edges per sample: {cs['edges']['mean']:.1f} ± {cs['edges']['std']:.1f} (range: {cs['edges']['min']}-{cs['edges']['max']})")
    print(f"Divisions per sample: {cs['divisions']['mean']:.1f} ± {cs['divisions']['std']:.1f} (range: {cs['divisions']['min']}-{cs['divisions']['max']})")
    
    if 'displacement' in cs:
        print(f"\n--- Cell Displacement ---")
        print(f"Mean displacement between frames: {cs['displacement']['mean_um']:.3f} ± {cs['displacement']['std_um']:.3f} µm")
        print(f"Range: {cs['displacement']['min_um']:.3f} - {cs['displacement']['max_um']:.3f} µm")
        print(f"Note: Maximum matching distance is 7.0 µm")
    
    print(f"\n--- Sample-by-Sample Details ---")
    for stats in aggregated['samples']:
        print(f"\n{stats['sample_id']}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}, Divisions: {stats['total_divisions']}")
        print(f"  Nodes/time: {stats['nodes_per_time']['mean']:.1f} ± {stats['nodes_per_time']['std']:.1f}")
        if stats['displacement']:
            print(f"  Displacement: {stats['displacement']['mean_um']:.3f} ± {stats['displacement']['std_um']:.3f} µm")
        if stats['density']:
            print(f"  Density: {stats['density']['cells_per_cubic_micrometer']:.6f} cells/µm³")


def main():
    """Run comprehensive dataset analysis."""
    data_root = "../train"
    catalog = DatasetCatalog(data_root)
    
    print("="*70)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("="*70)
    
    # Analyze first 10 samples with 20 timepoints each for speed
    aggregated = analyze_multiple_samples(catalog, num_samples=10, max_timepoints=20)
    
    # Print summary
    print_summary(aggregated)
    
    # Save full results
    output_path = Path("outputs/dataset_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
