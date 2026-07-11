"""Inspect ground truth division structure for a sample."""

import sys
sys.path.append('e:/kaggle/biohub-cell-tracking')

from data.dataset import DatasetCatalog
from data.models import TrackingGraph
from metrics.matching import match_nodes
from metrics.edge_jaccard import division_counts

def inspect_division(sample_id: str, data_root: str):
    """Inspect ground truth division structure."""
    catalog = DatasetCatalog(data_root)
    sample = catalog.load_sample(sample_id, load_volume=False)
    
    # Load ground truth
    if sample.has_ground_truth:
        gt_graph = sample.tracks
    else:
        print("Sample has no ground truth")
        return
    
    print(f"Ground truth graph: {len(gt_graph.nodes)} nodes, {len(gt_graph.edges)} edges")
    
    # Find division nodes
    gt_divisions = gt_graph.division_nodes()
    print(f"\nGround truth divisions: {len(gt_divisions)}")
    
    for divider_id in gt_divisions:
        divider = gt_graph.nodes[divider_id]
        print(f"\nDivision at node {divider_id}:")
        print(f"  Position: t={divider.t}, z={divider.z}, y={divider.y}, x={divider.x}")
        
        # Get parent lineage
        parent_sources = gt_graph.incoming(divider_id)
        print(f"  Parent sources: {parent_sources}")
        
        # Get children
        children = gt_graph.outgoing(divider_id)
        print(f"  Children: {children}")
        
        if len(children) == 2:
            child1 = gt_graph.nodes[children[0]]
            child2 = gt_graph.nodes[children[1]]
            print(f"  Child 1: t={child1.t}, z={child1.z}, y={child1.y}, x={child1.x}")
            print(f"  Child 2: t={child2.t}, z={child2.z}, y={child2.y}, x={child2.x}")
            
            # Compute distances
            from data.models import VoxelScale
            scale = sample.image_metadata.scale
            import numpy as np
            
            parent_pos = np.array([divider.z, divider.y, divider.x])
            child1_pos = np.array([child1.z, child1.y, child1.x])
            child2_pos = np.array([child2.z, child2.y, child2.x])
            
            dist1 = np.linalg.norm((child1_pos - parent_pos) * np.array([scale.z, scale.y, scale.x]))
            dist2 = np.linalg.norm((child2_pos - parent_pos) * np.array([scale.z, scale.y, scale.x]))
            dist12 = np.linalg.norm((child2_pos - child1_pos) * np.array([scale.z, scale.y, scale.x]))
            
            print(f"  Parent->Child1 distance: {dist1:.2f} um")
            print(f"  Parent->Child2 distance: {dist2:.2f} um")
            print(f"  Child1->Child2 distance: {dist12:.2f} um")
            
            # Compute angle
            v1 = child1_pos - parent_pos
            v2 = child2_pos - parent_pos
            angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
            angle_deg = np.degrees(angle_rad)
            print(f"  Division angle: {angle_deg:.1f}°")

if __name__ == "__main__":
    inspect_division("44b6_12dfb391", "../train")
