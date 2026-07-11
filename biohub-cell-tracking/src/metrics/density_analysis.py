"""Density-stratified evaluation for identifying score leakage by local cell density."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from data.models import CellNode, TrackingGraph, VoxelScale
from metrics.matching import match_nodes

logger = logging.getLogger(__name__)


@dataclass
class DensityBin:
    """Density bin for stratified analysis."""
    
    bin_id: int
    density_range: Tuple[float, float]  # (min, max) cells per unit volume
    gt_nodes: int
    pred_nodes: int
    matched_nodes: int
    edge_tp: int
    edge_fp: int
    edge_fn: int
    edge_jaccard: float
    
    @property
    def recall(self) -> float:
        """Node recall (matched / gt)."""
        return self.matched_nodes / self.gt_nodes if self.gt_nodes > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Node precision (matched / pred)."""
        return self.matched_nodes / self.pred_nodes if self.pred_nodes > 0 else 0.0


def compute_local_density(
    node: CellNode,
    all_nodes: List[CellNode],
    scale: VoxelScale,
    radius_um: float = 10.0
) -> float:
    """Compute local cell density around a node.
    
    Args:
        node: Center node
        all_nodes: All nodes in the sample
        scale: VoxelScale for physical distance
        radius_um: Radius for local neighborhood (micrometers)
    
    Returns:
        Local density (cells per cubic micrometer)
    """
    count = 0
    for other in all_nodes:
        if other.node_id == node.node_id:
            continue
        
        # Compute physical distance
        t1, z1, y1, x1 = node.physical_coords(scale)
        t2, z2, y2, x2 = other.physical_coords(scale)
        
        # Only consider same timepoint for spatial density
        if t1 != t2:
            continue
        
        distance = np.sqrt((z2 - z1)**2 + (y2 - y1)**2 + (x2 - x1)**2)
        
        if distance <= radius_um:
            count += 1
    
    # Volume of sphere
    volume = (4/3) * np.pi * (radius_um ** 3)
    
    return count / volume if volume > 0 else 0.0


def stratify_by_density(
    predicted: TrackingGraph,
    ground_truth: TrackingGraph,
    scale: VoxelScale,
    *,
    num_bins: int = 5,
    radius_um: float = 10.0
) -> List[DensityBin]:
    """Stratify evaluation results by local cell density.
    
    Args:
        predicted: Predicted tracking graph
        ground_truth: Ground truth tracking graph
        scale: VoxelScale
        num_bins: Number of density bins
        radius_um: Radius for local density computation
    
    Returns:
        List of DensityBin objects
    """
    # Compute local density for all GT nodes
    gt_nodes_list = list(ground_truth.iter_nodes())
    densities = []
    
    for node in gt_nodes_list:
        density = compute_local_density(node, gt_nodes_list, scale, radius_um)
        densities.append(density)
    
    # Create density bins
    if not densities:
        return []
    
    min_density = min(densities)
    max_density = max(densities)
    
    if min_density == max_density:
        # All nodes have same density
        bin_edges = [min_density - 1, max_density + 1]
    else:
        bin_edges = np.linspace(min_density, max_density, num_bins + 1).tolist()
    
    # Assign nodes to bins
    node_to_bin: Dict[int, int] = {}
    for node, density in zip(gt_nodes_list, densities):
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= density < bin_edges[i + 1]:
                node_to_bin[node.node_id] = i
                break
        else:
            # Handle edge case
            node_to_bin[node.node_id] = num_bins - 1
    
    # Match nodes
    matching = match_nodes(predicted, ground_truth, scale)
    
    # Compute edge counts per bin
    gt_edges = {(edge.source_id, edge.target_id) for edge in ground_truth.edges}
    gt_outgoing = {node_id: ground_truth.outgoing(node_id) for node_id in ground_truth.nodes}
    gt_incoming = {node_id: ground_truth.incoming(node_id) for node_id in ground_truth.nodes}
    
    bins: List[Dict] = [{} for _ in range(num_bins)]
    
    # Initialize bins
    for i in range(num_bins):
        bins[i] = {
            'gt_nodes': 0,
            'pred_nodes': 0,
            'matched_nodes': 0,
            'edge_tp': 0,
            'edge_fp': 0,
            'edge_fn': 0,
        }
    
    # Count GT nodes per bin
    for node in gt_nodes_list:
        bin_idx = node_to_bin[node.node_id]
        bins[bin_idx]['gt_nodes'] += 1
    
    # Count predicted nodes per bin (based on matched GT nodes)
    for pred_id, gt_id in matching.pred_to_gt.items():
        if gt_id in node_to_bin:
            bin_idx = node_to_bin[gt_id]
            bins[bin_idx]['pred_nodes'] += 1
            bins[bin_idx]['matched_nodes'] += 1
    
    # Count unmatched predicted nodes (assign to bin of nearest GT node)
    for pred_node in predicted.iter_nodes():
        if pred_node.node_id not in matching.pred_to_gt:
            # Find nearest GT node
            min_dist = float('inf')
            nearest_gt_id = None
            for gt_node in gt_nodes_list:
                t1, z1, y1, x1 = pred_node.physical_coords(scale)
                t2, z2, y2, x2 = gt_node.physical_coords(scale)
                dist = np.sqrt((t2 - t1)**2 + (z2 - z1)**2 + (y2 - y1)**2 + (x2 - x1)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_gt_id = gt_node.node_id
            
            if nearest_gt_id in node_to_bin:
                bin_idx = node_to_bin[nearest_gt_id]
                bins[bin_idx]['pred_nodes'] += 1
    
    # Count edges per bin
    for edge in predicted.edges:
        gt_source = matching.pred_to_gt.get(edge.source_id)
        gt_target = matching.pred_to_gt.get(edge.target_id)
        
        if gt_source is None or gt_target is None:
            continue
        
        # Determine bin (use source node's bin)
        bin_idx = node_to_bin.get(gt_source, 0)
        
        gt_pair = (gt_source, gt_target)
        if gt_pair in gt_edges:
            bins[bin_idx]['edge_tp'] += 1
        else:
            target_conflict = gt_target in gt_incoming and len(gt_incoming[gt_target]) > 0
            source_conflict = gt_source in gt_outgoing and len(gt_outgoing[gt_source]) > 0
            if target_conflict or source_conflict:
                bins[bin_idx]['edge_fp'] += 1
    
    # Count FN edges per bin
    for edge in ground_truth.edges:
        bin_idx = node_to_bin.get(edge.source_id, 0)
        bins[bin_idx]['edge_fn'] += 1
    
    # Create DensityBin objects
    density_bins = []
    for i in range(min(num_bins, len(bin_edges) - 1)):
        bin_data = bins[i]
        edge_jaccard = bin_data['edge_tp'] / (bin_data['edge_tp'] + bin_data['edge_fp'] + bin_data['edge_fn']) \
            if (bin_data['edge_tp'] + bin_data['edge_fp'] + bin_data['edge_fn']) > 0 else 1.0
        
        density_bins.append(DensityBin(
            bin_id=i,
            density_range=(bin_edges[i], bin_edges[i + 1]),
            gt_nodes=bin_data['gt_nodes'],
            pred_nodes=bin_data['pred_nodes'],
            matched_nodes=bin_data['matched_nodes'],
            edge_tp=bin_data['edge_tp'],
            edge_fp=bin_data['edge_fp'],
            edge_fn=bin_data['edge_fn'],
            edge_jaccard=edge_jaccard
        ))
    
    return density_bins


def print_density_report(bins: List[DensityBin]) -> None:
    """Print density-stratified evaluation report.
    
    Args:
        bins: List of DensityBin objects
    """
    logger.info("\n" + "="*70)
    logger.info("DENSITY-STRATIFIED EVALUATION REPORT")
    logger.info("="*70)
    
    for bin_data in bins:
        logger.info(
            f"\nBin {bin_data.bin_id}: Density [{bin_data.density_range[0]:.6f}, {bin_data.density_range[1]:.6f}) cells/µm³"
        )
        logger.info(f"  GT nodes: {bin_data.gt_nodes}")
        logger.info(f"  Pred nodes: {bin_data.pred_nodes}")
        logger.info(f"  Matched nodes: {bin_data.matched_nodes}")
        logger.info(f"  Node recall: {bin_data.recall:.4f}")
        logger.info(f"  Node precision: {bin_data.precision:.4f}")
        logger.info(f"  Edge TP: {bin_data.edge_tp}")
        logger.info(f"  Edge FP: {bin_data.edge_fp}")
        logger.info(f"  Edge FN: {bin_data.edge_fn}")
        logger.info(f"  Edge Jaccard: {bin_data.edge_jaccard:.4f}")
    
    # Identify worst-performing bins
    worst_bins = sorted(bins, key=lambda b: b.edge_jaccard)[:2]
    if worst_bins:
        logger.info("\n" + "="*70)
        logger.info("WORST-PERFORMING DENSITY REGIONS:")
        for bin_data in worst_bins:
            logger.info(
                f"  Bin {bin_data.bin_id} [{bin_data.density_range[0]:.6f}, {bin_data.density_range[1]:.6f}): "
                f"Edge Jaccard = {bin_data.edge_jaccard:.4f}, "
                f"Recall = {bin_data.recall:.4f}, "
                f"Precision = {bin_data.precision:.4f}"
            )
