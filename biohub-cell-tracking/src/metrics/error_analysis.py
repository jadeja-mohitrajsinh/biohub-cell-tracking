"""Edge error analysis tool for understanding tracking failures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

from data.models import TrackingGraph, VoxelScale, CellNode, TrackingEdge
from metrics.matching import NodeMatchingResult, match_nodes

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of tracking errors."""
    DETECTION_MISS = "detection_miss"
    WRONG_PARENT = "wrong_parent"
    DISTANCE_TOO_FAR = "distance_too_far"
    DUPLICATE_ASSIGNMENT = "duplicate_assignment"
    DIVISION_MISSED = "division_missed"
    TRACK_BROKEN = "track_broken"
    AMBIGUOUS_ASSIGNMENT = "ambiguous_assignment"
    OTHER = "other"


@dataclass
class EdgeError:
    """Error analysis for a single edge."""
    edge_type: str  # "FP" or "FN"
    source_id: int
    target_id: int
    error_type: ErrorType
    distance_um: float
    confidence: float
    context: Dict


class EdgeErrorAnalyzer:
    """Analyze tracking errors to understand failure modes."""
    
    def __init__(
        self,
        scale: VoxelScale,
        max_match_distance_um: float = 7.0,
    ) -> None:
        """Initialize edge error analyzer.
        
        Args:
            scale: VoxelScale for physical distance calculation
            max_match_distance_um: Maximum matching distance
        """
        self.scale = scale
        self.max_match_distance_um = max_match_distance_um
        
        logger.info("EdgeErrorAnalyzer initialized")
    
    def analyze_errors(
        self,
        predicted: TrackingGraph,
        ground_truth: TrackingGraph,
        matching: NodeMatchingResult,
    ) -> Tuple[List[EdgeError], Dict[str, int]]:
        """Analyze tracking errors in detail.
        
        Args:
            predicted: Predicted tracking graph
            ground_truth: Ground truth tracking graph
            matching: Node matching result
        
        Returns:
            Tuple of (list of errors, error type counts)
        """
        errors = []
        error_counts = {error_type.value: 0 for error_type in ErrorType}
        
        # Analyze false positive edges
        fp_edges = self._get_false_positive_edges(predicted, ground_truth, matching)
        for edge in fp_edges:
            error = self._analyze_false_positive(edge, predicted, ground_truth, matching)
            errors.append(error)
            error_counts[error.error_type.value] += 1
        
        # Analyze false negative edges
        fn_edges = self._get_false_negative_edges(predicted, ground_truth, matching)
        for edge in fn_edges:
            error = self._analyze_false_negative(edge, predicted, ground_truth, matching)
            errors.append(error)
            error_counts[error.error_type.value] += 1
        
        logger.info(f"Error analysis complete: {len(errors)} total errors")
        logger.info(f"Error breakdown: {error_counts}")
        
        return errors, error_counts
    
    def _get_false_positive_edges(
        self,
        predicted: TrackingGraph,
        ground_truth: TrackingGraph,
        matching: NodeMatchingResult,
    ) -> List[TrackingEdge]:
        """Get false positive edges.
        
        Args:
            predicted: Predicted graph
            ground_truth: Ground truth graph
            matching: Node matching
        
        Returns:
            List of false positive edges
        """
        gt_edges = {(edge.source_id, edge.target_id) for edge in ground_truth.edges}
        gt_outgoing = {node_id: ground_truth.outgoing(node_id) for node_id in ground_truth.nodes}
        gt_incoming = {node_id: ground_truth.incoming(node_id) for node_id in ground_truth.nodes}
        
        fp_edges = []
        
        for edge in predicted.edges:
            gt_source = matching.pred_to_gt.get(edge.source_id)
            gt_target = matching.pred_to_gt.get(edge.target_id)
            
            if gt_source is None or gt_target is None:
                # One or both nodes not matched - this is a detection issue
                fp_edges.append(edge)
                continue
            
            gt_pair = (gt_source, gt_target)
            if gt_pair not in gt_edges:
                # Check if this is a conflict
                target_conflict = gt_target in gt_incoming and len(gt_incoming[gt_target]) > 0
                source_conflict = gt_source in gt_outgoing and len(gt_outgoing[gt_source]) > 0
                
                if target_conflict or source_conflict:
                    fp_edges.append(edge)
        
        return fp_edges
    
    def _get_false_negative_edges(
        self,
        predicted: TrackingGraph,
        ground_truth: TrackingGraph,
        matching: NodeMatchingResult,
    ) -> List[TrackingEdge]:
        """Get false negative edges (missed ground truth edges).
        
        Args:
            predicted: Predicted graph
            ground_truth: Ground truth graph
            matching: Node matching
        
        Returns:
            List of false negative edges
        """
        pred_edges = {(edge.source_id, edge.target_id) for edge in predicted.edges}
        
        fn_edges = []
        
        for edge in ground_truth.edges:
            pred_source = matching.gt_to_pred.get(edge.source_id)
            pred_target = matching.gt_to_pred.get(edge.target_id)
            
            if pred_source is None or pred_target is None:
                # One or both nodes not matched - detection miss
                fn_edges.append(edge)
                continue
            
            pred_pair = (pred_source, pred_target)
            if pred_pair not in pred_edges:
                fn_edges.append(edge)
        
        return fn_edges
    
    def _analyze_false_positive(
        self,
        edge: TrackingEdge,
        predicted: TrackingGraph,
        ground_truth: TrackingGraph,
        matching: NodeMatchingResult,
    ) -> EdgeError:
        """Analyze a false positive edge.
        
        Args:
            edge: False positive edge
            predicted: Predicted graph
            ground_truth: Ground truth graph
            matching: Node matching
        
        Returns:
            EdgeError with analysis
        """
        source = predicted.nodes.get(edge.source_id)
        target = predicted.nodes.get(edge.target_id)
        
        if not source or not target:
            return EdgeError(
                edge_type="FP",
                source_id=edge.source_id,
                target_id=edge.target_id,
                error_type=ErrorType.OTHER,
                distance_um=0.0,
                confidence=0.0,
                context={"reason": "node_not_found"},
            )
        
        # Calculate distance
        distance_um = self._calculate_distance(source, target)
        
        # Get matched GT nodes
        gt_source = matching.pred_to_gt.get(edge.source_id)
        gt_target = matching.pred_to_gt.get(edge.target_id)
        
        # Determine error type
        if gt_source is None or gt_target is None:
            error_type = ErrorType.DETECTION_MISS
        elif distance_um > self.max_match_distance_um:
            error_type = ErrorType.DISTANCE_TOO_FAR
        else:
            # Check for wrong parent
            if gt_source in ground_truth.nodes:
                gt_outgoing = ground_truth.outgoing(gt_source)
                if len(gt_outgoing) > 0:
                    error_type = ErrorType.WRONG_PARENT
                else:
                    error_type = ErrorType.DUPLICATE_ASSIGNMENT
            else:
                error_type = ErrorType.OTHER
        
        return EdgeError(
            edge_type="FP",
            source_id=edge.source_id,
            target_id=edge.target_id,
            error_type=error_type,
            distance_um=distance_um,
            confidence=0.0,  # Would need to extract from detector
            context={
                "gt_source": gt_source,
                "gt_target": gt_target,
                "distance_um": distance_um,
            },
        )
    
    def _analyze_false_negative(
        self,
        edge: TrackingEdge,
        predicted: TrackingGraph,
        ground_truth: TrackingGraph,
        matching: NodeMatchingResult,
    ) -> EdgeError:
        """Analyze a false negative edge.
        
        Args:
            edge: False negative edge (ground truth edge)
            predicted: Predicted graph
            ground_truth: Ground truth graph
            matching: Node matching
        
        Returns:
            EdgeError with analysis
        """
        source = ground_truth.nodes.get(edge.source_id)
        target = ground_truth.nodes.get(edge.target_id)
        
        if not source or not target:
            return EdgeError(
                edge_type="FN",
                source_id=edge.source_id,
                target_id=edge.target_id,
                error_type=ErrorType.OTHER,
                distance_um=0.0,
                confidence=0.0,
                context={"reason": "node_not_found"},
            )
        
        # Calculate distance
        distance_um = self._calculate_distance(source, target)
        
        # Get predicted nodes
        pred_source = matching.gt_to_pred.get(edge.source_id)
        pred_target = matching.gt_to_pred.get(edge.target_id)
        
        # Determine error type
        if pred_source is None or pred_target is None:
            error_type = ErrorType.DETECTION_MISS
        elif distance_um > self.max_match_distance_um:
            error_type = ErrorType.DISTANCE_TOO_FAR
        else:
            # Check if this is a division
            gt_outgoing = ground_truth.outgoing(edge.source_id)
            if len(gt_outgoing) == 2:
                error_type = ErrorType.DIVISION_MISSED
            else:
                # Check if track is broken
                if pred_source in predicted.nodes:
                    pred_outgoing = predicted.outgoing(pred_source)
                    if len(pred_outgoing) == 0:
                        error_type = ErrorType.TRACK_BROKEN
                    else:
                        error_type = ErrorType.WRONG_PARENT
                else:
                    error_type = ErrorType.OTHER
        
        return EdgeError(
            edge_type="FN",
            source_id=edge.source_id,
            target_id=edge.target_id,
            error_type=error_type,
            distance_um=distance_um,
            confidence=0.0,
            context={
                "pred_source": pred_source,
                "pred_target": pred_target,
                "distance_um": distance_um,
                "is_division": len(ground_truth.outgoing(edge.source_id)) == 2,
            },
        )
    
    def _calculate_distance(
        self,
        node1: CellNode,
        node2: CellNode,
    ) -> float:
        """Calculate physical distance between two nodes.
        
        Args:
            node1: First cell node
            node2: Second cell node
        
        Returns:
            Physical distance in micrometers
        """
        dz = (node2.z - node1.z) * self.scale.z
        dy = (node2.y - node1.y) * self.scale.y
        dx = (node2.x - node1.x) * self.scale.x
        
        return np.sqrt(dz**2 + dy**2 + dx**2)
    
    def generate_error_report(
        self,
        errors: List[EdgeError],
        error_counts: Dict[str, int],
    ) -> str:
        """Generate a human-readable error report.
        
        Args:
            errors: List of edge errors
            error_counts: Error type counts
        
        Returns:
            Formatted error report
        """
        lines = [
            "=" * 80,
            "TRACKING ERROR ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Total errors: {len(errors)}",
            "",
            "ERROR BREAKDOWN:",
        ]
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"  {error_type}: {count}")
        
        lines.extend([
            "",
            "DETAILED ERRORS:",
            "",
        ])
        
        # Group by error type
        errors_by_type = {}
        for error in errors:
            if error.error_type.value not in errors_by_type:
                errors_by_type[error.error_type.value] = []
            errors_by_type[error.error_type.value].append(error)
        
        for error_type, type_errors in sorted(errors_by_type.items()):
            lines.append(f"\n{error_type.upper()} ({len(type_errors)} errors):")
            lines.append("-" * 40)
            
            # Show first 5 errors of this type
            for i, error in enumerate(type_errors[:5]):
                lines.append(
                    f"  {error.edge_type} edge {error.source_id} -> {error.target_id}: "
                    f"distance={error.distance_um:.2f}µm"
                )
            
            if len(type_errors) > 5:
                lines.append(f"  ... and {len(type_errors) - 5} more")
        
        lines.extend([
            "",
            "=" * 80,
        ])
        
        return "\n".join(lines)
    
    def suggest_improvements(
        self,
        error_counts: Dict[str, int],
    ) -> List[str]:
        """Suggest improvements based on error analysis.
        
        Args:
            error_counts: Error type counts
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        total_errors = sum(error_counts.values())
        
        if error_counts[ErrorType.DETECTION_MISS.value] / total_errors > 0.3:
            suggestions.append(
                "High detection miss rate: Improve detector sensitivity or "
                "adjust detection thresholds per embryo."
            )
        
        if error_counts[ErrorType.DISTANCE_TOO_FAR.value] / total_errors > 0.2:
            suggestions.append(
                "Many edges exceed distance threshold: Use adaptive tracking radius "
                "based on per-embryo motion statistics."
            )
        
        if error_counts[ErrorType.DIVISION_MISSED.value] / total_errors > 0.15:
            suggestions.append(
                "Many divisions missed: Implement explicit division detection "
                "and one-to-two linking."
            )
        
        if error_counts[ErrorType.WRONG_PARENT.value] / total_errors > 0.2:
            suggestions.append(
                "Many wrong parent assignments: Enhance cost function with "
                "appearance features and learned matching."
            )
        
        if error_counts[ErrorType.TRACK_BROKEN.value] / total_errors > 0.1:
            suggestions.append(
                "Track breaks detected: Implement gap closing and track continuity "
                "algorithms."
            )
        
        if not suggestions:
            suggestions.append("Error distribution looks reasonable. Focus on overall "
                             "system optimization and ensemble methods.")
        
        return suggestions
