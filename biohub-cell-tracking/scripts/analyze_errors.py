#!/usr/bin/env python
"""Error analysis script for tracking results."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DatasetCatalog
from pipeline.classical_pipeline import ClassicalPipeline
from metrics.evaluator import evaluate_graph
from metrics.matching import match_nodes
from metrics.error_analysis import EdgeErrorAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function to run error analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze tracking errors")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/enhanced_tracking.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../train"),
        help="Path to data directory"
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Sample ID to analyze"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Sample: {args.sample}")
    
    # Load sample with ground truth
    catalog = DatasetCatalog(args.data_root)
    sample = catalog.load_sample(args.sample, require_tracks=True)
    
    logger.info(f"Loaded sample {args.sample}")
    logger.info(f"Volume shape: {sample.image_metadata.shape}")
    logger.info(f"Scale: {sample.image_metadata.scale}")
    logger.info(f"GT nodes: {sample.tracks.num_nodes}")
    logger.info(f"GT edges: {sample.tracks.num_edges}")
    
    # Run pipeline
    pipeline = ClassicalPipeline(config)
    predicted_graph = pipeline.process(sample)
    
    logger.info(f"Predicted nodes: {predicted_graph.num_nodes}")
    logger.info(f"Predicted edges: {predicted_graph.num_edges}")
    
    # Evaluate
    result = evaluate_graph(
        predicted_graph,
        sample.tracks,
        sample.image_metadata.scale,
        max_distance_um=7.0,
    )
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Edge TP: {result['edge_tp']}")
    logger.info(f"Edge FP: {result['edge_fp']}")
    logger.info(f"Edge FN: {result['edge_fn']}")
    logger.info(f"Edge Jaccard: {result['edge_jaccard']:.4f}")
    logger.info(f"Adjusted Edge Jaccard: {result['adjusted_edge_jaccard']:.4f}")
    logger.info(f"Division TP: {result['division_tp']}")
    logger.info(f"Division FP: {result['division_fp']}")
    logger.info(f"Division FN: {result['division_fn']}")
    logger.info(f"Division Jaccard: {result['division_jaccard']:.4f}")
    logger.info(f"Total Score: {result['score']:.4f}")
    logger.info(f"Matched nodes: {result['matched_nodes']}")
    logger.info("="*60)
    
    # Error analysis
    logger.info("\nRunning detailed error analysis...")
    
    analyzer = EdgeErrorAnalyzer(
        scale=sample.image_metadata.scale,
        max_match_distance_um=7.0,
    )
    
    matching = match_nodes(predicted_graph, sample.tracks, sample.image_metadata.scale, max_distance_um=7.0)
    errors, error_counts = analyzer.analyze_errors(predicted_graph, sample.tracks, matching)
    
    # Generate report
    report = analyzer.generate_error_report(errors, error_counts)
    print("\n" + report)
    
    # Get suggestions
    suggestions = analyzer.suggest_improvements(error_counts)
    logger.info("\nSUGGESTED IMPROVEMENTS:")
    for suggestion in suggestions:
        logger.info(f"  - {suggestion}")
    
    # Save report to file
    output_path = Path("outputs/error_analysis") / f"{args.sample}_errors.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(report)
        f.write("\n\nSUGGESTED IMPROVEMENTS:\n")
        for suggestion in suggestions:
            f.write(f"  - {suggestion}\n")
    
    logger.info(f"\nError report saved to {output_path}")


if __name__ == "__main__":
    main()
