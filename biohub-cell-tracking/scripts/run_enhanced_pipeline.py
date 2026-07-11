#!/usr/bin/env python
"""Run enhanced tracking pipeline with multi-feature cost function."""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function to run enhanced pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced tracking pipeline")
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
        help="Sample ID to process"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate against ground truth if available"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Sample: {args.sample}")
    
    # Load sample
    catalog = DatasetCatalog(args.data_root)
    sample = catalog.load_sample(args.sample, require_tracks=args.evaluate)
    
    logger.info(f"Loaded sample {args.sample}")
    logger.info(f"Volume shape: {sample.image_metadata.shape}")
    logger.info(f"Scale: {sample.image_metadata.scale}")
    
    # Create pipeline
    pipeline = ClassicalPipeline(config)
    
    # Process sample
    logger.info("Starting enhanced pipeline processing...")
    predicted_graph = pipeline.process(sample)
    
    logger.info(f"Pipeline complete: {predicted_graph.num_nodes} nodes, {predicted_graph.num_edges} edges")
    
    # Evaluate if ground truth available
    if args.evaluate and sample.tracks:
        logger.info("Evaluating against ground truth...")
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
    else:
        logger.info("No ground truth available for evaluation")
    
    return predicted_graph


if __name__ == "__main__":
    main()
