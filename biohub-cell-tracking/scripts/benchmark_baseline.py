"""Benchmark the classical baseline on validation samples."""

from pathlib import Path
import logging
import yaml
import numpy as np

from data.dataset import DatasetCatalog
from metrics.evaluator import evaluate_graph
from metrics.density_analysis import stratify_by_density, print_density_report
from pipeline.classical_pipeline import ClassicalPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('outputs/benchmark.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def benchmark_sample(
    sample_id: str,
    catalog: DatasetCatalog,
    pipeline: ClassicalPipeline
) -> dict:
    """Benchmark pipeline on a single sample.
    
    Args:
        sample_id: Sample identifier
        catalog: Dataset catalog
        pipeline: Pipeline to benchmark
    
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking sample: {sample_id}")
    logger.info(f"{'='*60}")
    
    # Load sample
    sample = catalog.load_sample(sample_id, require_tracks=True, load_volume=False)
    
    logger.info(f"Ground truth: {sample.tracks.num_nodes} nodes, {sample.tracks.num_edges} edges")
    
    # Run pipeline
    predicted_graph = pipeline.process(sample)
    
    logger.info(f"Predicted: {predicted_graph.num_nodes} nodes, {predicted_graph.num_edges} edges")
    
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
    
    # Density-stratified analysis
    logger.info(f"\nDensity-stratified analysis:")
    density_bins = stratify_by_density(
        predicted_graph,
        sample.tracks,
        sample.image_metadata.scale,
        num_bins=5,
        radius_um=10.0
    )
    print_density_report(density_bins)
    
    return {
        'sample_id': sample_id,
        'gt_nodes': sample.tracks.num_nodes,
        'gt_edges': sample.tracks.num_edges,
        'pred_nodes': predicted_graph.num_nodes,
        'pred_edges': predicted_graph.num_edges,
        'edge_jaccard': result['edge_jaccard'],
        'adjusted_edge_jaccard': result['adjusted_edge_jaccard'],
        'division_jaccard': result['division_jaccard'],
        'score': result['score'],
        'matched_nodes': result['matched_nodes'],
    }


def main():
    """Run benchmark on validation samples."""
    data_root = "../train"
    config_path = "configs/classical_baseline.yaml"
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Initialize pipeline
    pipeline = ClassicalPipeline(config)
    
    # Load dataset
    catalog = DatasetCatalog(data_root)
    all_samples = catalog.list_samples()
    
    # Use first 5 samples for quick benchmark
    val_samples = all_samples[:5]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARKING {len(val_samples)} SAMPLES")
    logger.info(f"{'='*60}")
    
    results = []
    for sample_id in val_samples:
        try:
            result = benchmark_sample(sample_id, catalog, pipeline)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")
    
    for result in results:
        logger.info(
            f"{result['sample_id']}: "
            f"Score={result['score']:.4f}, "
            f"EdgeJ={result['edge_jaccard']:.4f}, "
            f"AdjEdgeJ={result['adjusted_edge_jaccard']:.4f}, "
            f"DivJ={result['division_jaccard']:.4f}"
        )
    
    # Compute averages
    if results:
        avg_score = np.mean([r['score'] for r in results])
        avg_edge_j = np.mean([r['edge_jaccard'] for r in results])
        avg_adj_edge_j = np.mean([r['adjusted_edge_jaccard'] for r in results])
        avg_div_j = np.mean([r['division_jaccard'] for r in results])
        
        logger.info(f"\nAverage:")
        logger.info(f"  Score: {avg_score:.4f}")
        logger.info(f"  Edge Jaccard: {avg_edge_j:.4f}")
        logger.info(f"  Adjusted Edge Jaccard: {avg_adj_edge_j:.4f}")
        logger.info(f"  Division Jaccard: {avg_div_j:.4f}")
    
    logger.info(f"\nBenchmark complete. Results saved to outputs/benchmark.log")


if __name__ == "__main__":
    main()
