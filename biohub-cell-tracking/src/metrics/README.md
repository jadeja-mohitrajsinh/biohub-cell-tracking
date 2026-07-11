# Metrics Module

## Purpose

Offline Edge Jaccard, adjusted Edge Jaccard, and Division Jaccard evaluation.

## API

- `match_nodes(predicted, ground_truth, scale)`
- `evaluate_graph(predicted, ground_truth, scale)`
- `evaluate_samples(predictions, samples)`

## Public Classes

- `EvaluationResult`, `NodeMatchingResult`, `CountResult`

## Roadmap

- Parity tests with official `tracking_cellmot` package
- Visualization hooks for TP/FP/FN edges
