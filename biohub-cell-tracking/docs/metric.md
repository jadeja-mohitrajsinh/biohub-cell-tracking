# Evaluation Metrics

Based on the [official competition metric description](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md).

## Edge Jaccard

Ground truth is **sparse** — only a subset of cells are annotated.

### Matching

1. Pair predicted and GT nodes by centroid distance in **physical micrometers**
2. Maximum match distance: **7 µm**
3. Optimal bipartite assignment (Hungarian algorithm)

### Edge scoring

- **TP:** Predicted edge connects two matched nodes that are connected in GT
- **FN:** GT edge with no matching predicted edge
- **FP:** Predicted edge where matched GT target has another incoming edge, or matched GT source has another outgoing edge
- Unmatched predicted nodes/edges in unannotated regions are ignored

```
edge_jaccard = TP / (TP + FP + FN)
```

### Adjusted edge Jaccard

Penalizes excess predicted nodes:

```
adjusted = max(0, edge_jaccard × (1 − 0.1 × (T_pred − T_true) / T_true))
```

`T_true` comes from GEFF metadata `estimated_number_of_nodes`.

## Division Jaccard

A division is a node with exactly two outgoing edges. GT divisions are matched with tolerance across daughter lineages and ±1 timepoint visibility.

- **TP:** Predicted fork satisfies parent-era coverage, both daughter lineages touched, single connected component, contains a fork
- **FN:** GT division without a matching predicted fork
- **FP:** Predicted division matched to annotated GT region but not paired to any GT division

```
division_jaccard = TP / (TP + FP + FN)
```

## Distance Scaling

Distances are computed in micrometers using OME-Zarr / GEFF axis scales, not raw voxel indices.

## Offline Validation

```bash
biohub evaluate <sample_id> --data-root ../train --baseline identity
```

Python API:

```python
from metrics.evaluator import evaluate_graph

result = evaluate_graph(predicted_graph, gt_graph, scale)
print(result["score"])
```

## Final Score

Micro-averaged across samples:

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```
