# Mitosis and Lineage (Planned)

## Cell Division

Encoded in GEFF as one source node with two outgoing edges to child detections at `t+1`.

## Parent-Child Graph

Lineage module will build directed acyclic graphs from tracking output and validate fork structure.

## Validation

- Out-degree == 2 at division nodes
- No overlapping parent assignments in matched regions
- Division Jaccard offline checks

## Edge Cases

- Late-visible splits (±1 timepoint tolerance in metric)
- Sparse GT ignores divisions in unannotated regions
- False parent links when only one daughter is detected

Phase 6 implementation.
