# Step 3: Evaluation Metric Deep Dive

## Metric Overview

**Final Score Formula:**
```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

**Key Insight:** Edge accuracy is 10× more important than division accuracy.

## Component 1: Edge Jaccard

### Node Matching
- **Algorithm:** Hungarian bipartite matching (optimal assignment)
- **Distance metric:** Physical Euclidean distance in micrometers
- **Maximum matching distance:** 7.0 µm
- **Cost matrix:** Distance for pairs ≤ 7 µm, 70 µm (10× max) for others

**Physical distance calculation:**
```
distance = sqrt((Δt × 1.0)² + (Δz × 1.625)² + (Δy × 0.40625)² + (Δx × 0.40625)²)
```

**Anisotropy handling:** Correctly accounts for different voxel spacings.

### Edge Classification

**True Positive (TP):**
- Predicted edge connects two matched nodes
- The matched GT nodes are connected by an edge in ground truth
- Both endpoints must be within 7 µm of GT nodes

**False Positive (FP):**
- Predicted edge connects matched nodes
- BUT matched GT target has another incoming edge OR
- Matched GT source has another outgoing edge
- **Rationale:** Prevents multiple predictions for same relationship

**False Negative (FN):**
- GT edge exists
- No predicted edge matches it
- **Rationale:** Missed relationships

**Edge Jaccard:**
```
edge_jaccard = TP / (TP + FP + FN)
```

### Adjusted Edge Jaccard

**Purpose:** Penalize over-prediction of nodes.

**Formula:**
```
adjusted = max(0, edge_jaccard × (1 − 0.1 × (T_pred − T_true) / T_true))
```

**Parameters:**
- `T_pred`: Number of predicted nodes
- `T_true`: Estimated true nodes from GEFF metadata
- `penalty_weight`: 0.1 (10% penalty per 100% over-prediction)

**Examples:**
- Perfect prediction (T_pred = T_true): adjusted = edge_jaccard
- 10% over-prediction: adjusted = edge_jaccard × 0.99
- 50% over-prediction: adjusted = edge_jaccard × 0.95
- 100% over-prediction: adjusted = edge_jaccard × 0.90
- 200% over-prediction: adjusted = edge_jaccard × 0.80

**Critical implication:** Conservative detection is better than aggressive detection.

## Component 2: Division Jaccard

### Division Definition
- A node with exactly two outgoing edges
- Represents mother → daughter A + daughter B

### Division Matching Criteria

**True Positive (TP) Division:**
1. Predicted component contains:
   - Matched parent-era nodes (mother + its ancestors)
   - Matched daughter lineage A nodes
   - Matched daughter lineage B nodes
2. Component has a fork (node with 2 outgoing edges)
3. All three conditions satisfied in single connected component

**Tolerance:**
- ±1 timepoint visibility for division timing
- Lineage-based matching (not exact node correspondence)
- Daughter lineages can be partially matched

**False Positive (FP) Division:**
- Predicted division node exists
- Matched to GT node in annotated region
- But not paired to any GT division

**False Negative (FN) Division:**
- GT division exists
- No predicted division satisfies TP criteria

**Division Jaccard:**
```
division_jaccard = TP / (TP + FP + FN)
```

### Division Weight in Final Score
- **Weight:** 0.1 (10% of edge weight)
- **Rationale:** Divisions are rare but biologically important
- **Practical impact:** Division Jaccard of 1.0 adds only 0.1 to final score

## Failure Mode Analysis

### Edge-Level Failures

**1. Identity Switches**
- **Scenario:** Cell A tracked as B, then B tracked as A
- **Result:** FP edge (A→B) + FN edge (A→A) + FN edge (B→B)
- **Impact:** Severe penalty (3 errors for 1 switch)
- **Mitigation:** Appearance consistency, motion smoothness

**2. Broken Tracks**
- **Scenario:** Detection fails at t=5, track splits into two
- **Result:** FN edge (before gap) + FP edge (after gap)
- **Impact:** Moderate penalty
- **Mitigation:** Track bridging, interpolation

**3. Merge Errors**
- **Scenario:** Two cells detected as one
- **Result:** FP edge (merged cell) + FN edges (two true cells)
- **Impact:** Moderate penalty
- **Mitigation:** Better segmentation

**4. Split Errors**
- **Scenario:** One cell detected as two
- **Result:** FP edges (two detections) + FN edge (true cell)
- **Impact:** Moderate penalty + over-prediction penalty
- **Mitigation:** Merge nearby detections

**5. Long-Range Links**
- **Scenario:** Link cells that moved > 7 µm
- **Result:** No match (nodes too far), edge ignored
- **Impact:** Wasted computation, no direct penalty
- **Mitigation:** Respect 7 µm threshold

### Node-Level Failures

**1. Over-prediction**
- **Scenario:** Detect cells in unlabeled regions
- **Result:** Adjusted edge Jaccard penalty
- **Impact:** Linear penalty based on excess nodes
- **Mitigation:** Conservative detection thresholds

**2. Under-prediction**
- **Scenario:** Miss cells in annotated regions
- **Result:** FN edges (no predictions for GT nodes)
- **Impact:** Direct edge Jaccard reduction
- **Mitigation:** Sensitive detection in annotated regions

**3. Localization Error**
- **Scenario:** Detect cell but centroid is > 7 µm from GT
- **Result:** No match, node ignored
- **Impact:** Lost opportunity for TP edges
- **Mitigation:** Accurate centroid localization

### Division-Level Failures

**1. Missed Division**
- **Scenario:** Division occurs but not detected
- **Result:** FN division
- **Impact:** 0.1 × (1 / (TP + FP + FN)) reduction
- **Mitigation:** Division-specific features

**2. False Division**
- **Scenario:** Two cells appear as division but aren't related
- **Result:** FP division
- **Impact:** 0.1 × (1 / (TP + FP + FN)) reduction
- **Mitigation:** Strict division criteria

**3. Timing Error**
- **Scenario:** Division detected at wrong time
- **Result:** May still count as TP (±1 tolerance)
- **Impact:** Minimal if within tolerance
- **Mitigation:** Temporal consistency checks

## Strategic Implications

### 1. Edge Accuracy is Paramount
- **Weight:** 1.0 vs 0.1 for divisions
- **Focus:** Get tracking right first
- **Accept:** Low division Jaccard initially

### 2. Conservative Detection
- **Penalty:** 10% per 100% over-prediction
- **Strategy:** Detect fewer cells with high confidence
- **Benefit:** Avoid adjusted Jaccard penalty

### 3. Localization Precision
- **Threshold:** 7 µm matching distance
- **Requirement:** Centroid accuracy < 7 µm
- **Margin:** Mean displacement is 2.1 µm, so 7 µm is generous

### 4. Identity Consistency
- **Cost:** Identity switches cause 3 errors
- **Priority:** Maintain cell identity over time
- **Method:** Appearance + motion consistency

### 5. Track Continuity
- **Problem:** Broken tracks cause FP + FN
- **Solution:** Track bridging, interpolation
- **Benefit:** Fewer, longer tracks are better

### 6. Division Detection is Secondary
- **Weight:** 0.1 in final score
- **Strategy:** Focus on tracking, add divisions later
- **Accept:** Division Jaccard will be low initially

## Optimization Strategy

### Phase 1: Maximize Edge Jaccard
1. Accurate cell detection (localization < 7 µm)
2. Conservative node count (avoid penalty)
3. Robust tracking (minimize identity switches)
4. Track continuity (minimize broken tracks)

### Phase 2: Add Division Detection
1. After edge Jaccard is optimized
2. Use heuristic rules based on trajectory splitting
3. Accept low division Jaccard due to sparse data
4. Focus on avoiding false positive divisions

### Phase 3: Fine-tuning
1. Balance detection sensitivity vs specificity
2. Optimize tracking parameters
3. Add division detection refinements
4. Validate on full validation set

## Metric Implementation Insights

### 1. Sparse Evaluation
- Only annotated regions are evaluated
- Predictions in unlabeled regions are ignored
- **Strategy:** Focus effort on annotated regions

### 2. Bipartite Matching
- Optimal assignment, not greedy
- **Benefit:** Best possible node matching
- **Cost:** O(n³) for n nodes (acceptable for sparse GT)

### 3. Physical Distance
- Accounts for anisotropic voxel spacing
- **Critical:** Must use physical coordinates, not voxel indices
- **Implementation:** Already handled correctly in codebase

### 4. Estimated True Nodes
- Comes from GEFF metadata
- **Fallback:** Uses actual GT node count if metadata missing
- **Implication:** Trust metadata when available

## Validation Strategy

### 1. Primary Metric: Adjusted Edge Jaccard
- **Target:** > 0.8 for competitive solution
- **Focus:** Tracking quality
- **Monitor:** Node count to avoid penalty

### 2. Secondary Metric: Division Jaccard
- **Target:** > 0.3 (given sparse data)
- **Focus:** Avoid false positives
- **Accept:** Low initial values

### 3. Diagnostic Metrics
- **Node match rate:** % of GT nodes matched
- **Edge TP/FP/FN:** Breakdown of edge errors
- **Track length distribution:** Measure track continuity
- **Identity switch rate:** Measure tracking consistency

## Next Steps

1. **Step 4:** Design detection approaches (classical vs ML)
2. **Step 5:** Design tracking approaches (simple to SOTA)
3. **Step 6:** Theoretical comparison of approaches
4. **Step 7:** Implement best baseline focusing on edge Jaccard
