# Tracking Improvements Guide

## Overview

Based on benchmark analysis showing **0.68 average score** with high variance (0.398-0.997), these improvements target the key bottlenecks:

- **Tracking cost function**: Distance-only → Multi-feature learned cost
- **Adaptive parameters**: Fixed thresholds → Per-embryo adaptation
- **Division detection**: Missing → Explicit geometric validation
- **Error analysis**: Unknown → Systematic failure mode analysis

## Key Improvements Implemented

### 1. Enhanced Cost Function (`src/tracking/enhanced_cost.py`)

**Problem**: Current cost = distance only, missing important cues.

**Solution**: Multi-feature cost combining:
- Physical distance (weighted 1.0)
- Detection confidence (weighted 0.3)
- Intensity similarity (weighted 0.2)
- Size similarity (weighted 0.1)
- Neighborhood density (weighted 0.1)
- Displacement components (weighted 0.2)

**Usage**:
```python
from tracking.enhanced_cost import EnhancedCostFunction

cost_function = EnhancedCostFunction(
    scale=sample.image_metadata.scale,
    config={
        "distance_weight": 1.0,
        "confidence_weight": 0.3,
        "intensity_weight": 0.2,
        # ... other weights
    }
)
```

### 2. Adaptive Tracking Radius (`src/tracking/enhanced_cost.py`)

**Problem**: Fixed 7μm radius doesn't work for all embryos (motion varies).

**Solution**: Learn motion statistics from data and adapt radius:
- Initial: mean_displacement + 2×std
- Updates online as tracking progresses
- Per-embryo adaptation based on GT statistics

**Usage**:
```python
from tracking.enhanced_cost import AdaptiveCostFunction

cost_function = AdaptiveCostFunction(
    scale=sample.image_metadata.scale,
    config={
        "mean_displacement_um": 2.0,  # Initial estimate
        "std_displacement_um": 1.0,
        "adaptive_radius": true,
        "radius_multiplier": 2.0,
    }
)

# Update during tracking
cost_function.update_motion_statistics(observed_displacements)
```

### 3. Enhanced Hungarian Tracker (`src/tracking/enhanced_hungarian.py`)

**Problem**: Basic Hungarian doesn't use volume features.

**Solution**: Enhanced tracker with:
- Automatic volume loading for feature extraction
- Multi-feature cost matrix construction
- Fallback to simple distance if features unavailable
- Motion statistics learning

**Usage**:
```python
from tracking.enhanced_hungarian import EnhancedHungarianTrackerWithSample

tracker = EnhancedHungarianTrackerWithSample(
    config={
        "use_adaptive": True,
        "neighborhood_radius": 5,
        "cost_function": {
            "distance_weight": 1.0,
            "confidence_weight": 0.3,
            # ...
        }
    },
    scale=sample.image_metadata.scale,
    sample=sample,
)

result = tracker.track(detections_by_time, confidences_by_time)
```

### 4. Division Detection (`src/lineage/division_detector.py`)

**Problem**: Division Jaccard = 0.0, no explicit division handling.

**Solution**: Geometric division validation:
- Detect nodes with 2 outgoing edges
- Validate parent-child distances (< 5μm)
- Validate child-child distance (< 4μm)
- Validate division angle (30°-150°)
- Calculate confidence score
- Find missing divisions (unassigned parents)

**Usage**:
```python
from lineage.division_detector import DivisionDetector, DivisionAwareTracker

detector = DivisionDetector(
    scale=sample.image_metadata.scale,
    config={
        "max_parent_child_distance_um": 5.0,
        "max_child_child_distance_um": 4.0,
        "min_division_angle_deg": 30.0,
        "max_division_angle_deg": 150.0,
    }
)

candidates = detector.detect_divisions(graph, detections_by_time)

# Or use division-aware tracker
tracker = DivisionAwareTracker(scale=sample.image_metadata.scale)
graph = tracker.post_process_divisions(graph, detections_by_time)
```

### 5. Per-Embryo Normalization (`src/preprocessing/embryo_normalization.py`)

**Problem**: Scores vary 0.398-0.997 due to embryo-specific characteristics.

**Solution**: Per-embryo analysis and adaptive configuration:
- Analyze intensity statistics per embryo
- Estimate cell count and density
- Calculate displacement statistics from GT
- Adapt tracking radius per embryo
- Adapt detection thresholds per embryo

**Usage**:
```python
from preprocessing.embryo_normalization import (
    EmbryoAnalyzer,
    AdaptiveNormalizer,
    PerEmbryoConfigurator,
)

# Analyze embryo
analyzer = EmbryoAnalyzer(scale=sample.image_metadata.scale)
stats = analyzer.analyze_sample(sample)

# Adaptive normalization
normalizer = AdaptiveNormalizer(config={"method": "percentile"})
normalized_volume = normalizer.normalize_volume(volume, stats)

# Adaptive configuration
configurator = PerEmbryoConfigurator(scale=sample.image_metadata.scale)
adapted_config = configurator.configure_for_sample(sample, base_config)
```

### 6. Error Analysis (`src/metrics/error_analysis.py`)

**Problem**: 242 FN edges but unknown root causes.

**Solution**: Systematic error categorization:
- Detection miss (node not matched)
- Wrong parent (incorrect assignment)
- Distance too far (exceeds threshold)
- Duplicate assignment (conflict)
- Division missed (one-to-two not detected)
- Track broken (gap in continuity)
- Ambiguous assignment

**Usage**:
```python
from metrics.error_analysis import EdgeErrorAnalyzer
from metrics.matching import match_nodes

analyzer = EdgeErrorAnalyzer(
    scale=sample.image_metadata.scale,
    max_match_distance_um=7.0,
)

matching = match_nodes(predicted, ground_truth, scale, max_distance_um=7.0)
errors, error_counts = analyzer.analyze_errors(predicted, ground_truth, matching)

# Generate report
report = analyzer.generate_error_report(errors, error_counts)
print(report)

# Get suggestions
suggestions = analyzer.suggest_improvements(error_counts)
for suggestion in suggestions:
    print(f"- {suggestion}")
```

### 7. Gap Closing (`src/tracking/gap_closing.py`)

**Problem**: Track breaks cause missing edges.

**Solution**: Gap closing and track continuity:
- Detect gaps (missing links between consecutive frames)
- Validate gaps (distance, velocity constraints)
- Close reasonable gaps with direct edges
- Resolve conflicts (multiple incoming edges)
- Merge short tracks

**Usage**:
```python
from tracking.gap_closing import GapCloser, PostProcessingPipeline

# Gap closing only
gap_closer = GapCloser(
    scale=sample.image_metadata.scale,
    config={
        "max_gap_length": 3,
        "max_gap_distance_um": 15.0,
        "max_gap_velocity_um_per_frame": 5.0,
    }
)
gaps = gap_closer.find_gaps(graph)
graph = gap_closer.close_gaps(graph, gaps, detections_by_time)

# Full post-processing pipeline
pipeline = PostProcessingPipeline(
    scale=sample.image_metadata.scale,
    config={
        "gap_closing": {...},
        "track_continuity": {...},
    }
)
graph = pipeline.process(graph, detections_by_time)
```

## Integration with Existing Pipeline

### Option 1: Use Enhanced Configuration

Create `configs/enhanced_tracking.yaml` (already provided):

```yaml
tracking:
  method: "enhanced_hungarian"
  use_adaptive: true
  cost_function:
    distance_weight: 1.0
    confidence_weight: 0.3
    # ...

division:
  enable: true
  # ...

normalization:
  enable: true
  # ...

post_processing:
  enable: true
  # ...
```

### Option 2: Modify Classical Pipeline

Update `src/pipeline/classical_pipeline.py`:

```python
from tracking.enhanced_hungarian import EnhancedHungarianTrackerWithSample
from lineage.division_detector import DivisionAwareTracker
from tracking.gap_closing import PostProcessingPipeline

# In __init__:
if detection_method == "enhanced_hungarian":
    self.tracker = EnhancedHungarianTrackerWithSample(
        config=self.tracking_config,
        scale=sample.image_metadata.scale,
        sample=sample,
    )

# In process:
# After tracking
if self.division_enabled:
    division_tracker = DivisionAwareTracker(scale=sample.image_metadata.scale)
    graph = division_tracker.post_process_divisions(graph, detections_by_time)

if self.post_processing_enabled:
    post_processor = PostProcessingPipeline(scale=sample.image_metadata.scale)
    graph = post_processor.process(graph, detections_by_time)
```

## Expected Impact

Based on the benchmark analysis:

| Component | Current | Target | Expected Gain |
|-----------|---------|--------|---------------|
| Edge Jaccard | 0.62 | 0.75 | +0.13 |
| Division Jaccard | 0.0 | 0.5 | +0.5 |
| Score | 0.68 | 0.80+ | +0.12+ |

**Key improvements**:
- **Enhanced cost**: Better linking decisions → reduce FN edges
- **Adaptive radius**: Handle embryo variation → reduce variance (0.398→0.997 → 0.70→0.85)
- **Division detection**: Explicit handling → Division Jaccard 0.0→0.5+
- **Gap closing**: Fix track breaks → reduce FN edges
- **Error analysis**: Target improvements → systematic iteration

## Recommended Workflow

1. **Run error analysis** on current baseline:
   ```bash
   python scripts/analyze_errors.py --sample 44b6_12dfb391
   ```

2. **Implement enhanced cost function** first (highest impact):
   ```bash
   python scripts/run_pipeline.py --config configs/enhanced_tracking.yaml
   ```

3. **Add adaptive tracking** to reduce variance:
   - Enable `use_adaptive: true` in config
   - Verify per-embryo statistics

4. **Enable division detection**:
   - Set `division.enable: true`
   - Validate on samples with known divisions

5. **Add post-processing**:
   - Enable `post_processing.enable: true`
   - Tune gap closing parameters

6. **Iterate with error analysis**:
   - Re-run error analysis after each change
   - Focus on remaining dominant error types

## Configuration Tuning

### Cost Function Weights

Start with defaults, then tune based on error analysis:
- If many `distance_too_far` errors: increase `distance_weight`
- If many `wrong_parent` errors: increase `confidence_weight`, `intensity_weight`
- If many `ambiguous_assignment` errors: increase `neighborhood_weight`

### Adaptive Parameters

- `radius_multiplier`: Controls gap between mean and max distance
  - Too low: miss legitimate links
  - Too high: increase false positives
- Start with 2.0, adjust based on FN/FP ratio

### Division Parameters

- `max_parent_child_distance_um`: Parent-daughter distance
  - Tighter (3-4μm): fewer false divisions, more missed
  - Looser (6-8μm): more divisions, more false positives
- `min_division_angle_deg`: Minimum separation angle
  - Higher (45°): more geometric constraints
  - Lower (20°): more divisions detected

## Next Steps

1. **Integrate enhanced tracker** into pipeline
2. **Run benchmark** on all samples
3. **Perform error analysis** on results
4. **Tune parameters** based on dominant error types
5. **Iterate** until target performance reached

The systematic approach of error analysis → targeted improvement → validation should efficiently close the gap from 0.68 to 0.80+.
