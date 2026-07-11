# Modular Architecture Design

## Overview

Production-quality, modular architecture for the BioHub cell tracking competition.

## Directory Structure

```
src/
├── detection/
│   ├── __init__.py
│   ├── base.py              # Base detector interface
│   ├── frangi_detector.py   # Frangi filter-based detection
│   ├── watershed.py         # Watershed segmentation
│   └── utils.py             # Detection utilities
├── tracking/
│   ├── __init__.py
│   ├── base.py              # Base tracker interface
│   ├── hungarian.py         # Hungarian matching tracker
│   ├── kalman.py           # Kalman filter tracker (future)
│   └── gap_closing.py      # Gap closing (future)
├── pipeline/
│   ├── __init__.py
│   ├── base.py             # Base pipeline interface
│   ├── classical_pipeline.py  # Frangi + Watershed + Hungarian
│   └── config.py           # Configuration management
├── utils/
│   ├── __init__.py
│   ├── logging.py          # Logging utilities
│   └── metrics.py          # Additional metrics
└── inference/
    ├── __init__.py
    └── runner.py           # Inference runner
```

## Module Responsibilities

### Detection Module
- **Responsibility:** Detect cell centroids in 3D volumes
- **Input:** 3D numpy array (Z, Y, X)
- **Output:** List of (z, y, x) centroids
- **Interface:** `Detector` base class with `detect(volume)` method

### Tracking Module
- **Responsibility:** Link detections across time
- **Input:** Dict mapping time -> list of detections
- **Output:** List of TrackingEdge objects
- **Interface:** `Tracker` base class with `track(detections_by_time)` method

### Pipeline Module
- **Responsibility:** Orchestrate detection + tracking
- **Input:** Sample data
- **Output:** TrackingGraph
- **Interface:** `Pipeline` base class with `process(sample)` method

### Inference Module
- **Responsibility:** Run inference on test data
- **Input:** Data directory
- **Output:** Submission files
- **Interface:** `run_inference(config)` function

## Design Principles

1. **Single Responsibility:** Each module has one clear purpose
2. **Dependency Injection:** Dependencies passed via constructor
3. **Configuration:** All parameters via config objects
4. **Logging:** Comprehensive logging at all levels
5. **Type Hints:** Full type annotations
6. **Error Handling:** Graceful error handling with clear messages
7. **Testing:** Unit tests for each module
8. **Documentation:** Docstrings for all public methods

## Configuration Management

Use YAML configuration files for all parameters:

```yaml
# configs/classical_baseline.yaml
detection:
  method: "frangi_watershed"
  frangi:
    scale_range: [1.0, 10.0]
    scale_step: 1.0
    beta: 0.5
    gamma: 15.0
  threshold:
    method: "adaptive"
    block_size: 51
    c: 0
  watershed:
    min_size: 10
    max_size: 1000

tracking:
  method: "hungarian"
  max_distance_um: 7.0
  max_distance_voxel:
    z: 4.3
    y: 17.2
    x: 17.2

pipeline:
  normalize: true
  quantile_low: 0.01
  quantile_high: 0.99
```

## Logging Strategy

- Use Python logging module
- Log level: INFO for normal operation, DEBUG for debugging
- Log format: `[timestamp] [level] [module] message`
- Log to file + console
- Log key metrics at each step

## Error Handling

- Try-except blocks around critical operations
- Clear error messages with context
- Graceful degradation where possible
- Validation of inputs and outputs

## Testing Strategy

- Unit tests for each module
- Integration tests for pipeline
- End-to-end tests on sample data
- Performance benchmarks

## Implementation Order

1. **Base classes:** Define interfaces
2. **Detection:** Frangi + Watershed
3. **Tracking:** Hungarian matching
4. **Pipeline:** Orchestration
5. **Configuration:** YAML config loading
6. **Inference:** Test data processing
7. **Testing:** Unit + integration tests
8. **Documentation:** Complete docstrings
