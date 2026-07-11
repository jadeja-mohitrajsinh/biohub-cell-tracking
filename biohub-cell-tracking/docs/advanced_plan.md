# Advanced Cell Tracking Plan

## Executive Summary

Current implementation provides solid classical baselines but lacks deep learning capabilities essential for competitive performance. This plan integrates proven state-of-the-art methods from literature while maintaining the modular architecture.

## Current State Analysis

### ✅ Implemented (Phase 1-3)
- **Data pipeline**: Zarr/GEFF readers with lazy loading
- **Classical detection**: Frangi, adaptive threshold, watershed with multi-scale analysis
- **Classical tracking**: Hungarian algorithm with physical distance matching
- **Metrics**: Edge/division Jaccard matching competition spec
- **Submission**: GEFF writer for Kaggle format

### ❌ Missing (Critical Gaps)
- **Deep learning detection**: No 3D U-Net, CellPose, or StarDist integration
- **Learned linking**: Pure distance-based, no optical flow or FFN
- **Division detection**: No explicit mitosis detection or lineage inference
- **Anisotropy handling**: No isotropic resampling or anisotropic augmentations
- **Training pipeline**: No augmentation, no iterative retraining
- **Advanced tracking**: No Ultrack/Trackastra integration

## Improved Architecture

### 1. Hybrid Detection Pipeline

```
Raw Volume → Preprocessing → Multi-Method Ensemble → Centroid Refinement
                    ↓                    ↓                    ↓
              Isotropic           3D U-Net              Sub-voxel
              Resampling          CellPose              Localization
                                  StarDist
                                  Classical
```

**Key Improvements:**
- **Isotropic resampling**: Resample Z from 1.625μm to 0.40625μm using spline interpolation
- **3D U-Net**: Train on sparse GEFF labels with center/periphery/background classes
- **Ensemble detection**: Combine deep + classical methods via NMS or learned fusion
- **Sub-voxel refinement**: Gaussian fitting or centroid refinement on detected regions

### 2. Advanced Linking with Motion Modeling

```
Detections t → Motion Prediction → Cost Matrix → Global Optimization → Tracks t→t+1
                  ↓                    ↓              ↓
            Optical Flow        Distance +      LAP with
            (3D U-Net)         Appearance      Division
                               Features       Constraints
```

**Key Improvements:**
- **Optical flow network**: 3D U-Net predicting voxel-wise displacement between frames
- **Appearance features**: Extract intensity/texture features around each centroid
- **Learned cost function**: Train small MLP to predict match probability from features
- **Global optimization**: Use LAP/ultrack for globally consistent tracks with division constraints
- **Division-aware linking**: Explicitly allow one-to-two matches with mitosis priors

### 3. Explicit Division Detection

```
Track Graph → Division Candidate Selection → Patch Classification → Lineage Validation
                    ↓                           ↓                    ↓
              Fork Detection              3D CNN Classifier      Graph Consistency
              (2 outgoing edges)          (Mitotic vs Normal)    Check
```

**Key Improvements:**
- **Candidate detection**: Identify potential divisions (forks in track graph)
- **Patch classifier**: 3D CNN classifying image patches around parent/daughters
- **Temporal consistency**: Validate division timing and daughter trajectories
- **Lineage reconstruction**: Build complete lineage trees from division events

### 4. Training Pipeline with Augmentation

```
GEFF Labels → Label Generation → Augmentation → 3D U-Net Training → Pseudo-labels
                  ↓                  ↓               ↓                 ↓
            Center/Periphery    Anisotropic      Loss:             Retrain
            Spheres             Flips/Rotations   Dice +            Iteratively
                                Elastic          Focal
                                Intensity
```

**Key Improvements:**
- **Label generation**: Convert sparse centroids to center/periphery spheres (3-5 voxel radius)
- **Anisotropic augmentations**: Respect physical scale (different Z vs XY transforms)
- **Elastic deformations**: Simulate cell movement and deformation
- **Intensity augmentation**: Contrast/brightness jitter, noise injection
- **Iterative training**: ELEPHANT-style retraining on confident pseudo-labels

## Implementation Roadmap

### Phase 4A: Deep Learning Detection (Priority: HIGH)

**Goal**: Implement 3D U-Net for cell center detection

**Tasks**:
1. Create `src/detection/unet_detector.py` with 3D U-Net architecture
2. Implement label generation from GEFF centroids (center/periphery spheres)
3. Create PyTorch dataset with lazy Zarr loading
4. Implement anisotropic-aware augmentation pipeline
5. Train on sparse labels with Dice + Focal loss
6. Add sub-voxel centroid refinement

**Dependencies**: `torch`, `monai` (for 3D U-Net), `torchio` (for augmentation)

**Timeline**: 2-3 weeks

### Phase 4B: Classical Integration (Priority: HIGH)

**Goal**: Integrate CellPose and StarDist as alternative detectors

**Tasks**:
1. Create `src/detection/cellpose_detector.py` wrapper
2. Create `src/detection/stardist_detector.py` wrapper
3. Implement ensemble fusion (NMS or learned weighting)
4. Benchmark all detectors on validation set

**Dependencies**: `cellpose`, `stardist`

**Timeline**: 1 week

### Phase 5A: Optical Flow Linking (Priority: HIGH)

**Goal**: Implement learned motion prediction for better linking

**Tasks**:
1. Create `src/tracking/flow_tracker.py` with 3D flow U-Net
2. Generate flow training data from GEFF tracks
3. Train flow network to predict displacement between frames
4. Integrate flow predictions into Hungarian cost matrix
5. Benchmark vs distance-only baseline

**Dependencies**: `torch`, `monai`

**Timeline**: 2-3 weeks

### Phase 5B: Advanced Tracking Integration (Priority: MEDIUM)

**Goal**: Integrate Ultrack/Trackastra for global optimization

**Tasks**:
1. Create `src/tracking/ultrack_tracker.py` wrapper
2. Create `src/tracking/trackastra_tracker.py` wrapper
3. Configure division-aware linking parameters
4. Benchmark vs Hungarian baseline

**Dependencies**: `ultrack`, `trackastra`

**Timeline**: 1 week

### Phase 6: Division Detection (Priority: HIGH)

**Goal**: Explicit mitosis detection and lineage reconstruction

**Tasks**:
1. Create `src/lineage/division_detector.py`
2. Implement fork detection from track graph
3. Create 3D CNN classifier for mitotic patches
4. Train classifier on GT divisions vs normal cells
5. Implement lineage tree reconstruction
6. Add division-specific validation

**Dependencies**: `torch`, `monai`

**Timeline**: 2 weeks

### Phase 7: Iterative Training (Priority: MEDIUM)

**Goal**: ELEPHANT-style iterative retraining on pseudo-labels

**Tasks**:
1. Create confidence estimation for detections
2. Implement pseudo-label generation from high-confidence predictions
3. Create iterative training loop
4. Validate improvement on held-out set

**Dependencies**: Existing training pipeline

**Timeline**: 1-2 weeks

### Phase 8: Ensemble and Optimization (Priority: MEDIUM)

**Goal**: Model ensembling and hyperparameter optimization

**Tasks**:
1. Implement ensemble of multiple detection methods
2. Implement ensemble of multiple tracking methods
3. Create hyperparameter search (Optuna or manual)
4. Cross-validation on training split
5. Final ensemble for submission

**Dependencies**: `optuna` (optional)

**Timeline**: 2 weeks

## Technical Specifications

### 3D U-Net Architecture

```python
# Input: (C, D, H, W) where C=1 (grayscale), D=64 (resampled Z), H=W=256
# Output: (3, D, H, W) - background, center, periphery probabilities

Architecture:
- Encoder: 4 levels with 3×3×3 convs, stride 2 pooling
- Bottleneck: 256 filters
- Decoder: 4 levels with upsampling + skip connections
- Output: 1×1×1 conv to 3 channels with softmax
```

### Label Generation

```python
# For each GT centroid at (t, z, y, x):
# - Center sphere: radius=2 voxels, label=1
# - Periphery sphere: radius=4 voxels, label=2 (excluding center)
# - Background: label=0
```

### Augmentation Strategy

```python
# Anisotropic-aware augmentations:
# - XY plane: rotations (±15°), flips, elastic deformations
# - Z axis: flips only (no rotation to preserve anisotropy)
# - Intensity: contrast (0.8-1.2), brightness (±0.1), Gaussian noise
# - Resampling: Random Z-scale (0.9-1.1) to simulate focus variation
```

### Optical Flow Network

```python
# Input: Concatenated frame t and t+1 along channel axis
# Output: 3-channel displacement field (dz, dy, dx) per voxel

Architecture:
- Similar to 3D U-Net but with regression head
- Loss: L1 + smoothness regularization
```

### Division Classifier

```python
# Input: 3D patch around candidate division (parent + daughters)
# Output: Probability of mitosis

Architecture:
- Small 3D CNN (3-4 layers)
- Trained on balanced dataset of divisions vs non-divisions
```

## Expected Performance Improvements

Based on literature and similar competitions:

| Method | Expected Edge Jaccard | Expected Division Jaccard |
|--------|----------------------|---------------------------|
| Current classical | 0.3-0.4 | 0.1-0.2 |
| + 3D U-Net detection | 0.5-0.6 | 0.2-0.3 |
| + Optical flow linking | 0.6-0.7 | 0.3-0.4 |
| + Division detection | 0.65-0.75 | 0.5-0.6 |
| + Ensemble | 0.75-0.85 | 0.6-0.7 |

## Resource Requirements

### Compute
- **Training**: 1-2 GPUs (16GB+ VRAM recommended for 3D U-Net)
- **Inference**: Single GPU sufficient, can run on CPU with slower performance
- **Storage**: 50-100GB for models, checkpoints, augmented data

### Software
- **Core**: Python 3.12, PyTorch 2.2+, MONAI 1.3+
- **Augmentation**: TorchIO for medical image augmentations
- **Tracking**: Ultrack, Trackastra, LAPtrack
- **Segmentation**: CellPose, StarDist

## Risk Mitigation

1. **Limited training data**: Use transfer learning from similar datasets, strong augmentation
2. **Anisotropy challenges**: Careful augmentation design, validate on physical metrics
3. **Division sparsity**: Oversample divisions in training, use focal loss
4. **Compute constraints**: Start with smaller models, use mixed precision training
5. **Overfitting**: Cross-validation, early stopping, regularization

## Success Metrics

- **Edge Jaccard**: Target >0.75 on validation set
- **Division Jaccard**: Target >0.60 on validation set
- **Runtime**: <5 minutes per sample for inference
- **Memory**: <8GB GPU memory for inference

## Next Steps

1. **Immediate**: Implement Phase 4A (3D U-Net detection)
2. **Parallel**: Set up training infrastructure and augmentation pipeline
3. **Validation**: Create proper validation split from training data
4. **Iteration**: Rapid prototyping and benchmarking at each phase
