# Step 4: Cell Detection Approaches Design

## Problem Constraints

**Key constraints from dataset analysis:**
1. **Extreme sparsity:** < 1% of cells are annotated
2. **Over-prediction penalty:** 10% penalty per 100% excess nodes
3. **Conservative requirement:** Better to miss cells than over-predict
4. **Localization accuracy:** Must be < 7 µm from true centroid
5. **High density:** Cells are tightly packed in some regions
6. **Variable intensity:** Fluorescence varies across cells/time

**Implication:** Supervised deep learning is not feasible due to sparse labels. Classical or self-supervised methods are preferred.

## Approach 1: Classical Blob Detection

### 1.1 Laplacian of Gaussian (LoG)

**How it works:**
- Convolve image with Laplacian of Gaussian at multiple scales
- Find local maxima in scale-space
- Scale corresponds to blob size

**Advantages:**
- Well-established for blob detection
- Scale-invariant
- No training required
- Fast computation
- Good theoretical foundation

**Disadvantages:**
- Sensitive to noise
- Assumes Gaussian blob shape (cells may not be Gaussian)
- May merge nearby cells
- Parameter tuning required (scale range, threshold)

**Expected performance:**
- **Precision:** Medium (may detect false positives)
- **Recall:** Medium (may miss non-Gaussian cells)
- **Localization:** Good (sub-pixel accuracy possible)
- **Speed:** Fast
- **Expected leaderboard impact:** 0.3-0.5 edge Jaccard

**Computational complexity:**
- O(N × S) where N = pixels, S = number of scales
- For 256×256×64 volume: ~4M operations per scale
- Typical: 5-10 scales → 20-40M operations per timepoint

**Implementation complexity:**
- Low (scipy.ndimage provides implementation)
- Parameter tuning required

### 1.2 Difference of Gaussian (DoG)

**How it works:**
- Approximation of LoG using difference of two Gaussians
- Scale-space extrema detection
- Used in SIFT keypoint detection

**Advantages:**
- Faster than LoG (approximation)
- Scale-invariant
- No training required
- Good for multi-scale detection

**Disadvantages:**
- Approximation error vs LoG
- Same limitations as LoG regarding shape assumptions
- Parameter tuning required

**Expected performance:**
- Similar to LoG but faster
- **Expected leaderboard impact:** 0.3-0.5 edge Jaccard

**Computational complexity:**
- O(N × S) but with smaller constant than LoG
- ~2× faster than LoG

### 1.3 Hessian-Based Detection (Frangi filter)

**How it works:**
- Use Hessian matrix eigenvalues to detect tube-like/blob-like structures
- Vesselness/brightness enhancement
- Originally designed for vessel detection

**Advantages:**
- Handles anisotropic structures well
- Works well with elongated cells
- Enhances contrast
- No training required

**Disadvantages:**
- Designed for vessels, not cells
- May not handle spherical cells well
- Parameter tuning required
- Computationally expensive (Hessian calculation)

**Expected performance:**
- **Precision:** Medium-High (good for elongated cells)
- **Recall:** Medium (may miss spherical cells)
- **Localization:** Good
- **Speed:** Medium
- **Expected leaderboard impact:** 0.4-0.6 edge Jaccard

**Computational complexity:**
- O(N × S × 9) for Hessian calculation
- Slower than LoG/DoG

## Approach 2: Threshold-Based Segmentation

### 2.1 Simple Threshold + Connected Components

**How it works:**
- Apply intensity threshold
- Find connected components
- Compute centroid of each component
- Filter by size

**Advantages:**
- Extremely simple
- Very fast
- No training required
- Easy to implement
- Good for high-contrast images

**Disadvantages:**
- Sensitive to threshold choice
- Merges nearby cells
- Poor performance with variable intensity
- No shape modeling
- Requires size filtering

**Expected performance:**
- **Precision:** Low-Medium (many false positives)
- **Recall:** Low-Medium (misses low-intensity cells)
- **Localization:** Medium (centroid depends on segmentation)
- **Speed:** Very fast
- **Expected leaderboard impact:** 0.2-0.4 edge Jaccard

**Computational complexity:**
- O(N) for threshold + O(N) for connected components
- Very fast

### 2.2 Adaptive Threshold

**How it works:**
- Local thresholding (e.g., Otsu, adaptive Gaussian)
- Handles intensity variations
- Connected components on thresholded result

**Advantages:**
- Handles variable intensity better
- Still fast
- No training required
- Robust to illumination changes

**Disadvantages:**
- Still merges nearby cells
- Parameter tuning required (window size)
- May create false positives in homogeneous regions
- Computationally slower than global threshold

**Expected performance:**
- **Precision:** Medium
- **Recall:** Medium-High (better for variable intensity)
- **Localization:** Medium
- **Speed:** Fast
- **Expected leaderboard impact:** 0.3-0.5 edge Jaccard

### 2.3 Watershed Segmentation

**How it works:**
- Use intensity as topographic surface
- Find regional minima as seeds
- Flood from seeds to create regions
- Merge/split regions based on criteria

**Advantages:**
- Handles touching cells better
- Can separate merged cells
- Flexible (many variants)
- No training required

**Disadvantages:**
- Sensitive to seed placement
- Over-segmentation common
- Parameter tuning required
- Computationally expensive
- May require preprocessing (denoising, gradient)

**Expected performance:**
- **Precision:** Medium-High (better separation)
- **Recall:** Medium (over-segmentation issues)
- **Localization:** Good
- **Speed:** Medium-Slow
- **Expected leaderboard impact:** 0.4-0.6 edge Jaccard

**Computational complexity:**
- O(N log N) for priority-based watershed
- Slower than threshold-based methods

## Approach 3: Machine Learning (Self-Supervised/Semi-Supervised)

### 3.1 Cellpose (Pre-trained)

**How it works:**
- Pre-trained deep learning model for cell segmentation
- Generalizes across cell types
- Uses cytoplasm/nucleus markers

**Advantages:**
- State-of-the-art performance
- Pre-trained on diverse cell types
- Handles touching cells well
- Robust to variability
- Actively maintained

**Disadvantages:**
- Requires GPU for speed
- Large model size
- May overfit to training distribution
- Not optimized for this specific data
- May over-predict (violates conservative requirement)

**Expected performance:**
- **Precision:** High (good generalization)
- **Recall:** High (sensitive detection)
- **Localization:** Excellent
- **Speed:** Slow on CPU, fast on GPU
- **Expected leaderboard impact:** 0.6-0.8 edge Jaccard (if tuned conservatively)

**Computational complexity:**
- O(N) with large constant (deep network)
- GPU recommended for real-time performance

**Implementation complexity:**
- Medium (pip install, API usage)
- Requires careful threshold tuning for conservative detection

### 3.2 StarDist (Pre-trained)

**How it works:**
- Star-convex polygon representation
- Pre-trained on various cell types
- Handles diverse shapes well

**Advantages:**
- Handles irregular shapes
- Pre-trained models available
- Good for diverse cell morphologies
- Active development

**Disadvantages:**
- Assumes star-convex shape (may not hold)
- Requires GPU for speed
- May over-predict
- Not optimized for this specific data

**Expected performance:**
- **Precision:** High
- **Recall:** High
- **Localization:** Excellent
- **Speed:** Slow on CPU, fast on GPU
- **Expected leaderboard impact:** 0.6-0.8 edge Jaccard

### 3.3 Self-Supervised Contrastive Learning

**How it works:**
- Train encoder to recognize similar cells across time
- Use temporal consistency as supervision signal
- No manual labels required
- Fine-tune on sparse annotations

**Advantages:**
- Leverages unlabeled data
- Adapts to specific dataset
- No over-prediction if tuned conservatively
- State-of-the-art potential

**Disadvantages:**
- Complex implementation
- Requires significant compute
- Long training time
- Many hyperparameters
- Risk of overfitting to training distribution

**Expected performance:**
- **Precision:** High (if tuned well)
- **Recall:** High
- **Localization:** Excellent
- **Speed:** Slow training, fast inference
- **Expected leaderboard impact:** 0.7-0.9 edge Jaccard

**Computational complexity:**
- Training: O(E × N × B) where E = epochs, N = samples, B = batch size
- Very expensive (days of GPU time)

## Approach 4: Hybrid Methods

### 4.1 Classical + ML Refinement

**How it works:**
- Use classical method for initial detection
- Use ML to filter/refine detections
- Conservative ML classifier to reduce false positives

**Advantages:**
- Best of both worlds
- Classical method provides candidates
- ML reduces false positives
- Can be conservative
- Less training data required

**Disadvantages:**
- More complex pipeline
- Still requires some training
- Parameter tuning for both stages
- May inherit classical method limitations

**Expected performance:**
- **Precision:** High (ML filtering)
- **Recall:** Medium-High (classical baseline)
- **Localization:** Good
- **Speed:** Medium
- **Expected leaderboard impact:** 0.5-0.7 edge Jaccard

### 4.2 Multi-Method Ensemble

**How it works:**
- Run multiple detection methods
- Combine results (union or intersection)
- Use consensus to reduce false positives

**Advantages:**
- Robust to method-specific failures
- Can use intersection for high precision
- Flexible combination strategies
- No training required

**Disadvantages:**
- Computationally expensive (multiple methods)
- Complex combination logic
- May still over-predict with union
- May under-predict with intersection

**Expected performance:**
- **Precision:** High (with intersection)
- **Recall:** Medium (with intersection)
- **Localization:** Good
- **Speed:** Slow (multiple methods)
- **Expected leaderboard impact:** 0.5-0.7 edge Jaccard

## Theoretical Comparison

### Performance vs Complexity Matrix

| Approach | Precision | Recall | Speed | Complexity | Expected Score |
|----------|-----------|--------|-------|------------|----------------|
| LoG | Medium | Medium | Fast | Low | 0.3-0.5 |
| DoG | Medium | Medium | Fast | Low | 0.3-0.5 |
| Frangi | Med-High | Medium | Medium | Low | 0.4-0.6 |
| Threshold | Low-Med | Low-Med | Very Fast | Very Low | 0.2-0.4 |
| Adaptive Thresh | Medium | Med-High | Fast | Low | 0.3-0.5 |
| Watershed | Med-High | Medium | Medium | Medium | 0.4-0.6 |
| Cellpose | High | High | Slow (CPU) | Medium | 0.6-0.8 |
| StarDist | High | High | Slow (CPU) | Medium | 0.6-0.8 |
| Self-Supervised | High | High | Fast (after train) | High | 0.7-0.9 |
| Classical+ML | High | Med-High | Medium | Medium | 0.5-0.7 |
| Ensemble | High | Medium | Slow | Medium | 0.5-0.7 |

### Suitability for Competition Constraints

**Constraint: Conservative detection (avoid over-prediction)**
- **Best:** Classical+ML (ML filtering reduces FP)
- **Good:** Ensemble (intersection reduces FP)
- **Risky:** Pure ML (may over-predict)
- **Acceptable:** Classical with high thresholds

**Constraint: Sparse annotations (< 1% labeled)**
- **Best:** Classical (no training required)
- **Good:** Pre-trained ML (generalizes)
- **Possible:** Self-supervised (uses unlabeled data)
- **Poor:** Supervised ML (insufficient labels)

**Constraint: Kaggle runtime (12 hours max)**
- **Best:** Classical (fast)
- **Good:** Pre-trained ML (fast inference)
- **Possible:** Self-supervised (if pre-trained)
- **Poor:** Training from scratch

**Constraint: CPU/GPU availability**
- **Best:** Classical (CPU-friendly)
- **Good:** Pre-trained ML (GPU optional)
- **Poor:** Training methods (requires GPU)

## Recommended Approach

### Phase 1: Quick Baseline (Classical)
**Method:** Adaptive threshold + connected components + size filtering
**Rationale:**
- Fast to implement
- No training required
- Good starting point
- Easy to tune conservatively

**Expected score:** 0.3-0.4 edge Jaccard

### Phase 2: Improved Classical
**Method:** Frangi filter + adaptive threshold + watershed
**Rationale:**
- Better handling of touching cells
- Improved localization
- Still fast
- No training required

**Expected score:** 0.4-0.6 edge Jaccard

### Phase 3: ML-Enhanced
**Method:** Classical candidates + Cellpose (conservative threshold)
**Rationale:**
- Leverage state-of-the-art segmentation
- Conservative threshold to avoid over-prediction
- GPU acceleration if available
- Best potential performance

**Expected score:** 0.6-0.8 edge Jaccard

### Phase 4: Advanced (if time permits)
**Method:** Self-supervised pre-training + fine-tuning on sparse labels
**Rationale:**
- Maximum performance potential
- Adapts to specific dataset
- Leverages unlabeled data
- Conservative fine-tuning

**Expected score:** 0.7-0.9 edge Jaccard

## Implementation Priority

1. **Immediate:** Adaptive threshold + connected components (baseline)
2. **Short-term:** Frangi filter + watershed (improvement)
3. **Medium-term:** Cellpose with conservative tuning (ML enhancement)
4. **Long-term:** Self-supervised approach (if competitive)

## Parameter Tuning Strategy

### Classical Methods
- **Threshold:** Start high, lower gradually
- **Size range:** Filter by expected cell size
- **Scale range:** For multi-scale methods
- **Validation:** Use adjusted edge Jaccard on validation set

### ML Methods
- **Confidence threshold:** Tune for conservative detection
- **Preprocessing:** Normalize intensity distribution
- **Post-processing:** Size filtering, NMS
- **Validation:** Monitor node count vs estimated true nodes

## Failure Mode Mitigation

### Over-prediction
- **Classical:** Increase thresholds, add size filters
- **ML:** Increase confidence threshold, add post-processing filters
- **Both:** Monitor node count, apply penalty-aware optimization

### Under-prediction
- **Classical:** Lower thresholds, reduce size filters
- **ML:** Lower confidence threshold
- **Both:** Focus on annotated regions

### Poor Localization
- **Classical:** Use sub-pixel refinement, centroid calculation
- **ML:** Use model's uncertainty estimates
- **Both:** Post-processing refinement

## Next Steps

1. **Step 5:** Design tracking approaches
2. **Step 6:** Theoretical comparison of detection + tracking combinations
3. **Step 7:** Implement best detection + tracking baseline
