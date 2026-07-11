# Revised Roadmap for 0.93+ Competition Score

Based on external review, this roadmap prioritizes the highest-impact components for reaching 0.93+ score.

## Priority Order

### Phase 1: Perfect Validation ⭐⭐⭐⭐⭐
**Embryo-disjoint validation split**
- The competition is embryo-disjoint, so validation must also be embryo-disjoint
- Random validation can make you think a model is better than it really is
- Monitor Edge Jaccard and Division Jaccard during validation (not just Dice)

### Phase 2: Train 3D U-Net Detector ⭐⭐⭐⭐⭐
**Deep supervision architecture**
- Add auxiliary outputs at 1/2, 1/4, 1/8 resolution
- Train with combined Dice + Focal loss
- Use anisotropic-aware augmentation
- Improve detector until local Edge Jaccard stops improving

### Phase 3: Learned Association Network ⭐⭐⭐⭐⭐
**Predict P(match) instead of hand-tuned cost**
- Input: Cell A features + Cell B features
- Output: Probability they are the same cell
- Features: distance, motion, confidence, local density, intensity
- Hungarian minimizes 1-P(match) instead of manually engineered cost

### Phase 4: Division Handling ⭐⭐⭐⭐
**Motion + geometry + graph constraints (no CNN initially)**
- Use motion patterns to identify division candidates
- Geometric validation (parent-child distance, child-child distance, division angle)
- Graph constraints (fork detection in track graph)
- Postpone CNN classifier for later

### Phase 5: Graph Optimization ⭐⭐⭐⭐
**Global track optimization**
- Resolve conflicts and merge short tracks
- Gap closing with velocity constraints
- Track continuity optimization

### Phase 6: Test-Time Augmentation ⭐⭐⭐
**Cheap improvement (+0.01 to +0.03)**
- Spatial flips (XY axes)
- Intensity variations
- Ensemble predictions

### Phase 7: Model Ensemble ⭐⭐⭐
**Multiple model variants**
- Different U-Net architectures
- Different training seeds
- Weighted averaging of predictions

## Components Removed from Initial Roadmap

### Optical Flow Network
- Too much engineering effort
- Unknown improvement
- Hard to train

### Pseudo-label Training
- Don't know if detector or tracker is limiting performance yet
- Postpone until baseline is solid

### CellPose Integration
- Competition labels centroids, not masks
- Segmentation excellence doesn't necessarily translate to better centroid tracking
- Skip initially

### Multiple External Trackers (Ultrack, Trackastra)
- Don't integrate all of them
- Build one very strong pipeline instead
- Reduce engineering complexity

## Key Insight

Before adding 20 tracking tricks, first make sure the detector is excellent. The error analysis showed that detection misconfiguration was the primary bottleneck.

## Expected Timeline

- **Phase 1-2**: 2-3 weeks (validation + U-Net training)
- **Phase 3-4**: 2-3 weeks (learned association + division)
- **Phase 5-7**: 1-2 weeks (optimization + TTA + ensemble)

Total: 5-8 weeks to reach 0.90+ score.
