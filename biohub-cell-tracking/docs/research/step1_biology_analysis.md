# Step 1: Biological Context Analysis

## Zebrafish Embryonic Development

### Biological System
- **Organism:** Zebrafish (Danio rerio) embryo
- **Developmental stage:** Early embryogenesis
- **Imaging modality:** 3D fluorescence microscopy over time
- **Temporal resolution:** 100 timepoints (likely spanning several hours)
- **Spatial resolution:** 64 Z-slices × 256×256 XY pixels

### Cell Division Patterns in Early Embryogenesis

#### Mitosis Characteristics
1. **Rapid divisions:** Early zebrafish embryos undergo rapid synchronous divisions
2. **Spatial organization:** Cells are densely packed in a spherical/ellipsoidal embryo
3. **Division orientation:** Can be random or oriented depending on developmental stage
4. **Timing:** Divisions occur at specific time intervals, not uniformly distributed

#### Cell Behavior
- **Migration:** Cells move significantly during development
- **Shape changes:** Cells deform before/during/after division
- **Density:** High cell density creates challenges for segmentation
- **Apoptosis:** Some cells may die (though less common in early stages)

### Implications for Tracking

#### Detection Challenges
- **High density:** Cells are tightly packed, making boundary detection difficult
- **Variable intensity:** Fluorescence intensity varies across cells and time
- **Shape variability:** Cells are not perfect spheres; they deform during division
- **Partial visibility:** Cells at embryo boundaries may be partially visible

#### Tracking Challenges
- **Rapid motion:** Cells can move significantly between consecutive frames
- **Appearance changes:** Cell appearance changes during division cycle
- **Occlusions:** Cells may be temporarily occluded by neighbors
- **Division timing:** Divisions don't always occur at exact frame boundaries

#### Division Detection Challenges
- **Mother-daughter similarity:** Daughters may look similar to mother initially
- **Spatial proximity:** Daughters remain close to division site
- **Temporal ambiguity:** Exact division frame may be between timepoints
- **False positives:** Cell pairs may appear as divisions without actual mitosis

### Physical Scale Considerations

**Voxel dimensions:**
- T: 1.0 µm (time)
- Z: 1.625 µm (depth)
- Y: 0.40625 µm (vertical)
- X: 0.40625 µm (horizontal)

**Anisotropy:**
- Z-resolution is 4× coarser than XY
- This affects detection and tracking algorithms
- Must account for anisotropy in distance calculations

**Maximum matching distance: 7 µm**
- This is approximately:
  - 4.3 voxels in Z
  - 17.2 voxels in Y/X
- Cells can move significantly between frames

### Sparse Annotation Strategy

**Why sparse?**
- Manual annotation of all cells is infeasible
- Representative regions are annotated
- Focus on challenging areas (divisions, dense regions)

**Implications:**
- Cannot train on full supervision
- Must handle unlabeled regions
- Over-prediction in unlabeled regions is penalized
- Need to estimate true cell count from metadata

### Key Biological Insights for Algorithm Design

1. **Exploit temporal consistency:** Cells don't teleport; motion is smooth
2. **Division is a local event:** Look for splitting trajectories
3. **Density varies:** Need adaptive algorithms for different regions
4. **Anisotropy matters:** Must use physical distances, not voxeldistances
5. **Appearance changes:** Cells change during cell cycle
6. **Early divisions are synchronous:** May have regular timing patterns

### Failure Modes to Anticipate

1. **Identity switches:** When cells cross paths
2. **Missed divisions:** When division occurs between frames
3. **False divisions:** When two cells appear close but aren't related
4. **Broken tracks:** When detection fails temporarily
5. **Merge errors:** When two cells are detected as one
6. **Split errors:** When one cell is detected as two

### Research Questions

1. What is the typical cell size in voxels?
2. What is the typical cell displacement between frames?
3. How many divisions occur per timepoint?
4. What is the cell density distribution?
5. Are there spatial patterns to divisions?
6. How does intensity vary across the cell cycle?

These questions will be answered in Step 2 (Dataset Analysis).
