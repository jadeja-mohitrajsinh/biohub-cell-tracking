# Competition Overview

**Name:** Biohub — Cell Tracking During Development

**Task:** Detect and track cells in 3D+time fluorescence microscopy of developing zebrafish embryos.

**Submission format:** GEFF graphs (`{sample_id}.geff`) with nodes at `(t, z, y, x)` and directed temporal edges. Divisions are one parent with two outgoing edges.

**Metric:** Combined adjusted edge Jaccard and division Jaccard. See [metric.md](metric.md).

**Official resources:**

- [Kaggle competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- [Metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- [Baseline repository](https://github.com/royerlab/kaggle-cell-tracking-competition)
