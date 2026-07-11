# Data Flow

```text
../train/{sample}.zarr
        │
        ▼
 ZarrVolumeReader.read_timepoint(t)
        │
        ▼
 EmbryoDataset.get_timepoint(t) ──► visualization / preprocessing
        │
../train/{sample}.geff
        │
        ▼
 GeffReader.read_graph() ──► TrackingGraph
        │
        ├──► metrics.evaluate_graph()
        └──► SubmissionGenerator.write_sample()
```

Configuration flows from `configs/default.yaml` through `utils.config.load_config()` into CLI commands and experiment scripts.
