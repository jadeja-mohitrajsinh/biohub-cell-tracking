# Pipeline

End-to-end flow from raw competition data to Kaggle submission.

1. **Ingest** — `DatasetCatalog` discovers `{id}.zarr` + `{id}.geff`
2. **Load** — lazy timepoint streaming via `ZarrVolumeReader`
3. **Preprocess** — normalization (optional, Phase 2)
4. **Detect / Segment** — Phase 4
5. **Track** — Phase 3–5
6. **Evaluate** — offline edge/division Jaccard
7. **Submit** — `SubmissionGenerator` writes test GEFF files

Phase 1 implements steps 1, 2, 6, and 7.
