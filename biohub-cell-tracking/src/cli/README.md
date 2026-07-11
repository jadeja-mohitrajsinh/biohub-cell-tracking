# CLI Module

## Purpose

Command-line entry point for dataset inspection, evaluation, and submission.

## API

```bash
biohub list --data-root ../train
biohub inspect <sample_id> --data-root ../train
biohub evaluate <sample_id> --data-root ../train
biohub submit <sample_id> --data-root ../train
```

## Roadmap

- Batch evaluation across splits
- Visualization subcommands
