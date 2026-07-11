# Contributing

## Workflow

1. Create a feature branch for each milestone.
2. Keep modules small and typed.
3. Add unit tests for core logic.
4. Update module README and docs when APIs change.

## Code Style

- Type hints on public functions
- NumPy-style docstrings
- `ruff` for linting
- Prefer focused diffs over broad refactors

## Testing

```bash
pytest
```

Integration tests expect local competition data at `../train`.

## Documentation

- Update `docs/` when behavior or metrics change
- Record experiments in `docs/experiments.md`
