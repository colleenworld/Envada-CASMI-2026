# MS2DeepScore feature environment

This directory is intentionally isolated from the main CASMI environment.

The main project currently targets Python 3.14. MS2DeepScore 2.11.0 is
officially tested with Python 3.11-3.13, so feature generation runs under
Python 3.13 and writes Parquet files consumed by the main ranker.

## Create/sync

```bash
uv python install 3.13
uv sync --project tools/ms2deepscore --python 3.13
```

## Verify

```bash
uv run --project tools/ms2deepscore python -c \
  "import ms2deepscore, sys; print(sys.version); print(ms2deepscore.__version__)"
```

Do not add MS2DeepScore to the repository root `pyproject.toml`.
