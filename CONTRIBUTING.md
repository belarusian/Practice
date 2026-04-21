# Contributing

## Development Setup

Use `uv` for local development.

```bash
uv sync --extra dev
PYTHONPATH=src pytest -q
python3 -m compileall src tests
```

Install optional stacks only when you need them:

```bash
uv sync --extra serving
uv sync --extra training
uv sync --extra transformer
uv sync --extra vector
uv sync --extra workflow
```

## Branching

Use a focused branch per change. Keep runtime proof artifacts under `artifacts/`; that directory is ignored and should not be committed.

## Sunny / Private Lab Work

The `ops/sunny` scripts are examples of repo-backed host operations. Public defaults use placeholder hosts and paths. For a private lab checkout, copy:

```bash
cp ops/sunny/lab.env.example ops/sunny/lab.env
```

Then edit `ops/sunny/lab.env` with your own SSH targets, proxy, and host paths. Do not commit that file.

## Validation

Local tests prove code behavior that does not depend on a specific machine. Host-specific GPU validation should use the report-producing scripts under `ops/sunny` or an equivalent private-lab adaptation.
