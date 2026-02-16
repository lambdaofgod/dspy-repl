# Releasing dspy-repl

This project uses GitHub Actions and PyPI Trusted Publishing.

## One-time setup

1. In PyPI, create a Trusted Publisher for this repository.
2. In GitHub, create environments:
   - `testpypi`
   - `pypi`
3. Grant the workflow permission to request OIDC tokens (`id-token: write` is already set in workflow files).
4. Optionally protect the `pypi` environment with required reviewers.

## Validate locally

```bash
python -m pip install -e ".[dev]"
python -m pip install build twine
ruff check src tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m build
python -m twine check --strict dist/*
```

## Publish flow

- TestPyPI dry run (manual): run the `Publish` workflow with `target=testpypi`.
- Production publish: push a tag like `v0.1.0`.

```bash
git tag v0.1.0
git push origin v0.1.0
```

The tag triggers `publish.yml`, which builds artifacts, validates metadata, then publishes to PyPI via Trusted Publishing.
