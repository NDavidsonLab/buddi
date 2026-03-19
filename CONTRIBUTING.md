# Contributing

## Development Environment

This project uses `uv` for local environment management, packaging, and CI.

For a standard development environment:

```bash
uv sync --frozen
```

That installs the default dependency groups declared in [pyproject.toml](/Users/buntend/Documents/work/buddi_v2/pyproject.toml), which currently include both `dev` and `docs`.

If you need to rebuild the environment from scratch, remove your virtual environment and run `uv sync --frozen` again.

## Local Workflow

Before opening a PR, run the same core checks the repository expects:

```bash
uv run --frozen pre-commit run --all-files
uv run --frozen pytest
```

Notes:

- The current pytest suite is notebook-focused and skips the slow notebook executions unless you explicitly pass `--run-slow`.
- You can run the notebook execution suite locally with:

```bash
uv run --frozen pytest tests/test_notebooks.py -v --durations=0 --run-slow
```

- If you want the helper Jupyter kernel and lab launcher from the project tasks:

```bash
uv run --frozen poe buddi-lab
```

## Documentation

Docs are built with Sphinx from `docs/src`.

To build the docs locally:

```bash
uv run --frozen --group docs sphinx-build -b html docs/src docs/build
```

The docs publishing workflow runs on pushes to `main` and on published GitHub releases.

## Releases

This branch uses GitHub Releases as the trigger for publishing to PyPI.

The release flow is:

1. Changes merge to `main`.
1. The `release drafter` workflow updates the draft GitHub release notes automatically.
1. When you are ready to ship, publish a GitHub Release from the draft.
1. Publishing the release triggers `.github/workflows/pypi.yml`.
1. That workflow checks out the repo, fetches tags, runs `uv build`, and publishes the generated distributions to PyPI using trusted publishing.

In practice, that means:

- Make sure the branch is merged to `main`.
- Confirm CI is green.
- Review the drafted release notes.
- Publish the GitHub Release.
- Do not upload artifacts to PyPI manually unless the automated release flow is unavailable and you have explicitly decided to bypass it.

## Versioning

Package versions are derived with `setuptools-scm`.

A generated version file lives at `src/buddi/_version.py` when release artifacts are built from git metadata and tags rather than by manually editing a version string in source.

## Pull Requests

Prefer small, reviewable PRs.

For code or notebook changes:

- include the relevant test command(s) in the PR description
- note whether notebook execution was run
- call out any platform-specific assumptions, especially around TensorFlow and local Jupyter usage
