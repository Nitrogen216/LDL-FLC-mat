# Repository Guidelines

## Project Structure & Module Organization
- Root folders: `python/` (primary code) and `matlab/` (placeholder).
- Python entry points: `python/run_LDLFC.py`, `python/run_LDLFCC.py`, plus other experiment scripts (e.g., `run_LDLLRR.py`, `run_LDLSCL.py`).
- Core modules: `python/ldl_models.py`, `python/ldl_metrics.py`, `python/ldl_flc.py`, `python/util.py`.
- Datasets (processed) live under subfolders of `python/` (e.g., `RAF_ML/`, `SJAFFE/`, `Gene/`, `Flickr_ldl/`). Cite the original papers if used.

## Build, Test, and Development Commands
- Run LDL-FC: `python python/run_LDLFC.py`
- Run LDL-FCC: `python python/run_LDLFCC.py`
- Run other baselines: `python python/run_LDLLRR.py`, `python python/run_LDLSCL.py`, etc.
- Environment: Python 3.8+ recommended. Create a venv: `python -m venv .venv && source .venv/bin/activate` and install project requirements as needed.

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indentation, 88–120 char line length.
- Names: `snake_case` for functions/variables, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Imports: standard lib, third-party, then local; prefer absolute module paths within `python/`.
- Formatting/Linting (recommended): `black`, `isort`, `ruff` or `flake8`. Keep diffs minimal and consistent with existing style.

## Testing Guidelines
- Framework: prefer `pytest` for new tests.
- Location: add tests under `python/tests/` mirroring module paths (e.g., `python/tests/test_ldl_metrics.py`).
- Conventions: name files `test_*.py`; aim to cover core metrics and model behaviors. Use small, deterministic fixtures.
- Run tests: `pytest -q` from repo root (after installing dev deps).

## Commit & Pull Request Guidelines
- Commits: concise imperative messages (e.g., "Add LDL-FC training loop"). Group related changes; avoid mixed refactors + features.
- PRs: include a clear description, motivation, and any dataset/setup notes. Link issues when applicable and add before/after metrics or logs for model changes.
- Size: prefer small, focused PRs. Include usage examples (commands and paths) in the description.

## Security & Configuration Tips
- Do not commit large raw datasets or secrets. Use `.gitignore` for generated artifacts.
- Set seeds where available for reproducibility; document non-deterministic steps.
