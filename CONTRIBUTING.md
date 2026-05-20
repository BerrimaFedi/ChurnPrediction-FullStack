# Contributing

Thanks for contributing to the Churn Prediction project. This short guide helps
contributors get a local development environment up and running and describes
basic workflows (running experiments, testing, and CI).

## Development environment

1. Create and activate a virtual environment (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start MLflow UI in a separate terminal (optional but recommended):

```powershell
python -m mlflow ui --port 5000
```

3. Run a quick experiment (example):

```powershell
python src/train.py
```

## Coding standards
- Keep code simple and focused in `src/`.
- Run tests with `pytest` and check style with `flake8`.

## Running tests and lint

```powershell
pytest -q
flake8 src
```

## Committing
- The repo includes a pre-commit model validation hook created by `tache5.py`.
  Ensure MLflow is reachable if you want the hook to validate model metrics.

## Pull requests
- Open PRs targeting `main` with a short description and test coverage for changes.
- Describe any MLflow artifacts or large outputs produced by experiments.

## CI
This repository includes a simple GitHub Actions workflow `.github/workflows/ci.yml`
that installs dependencies and runs tests and linting on push and PR.
