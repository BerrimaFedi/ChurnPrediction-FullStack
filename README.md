# Churn Prediction — Telco Customers

Compact MLOps project to train, evaluate and serve models predicting customer churn
for a telecom dataset. The repository includes preprocessing, experiments, model
registration with MLflow, drift detection and a small serving/test client.

## Quick overview
- Data: `data/raw/Telecom Customers Churn.csv`
- Main scripts and utilities: see `src/` (preprocessing, training, evaluation, serving)
- Experiments & artifacts: MLflow (tracks runs and stores model artifacts)
- Frontend: `ml-front/` (React scaffold)

## Prerequisites
- Python 3.8+ (code has been developed on Windows; some scripts use Windows-style paths)
- MLflow server (UI + tracking) running at `http://127.0.0.1:5000` for full functionality
- Recommended: create a virtualenv and install dependencies

Example (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: `requirements.txt` is present in the repo root. If it's empty, install the
packages used in `src/` (pandas, scikit-learn, mlflow, fastapi, uvicorn, matplotlib,
seaborn, xgboost, evidently, scipy, requests).

## Useful commands
- Start MLflow UI (in a separate terminal):

```powershell
python -m mlflow ui --port 5000
```

- Run experiments / training (examples):

```powershell
# Train multiple baseline models and log to MLflow
python src/train.py

# Run the longer experiment pipelines and registry logic
python src/tache4v2.py
python src/tache5.py
```

- Serve a registered model (MLflow model serving) — example using the registered
  model `churn_adaboost_production` promoted to Production:

```powershell
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/churn_adaboost_production/Production" --port 1234 --no-conda
```

- Alternatively run the FastAPI wrapper that loads a model from the registry:

```powershell
python src/serve_api.py
# or
uvicorn src.serve_api:app --host 127.0.0.1 --port 1234
```

- Test the API (client script):

```powershell
python src/test_api.py
```

## Key files
- `src/preprocessing.py`: data cleaning, encoding and train/test split.
- `src/train.py`: trains multiple models, logs runs and artifacts to MLflow.
- `src/evaluate.py`: evaluation helpers, plotting and model comparison.
- `src/tache4.py`, `src/tache4v2.py`, `src/tache5.py`: extended experiments, model
  registry operations, drift detection and the test client generator.
- `src/serve_api.py`: FastAPI endpoint that loads the production model from MLflow.
- `mlartifacts/`, `mlartifacts/models/`: saved model artifacts and versions.

## Data
Place the dataset at `data/raw/Telecom Customers Churn.csv`. Many scripts hardcode
this path; adjust `DATA_PATH` variables in `src/*` if you move the data.

## MLflow & Model Registry
The project expects an MLflow tracking server at `http://127.0.0.1:5000`. The
pipeline in `src/tache5.py` will register the best AdaBoost run under the name
`churn_adaboost_production` and attempt to promote it to Staging/Production.

## Drift detection
`src/tache5.py` implements KS-based drift checks and produces `drift_report.html`.
If Evidently is available it will produce a richer report; otherwise a KS fallback
HTML is generated.

## Frontend
The `ml-front/` directory contains a Create React App skeleton. To run:

```bash
cd ml-front
npm install
npm start
```

## Makefile
There is a `Makefile` with convenience targets: `setup`, `mlflow-ui`, `tache4`,
`tache5`, `serve`, `test`, `pipeline` and `clean`. On Windows you can run these
from PowerShell using `make <target>` if `make` is available (WSL or Make for Windows).


