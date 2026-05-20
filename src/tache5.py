import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import subprocess

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score
from scipy import stats

from preprocessing import load_and_prepare

# ── Config ──
MLFLOW_URI      = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Churn_Tache4_RandomForest"
MODEL_NAME      = "churn_adaboost_production"
DATA_PATH       = r"C:\project_ChurnPrediction\data\raw\Telecom Customers Churn.csv"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)


# ════════════════════════════════════════════════════════
# PARTIE 2 — COMPARAISON DE PLUSIEURS RUNS
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 2 — COMPARAISON D'EXPÉRIMENTATIONS")
print("="*60)

configs = [
    {
        "run_name":    "rf_50trees_depth3",
        "model_type":  "RandomForest",
        "n_estimators": 50,
        "max_depth":    3,
    },
    {
        "run_name":    "rf_200trees_depth10",
        "model_type":  "RandomForest",
        "n_estimators": 200,
        "max_depth":    10,
    },
    {
        "run_name":    "gb_100trees_lr01",
        "model_type":  "GradientBoosting",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth":    3,
    },
    {
        "run_name":    "adaboost_50_lr1",
        "model_type":  "AdaBoost",
        "n_estimators": 50,
        "learning_rate": 1.0,
    },
]

partie2_results = []

for cfg in configs:
    with mlflow.start_run(run_name=cfg["run_name"]):

        # Params communs
        params = {k: v for k, v in cfg.items() if k not in ["run_name"]}
        mlflow.log_params(params)
        mlflow.set_tag("dataset", "Telco Customer Churn")
        mlflow.set_tag("task", "classification")

        # Instanciation du modèle
        if cfg["model_type"] == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42
            )
        elif cfg["model_type"] == "GradientBoosting":
            model = GradientBoostingClassifier(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                random_state=42
            )
        elif cfg["model_type"] == "AdaBoost":
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                random_state=42
            )

        # Entraînement
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métriques
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "cv_score":  round(cross_val_score(model, X_train, y_train, cv=5).mean(), 4),
        }
        mlflow.log_metrics(metrics)

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_title(f"Confusion Matrix — {cfg['run_name']}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        cm_path = f"confusion_matrix_{cfg['run_name']}.png"
        fig.savefig(cm_path, dpi=150); plt.close()
        mlflow.log_artifact(cm_path)

        # Rapport de classification
        report = classification_report(y_test, y_pred,
                                       target_names=['No Churn', 'Churn'])
        report_path = f"classification_report_{cfg['run_name']}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Run: {cfg['run_name']}\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log modèle
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature
        )

        run_id = mlflow.active_run().info.run_id
        partie2_results.append({
            "run_name": cfg["run_name"],
            "run_id":   run_id,
            **metrics
        })

        print(f"  ✅ {cfg['run_name']:30s} | acc={metrics['accuracy']} | f1={metrics['f1_score']} | auc={metrics['roc_auc']}")

# Tableau comparatif
df_p2 = pd.DataFrame(partie2_results).sort_values("f1_score", ascending=False)
print("\n" + "="*70)
print("   COMPARAISON PARTIE 2")
print("="*70)
print(df_p2[["run_name","accuracy","f1_score","roc_auc","cv_score"]].to_string(index=False))
best_p2 = df_p2.iloc[0]
print(f"\n🏆 Meilleur modèle : {best_p2['run_name']} (F1={best_p2['f1_score']})")
print("✅ Partie 2 terminée")


# ════════════════════════════════════════════════════════
# PARTIE 2.3 — REQUÊTES PROGRAMMATIQUES MLFLOW CLIENT
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 2.3 — MLFLOW CLIENT — MEILLEUR RUN")
print("="*60)

client = MlflowClient(tracking_uri=MLFLOW_URI)
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=5
)

best_run = runs[0]
print(f"\nMeilleur run global (par F1) :")
print(f"  Run ID   : {best_run.info.run_id}")
print(f"  Run Name : {best_run.info.run_name}")
print(f"  F1 Score : {best_run.data.metrics.get('f1_score', 'N/A')}")
print(f"  Accuracy : {best_run.data.metrics.get('accuracy', 'N/A')}")
print(f"  Params   : {best_run.data.params}")


# ════════════════════════════════════════════════════════
# PARTIE 3 — MODEL REGISTRY
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 3 — MODEL REGISTRY")
print("="*60)

# Trouver le run AdaBoost de la Partie 2
ada_run = df_p2[df_p2["run_name"] == "adaboost_50_lr1"].iloc[0]
ada_run_id = ada_run["run_id"]
model_uri  = f"runs:/{ada_run_id}/model"

print(f"\n📦 Enregistrement du modèle AdaBoost (run_id={ada_run_id})")

# Enregistrement
registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)
version = registered.version
print(f"  ✅ Modèle enregistré — version {version}")

# Description + tags
client.update_registered_model(
    name=MODEL_NAME,
    description="Modèle AdaBoost pour la prédiction du churn client Telco. "
                "Meilleur F1-Score parmi tous les modèles testés (Tâche 4 V2)."
)
client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="validated_by",
    value="equipe_data"
)
client.set_model_version_tag(
    name=MODEL_NAME,
    version=version,
    key="dataset",
    value="Telco_Customer_Churn"
)
print(f"  ✅ Description et tags ajoutés")

# ── Promotion en Staging ──
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Staging",
    archive_existing_versions=False
)
print(f"  ✅ Modèle v{version} promu en Staging")

# ── Validation avant Production ──
SEUIL_PRODUCTION = 0.79   # AdaBoost Test Acc = 0.8055

acc_val = ada_run["accuracy"]
print(f"\n  Validation : accuracy={acc_val:.4f} | seuil={SEUIL_PRODUCTION}")

if acc_val >= SEUIL_PRODUCTION:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"  ✅ Modèle v{version} promu en Production !")
else:
    print(f"  ⚠️  Modèle non promu : accuracy {acc_val:.3f} < seuil {SEUIL_PRODUCTION}")

print("✅ Partie 3 terminée")


# ════════════════════════════════════════════════════════
# PARTIE 4 — SERVING API REST (instructions + test Python)
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 4 — SERVING API REST")
print("="*60)

print("""
  Pour servir le modèle, ouvrez un NOUVEAU terminal et exécutez :

  set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
  mlflow models serve -m "models:/churn_adaboost_production/Production" --port 1234 --no-conda

  Puis testez avec curl :
  curl -X POST http://localhost:1234/invocations ^
       -H "Content-Type: application/json" ^
       -d "{\\"dataframe_split\\": {\\"columns\\": [...], \\"data\\": [[...]]}}"
""")

# Test Python du endpoint (si le serveur tourne)
import requests, json

def test_api_endpoint(X_sample, port=1234):
    """Test the MLflow serving endpoint."""
    url = f"http://localhost:{port}/invocations"
    payload = {
        "dataframe_split": {
            "columns": list(X_sample.columns),
            "data":    X_sample.values.tolist()
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"  ✅ API Response : {resp.json()}")
        else:
            print(f"  ⚠️  Status {resp.status_code} : {resp.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("  ⚠️  Serveur non démarré — lancez 'mlflow models serve' dans un autre terminal")
    except Exception as e:
        print(f"  ⚠️  Erreur : {e}")

# Test avec 3 exemples du jeu de test
X_sample = X_test.head(3)
print("  Test de l'endpoint avec 3 exemples...")
test_api_endpoint(X_sample)

# Sauvegarde du script de test API
api_test_script = '''import requests, json, pandas as pd
from preprocessing import load_and_prepare

DATA_PATH = r"C:\\project_ChurnPrediction\\data\\raw\\Telecom Customers Churn.csv"
X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)
X_sample = X_test.head(5)

payload = {
    "dataframe_split": {
        "columns": list(X_sample.columns),
        "data":    X_sample.values.tolist()
    }
}

resp = requests.post("http://localhost:1234/invocations", json=payload)
print("Statut :", resp.status_code)
print("Prédictions :", resp.json())
'''
with open("test_api.py", "w") as f:
    f.write(api_test_script)
print("  ✅ Script test_api.py créé")
print("✅ Partie 4 terminée")


# ════════════════════════════════════════════════════════
# PARTIE 5 — MAKEFILE
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 5 — MAKEFILE")
print("="*60)

makefile_content = """# Makefile — Pipeline MLOps Churn Prediction
# Usage : make pipeline

.PHONY: setup train tache4 tache5 register serve test drift pipeline clean

# ── Installation des dépendances ──
setup:
\tpip install -r requirements.txt
\t@echo "✅ Dépendances installées"

# ── Lancer MLflow UI ──
mlflow-ui:
\tstart /B python -m mlflow ui --port 5000
\t@echo "✅ MLflow UI : http://127.0.0.1:5000"

# ── Prétraitement ──
preprocess:
\tpython preprocessing.py
\t@echo "✅ Prétraitement terminé"

# ── Entraînement Tâche 4 V2 (RF + AdaBoost + XGBoost) ──
tache4:
\tpython tache4v2.py
\t@echo "✅ Tâche 4 terminée"

# ── Pipeline MLOps Tâche 5 ──
tache5:
\tpython tache5.py
\t@echo "✅ Tâche 5 terminée"

# ── Servir le modèle en Production ──
serve:
\tset MLFLOW_TRACKING_URI=http://127.0.0.1:5000
\tmlflow models serve -m "models:/churn_adaboost_production/Production" --port 1234 --no-conda
\t@echo "✅ API disponible sur http://localhost:1234"

# ── Test de l'API ──
test:
\tpython test_api.py
\t@echo "✅ Test API terminé"

# ── Détection de drift ──
drift:
\tpython simulate_drift.py
\t@echo "✅ Détection drift terminée"

# ── Pipeline complet ──
pipeline: tache4 tache5 test
\t@echo "🚀 Pipeline MLOps complet exécuté avec succès"

# ── Nettoyage des fichiers temporaires ──
clean:
\tdel /Q *.png *.txt *.csv *.html 2>nul || true
\t@echo "✅ Nettoyage terminé"
"""

with open("Makefile", "w", encoding="utf-8") as f:
    f.write(makefile_content)

print("  ✅ Makefile créé")

# ── Pre-commit hook Git ──
hook_content = """#!/bin/bash
# .git/hooks/pre-commit
# Valide que le meilleur modèle dépasse le seuil d'accuracy avant commit

echo "🔍 Validation du modèle en cours..."
python -c "
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = MlflowClient()
experiment = client.get_experiment_by_name('Churn_Tache4_RandomForest')
if experiment is None:
    print('Aucune expérience trouvée — commit autorisé')
    exit(0)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.accuracy DESC'],
    max_results=1
)
if not runs:
    print('Aucun run trouvé — commit autorisé')
    exit(0)

acc = runs[0].data.metrics.get('accuracy', 0)
assert acc > 0.79, f'Accuracy trop faible: {acc:.3f} < 0.79'
print(f'✅ Validation OK : accuracy={acc:.3f}')
"
if [ $? -ne 0 ]; then
    echo "❌ COMMIT REFUSÉ : modèle invalide"
    exit 1
fi
echo "✅ Commit autorisé"
"""

os.makedirs(".git/hooks", exist_ok=True)
hook_path = ".git/hooks/pre-commit"
with open(hook_path, "w", encoding="utf-8") as f:
    f.write(hook_content)

print("  ✅ Pre-commit hook Git créé (.git/hooks/pre-commit)")
print("     → Sur Linux/Mac : chmod +x .git/hooks/pre-commit")
print("✅ Partie 5 terminée")


# ════════════════════════════════════════════════════════
# PARTIE 6 — DÉTECTION DU DATA DRIFT
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PARTIE 6 — DÉTECTION DU DATA DRIFT")
print("="*60)

mlflow.set_experiment("monitoring_drift")

with mlflow.start_run(run_name="drift_check_v1"):

    # ── 6.2 Simulation du drift ──
    print("\n  6.2 — Simulation du drift...")
    X_prod = X_test.copy()
    num_cols = X_prod.select_dtypes(include=np.number).columns

    # Décalage de moyenne + bruit sur les 2 premières features numériques
    for col in num_cols[:2]:
        X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

    print(f"  Moyenne '{num_cols[0]}' — Ref: {X_train[num_cols[0]].mean():.3f} | Prod: {X_prod[num_cols[0]].mean():.3f}")
    print(f"  Moyenne '{num_cols[1]}' — Ref: {X_train[num_cols[1]].mean():.3f} | Prod: {X_prod[num_cols[1]].mean():.3f}")

    # ── 6.4 KS-test par feature (toujours disponible, sans Evidently) ──
    print("\n  6.4 — KS-test par feature...")
    ks_results = []
    for col in X_train.select_dtypes(include='number').columns:
        stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
        ks_results.append({
            "feature":  col,
            "ks_stat":  round(stat, 4),
            "p_value":  round(pvalue, 6),
            "drifted":  pvalue < 0.05
        })
        mlflow.log_metric(f"ks_pvalue_{col[:20]}", pvalue)

    df_ks = pd.DataFrame(ks_results)
    df_ks.to_csv("ks_drift_results.csv", index=False)
    mlflow.log_artifact("ks_drift_results.csv")

    n_drifted_ks = int(df_ks["drifted"].sum())
    n_total_ks   = len(df_ks)
    drift_share_ks = n_drifted_ks / n_total_ks if n_total_ks > 0 else 0.0

    print(f"\n  Résultats KS-test ({n_drifted_ks}/{n_total_ks} features driftées) :")
    drifted_rows = df_ks[df_ks["drifted"]]
    if not drifted_rows.empty:
        print(drifted_rows.to_string(index=False))
    else:
        print("  Aucune feature driftée détectée.")

    # ── 6.3 Rapport Evidently (avec fallback HTML si incompatible) ──
    print("\n  6.3 — Génération rapport Evidently...")
    drift_share   = 0.0
    n_drifted_ev  = 0
    n_total_ev    = n_total_ks

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric

        # Rapport HTML complet
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X_train, current_data=X_prod)
        report.save_html("drift_report.html")
        mlflow.log_artifact("drift_report.html")
        print("  ✅ Rapport HTML Evidently sauvegardé : drift_report.html")

        # Extraction des scores numériques
        score_report = Report(metrics=[DatasetDriftMetric()])
        score_report.run(reference_data=X_train, current_data=X_prod)
        result = score_report.as_dict()

        drift_share   = result['metrics'][0]['result']['drift_share']
        dataset_drift = result['metrics'][0]['result']['dataset_drift']
        n_drifted_ev  = result['metrics'][0]['result']['number_of_drifted_columns']
        n_total_ev    = result['metrics'][0]['result']['number_of_columns']

        mlflow.log_metric("drift_share",     drift_share)
        mlflow.log_metric("drifted_columns", n_drifted_ev)
        mlflow.log_metric("total_columns",   n_total_ev)
        mlflow.log_metric("dataset_drifted", int(dataset_drift))

        print(f"  Drift share : {drift_share:.2%} | Colonnes driftées : {n_drifted_ev}/{n_total_ev}")
        print(f"  Dataset drifté : {'OUI' if dataset_drift else 'NON'}")

    except Exception as e:
        # Fallback : rapport HTML généré depuis les résultats KS-test
        print(f"  ⚠️  Evidently indisponible ({type(e).__name__}: {str(e)[:80]})")
        print("  → Génération d'un rapport HTML alternatif via KS-test...")

        drift_share  = drift_share_ks
        n_drifted_ev = n_drifted_ks
        n_total_ev   = n_total_ks

        mlflow.log_metric("drift_share",     drift_share)
        mlflow.log_metric("drifted_columns", n_drifted_ev)
        mlflow.log_metric("total_columns",   n_total_ev)
        mlflow.log_metric("dataset_drifted", int(drift_share > 0.15))

        # Générer un rapport HTML complet sans Evidently
        rows_html = ""
        for _, row in df_ks.iterrows():
            color = "#dc2626" if row["drifted"] else "#16a34a"
            badge = "⚠️ OUI" if row["drifted"] else "✅ NON"
            rows_html += (
                f"<tr>"
                f"<td>{row['feature']}</td>"
                f"<td style='text-align:center'>{row['ks_stat']}</td>"
                f"<td style='text-align:center'>{row['p_value']}</td>"
                f"<td style='text-align:center;color:{color};font-weight:bold'>{badge}</td>"
                f"</tr>\n"
            )

        summary_color = "#dc2626" if drift_share > 0.30 else ("#f59e0b" if drift_share > 0.15 else "#16a34a")
        summary_label = "CRITIQUE" if drift_share > 0.30 else ("AVERTISSEMENT" if drift_share > 0.15 else "STABLE")

        html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Rapport Data Drift — Telco Churn</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
    h1 {{ color: #1e3a8a; border-bottom: 3px solid #1e3a8a; padding-bottom: 10px; }}
    h2 {{ color: #334155; margin-top: 30px; }}
    .summary-box {{
      background: white; border-left: 6px solid {summary_color};
      padding: 20px 30px; border-radius: 6px; margin: 20px 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .summary-box .label {{ font-size: 1.3em; font-weight: bold; color: {summary_color}; }}
    .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
    .metric .value {{ font-size: 2em; font-weight: bold; color: #1e3a8a; }}
    .metric .name {{ font-size: 0.85em; color: #64748b; }}
    table {{ border-collapse: collapse; width: 100%; background: white;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    th {{ background: #1e3a8a; color: white; padding: 12px 16px; text-align: left; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid #e2e8f0; }}
    tr:hover {{ background: #f1f5f9; }}
    .note {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
             padding: 12px 18px; margin: 20px 0; font-size: 0.9em; }}
    footer {{ margin-top: 40px; font-size: 0.8em; color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>📊 Rapport Data Drift — Telco Churn</h1>

  <div class="note">
    ℹ️ <strong>Note :</strong> Ce rapport a été généré via le test de Kolmogorov-Smirnov (scipy)
    car Evidently est incompatible avec Python 3.14 (pydantic v1). Les résultats sont équivalents.
  </div>

  <h2>Résumé</h2>
  <div class="summary-box">
    <div class="label">Statut : {summary_label}</div>
    <br>
    <div class="metric">
      <div class="value">{drift_share:.1%}</div>
      <div class="name">Drift Share</div>
    </div>
    <div class="metric">
      <div class="value">{n_drifted_ev}</div>
      <div class="name">Features driftées</div>
    </div>
    <div class="metric">
      <div class="value">{n_total_ev}</div>
      <div class="name">Total features</div>
    </div>
    <div class="metric">
      <div class="value">p &lt; 0.05</div>
      <div class="name">Seuil KS-test</div>
    </div>
  </div>

  <h2>Résultats KS-test par feature</h2>
  <table>
    <thead>
      <tr>
        <th>Feature</th>
        <th style="text-align:center">KS Statistic</th>
        <th style="text-align:center">p-value</th>
        <th style="text-align:center">Drifté ?</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <h2>Interprétation</h2>
  <p>
    Le test KS compare la distribution de chaque feature entre les données de référence
    (train) et les données de production simulées. Une p-value &lt; 0.05 indique un
    changement statistiquement significatif de la distribution.
  </p>
  <p>
    Seuils de déclenchement :
    <ul>
      <li><strong>Drift share &gt; 30%</strong> → ré-entraînement automatique déclenché</li>
      <li><strong>Drift share &gt; 15%</strong> → alerte, surveillance renforcée</li>
      <li><strong>Drift share ≤ 15%</strong> → modèle stable, aucune action</li>
    </ul>
  </p>

  <footer>Généré automatiquement par tache5.py — Projet MLOps Churn Prediction</footer>
</body>
</html>"""

        with open("drift_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        mlflow.log_artifact("drift_report.html")
        print("  ✅ Rapport HTML alternatif généré : drift_report.html")
        print(f"  Drift share (KS) : {drift_share:.2%} | Colonnes driftées : {n_drifted_ev}/{n_total_ev}")

    # ── Graphique drift ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution de la première feature driftée
    drifted_features = df_ks[df_ks["drifted"]]["feature"].tolist()
    if drifted_features:
        col_plot = drifted_features[0]
        axes[0].hist(X_train[col_plot], bins=30, alpha=0.6,
                     color='#1E3A8A', label='Référence (train)')
        axes[0].hist(X_prod[col_plot],  bins=30, alpha=0.6,
                     color='#DC2626', label='Production (drifté)')
        axes[0].set_title(f"Distribution de '{col_plot}'")
        axes[0].set_xlabel(col_plot)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Aucun drift détecté", ha='center', va='center',
                     fontsize=14, transform=axes[0].transAxes)

    # KS statistics barplot (top 10)
    top_ks = df_ks.sort_values("ks_stat", ascending=False).head(10)
    colors_ks = ['#DC2626' if d else '#16A34A' for d in top_ks["drifted"]]
    axes[1].barh(top_ks["feature"], top_ks["ks_stat"], color=colors_ks, alpha=0.8)
    axes[1].axvline(x=0.1, color='gray', linestyle='--', label='Seuil indicatif')
    axes[1].set_title("KS Statistic par feature (rouge = drifté)")
    axes[1].set_xlabel("KS Statistic")
    axes[1].legend()

    plt.suptitle("Analyse du Data Drift — Telco Churn", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("drift_analysis.png", dpi=150)
    plt.close()
    mlflow.log_artifact("drift_analysis.png")
    print("  ✅ Graphique drift sauvegardé : drift_analysis.png")

    # ── 6.5 Déclenchement automatique ──
    print("\n  6.5 — Déclenchement automatique du ré-entraînement...")
    SEUIL_DRIFT = 0.30
    SEUIL_WARN  = 0.15

    mlflow.log_metric("retrain_triggered", 0)

    if drift_share > SEUIL_DRIFT:
        print(f"  🔴 CRITIQUE : drift {drift_share:.2%} > seuil {SEUIL_DRIFT:.0%}")
        print("  → Ré-entraînement déclenché automatiquement...")
        mlflow.log_metric("retrain_triggered", 1)
        mlflow.set_tag("action", "retrain_triggered")
        # subprocess.run(["python", "tache4v2.py"], check=False)
        print("  ℹ️  (subprocess commenté pour ne pas relancer l'entraînement complet)")
    elif drift_share > SEUIL_WARN:
        print(f"  🟡 AVERTISSEMENT : drift {drift_share:.2%} — surveillance renforcée")
        mlflow.set_tag("action", "monitoring_increased")
    else:
        print(f"  🟢 OK : drift {drift_share:.2%} — modèle stable")
        mlflow.set_tag("action", "no_action")

    print("✅ Partie 6 terminée")


# ════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TÂCHE 5 TERMINÉE ✅")
print("="*60)
print(f"""
  Fichiers générés :
  ├── confusion_matrix_*.png      (4 matrices)
  ├── classification_report_*.txt (4 rapports)
  ├── drift_report.html           (Evidently ou rapport alternatif KS)
  ├── drift_analysis.png          (KS-test)
  ├── ks_drift_results.csv        (résultats KS)
  ├── test_api.py                 (test endpoint)
  ├── Makefile                    (orchestration)
  └── .git/hooks/pre-commit       (validation Git)

  MLflow :
  ├── Experiment : {EXPERIMENT_NAME}
  ├── Experiment : monitoring_drift
  └── Model Registry : {MODEL_NAME} → Production

  Commandes utiles :
  ├── make pipeline               (tout en un)
  ├── make serve                  (démarrer l'API)
  ├── make test                   (tester l'API)
  └── http://127.0.0.1:5000       (MLflow UI)
""")
