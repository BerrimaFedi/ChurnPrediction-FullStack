# Makefile — Pipeline MLOps Churn Prediction
# Usage : make pipeline

.PHONY: setup train tache4 tache5 register serve test drift pipeline clean

# ── Installation des dépendances ──
setup:
	pip install -r requirements.txt
	@echo "✅ Dépendances installées"

# ── Lancer MLflow UI ──
mlflow-ui:
	start /B python -m mlflow ui --port 5000
	@echo "✅ MLflow UI : http://127.0.0.1:5000"

# ── Prétraitement ──
preprocess:
	python preprocessing.py
	@echo "✅ Prétraitement terminé"

# ── Entraînement Tâche 4 V2 (RF + AdaBoost + XGBoost) ──
tache4:
	python tache4v2.py
	@echo "✅ Tâche 4 terminée"

# ── Pipeline MLOps Tâche 5 ──
tache5:
	python tache5.py
	@echo "✅ Tâche 5 terminée"

# ── Servir le modèle en Production ──
serve:
	set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
	mlflow models serve -m "models:/churn_adaboost_production/Production" --port 1234 --no-conda
	@echo "✅ API disponible sur http://localhost:1234"

# ── Test de l'API ──
test:
	python test_api.py
	@echo "✅ Test API terminé"

# ── Détection de drift ──
drift:
	python simulate_drift.py
	@echo "✅ Détection drift terminée"

# ── Pipeline complet ──
pipeline: tache4 tache5 test
	@echo "🚀 Pipeline MLOps complet exécuté avec succès"

# ── Nettoyage des fichiers temporaires ──
clean:
	del /Q *.png *.txt *.csv *.html 2>nul || true
	@echo "✅ Nettoyage terminé"
