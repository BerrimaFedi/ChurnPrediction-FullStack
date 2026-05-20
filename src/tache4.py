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

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score

from preprocessing import load_and_prepare

# ── MLflow ──
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Churn_Tache4_RandomForest")

DATA_PATH = r"C:\project_ChurnPrediction\data\raw\Telecom Customers Churn.csv"
X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)

# ════════════════════════════════════════════
# QUESTION 1 — FEATURE IMPORTANCE
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q1 — FEATURE IMPORTANCE")
print("="*55)

with mlflow.start_run(run_name="Q1_Feature_Importance"):

    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                random_state=42)
    rf.fit(X_train, y_train)

    # Importance des features
    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    # Top 3
    top3 = importances.head(3)
    print("\nTop 10 features les plus importantes :")
    print(importances.head(10).to_string())
    print(f"\n🏆 Top 3 : {list(top3.index)}")

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1E3A8A' if i < 3 else '#93C5FD'
              for i in range(len(importances.head(15)))]
    importances.head(15).plot(kind='barh', ax=ax, color=colors[::-1])
    ax.set_title("Feature Importance — Random Forest (Top 15)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Importance")
    ax.axvline(x=importances.mean(), color='red',
               linestyle='--', label=f'Moyenne ({importances.mean():.4f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig("q1_feature_importance.png", dpi=150)
    plt.close()

    mlflow.log_metric("top1_importance", importances.iloc[0])
    mlflow.log_metric("top2_importance", importances.iloc[1])
    mlflow.log_metric("top3_importance", importances.iloc[2])
    mlflow.log_artifact("q1_feature_importance.png")
    mlflow.set_tag("top3_features", str(list(top3.index)))

    print("✅ Q1 terminée — graphique sauvegardé")


# ════════════════════════════════════════════
# QUESTION 2 — STABILITÉ (random_state)
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q2 — STABILITÉ DES PRÉDICTIONS")
print("="*55)

stability_results = []

with mlflow.start_run(run_name="Q2_Stability_RandomState"):

    for seed in [0, 1, 7, 21, 42, 99, 123, 200, 777, 1000]:
        rf_s = RandomForestClassifier(n_estimators=100,
                                      max_depth=10,
                                      random_state=seed)
        rf_s.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf_s.predict(X_test))
        f1  = f1_score(y_test, rf_s.predict(X_test), zero_division=0)
        stability_results.append({"seed": seed,
                                   "accuracy": round(acc, 4),
                                   "f1": round(f1, 4)})

    df_stab = pd.DataFrame(stability_results)
    print("\nRésultats stabilité :")
    print(df_stab.to_string(index=False))
    print(f"\nAccuracy — Mean: {df_stab['accuracy'].mean():.4f} | Std: {df_stab['accuracy'].std():.4f}")
    print(f"F1       — Mean: {df_stab['f1'].mean():.4f}       | Std: {df_stab['f1'].std():.4f}")

    # Graphique stabilité
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(df_stab['seed'], df_stab['accuracy'], 'o-', color='#1E3A8A')
    ax1.axhline(df_stab['accuracy'].mean(), color='red',
                linestyle='--', label=f"Moyenne={df_stab['accuracy'].mean():.4f}")
    ax1.fill_between(df_stab['seed'],
                     df_stab['accuracy'].mean() - df_stab['accuracy'].std(),
                     df_stab['accuracy'].mean() + df_stab['accuracy'].std(),
                     alpha=0.2, color='blue')
    ax1.set_title("Accuracy selon random_state")
    ax1.set_xlabel("random_state")
    ax1.legend()

    ax2.plot(df_stab['seed'], df_stab['f1'], 's-', color='#16A34A')
    ax2.axhline(df_stab['f1'].mean(), color='red',
                linestyle='--', label=f"Moyenne={df_stab['f1'].mean():.4f}")
    ax2.fill_between(df_stab['seed'],
                     df_stab['f1'].mean() - df_stab['f1'].std(),
                     df_stab['f1'].mean() + df_stab['f1'].std(),
                     alpha=0.2, color='green')
    ax2.set_title("F1-Score selon random_state")
    ax2.set_xlabel("random_state")
    ax2.legend()

    plt.suptitle("Stabilité du Random Forest", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("q2_stability.png", dpi=150)
    plt.close()

    mlflow.log_metric("accuracy_mean", df_stab['accuracy'].mean())
    mlflow.log_metric("accuracy_std",  df_stab['accuracy'].std())
    mlflow.log_metric("f1_mean",       df_stab['f1'].mean())
    mlflow.log_metric("f1_std",        df_stab['f1'].std())
    mlflow.log_artifact("q2_stability.png")

    print("✅ Q2 terminée")


# ════════════════════════════════════════════
# QUESTION 3 — ANALYSE DES ERREURS
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q3 — ANALYSE DES ERREURS")
print("="*55)

with mlflow.start_run(run_name="Q3_Error_Analysis"):

    rf_main = RandomForestClassifier(n_estimators=100,
                                     max_depth=10, random_state=42)
    rf_main.fit(X_train, y_train)
    y_pred  = rf_main.predict(X_test)
    y_proba = rf_main.predict_proba(X_test)[:, 1]

    # Trouver les erreurs
    X_test_df = X_test.copy()
    X_test_df['y_true']  = y_test.values
    X_test_df['y_pred']  = y_pred
    X_test_df['proba']   = y_proba
    X_test_df['correct'] = (X_test_df['y_true'] == X_test_df['y_pred'])

    errors = X_test_df[~X_test_df['correct']].copy()

    # Faux Positifs (prédit Churn, réalité Stay)
    fp = errors[errors['y_pred'] == 1].sort_values('proba', ascending=False)
    # Faux Négatifs (prédit Stay, réalité Churn)
    fn = errors[errors['y_pred'] == 0].sort_values('proba', ascending=True)

    print(f"\nTotal erreurs : {len(errors)} / {len(X_test)}")
    print(f"  Faux Positifs (FP) : {len(fp)}")
    print(f"  Faux Négatifs (FN) : {len(fn)}")

    # Analyse des features clés sur les erreurs
    key_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    print("\n--- 3 exemples de Faux Positifs (prédit Churn, réalité Stay) ---")
    print(fp[key_cols + ['proba']].head(3).to_string())

    print("\n--- 3 exemples de Faux Négatifs (prédit Stay, réalité Churn) ---")
    print(fn[key_cols + ['proba']].head(3).to_string())

    # Comparaison erreurs vs correct
    print("\n--- Statistiques : Erreurs vs Prédictions correctes ---")
    for col in key_cols:
        err_mean  = errors[col].mean()
        corr_mean = X_test_df[X_test_df['correct']][col].mean()
        print(f"  {col:20s} | Erreurs: {err_mean:.3f} | Corrects: {corr_mean:.3f}")

    # Graphique distribution des probabilités
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(X_test_df[X_test_df['correct']]['proba'],
                 bins=30, alpha=0.7, color='#16A34A', label='Corrects')
    axes[0].hist(errors['proba'], bins=30, alpha=0.7,
                 color='#DC2626', label='Erreurs')
    axes[0].set_title("Distribution des probabilités")
    axes[0].set_xlabel("Probabilité prédite (Churn)")
    axes[0].legend()

    axes[1].scatter(errors[errors['y_pred']==1]['proba'],
                    errors[errors['y_pred']==1]['tenure'],
                    alpha=0.5, color='#F59E0B', label='Faux Positifs')
    axes[1].scatter(errors[errors['y_pred']==0]['proba'],
                    errors[errors['y_pred']==0]['tenure'],
                    alpha=0.5, color='#7C3AED', label='Faux Négatifs')
    axes[1].set_title("Erreurs : Probabilité vs Tenure")
    axes[1].set_xlabel("Probabilité Churn")
    axes[1].set_ylabel("Tenure (standardisé)")
    axes[1].legend()

    plt.suptitle("Analyse des Erreurs de Classification", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("q3_errors.png", dpi=150)
    plt.close()

    mlflow.log_metric("total_errors",      len(errors))
    mlflow.log_metric("false_positives",   len(fp))
    mlflow.log_metric("false_negatives",   len(fn))
    mlflow.log_metric("error_rate",        round(len(errors)/len(X_test), 4))
    mlflow.log_artifact("q3_errors.png")

    print("✅ Q3 terminée")


# ════════════════════════════════════════════
# QUESTION 4 — BIAIS / VARIANCE
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q4 — BIAIS ET VARIANCE")
print("="*55)

bv_results = []

configs = [
    (10,  2),   # underfitting probable
    (10,  5),
    (50,  5),
    (100, 5),
    (100, 10),  # équilibré
    (100, 15),
    (100, None),  # overfitting probable
    (200, 10),
    (200, None),
]

with mlflow.start_run(run_name="Q4_Bias_Variance"):

    for n_est, depth in configs:
        rf_bv = RandomForestClassifier(n_estimators=n_est,
                                       max_depth=depth,
                                       random_state=42)
        rf_bv.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, rf_bv.predict(X_train))
        test_acc  = accuracy_score(y_test,  rf_bv.predict(X_test))
        bias      = round(1 - test_acc, 4)
        variance  = round(train_acc - test_acc, 4)

        bv_results.append({
            "n_estimators": n_est,
            "max_depth":    str(depth) if depth else "None",
            "Train Acc":    round(train_acc, 4),
            "Test Acc":     round(test_acc, 4),
            "Biais":        bias,
            "Variance":     variance,
        })

        depth_str = str(depth) if depth else "None"
        mlflow.log_metric(f"train_acc_n{n_est}_d{depth_str}", train_acc)
        mlflow.log_metric(f"test_acc_n{n_est}_d{depth_str}",  test_acc)

    df_bv = pd.DataFrame(bv_results)
    print("\nTableau Biais / Variance :")
    print(df_bv.to_string(index=False))

    # Graphique Biais/Variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_labels = [f"n={r['n_estimators']}\nd={r['max_depth']}"
                for _, r in df_bv.iterrows()]

    axes[0].plot(range(len(df_bv)), df_bv['Train Acc'], 'o-',
                 color='#1E3A8A', label='Train Accuracy')
    axes[0].plot(range(len(df_bv)), df_bv['Test Acc'], 's-',
                 color='#DC2626', label='Test Accuracy')
    axes[0].set_xticks(range(len(df_bv)))
    axes[0].set_xticklabels(x_labels, fontsize=8)
    axes[0].set_title("Train vs Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(df_bv)), df_bv['Variance'],
                color='#F59E0B', alpha=0.8, label='Variance (Train-Test gap)')
    axes[1].set_xticks(range(len(df_bv)))
    axes[1].set_xticklabels(x_labels, fontsize=8)
    axes[1].set_title("Variance (gap Train-Test)")
    axes[1].set_ylabel("Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Analyse Biais/Variance — Random Forest",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("q4_bias_variance.png", dpi=150)
    plt.close()

    mlflow.log_artifact("q4_bias_variance.png")
    print("✅ Q4 terminée")

# ════════════════════════════════════════════
# QUESTION 5 — RF vs DECISION TREE
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q5 — RF vs DECISION TREE")
print("="*55)

with mlflow.start_run(run_name="Q5_RF_vs_DecisionTree"):

    # Random Forest
    rf_final = RandomForestClassifier(n_estimators=100,
                                      max_depth=10, random_state=42)
    rf_final.fit(X_train, y_train)
    rf_pred  = rf_final.predict(X_test)
    rf_train = accuracy_score(y_train, rf_final.predict(X_train))
    rf_test  = accuracy_score(y_test,  rf_pred)
    rf_f1    = f1_score(y_test, rf_pred, zero_division=0)
    rf_cv    = cross_val_score(rf_final, X_train, y_train, cv=5).mean()

    # Decision Tree (même profondeur)
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred  = dt.predict(X_test)
    dt_train = accuracy_score(y_train, dt.predict(X_train))
    dt_test  = accuracy_score(y_test,  dt_pred)
    dt_f1    = f1_score(y_test, dt_pred, zero_division=0)
    dt_cv    = cross_val_score(dt, X_train, y_train, cv=5).mean()

    comparison = {
        "Métrique":      ["Train Acc", "Test Acc", "F1-Score", "CV Score (5-fold)", "Variance (Train-Test)"],
        "Random Forest": [rf_train, rf_test, rf_f1, rf_cv, rf_train - rf_test],
        "Decision Tree": [dt_train, dt_test, dt_f1, dt_cv, dt_train - dt_test],
    }
    df_comp = pd.DataFrame(comparison)
    df_comp["RF > DT"] = df_comp["Random Forest"] > df_comp["Decision Tree"]
    print("\n" + df_comp.to_string(index=False))

    # ── Graphique : 3 colonnes ──
    metrics = ["Train Acc", "Test Acc", "F1-Score", "CV Score"]
    rf_vals = [rf_train, rf_test, rf_f1, rf_cv]
    dt_vals = [dt_train, dt_test, dt_f1, dt_cv]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ── axes[0] : barplot métriques ──
    bars1 = axes[0].bar(x - width/2, rf_vals, width,
                        label='Random Forest', color='#1E3A8A')
    bars2 = axes[0].bar(x + width/2, dt_vals, width,
                        label='Decision Tree', color='#F59E0B')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("RF vs Decision Tree — Métriques")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)

    # ── axes[1] : Confusion Matrix Random Forest ──
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[1].set_title("Confusion Matrix — Random Forest")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # ── axes[2] : Confusion Matrix Decision Tree ──
    cm_dt = confusion_matrix(y_test, dt_pred)
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[2].set_title("Confusion Matrix — Decision Tree")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    plt.suptitle("Random Forest vs Decision Tree", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("q5_rf_vs_dt.png", dpi=150)
    plt.close()

    mlflow.log_metric("RF_test_acc", rf_test)
    mlflow.log_metric("DT_test_acc", dt_test)
    mlflow.log_metric("RF_f1",       rf_f1)
    mlflow.log_metric("DT_f1",       dt_f1)
    mlflow.log_metric("RF_cv",       rf_cv)
    mlflow.log_metric("DT_cv",       dt_cv)
    mlflow.log_artifact("q5_rf_vs_dt.png")

    print(f"\n🏆 Meilleur Test Acc : {'RF' if rf_test > dt_test else 'DT'}")
    print(f"🏆 Meilleur F1      : {'RF' if rf_f1  > dt_f1  else 'DT'}")
    print(f"🏆 Meilleur CV      : {'RF' if rf_cv  > dt_cv  else 'DT'}")
    print("✅ Q5 terminée")


print("\n" + "="*55)
print("  TÂCHE 4 TERMINÉE ✅")
print("  Voir résultats : http://127.0.0.1:5000")
print("="*55)