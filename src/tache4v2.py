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

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

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

    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    top3 = importances.head(3)
    print("\nTop 10 features les plus importantes :")
    print(importances.head(10).to_string())
    print(f"\n🏆 Top 3 : {list(top3.index)}")

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

    X_test_df = X_test.copy()
    X_test_df['y_true']  = y_test.values
    X_test_df['y_pred']  = y_pred
    X_test_df['proba']   = y_proba
    X_test_df['correct'] = (X_test_df['y_true'] == X_test_df['y_pred'])

    errors = X_test_df[~X_test_df['correct']].copy()
    fp = errors[errors['y_pred'] == 1].sort_values('proba', ascending=False)
    fn = errors[errors['y_pred'] == 0].sort_values('proba', ascending=True)

    print(f"\nTotal erreurs : {len(errors)} / {len(X_test)}")
    print(f"  Faux Positifs (FP) : {len(fp)}")
    print(f"  Faux Négatifs (FN) : {len(fn)}")

    key_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    print("\n--- 3 exemples de Faux Positifs (prédit Churn, réalité Stay) ---")
    print(fp[key_cols + ['proba']].head(3).to_string())

    print("\n--- 3 exemples de Faux Négatifs (prédit Stay, réalité Churn) ---")
    print(fn[key_cols + ['proba']].head(3).to_string())

    print("\n--- Statistiques : Erreurs vs Prédictions correctes ---")
    for col in key_cols:
        err_mean  = errors[col].mean()
        corr_mean = X_test_df[X_test_df['correct']][col].mean()
        print(f"  {col:20s} | Erreurs: {err_mean:.3f} | Corrects: {corr_mean:.3f}")

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

    plt.suptitle("Analyse des Erreurs de Classification",
                 fontsize=13, fontweight='bold')
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
    (10,  2),
    (10,  5),
    (50,  5),
    (100, 5),
    (100, 10),
    (100, 15),
    (100, None),
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

    rf_final = RandomForestClassifier(n_estimators=100,
                                      max_depth=10, random_state=42)
    rf_final.fit(X_train, y_train)
    rf_pred  = rf_final.predict(X_test)
    rf_train = accuracy_score(y_train, rf_final.predict(X_train))
    rf_test  = accuracy_score(y_test,  rf_pred)
    rf_f1    = f1_score(y_test, rf_pred, zero_division=0)
    rf_cv    = cross_val_score(rf_final, X_train, y_train, cv=5).mean()

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

    metrics = ["Train Acc", "Test Acc", "F1-Score", "CV Score"]
    rf_vals = [rf_train, rf_test, rf_f1, rf_cv]
    dt_vals = [dt_train, dt_test, dt_f1, dt_cv]
    x     = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

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

    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[1].set_title("Confusion Matrix — Random Forest")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

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


# ════════════════════════════════════════════
# QUESTION 6 — ADABOOST
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q6 — ADABOOST")
print("="*55)

with mlflow.start_run(run_name="Q6_AdaBoost"):

    ada_results = []

    configs_ada = [
        {"n_estimators": 50,  "learning_rate": 1.0},
        {"n_estimators": 100, "learning_rate": 1.0},
        {"n_estimators": 100, "learning_rate": 0.5},
        {"n_estimators": 200, "learning_rate": 0.5},
        {"n_estimators": 200, "learning_rate": 0.1},
    ]

    for cfg in configs_ada:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            random_state=42
        )
        ada.fit(X_train, y_train)
        ada_pred  = ada.predict(X_test)
        ada_proba = ada.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, ada.predict(X_train))
        test_acc  = accuracy_score(y_test, ada_pred)
        f1        = f1_score(y_test, ada_pred, zero_division=0)
        auc       = roc_auc_score(y_test, ada_proba)
        cv        = cross_val_score(ada, X_train, y_train, cv=5).mean()

        ada_results.append({
            "n_estimators":  cfg["n_estimators"],
            "learning_rate": cfg["learning_rate"],
            "Train Acc":     round(train_acc, 4),
            "Test Acc":      round(test_acc, 4),
            "F1":            round(f1, 4),
            "AUC":           round(auc, 4),
            "CV":            round(cv, 4),
        })

    df_ada = pd.DataFrame(ada_results)
    print("\nRésultats AdaBoost :")
    print(df_ada.to_string(index=False))

    # Meilleure config par F1
    best_ada_row = df_ada.loc[df_ada['F1'].idxmax()]
    best_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=int(best_ada_row['n_estimators']),
        learning_rate=best_ada_row['learning_rate'],
        random_state=42
    )
    best_ada.fit(X_train, y_train)
    best_ada_pred  = best_ada.predict(X_test)
    best_ada_proba = best_ada.predict_proba(X_test)[:, 1]

    print(f"\n🏆 Meilleure config AdaBoost : n={int(best_ada_row['n_estimators'])}, lr={best_ada_row['learning_rate']}")
    print(f"   Test Acc={best_ada_row['Test Acc']} | F1={best_ada_row['F1']} | AUC={best_ada_row['AUC']}")

    # ── Graphique AdaBoost ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Courbe d'apprentissage (staging scores)
    staged_scores = [accuracy_score(y_test, pred)
                     for pred in best_ada.staged_predict(X_test)]
    axes[0].plot(range(1, len(staged_scores)+1), staged_scores,
                 color='#DC2626', lw=2)
    axes[0].set_title(f"Courbe d'apprentissage AdaBoost\n(n={int(best_ada_row['n_estimators'])}, lr={best_ada_row['learning_rate']})")
    axes[0].set_xlabel("Nombre d'estimateurs")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=max(staged_scores), color='gray',
                    linestyle='--', label=f"Max={max(staged_scores):.4f}")
    axes[0].legend()

    # Confusion Matrix
    cm_ada = confusion_matrix(y_test, best_ada_pred)
    sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[1].set_title("Confusion Matrix — AdaBoost")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # Comparaison des configs
    x_labels = [f"n={r['n_estimators']}\nlr={r['learning_rate']}"
                for _, r in df_ada.iterrows()]
    x = np.arange(len(df_ada))
    width = 0.35
    axes[2].bar(x - width/2, df_ada['Test Acc'], width,
                label='Test Accuracy', color='#DC2626', alpha=0.8)
    axes[2].bar(x + width/2, df_ada['F1'], width,
                label='F1-Score', color='#7C3AED', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels, fontsize=8)
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("Comparaison des configs AdaBoost")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("AdaBoost — Analyse complète", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("q6_adaboost.png", dpi=150)
    plt.close()

    # Log MLflow
    mlflow.log_param("best_n_estimators",  int(best_ada_row['n_estimators']))
    mlflow.log_param("best_learning_rate", best_ada_row['learning_rate'])
    mlflow.log_metric("ada_test_acc", best_ada_row['Test Acc'])
    mlflow.log_metric("ada_f1",       best_ada_row['F1'])
    mlflow.log_metric("ada_auc",      best_ada_row['AUC'])
    mlflow.log_metric("ada_cv",       best_ada_row['CV'])
    mlflow.log_artifact("q6_adaboost.png")

    print("✅ Q6 AdaBoost terminée")


# ════════════════════════════════════════════
# QUESTION 7 — XGBOOST
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q7 — XGBOOST")
print("="*55)

with mlflow.start_run(run_name="Q7_XGBoost"):

    xgb_results = []

    configs_xgb = [
        {"n_estimators": 100, "max_depth": 3,  "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 100, "max_depth": 5,  "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 100, "max_depth": 3,  "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3,  "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 5,  "learning_rate": 0.01, "subsample": 0.8},
    ]

    for cfg in configs_xgb:
        xgb = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        xgb_pred  = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, xgb.predict(X_train))
        test_acc  = accuracy_score(y_test, xgb_pred)
        f1        = f1_score(y_test, xgb_pred, zero_division=0)
        auc       = roc_auc_score(y_test, xgb_proba)
        cv        = cross_val_score(xgb, X_train, y_train, cv=5).mean()

        xgb_results.append({
            "n_estimators":  cfg["n_estimators"],
            "max_depth":     cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "subsample":     cfg["subsample"],
            "Train Acc":     round(train_acc, 4),
            "Test Acc":      round(test_acc, 4),
            "F1":            round(f1, 4),
            "AUC":           round(auc, 4),
            "CV":            round(cv, 4),
        })

    df_xgb = pd.DataFrame(xgb_results)
    print("\nRésultats XGBoost :")
    print(df_xgb.to_string(index=False))

    # Meilleure config par F1
    best_xgb_row = df_xgb.loc[df_xgb['F1'].idxmax()]
    best_xgb = XGBClassifier(
        n_estimators=int(best_xgb_row['n_estimators']),
        max_depth=int(best_xgb_row['max_depth']),
        learning_rate=best_xgb_row['learning_rate'],
        subsample=best_xgb_row['subsample'],
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    best_xgb.fit(X_train, y_train)
    best_xgb_pred  = best_xgb.predict(X_test)
    best_xgb_proba = best_xgb.predict_proba(X_test)[:, 1]

    print(f"\n🏆 Meilleure config XGBoost : n={int(best_xgb_row['n_estimators'])}, depth={int(best_xgb_row['max_depth'])}, lr={best_xgb_row['learning_rate']}")
    print(f"   Test Acc={best_xgb_row['Test Acc']} | F1={best_xgb_row['F1']} | AUC={best_xgb_row['AUC']}")

    # Feature importance XGBoost
    xgb_importances = pd.Series(best_xgb.feature_importances_,
                                 index=X_train.columns).sort_values(ascending=False)

    # ── Graphique XGBoost ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Feature importance Top 10
    xgb_importances.head(10).plot(kind='barh', ax=axes[0],
                                   color='#16A34A')
    axes[0].set_title("Feature Importance — XGBoost (Top 10)")
    axes[0].set_xlabel("Importance")
    axes[0].invert_yaxis()

    # Confusion Matrix
    cm_xgb = confusion_matrix(y_test, best_xgb_pred)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[1].set_title("Confusion Matrix — XGBoost")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # Courbe ROC XGBoost vs RF vs AdaBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, best_xgb_proba)
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_final.predict_proba(X_test)[:, 1])
    fpr_ada, tpr_ada, _ = roc_curve(y_test, best_ada_proba)

    auc_xgb = roc_auc_score(y_test, best_xgb_proba)
    auc_rf  = roc_auc_score(y_test, rf_final.predict_proba(X_test)[:, 1])
    auc_ada = roc_auc_score(y_test, best_ada_proba)

    axes[2].plot(fpr_xgb, tpr_xgb, color='#16A34A', lw=2,
                 label=f"XGBoost  AUC={auc_xgb:.4f}")
    axes[2].plot(fpr_rf,  tpr_rf,  color='#1E3A8A', lw=2,
                 label=f"RF       AUC={auc_rf:.4f}")
    axes[2].plot(fpr_ada, tpr_ada, color='#DC2626', lw=2,
                 label=f"AdaBoost AUC={auc_ada:.4f}")
    axes[2].plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("Courbe ROC — XGBoost vs RF vs AdaBoost")
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("XGBoost — Analyse complète", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("q7_xgboost.png", dpi=150)
    plt.close()

    # Log MLflow
    mlflow.log_param("best_n_estimators",  int(best_xgb_row['n_estimators']))
    mlflow.log_param("best_max_depth",     int(best_xgb_row['max_depth']))
    mlflow.log_param("best_learning_rate", best_xgb_row['learning_rate'])
    mlflow.log_param("best_subsample",     best_xgb_row['subsample'])
    mlflow.log_metric("xgb_test_acc", best_xgb_row['Test Acc'])
    mlflow.log_metric("xgb_f1",       best_xgb_row['F1'])
    mlflow.log_metric("xgb_auc",      best_xgb_row['AUC'])
    mlflow.log_metric("xgb_cv",       best_xgb_row['CV'])
    mlflow.log_artifact("q7_xgboost.png")

    print("✅ Q7 XGBoost terminée")


# ════════════════════════════════════════════
# QUESTION 8 — COMPARAISON FINALE
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  Q8 — COMPARAISON FINALE : RF vs DT vs AdaBoost vs XGBoost")
print("="*55)

with mlflow.start_run(run_name="Q8_Final_Comparison"):

    # Recalcul propre de tous les meilleurs modèles
    models = {
        "Random Forest": (rf_final,   rf_final.predict_proba(X_test)[:, 1]),
        "Decision Tree": (dt,          None),
        "AdaBoost":      (best_ada,    best_ada_proba),
        "XGBoost":       (best_xgb,    best_xgb_proba),
    }

    final_results = []
    for name, (model, proba) in models.items():
        pred      = model.predict(X_test)
        test_acc  = accuracy_score(y_test, pred)
        f1        = f1_score(y_test, pred, zero_division=0)
        auc       = roc_auc_score(y_test, proba) if proba is not None else 0.0
        cv        = cross_val_score(model, X_train, y_train, cv=5).mean()
        train_acc = accuracy_score(y_train, model.predict(X_train))

        final_results.append({
            "Modèle":     name,
            "Train Acc":  round(train_acc, 4),
            "Test Acc":   round(test_acc, 4),
            "F1-Score":   round(f1, 4),
            "AUC":        round(auc, 4),
            "CV (5-fold)":round(cv, 4),
            "Variance":   round(train_acc - test_acc, 4),
        })

    df_final = pd.DataFrame(final_results).sort_values("F1-Score", ascending=False)
    df_final.reset_index(drop=True, inplace=True)

    print("\n" + "="*75)
    print("   TABLEAU COMPARATIF FINAL — TOUS LES MODÈLES")
    print("="*75)
    print(df_final.to_string(index=False))
    print("="*75)

    best = df_final.iloc[0]
    print(f"\n🏆 Meilleur modèle global : {best['Modèle']}")
    print(f"   F1={best['F1-Score']} | AUC={best['AUC']} | Test Acc={best['Test Acc']}")

    # ── Graphique comparaison finale ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    model_names = df_final['Modèle']
    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#1E3A8A', '#DC2626', '#16A34A', '#F59E0B']

    metrics_to_plot = ['Test Acc', 'F1-Score', 'AUC', 'CV (5-fold)']
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        axes[0].bar(x + i*width, df_final[metric], width,
                    label=metric, color=color, alpha=0.85)

    axes[0].set_xticks(x + width*1.5)
    axes[0].set_xticklabels(model_names, fontsize=10)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_title("Comparaison des métriques — Tous modèles")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Radar-style : variance
    axes[1].barh(df_final['Modèle'], df_final['Variance'],
                 color=['#1E3A8A', '#DC2626', '#16A34A', '#F59E0B'][:len(df_final)],
                 alpha=0.8)
    axes[1].set_xlabel("Variance (Train - Test)")
    axes[1].set_title("Overfitting par modèle (plus bas = mieux)")
    axes[1].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(df_final['Variance']):
        axes[1].text(v + 0.002, i, f"{v:.4f}", va='center', fontsize=9)

    plt.suptitle("Comparaison Finale : RF vs DT vs AdaBoost vs XGBoost",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("q8_final_comparison.png", dpi=150)
    plt.close()

    # Log MLflow
    for _, row in df_final.iterrows():
        prefix = row['Modèle'].lower().replace(' ', '_')
        mlflow.log_metric(f"{prefix}_test_acc", row['Test Acc'])
        mlflow.log_metric(f"{prefix}_f1",       row['F1-Score'])
        mlflow.log_metric(f"{prefix}_auc",      row['AUC'])

    mlflow.set_tag("best_model", best['Modèle'])
    mlflow.log_artifact("q8_final_comparison.png")

    print("✅ Q8 Comparaison finale terminée")


print("\n" + "="*55)
print("  TÂCHE 4 V2 TERMINÉE ✅")
print("  Voir résultats : http://127.0.0.1:5000")
print("="*55)
