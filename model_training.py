"""
Pipeline de antrenare si evaluare — Logistic Regression, Random Forest, XGBoost
IEEE-CIS Fraud Detection Dataset
"""
import pandas as pd
import numpy as np
import os
import pickle
import json
import shutil
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    precision_recall_curve, auc as pr_auc, matthews_corrcoef,
)
warnings.filterwarnings("ignore")


def load_ready_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "data", "processed", "model_ready_data.pkl")
    print(f"Incarcare matrice de antrenare/testare din: {input_path}")
    return pd.read_pickle(input_path), base_dir


def evaluate_model(name, model, X_test, y_test):
    """Evalueaza un model antrenat si tipareste raportul complet de metrici."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    prec_v, rec_v, _ = precision_recall_curve(y_test, y_proba)
    auc_pr  = pr_auc(rec_v, prec_v)
    mcc     = matthews_corrcoef(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"REZULTATE — {name}")
    print(f"{'='*55}")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"  AUC-PR   : {auc_pr:.4f}   (mai relevant pe date dezechilibrate)")
    print(f"  MCC      : {mcc:.4f}      (Coef. Matthews — robust la dezechilibru)")
    print(f"\nRaport de clasificare detaliat:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitima (0)", "Frauda (1)"],
                                digits=4))
    print("Matrice de confuzie:")
    print(confusion_matrix(y_test, y_pred))

    return {"AUC-ROC": auc_roc, "AUC-PR": auc_pr, "MCC": mcc}


def train_and_save(models_dict, X_train, y_train, X_test, y_test, models_dir):
    summary = {}
    for name, model in models_dict.items():
        print(f"\n--- Antrenare: {name} ---")
        model.fit(X_train, y_train)
        summary[name] = evaluate_model(name, model, X_test, y_test)

        # Persistenta model
        if name == "XGBoost" and hasattr(model, "save_model"):
            path = os.path.join(models_dir, "xgboost.json")
            model.save_model(path)
        else:
            fname = name.replace(" ", "_").lower() + ".pkl"
            path  = os.path.join(models_dir, fname)
            with open(path, "wb") as f:
                pickle.dump(model, f)
        print(f"  -> Salvat la: {path}")

    # Tabel sumar comparativ
    print(f"\n{'='*55}")
    print("SUMAR COMPARATIV MODELE")
    print(f"{'='*55}")
    print(f"{'Model':<22} {'AUC-ROC':>9} {'AUC-PR':>9} {'MCC':>9}")
    print("-" * 55)
    for nm, vals in summary.items():
        print(f"{nm:<22} {vals['AUC-ROC']:>9.4f} {vals['AUC-PR']:>9.4f} {vals['MCC']:>9.4f}")

    return summary


if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), base_dir = load_ready_data()
    models_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Calculam scale_pos_weight pentru XGBoost din setul complet de antrenare
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / pos, 2)
    print(f"\nDistributie clase (train): {neg} negative / {pos} pozitive => scale_pos_weight = {spw}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, n_jobs=-1,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, eval_metric="logloss",
            scale_pos_weight=spw,
        ),
    }

    summary = train_and_save(models, X_train, y_train, X_test, y_test, models_dir)
    print("\nToate modelele au fost antrenate si salvate cu succes!")

    # Salveaza metrics.json in radacina repo (folosit de Streamlit Cloud ca fallback)
    metrics_full = {}
    for name, model in models.items():
        yp = model.predict_proba(X_test)[:, 1]
        yd = model.predict(X_test)
        pc, rc, _ = precision_recall_curve(y_test, yp)
        metrics_full[name] = {
            "AUC-ROC":      round(roc_auc_score(y_test, yp), 4),
            "AUC-PR":       round(pr_auc(rc, pc), 4),
            "F1 (Fraudă)":  round(f1_score(y_test, yd, pos_label=1, zero_division=0), 4),
            "Precizie":     round(precision_score(y_test, yd, pos_label=1, zero_division=0), 4),
            "Recall":       round(recall_score(y_test, yd, pos_label=1, zero_division=0), 4),
            "MCC":          round(matthews_corrcoef(y_test, yd), 4),
        }
    metrics_path = os.path.join(base_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, indent=2, ensure_ascii=False)
    print(f"Metrici salvate: {metrics_path}")

    # Copiaza xgboost.json in radacina repo pentru Streamlit Cloud
    src = os.path.join(models_dir, "xgboost.json")
    dst = os.path.join(base_dir, "xgboost.json")
    shutil.copy2(src, dst)
    print(f"xgboost.json copiat in radacina repo.")
