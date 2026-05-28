"""
Pipeline de antrenare si evaluare — Logistic Regression, Random Forest, XGBoost
IEEE-CIS Fraud Detection Dataset
"""
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, auc as pr_auc, matthews_corrcoef,
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

    # Calculam scale_pos_weight pentru XGBoost
    # Raportul neg/pos reflecta dezechilibrul real (~28:1 pentru IEEE-CIS)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / pos, 2)
    print(f"\nDistributie clase (train): {neg} negative / {pos} pozitive => scale_pos_weight = {spw}")

    # Esantionare 20% din antrenare (optimizare RAM pentru PoC local)
    # In mediul de productie se antreaza pe intregul set.
    sample_size = int(len(X_train) * 0.20)
    X_sub = X_train.iloc[:sample_size]
    y_sub = y_train.iloc[:sample_size]

    # Recalculam spw pe esantion
    neg_s = int((y_sub == 0).sum())
    pos_s = int((y_sub == 1).sum())
    spw_s = round(neg_s / pos_s, 2) if pos_s > 0 else spw
    print(f"Esantion antrenare (20%): {neg_s} negative / {pos_s} pozitive => scale_pos_weight = {spw_s}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, n_jobs=-1,
            class_weight="balanced",   # echivalent cu scale_pos_weight pentru LogReg
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1,
            class_weight="balanced",   # corecteaza dezechilibrul de clasa
            random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, eval_metric="logloss",
            scale_pos_weight=spw_s,    # FIX: parametru pentru dezechilibru clasa
        ),
    }

    train_and_save(models, X_sub, y_sub, X_test, y_test, models_dir)
    print("\nToate modelele au fost antrenate si salvate cu succes!")
