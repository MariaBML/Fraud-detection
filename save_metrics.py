# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import json
import os
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, matthews_corrcoef,
    precision_recall_curve, auc as pr_auc,
)

base = os.path.dirname(os.path.abspath(__file__))

X_train, X_test, y_train, y_test = pd.read_pickle(
    os.path.join(base, "data", "processed", "model_ready_data.pkl")
)

models = {}
with open(os.path.join(base, "data", "models", "logistic_regression.pkl"), "rb") as f:
    models["Logistic Regression"] = pickle.load(f)
with open(os.path.join(base, "data", "models", "random_forest.pkl"), "rb") as f:
    models["Random Forest"] = pickle.load(f)
xgb_m = xgb.XGBClassifier()
xgb_m.load_model(os.path.join(base, "data", "models", "xgboost.json"))
models["XGBoost"] = xgb_m

out = {}
for name, model in models.items():
    yp = model.predict_proba(X_test)[:, 1]
    yd = model.predict(X_test)
    pc, rc, _ = precision_recall_curve(y_test, yp)
    out[name] = {
        "AUC-ROC":     round(roc_auc_score(y_test, yp), 4),
        "AUC-PR":      round(pr_auc(rc, pc), 4),
        "F1 (Fraudă)": round(f1_score(y_test, yd, pos_label=1, zero_division=0), 4),
        "Precizie":    round(precision_score(y_test, yd, pos_label=1, zero_division=0), 4),
        "Recall":      round(recall_score(y_test, yd, pos_label=1, zero_division=0), 4),
        "MCC":         round(matthews_corrcoef(y_test, yd), 4),
    }
    print(f"{name}: AUC-ROC={out[name]['AUC-ROC']} AUC-PR={out[name]['AUC-PR']} MCC={out[name]['MCC']}")

with open(os.path.join(base, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Salvat: metrics.json")

# Copiaza xgboost.json din data/models/ in radacina repo (pentru Streamlit Cloud)
import shutil
shutil.copy2(
    os.path.join(base, "data", "models", "xgboost.json"),
    os.path.join(base, "xgboost.json")
)
print("Copiat: xgboost.json -> radacina repo")
