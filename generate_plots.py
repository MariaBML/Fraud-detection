"""
Generare grafice comparative pentru analiza fraudei bancare.
Necesita setul complet de date (data/processed/model_ready_data.pkl).
Figurile sunt salvate in figures/ la rezolutie 300 DPI.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import shap
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURARE DIRECTOARE
# ─────────────────────────────────────────────────────────────────
base_dir    = os.path.dirname(os.path.abspath(__file__))
data_path   = os.path.join(base_dir, "data", "processed", "model_ready_data.pkl")
models_dir  = os.path.join(base_dir, "data", "models")
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# INCARCARE DATE SI MODELE
# ─────────────────────────────────────────────────────────────────
print("1. Incarcare date de testare...")
_, X_test, _, y_test = pd.read_pickle(data_path)

print("2. Incarcare modele antrenate...")
models = {}

with open(os.path.join(models_dir, "logistic_regression.pkl"), "rb") as f:
    models["Logistic Regression"] = pickle.load(f)

with open(os.path.join(models_dir, "random_forest.pkl"), "rb") as f:
    models["Random Forest"] = pickle.load(f)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(models_dir, "xgboost.json"))
models["XGBoost"] = xgb_model

COLORS = {
    "Logistic Regression": "#D97706",
    "Random Forest":       "#1E6FD9",
    "XGBoost":             "#059669",
}
MARKERS = {
    "Logistic Regression": "o",
    "Random Forest":       "s",
    "XGBoost":             "^",
}

# ─────────────────────────────────────────────────────────────────
# GRAFICUL 1 — Curba ROC
# ─────────────────────────────────────────────────────────────────
print("\n[Grafic 1] Curba ROC comparativa...")
plt.figure(figsize=(9, 7))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2.2, color=COLORS[name],
             label=f"{name}  (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "--", color="#CBD5E1", lw=1.5, label="Esantionare aleatoare")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel("Rata de Alarme False (False Positive Rate)", fontsize=12)
plt.ylabel("Rata de Detectie (True Positive Rate)",     fontsize=12)
plt.title("Compararea Performantei Modelelor (Curba ROC)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "ROC_Curve_Comparison.png"), dpi=300)
plt.close()
print("   -> Salvat: ROC_Curve_Comparison.png")

# ─────────────────────────────────────────────────────────────────
# GRAFICUL 2 — Matrici de Confuzie
# ─────────────────────────────────────────────────────────────────
print("[Grafic 2] Matrici de confuzie...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                cbar=False, annot_kws={"size": 14})
    axes[idx].set_title(name, fontsize=14, fontweight="bold", pad=12)
    axes[idx].set_xlabel("Predictia Modelului", fontsize=11)
    axes[idx].set_ylabel("Eticheta Reala",       fontsize=11)
    axes[idx].set_xticklabels(["Legitima (0)", "Frauda (1)"], fontsize=10)
    axes[idx].set_yticklabels(["Legitima (0)", "Frauda (1)"], fontsize=10, rotation=0)

fig.suptitle("Matrici de Confuzie — Comparatie Modele", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "Confusion_Matrices.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   -> Salvat: Confusion_Matrices.png")

# ─────────────────────────────────────────────────────────────────
# GRAFICUL 3 — SHAP Summary Plot (Random Forest)
# ─────────────────────────────────────────────────────────────────
print("[Grafic 3] SHAP Summary Plot (Random Forest, esantion 2000)...")
rf_model = models["Random Forest"]
X_sample = X_test.sample(2000, random_state=42)

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

# sklearn RandomForest returneaza lista [clasa0, clasa1]; luam clasa Frauda (1)
shap_fraud = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_fraud, X_sample, show=False, max_display=20,
                  plot_type="dot", alpha=0.7)
plt.title("Impactul Global al Variabilelor asupra Probabilitatii de Frauda (SHAP)",
          fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "SHAP_Summary_Plot.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   -> Salvat: SHAP_Summary_Plot.png")

# ─────────────────────────────────────────────────────────────────
# GRAFICUL 4 — Curba Precision-Recall
# Pe date dezechilibrate (frauda ~3.5%), curba PR e mai informativa decat ROC
# (Davis & Goadrich, ICML 2006).
# ─────────────────────────────────────────────────────────────────
print("[Grafic 4] Curba Precision-Recall comparativa...")
fig, ax = plt.subplots(figsize=(9, 7))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
    aucpr = auc(rec, prec)

    ax.plot(rec, prec, lw=2.2, color=COLORS[name],
            label=f"{name}  (AUC-PR = {aucpr:.3f})")

# Baseline: clasificator aleator = rata de frauda in setul de test
baseline = y_test.mean()
ax.axhline(y=baseline, color="#94A3B8", lw=1.5, linestyle="--",
           label=f"Baseline aleator ({baseline:.3f})")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("Recall (Rata de Detectie a Fraudei)",    fontsize=12)
ax.set_ylabel("Precizie (Acuratete Alarme Pozitive)",   fontsize=12)
ax.set_title("Curba Precision-Recall — Comparatie Modele\n"
             "(Metrica recomandata pentru date dezechilibrate)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax.grid(alpha=0.25)

# Referinta metodologica — plasata deasupra axei X, nu peste curbe
fig.text(0.5, -0.04,
         "Ref: Davis & Goadrich, ICML 2006; He & Garcia, IEEE TKDE 2009",
         ha="center", fontsize=8, color="#94A3B8", style="italic")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "PrecisionRecall_Comparison.png"), dpi=300)
plt.close()
print("   -> Salvat: PrecisionRecall_Comparison.png")

print(f"\nToate graficele au fost salvate in: {figures_dir}")
