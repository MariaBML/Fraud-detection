"""
FinTech Fraud Detection — Demonstrator XAI
Balaban Maria-Laura, 2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, random, pickle, warnings
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, auc as pr_auc,
    roc_curve, matthews_corrcoef,
)
warnings.filterwarnings("ignore")

# NumPy compatibility patch (Streamlit Cloud)
if not hasattr(np, "int"):
    np.int = int

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinTech Fraud XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* App background */
.stApp { background: #F1F5F9; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0; max-width: 1180px; }

/* ── HEADER ── */
.app-header {
    background: linear-gradient(100deg, #0D1B2A 0%, #1E3A5F 60%, #0D1B2A 100%);
    padding: 22px 32px 18px;
    border-radius: 0 0 16px 16px;
    margin-bottom: 22px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.header-title { color: #F1F5F9; font-size: 22px; font-weight: 700; margin: 0 0 4px 0; }
.header-sub  { color: #94AABF; font-size: 12px; margin: 0; }
.badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
.badge {
    background: rgba(30,111,217,.14); border: 1px solid #1E6FD9;
    color: #60A5FA; font-size: 10px; font-weight: 700;
    letter-spacing: .8px; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
}
.badge-g { background: rgba(5,150,105,.14); border-color: #059669; color: #34D399; }
.badge-r { background: rgba(220,38,38,.14);  border-color: #DC2626; color: #F87171; }

/* ── KPI CARDS ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.kpi-card {
    background: #fff; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 18px 16px; position: relative; overflow: hidden;
}
.kpi-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1E6FD9, #06B6D4);
}
.kpi-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px; color: #64748B; font-weight: 600; margin-bottom: 8px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #0D1B2A; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #94A3B8; margin-top: 5px; }
.kpi-accent { color: #1E6FD9; }
.kpi-red    { color: #DC2626; }
.kpi-green  { color: #059669; }

/* ── SECTION TITLE ── */
.sec-title {
    font-size: 13px; font-weight: 700; color: #0D1B2A; text-transform: uppercase;
    letter-spacing: 1px; padding-bottom: 8px;
    border-bottom: 2px solid #E2E8F0; margin-bottom: 14px; margin-top: 6px;
}

/* ── COMPARISON TABLE ── */
.cmp-table { width:100%; border-collapse: collapse; font-size: 13px; }
.cmp-table th {
    background: #0D1B2A; color: #94AABF; padding: 10px 14px; text-align: center;
    font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
}
.cmp-table th:first-child { text-align: left; }
.cmp-table td { padding: 10px 14px; border-bottom: 1px solid #F1F5F9; color: #334155; text-align: center; }
.cmp-table td:first-child { text-align: left; font-weight: 600; color: #0D1B2A; }
.cmp-table tr:hover td { background: #F8FAFC; }
.cmp-table .best { color: #059669; font-weight: 700; }
.cmp-table .worst { color: #94A3B8; }

/* ── TRANSACTION TABLE ── */
.tx-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.tx-table td { padding: 7px 10px; border-bottom: 1px solid #F1F5F9; vertical-align: middle; }
.tx-table td:first-child { color: #64748B; font-size: 11px; width: 55%; }
.tx-table td:last-child  { color: #0D1B2A; font-weight: 600; text-align: right; }

/* ── DECISION BOXES ── */
.dec-legit {
    background: linear-gradient(135deg,rgba(5,150,105,.07),rgba(5,150,105,.02));
    border: 2px solid #059669; border-radius: 14px; padding: 22px; text-align: center;
}
.dec-fraud {
    background: linear-gradient(135deg,rgba(220,38,38,.07),rgba(220,38,38,.02));
    border: 2px solid #DC2626; border-radius: 14px; padding: 22px; text-align: center;
}
.dec-icon  { font-size: 38px; margin-bottom: 6px; }
.dec-lbl   { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748B; }
.dec-txt   { font-size: 18px; font-weight: 700; margin: 6px 0 4px; }
.dec-real  { font-size: 12px; color: #64748B; margin-top: 8px; }

/* ── SHAP HEADER ── */
.shap-box {
    background: #fff; border: 1px solid #E2E8F0; border-left: 4px solid #1E6FD9;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
}
.shap-box h4 { margin: 0 0 4px; color: #0D1B2A; font-size: 14px; }
.shap-box p  { margin: 0; color: #64748B; font-size: 12px; line-height: 1.6; }

/* ── EU BANNER ── */
.eu-banner {
    background: #fff; border: 1px solid #BFDBFE; border-left: 4px solid #1E6FD9;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 18px;
    font-size: 12px; color: #334155; line-height: 1.7;
}
.eu-banner strong { color: #1E3A5F; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff; border-radius: 12px; padding: 5px; gap: 4px;
    border: 1px solid #E2E8F0; box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.stTabs [data-baseweb="tab"] {
    color: #64748B; border-radius: 9px; font-weight: 600; font-size: 13px; padding: 9px 20px;
}
.stTabs [aria-selected="true"] { background: #0D1B2A !important; color: #fff !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 18px; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #1E6FD9, #0284C7);
    color: #fff; border: none; border-radius: 10px;
    font-weight: 600; font-size: 14px; padding: 12px 24px;
    width: 100%; transition: all .2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1A5FC4, #0369A1);
    box-shadow: 0 4px 14px rgba(30,111,217,.35);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* Spinner */
.stSpinner > div { border-top-color: #1E6FD9 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
COL_LABELS = {
    "TransactionID": "ID Tranzacție", "TransactionDT": "Timestamp (sec)",
    "TransactionAmt": "Sumă ($)", "ProductCD": "Cod Produs",
    "card1": "Card Feature 1", "card2": "Card Feature 2",
    "card3": "Card Feature 3", "card4": "Card Feature 4",
    "card5": "Card Feature 5", "card6": "Card Feature 6",
    "addr1": "Cod Poștal (cumpărător)", "addr2": "Cod Poștal (destinatar)",
    "dist1": "Distanță (mi)", "P_emaildomain": "Email cumpărător",
    "R_emaildomain": "Email destinatar", "C1": "Asocieri card (C1)",
    "C2": "Asocieri (C2)", "D1": "Zile de la ultima tranzacție",
    "M1": "Match (M1)", "M2": "Match (M2)",
}
PRIORITY = ["TransactionAmt", "ProductCD", "card1", "card2", "card3",
            "card4", "card5", "card6", "P_emaildomain", "addr1",
            "dist1", "C1", "C2", "D1"]


def tx_table_html(row):
    cols = [c for c in PRIORITY if c in row.columns][:14]
    rows_html = ""
    for c in cols:
        val = row[c].values[0]
        val_str = f"{val:,.4g}" if isinstance(val, (float, np.floating)) else str(val)
        rows_html += f"<tr><td>{COL_LABELS.get(c, c)}</td><td>{val_str}</td></tr>"
    extra = row.shape[1] - len(cols)
    rows_html += (f"<tr><td colspan='2' style='color:#CBD5E1;font-style:italic;font-size:11px;'>"
                  f"... și {extra} variabile anonimizate (V1–V339)</td></tr>")
    return f"<table class='tx-table'>{rows_html}</table>"


def make_gauge(prob: float) -> plt.Figure:
    """Semi-circle risk gauge, neutral white background (matplotlib/SHAP safe)."""
    fig, ax = plt.subplots(figsize=(4.4, 2.7), facecolor="#FFFFFF")
    r_o, r_i = 1.0, 0.60
    for (s, e, col) in [(0.0, 0.3, "#059669"), (0.3, 0.6, "#D97706"), (0.6, 1.0, "#DC2626")]:
        t = np.linspace(np.pi * (1 - s), np.pi * (1 - e), 120)
        xs = np.concatenate([r_o * np.cos(t), r_i * np.cos(t[::-1])])
        ys = np.concatenate([r_o * np.sin(t), r_i * np.sin(t[::-1])])
        ax.fill(xs, ys, color=col, alpha=0.88)
    angle = np.pi * (1 - prob)
    ax.annotate("", xy=(0.82 * np.cos(angle), 0.82 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#0D1B2A", lw=2.5, mutation_scale=14))
    ax.plot(0, 0, "o", color="#0D1B2A", markersize=9, zorder=10)
    col_txt = "#059669" if prob < 0.3 else ("#D97706" if prob < 0.6 else "#DC2626")
    ax.text(0, -0.22, f"{prob * 100:.1f}%", ha="center", fontsize=22,
            fontweight="bold", color=col_txt)
    ax.text(0, -0.46, "probabilitate de fraudă", ha="center", fontsize=9, color="#64748B")
    ax.text(-1.1, 0.06, "0%",    ha="right", color="#059669", fontsize=8, fontweight="700")
    ax.text(1.1,  0.06, "100%",  ha="left",  color="#DC2626", fontsize=8, fontweight="700")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.65, 1.12)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# LOADING (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.abspath(__file__))

    # XGBoost (obligatoriu)
    xgb_path = os.path.join(base, "xgboost.json")
    if not os.path.exists(xgb_path):
        st.error(f"'xgboost.json' lipsește din {base}"); st.stop()
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    # Date test
    for fname, note in [("model_ready_data.pkl", "eșantion de evaluare"),
                         ("model_ready_data_deploy.pkl", "eșantion de evaluare")]:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            _, X_test, _, y_test = pd.read_pickle(p)
            return xgb_model, X_test, y_test, note, base

    st.error("Nu s-a găsit niciun fișier de date. Rulați feature_engineering.py."); st.stop()


@st.cache_resource
def get_explainer(_model):
    """Cache SHAP TreeExplainer — calculat o singură dată la pornire."""
    return shap.TreeExplainer(_model)


@st.cache_data
def load_sklearn_models(base):
    models = {}
    for name, fname in [("Logistic Regression", "logistic_regression.pkl"),
                         ("Random Forest",       "random_forest.pkl")]:
        p = os.path.join(base, "data", "models", fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                models[name] = pickle.load(f)
    return models


@st.cache_data
def compute_metrics(_xgb, _X, _y, _sk):
    """Calculează AUC-ROC, AUC-PR, F1, Precizie, Recall, MCC pentru toate modelele disponibile."""
    all_m = {**_sk, "XGBoost": _xgb}
    out = {}
    for name, model in all_m.items():
        yp = model.predict_proba(_X)[:, 1]
        yd = model.predict(_X)
        prec_c, rec_c, _ = precision_recall_curve(_y, yp)
        out[name] = {
            "AUC-ROC":      round(roc_auc_score(_y, yp), 4),
            "AUC-PR":       round(pr_auc(rec_c, prec_c), 4),
            "F1 (Fraudă)":  round(f1_score(_y, yd, pos_label=1, zero_division=0), 4),
            "Precizie":     round(precision_score(_y, yd, pos_label=1, zero_division=0), 4),
            "Recall":       round(recall_score(_y, yd, pos_label=1, zero_division=0), 4),
            "MCC":          round(matthews_corrcoef(_y, yd), 4),
        }
    return out


# ─────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────
xgb_model, X_test, y_test, data_note, BASE = load_assets()
sk_models = load_sklearn_models(BASE)
explainer  = get_explainer(xgb_model)

if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = None

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <div>
    <p class="header-title">🛡️ FinTech Fraud Detection — Demonstrator XAI</p>
    <p class="header-sub">
      Disertație · Știința Datelor și Inteligență Artificială ·
      Universitatea „Titu Maiorescu" București · 2026
    </p>
    <div class="badge-row">
      <span class="badge">EU AI Act</span>
      <span class="badge">XGBoost + SHAP</span>
      <span class="badge g">IEEE-CIS Dataset</span>
      <span class="badge r">High-Risk AI System</span>
    </div>
  </div>
  <div style="text-align:right;color:#94AABF;font-size:11px;align-self:flex-end;">
    {len(X_test):,} tranzacții · {data_note}<br>
    {X_test.shape[1]} variabile · 3 algoritmi comparați
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Dashboard & Rezultate",
    "🔍  Simulare Tranzacție",
    "🧠  Raport XAI Global",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab1:
    n_total = len(y_test)
    n_fraud = int(y_test.sum())
    fraud_pct = n_fraud / n_total * 100

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Tranzacții (set test)</div>
        <div class="kpi-value kpi-accent">{n_total:,}</div>
        <div class="kpi-sub">split temporal 80/20 · IEEE-CIS Dataset</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Tranzacții frauduloase</div>
        <div class="kpi-value kpi-red">{n_fraud:,}</div>
        <div class="kpi-sub">{fraud_pct:.2f}% — dezechilibru sever de clasă</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Variabile (features)</div>
        <div class="kpi-value kpi-accent">{X_test.shape[1]}</div>
        <div class="kpi-sub">reținute din ~432 originale; coloane cu &gt;75% NaN eliminate</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Algoritmi comparați</div>
        <div class="kpi-value kpi-accent">3</div>
        <div class="kpi-sub">Logistic Regression · Random Forest · XGBoost</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics table
    st.markdown('<div class="sec-title">Comparație Performanță Algoritmi</div>', unsafe_allow_html=True)
    metrics = compute_metrics(xgb_model, X_test, y_test, sk_models)
    MK = ["AUC-ROC", "AUC-PR", "F1 (Fraudă)", "Precizie", "Recall", "MCC"]
    ORDER = ["Logistic Regression", "Random Forest", "XGBoost"]
    avail = [m for m in ORDER if m in metrics]
    best  = {k: max(metrics[m][k] for m in avail) for k in MK}

    hdr = "".join(f"<th>{m}</th>" for m in avail)
    body = ""
    for k in MK:
        row = f"<td><strong>{k}</strong></td>"
        for m in avail:
            v = metrics[m][k]
            css = ' class="best"' if v >= best[k] - 1e-4 else ""
            row += f"<td{css}>{v:.4f}</td>"
        body += f"<tr>{row}</tr>"

    st.markdown(f"""
    <table class="cmp-table">
      <thead><tr><th>Metrică</th>{hdr}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    <p style="font-size:11px;color:#94A3B8;margin-top:8px;">
      ✦ Valorile <span style="color:#059669;font-weight:700;">verzi</span>
      indică cel mai bun scor per metrică · Evaluat pe {data_note}
      {"· <em>Logistic Regression și Random Forest necesită fișierele pkl locale pentru comparație completă</em>" if len(sk_models) < 2 else ""}
    </p>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns(2)
    figs_dir = os.path.join(BASE, "figures")

    with c1:
        st.markdown('<div class="sec-title">Curba ROC — Comparație</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:12px;color:#64748B;margin-bottom:10px;line-height:1.6;">
        <strong>Curba ROC</strong> (Receiver Operating Characteristic) reprezintă rata de detecție
        (True Positive Rate) față de rata alarmelor false (False Positive Rate) la toate pragurile posibile.
        <strong>AUC = 1.0</strong> înseamnă clasificator perfect; <strong>AUC = 0.5</strong> echivalează
        cu o ghicire aleatoare. Random Forest obține cel mai bun AUC-ROC (0.829) pe acest set de date.
        </p>""", unsafe_allow_html=True)
        roc_png = os.path.join(figs_dir, "ROC_Curve_Comparison.png")
        if os.path.exists(roc_png):
            st.image(roc_png, use_container_width=True)
        elif n_total >= 5000:
            # Suficient de multe cazuri pentru o curba neteda — genereaza dinamic
            fig_r, ax_r = plt.subplots(figsize=(6, 5))
            colors = {"Logistic Regression": "#D97706",
                      "Random Forest": "#1E6FD9",
                      "XGBoost": "#059669"}
            for nm in avail:
                model = sk_models.get(nm, xgb_model)
                yp = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, yp)
                auc_v = roc_auc_score(y_test, yp)
                # Interpolare pe grilă fină pentru afișare netedă
                fpr_fine = np.linspace(0, 1, 1000)
                tpr_fine = np.interp(fpr_fine, fpr, tpr)
                ax_r.plot(fpr_fine, tpr_fine, lw=2.2, color=colors.get(nm, "#999"),
                          label=f"{nm}  (AUC = {auc_v:.3f})")
            ax_r.plot([0, 1], [0, 1], "--", color="#CBD5E1", lw=1.5)
            ax_r.set_xlabel("False Positive Rate", fontsize=10)
            ax_r.set_ylabel("True Positive Rate",  fontsize=10)
            ax_r.legend(fontsize=9)
            ax_r.grid(alpha=0.2)
            fig_r.tight_layout()
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)
        else:
            # Prea putine cazuri de frauda pentru o curba reprezentativa
            st.markdown("""
            <div style="background:#FEF9C3;border:1px solid #FDE047;border-radius:10px;
                        padding:20px;text-align:center;color:#713F12;">
              <div style="font-size:22px;margin-bottom:8px;">📊</div>
              <div style="font-weight:700;margin-bottom:6px;">Grafic disponibil cu setul complet</div>
              <div style="font-size:12px;line-height:1.6;">
                Curba ROC necesită minim 5.000 tranzacții pentru o reprezentare vizuală corectă.<br>
                Rulați <code>generate_plots.py</code> local, apoi <code>git push figures/</code>
                pentru a activa graficul pe această pagină.
              </div>
            </div>
            """, unsafe_allow_html=True)
            # Arata cel putin AUC-ul numeric
            yp = xgb_model.predict_proba(X_test)[:, 1]
            auc_v = roc_auc_score(y_test, yp)
            st.metric("AUC-ROC (XGBoost, eșantion demo)", f"{auc_v:.4f}")

    with c2:
        st.markdown('<div class="sec-title">Matrice de Confuzie</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:12px;color:#64748B;margin-bottom:10px;line-height:1.6;">
        Matricea de confuzie arată distribuția predicțiilor: <strong>TP</strong> (fraude detectate corect),
        <strong>TN</strong> (tranzacții legitime acceptate), <strong>FP</strong> (alarme false — cost operațional)
        și <strong>FN</strong> (fraude ratate — cost financiar direct). Pe date cu dezechilibru sever (~3.5% fraudă),
        un model care prezice mereu „legitim" ar atinge 96.5% acuratețe — de aceea matricea de confuzie
        este mai informativă decât acuratețea simplă.
        </p>""", unsafe_allow_html=True)
        cm_png = os.path.join(figs_dir, "Confusion_Matrices.png")
        if os.path.exists(cm_png):
            st.image(cm_png, use_container_width=True)
        else:
            n = len(avail)
            fig_c, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
            if n == 1:
                axes = [axes]
            for i, nm in enumerate(avail):
                model = sk_models.get(nm, xgb_model)
                cm = confusion_matrix(y_test, model.predict(X_test))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            ax=axes[i], cbar=False, annot_kws={"size": 13})
                axes[i].set_title(nm, fontsize=12)
                axes[i].set_xlabel("Predicție"); axes[i].set_ylabel("Actual")
                axes[i].set_xticklabels(["Legitim", "Fraudă"])
                axes[i].set_yticklabels(["Legitim", "Fraudă"], rotation=0)
            fig_c.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)

    pr_png = os.path.join(figs_dir, "PrecisionRecall_Comparison.png")
    if os.path.exists(pr_png):
        st.markdown('<div class="sec-title">Curba Precision-Recall</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:12px;color:#64748B;margin-bottom:10px;line-height:1.6;">
        Pe seturi cu dezechilibru sever de clasă, <strong>Curba Precision-Recall</strong> este mai informativă
        decât curba ROC (Davis &amp; Goadrich, ICML 2006). <strong>Precizia</strong> măsoară câte dintre alarmele
        ridicate sunt fraude reale; <strong>Recall-ul</strong> măsoară câte fraude reale au fost detectate.
        Linia punctată reprezintă performanța unui clasificator aleator (bazeline = rata de fraudă ≈ 3.5%).
        XGBoost obține cel mai bun <strong>AUC-PR (0.377)</strong>, urmat de Random Forest (0.347).
        </p>""", unsafe_allow_html=True)
        st.image(pr_png, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — SIMULARE TRANZACȚIE
# ══════════════════════════════════════════════════════════════════
with tab2:
    col_L, col_R = st.columns([1, 1], gap="large")

    with col_L:
        st.markdown('<div class="sec-title">Profil Tranzacție</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:12px;color:#64748B;margin-bottom:12px;line-height:1.6;">
        Tabelul prezintă cele mai relevante caracteristici ale tranzacției selectate.
        <strong>TransactionAmt</strong> este suma în dolari; câmpurile <strong>card1–card6</strong> identifică
        cardul și emitentul; <strong>P_emaildomain</strong> este domeniul de email al cumpărătorului;
        <strong>C1–C2</strong> numără asocierile istorice ale cardului; <strong>D1</strong> reprezintă zilele
        scurse de la ultima tranzacție. Cele ~{extra} variabile anonimizate (V1–V339) sunt caracteristici
        comportamentale agregate de Vesta Financial, protejate prin acord de confidențialitate.
        </p>""".format(extra=max(0, X_test.shape[1] - 14)), unsafe_allow_html=True)

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🎲  Tranzacție aleatoare"):
                st.session_state.sim_idx = random.randint(0, len(X_test) - 1)
        with c_btn2:
            if st.button("⚠️  Caz de fraudă"):
                fraud_idx = y_test[y_test == 1].index.tolist()
                if fraud_idx:
                    chosen = random.choice(fraud_idx)
                    st.session_state.sim_idx = X_test.index.get_loc(chosen)

        if st.session_state.sim_idx is not None:
            tx = X_test.iloc[[st.session_state.sim_idx]]
            st.markdown(tx_table_html(tx), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#94A3B8;
                        background:#fff;border-radius:12px;border:1px dashed #E2E8F0;">
              <div style="font-size:36px;margin-bottom:10px;">🎲</div>
              <div style="font-size:13px;">Apasă un buton pentru a simula o tranzacție</div>
            </div>""", unsafe_allow_html=True)

    with col_R:
        if st.session_state.sim_idx is not None:
            idx = st.session_state.sim_idx
            tx   = X_test.iloc[[idx]]
            real = y_test.iloc[idx]
            pred = xgb_model.predict(tx)[0]
            prob = float(xgb_model.predict_proba(tx)[0][1])

            st.markdown('<div class="sec-title">Decizie Algoritmică (XGBoost)</div>',
                        unsafe_allow_html=True)

            fig_g = make_gauge(prob)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

            st.markdown("""
            <p style="font-size:11px;color:#64748B;text-align:center;
                       margin:-6px 0 14px;line-height:1.6;">
              Manometrul indică probabilitatea de fraudă estimată de model pentru această tranzacție.<br>
              <span style="color:#059669;font-weight:600;">■ 0–30%</span> risc scăzut &nbsp;·&nbsp;
              <span style="color:#D97706;font-weight:600;">■ 30–60%</span> risc mediu &nbsp;·&nbsp;
              <span style="color:#DC2626;font-weight:600;">■ 60–100%</span> risc ridicat
            </p>""", unsafe_allow_html=True)

            if pred == 1:
                if real == 1:
                    real_label = "⚠️ Fraudă confirmată"
                    real_style = ("font-size:13px;font-weight:600;color:#059669;"
                                  "margin-top:10px;")
                else:
                    real_label = "✅ Tranzacție legitimă — <strong>fals pozitiv</strong>"
                    real_style = "font-size:13px;color:#D97706;margin-top:10px;"
                st.markdown(f"""
                <div class="dec-fraud">
                  <div class="dec-icon">⛔</div>
                  <div class="dec-lbl">Decizie XGBoost</div>
                  <div class="dec-txt" style="color:#DC2626;">TRANZACȚIE BLOCATĂ</div>
                  <div style="{real_style}">Etichetă reală: {real_label}</div>
                </div>""", unsafe_allow_html=True)
            else:
                if real == 1:
                    real_label = "⚠️ Fraudă nedetectată — fals negativ"
                    real_style = ("font-size:15px;font-weight:700;color:#DC2626;"
                                  "background:rgba(220,38,38,0.10);border-radius:8px;"
                                  "padding:10px 14px;margin-top:12px;"
                                  "border-left:4px solid #DC2626;display:block;")
                else:
                    real_label = "✅ Tranzacție legitimă"
                    real_style = "font-size:12px;color:#64748B;margin-top:8px;"
                st.markdown(f"""
                <div class="dec-legit">
                  <div class="dec-icon">✅</div>
                  <div class="dec-lbl">Decizie XGBoost</div>
                  <div class="dec-txt" style="color:#059669;">TRANZACȚIE AUTORIZATĂ</div>
                  <div style="{real_style}">Etichetă reală: {real_label}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#94A3B8;
                        background:#fff;border-radius:12px;border:1px dashed #E2E8F0;">
              <div style="font-size:36px;margin-bottom:10px;">🛡️</div>
              <div>Simulează o tranzacție pentru a vedea scorul de risc</div>
            </div>""", unsafe_allow_html=True)

    # SHAP — full width
    if st.session_state.sim_idx is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="shap-box">
          <h4>🔍 Justificarea Deciziei — SHAP (eXplainable AI)</h4>
          <p>
            Conform <strong>EU AI Act Art. 13</strong> și <strong>GDPR Art. 22</strong>,
            orice decizie automată cu impact financiar trebuie însoțită de o explicație inteligibilă.
            Graficul arată contribuția fiecărei variabile la scorul de risc final (f(x)).
            <strong>Roșu</strong> = crește probabilitatea de fraudă &nbsp;·&nbsp;
            <strong>Albastru</strong> = reduce probabilitatea de fraudă.
          </p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Se calculează explicația SHAP (TreeExplainer)..."):
            tx = X_test.iloc[[st.session_state.sim_idx]]
            sv = explainer(tx)
            # Handle shape: (1, n_features) for binary XGBoost
            if hasattr(sv, "shape") and len(sv.shape) == 3:
                sv = sv[:, :, 1]

            fig_s, ax_s = plt.subplots()
            shap.plots.waterfall(sv[0], max_display=14, show=False)
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)

        st.markdown("""
        <p style="font-size:11px;color:#94A3B8;text-align:center;margin-top:4px;">
        Tranzacțiile cu risc scăzut au contribuții SHAP de magnitudine mică — apasă
        <em>⚠️ Caz de fraudă</em> pentru valori mai expresive.
        </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — XAI GLOBAL
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="eu-banner">
      <strong>🇪🇺 Conformitate EU AI Act (Regulamentul 2024/1689)</strong><br>
      Sistemele de scoring financiar automatizat se încadrează în categoria
      <strong>High-Risk AI Systems</strong> (Anexa III, pct. 5b).
      Art. 13 (Transparență) impune furnizarea de explicații inteligibile pentru orice decizie automatizată
      cu impact asupra persoanelor fizice. SHAP (SHapley Additive exPlanations) satisface această cerință
      prin atribuiri de contribuție garantat unice, consistente și corecte din punct de vedere matematic
      (Lundberg &amp; Lee, NeurIPS 2017 — teoria jocurilor cooperative, valorile Shapley).
    </div>
    """, unsafe_allow_html=True)

    c_shap, c_guide = st.columns([2, 1], gap="large")

    with c_shap:
        st.markdown('<div class="sec-title">Importanța Globală a Variabilelor (SHAP Summary)</div>',
                    unsafe_allow_html=True)
        shap_png = os.path.join(BASE, "figures", "SHAP_Summary_Plot.png")
        if os.path.exists(shap_png):
            st.image(shap_png, use_container_width=True)
        else:
            st.info("Rulați `generate_plots.py` local pentru a genera graficul SHAP global "
                    "(necesită setul complet de date).")
            st.code("python generate_plots.py", language="bash")

    with c_guide:
        st.markdown('<div class="sec-title">Ghid Interpretare</div>', unsafe_allow_html=True)
        st.markdown("""
**Axa X — Valoarea SHAP**
- Pozitivă (+) → crește riscul de fraudă
- Negativă (−) → reduce riscul de fraudă

**Culoarea punctelor**
- 🔴 Roșu = valoare mare a variabilei
- 🔵 Albastru = valoare mică a variabilei

**Variabile cheie detectate**
- `TransactionAmt` — sume atipice
- `card1`, `card2` — frecvența cardului
- `P_emaildomain` — domeniu suspect
- `C1`, `C2` — asocieri anormale
- `D1` — intervale temporale neobișnuite

**Baza matematică**

Valoarea Shapley φᵢ(v):

φᵢ = Σ [|S|!(n−|S|−1)!/n!] · [v(S∪{i})−v(S)]
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Importanța Nativă XGBoost (Gain)</div>',
                unsafe_allow_html=True)

    imp = xgb_model.get_booster().get_score(importance_type="gain")
    top15 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
    feats, vals = zip(*top15)

    fig_i, ax_i = plt.subplots(figsize=(10, 4))
    bar_colors = ["#059669" if v >= vals[0] * 0.5 else "#1E6FD9" for v in vals]
    ax_i.barh(feats[::-1], vals[::-1], color=bar_colors[::-1], alpha=0.88)
    ax_i.set_xlabel("Gain (importanță cumulativă)", fontsize=10)
    ax_i.grid(axis="x", alpha=0.2)
    ax_i.tick_params(labelsize=10)
    fig_i.tight_layout()
    st.pyplot(fig_i, use_container_width=True)
    plt.close(fig_i)
