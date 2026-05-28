# FinTech Fraud Detection — Demonstrator XAI

**Disertație · Master Știința Datelor și Inteligență Artificială**  
Universitatea „Titu Maiorescu" București · 2026  
Autor: Balaban Maria-Laura | Coord.: Conf. univ. dr. Daniela Joița

---

## Descriere

Aplicație demonstrativă pentru detectarea tranzacțiilor bancare frauduloase, cu integrare XAI (eXplainable AI) pentru conformitatea cu EU AI Act și GDPR.

Compară 3 algoritmi de clasificare pe setul IEEE-CIS Fraud Detection (Kaggle, 590.540 rânduri):

| Model | AUC-ROC | AUC-PR | MCC |
|---|---|---|---|
| Logistic Regression | 0.666 | 0.056 | 0.012 |
| Random Forest | **0.829** | 0.347 | 0.310 |
| XGBoost | 0.803 | **0.377** | **0.391** |

Explicabilitatea deciziilor este asigurată prin **SHAP** (SHapley Additive exPlanations, Lundberg & Lee 2017), conform cerințelor Art. 13 din EU AI Act și Art. 22 GDPR.

---

## Demo live

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-maria.streamlit.app/)

---

## Structură proiect

```
├── app.py                      # Aplicația Streamlit (dashboard, simulare, raport XAI)
├── data_ingestion.py           # Merge CSV-uri + optimizare RAM
├── feature_engineering.py      # Split temporal, imputare, label encoding
├── model_training.py           # Antrenare LR / RF / XGBoost + evaluare metrici
├── generate_plots.py           # Grafice comparative la 300 DPI (pentru disertație)
├── small sample.py             # Eșantion stratificat 5.000 rânduri pentru deploy
├── xgboost.json                # Model XGBoost serializat (folosit de Streamlit Cloud)
├── model_ready_data.pkl        # Eșantion de evaluare (5.000 rânduri, stratificat)
├── figures/                    # Grafice pre-generate: ROC, Confusion Matrix, SHAP, PR
├── requirements.txt
└── .streamlit/
    └── config.toml             # Temă UI (culori, font)
```

---

## Rulare locală

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline complet (antrenare de la zero)

```bash
# 1. Descarcă datele IEEE-CIS din Kaggle și plasează-le în data/raw/
python data_ingestion.py         # merge + compresie → data/processed/train_merged_optimized.pkl

# 2. Inginerie caracteristici + separare temporală pe TransactionDT (80/20)
python feature_engineering.py    # → data/processed/model_ready_data.pkl

# 3. Antrenare și evaluare 3 modele
python model_training.py         # → data/models/

# 4. Generare grafice comparative (necesită setul complet de date)
python generate_plots.py         # → figures/

# 5. Creare eșantion stratificat pentru deploy pe Streamlit Cloud
python "small sample.py"         # → model_ready_data.pkl (rădăcina repo)

# 6. Pornire aplicație
streamlit run app.py
```

### Date necesare (excluse din repo — dimensiune prea mare)

Descarcă de pe Kaggle: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)

Plasează în `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`

---

## Stack tehnic

| Componentă | Tehnologie |
|---|---|
| Limbaj | Python 3.10 |
| ML | Scikit-Learn 1.3, XGBoost 2.0 |
| Explicabilitate | SHAP 0.44 (TreeExplainer) |
| Interfață | Streamlit 1.32 |
| Vizualizare | Matplotlib 3.8, Seaborn 0.13 |
| Dataset | IEEE-CIS Fraud Detection — Kaggle (590.540 rânduri) |
