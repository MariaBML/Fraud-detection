# FinTech Fraud Detection — Demonstrator XAI

**Disertație · Master Știința Datelor și Inteligență Artificială**  
Universitatea „Titu Maiorescu" București · 2026  
Autor: Balaban Maria-Laura | Coord.: Conf. univ. dr. Daniela Joița

---

## Descriere

Aplicație demonstrativă pentru detectarea tranzacțiilor bancare frauduloase,
cu integrare XAI (eXplainable AI) pentru conformitatea cu EU AI Act.

Compară 3 algoritmi de clasificare pe setul IEEE-CIS Fraud Detection (Kaggle):
- **Logistic Regression** — model interpretabil intrinsec
- **Random Forest** — ansamblu bazat pe bagging
- **XGBoost** — gradient boosting, model principal de producție

Explicabilitatea deciziilor este asigurată prin **SHAP** (SHapley Additive exPlanations),
conform cerințelor Art. 13 & 14 din EU AI Act și Art. 22 GDPR.

---

## Rulare locală

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Flux complet (antrenare de la zero)

```bash
# 1. Descarcă datele IEEE-CIS din Kaggle în data/raw/
python data_ingestion.py       # merge + compresie RAM

# 2. Inginerie caracteristici + split temporal pe TransactionDT
python feature_engineering.py

# 3. Antrenare + evaluare 3 modele
python model_training.py

# 4. Generare grafice pentru disertație (DPI 300)
python generate_plots.py

# 5. Copiaza in radacina pentru deploy:
# cp data/models/xgboost.json .
# python "small sample.py"   (creeaza model_ready_data_deploy.pkl)

# 6. Pornire app
streamlit run app.py
```

## Date necesare (nu sunt incluse în repo — dimensiune prea mare)

Descarcă de pe Kaggle: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

Plasează în `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`

---

## Stack tehnic

| Componentă | Tehnologie |
|---|---|
| Limbaj | Python 3.10 |
| ML Framework | Scikit-Learn, XGBoost |
| Explicabilitate | SHAP (TreeExplainer) |
| Interfață | Streamlit |
| Vizualizare | Matplotlib, Seaborn |

## Deploy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-maria.streamlit.app/)
