import streamlit as st
import pandas as pd
import os
import shap
import random
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# PATCH pentru a evita erori de compatibilitate NumPy în cloud
if not hasattr(np, 'int'):
    np.int = int

st.set_page_config(page_title="FinTech Fraud XAI", layout="wide")

st.title("Sistem de Detecție a Fraudelor Bancare")
st.markdown("**Demonstrator XAI (eXplainable AI) pentru conformitatea cu EU AI Act**")

@st.cache_resource
def load_assets():
    # Determinăm folderul curent (unde se află app.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Căutăm fișierele în folderul principal
    model_path = os.path.join(base_dir, 'xgboost.json')
    data_path = os.path.join(base_dir, 'model_ready_data.pkl')
    
    # Verificare critică: dacă nu există, anunțăm clar
    if not os.path.exists(model_path):
        st.error(f"Eroare: Nu am găsit 'xgboost.json' în {base_dir}")
        st.stop()
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Încărcare set test
    _, X_test, _, y_test = pd.read_pickle(data_path)
    return model, X_test, y_test

# Încărcare
model, X_test, y_test = load_assets()

# Sidebar
st.sidebar.header("Panou de Control")
if st.sidebar.button("Simulează o Tranzacție Aleatoare", type="primary"):
    idx = random.choice(range(len(X_test)))
    transaction = X_test.iloc[[idx]]
    actual_label = y_test.iloc[idx]
    
    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Profil Tranzacție")
        st.dataframe(transaction.T.head(10), use_container_width=True)
        
    with col2:
        st.subheader("Decizie Algoritmică (XGBoost)")
        if prediction == 1:
            st.error(f"⚠️ TRANZACȚIE BLOCATĂ (Risc Fraudă: {probability * 100:.2f}%)")
        else:
            st.success(f"✅ TRANZACȚIE AUTORIZATĂ (Risc Fraudă: {probability * 100:.2f}%)")
        st.write(f"**Eticheta reală:** {'Fraudă' if actual_label == 1 else 'Legitimă'}")

    st.divider()
    st.subheader("Justificarea Deciziei (SHAP XAI)")
    
    with st.spinner('Se calculează explicația SHAP...'):
        explainer = shap.Explainer(model.predict_proba, X_test.iloc[:50])
        shap_values = explainer(transaction)
        
        # Selectăm clasa 1 (Fraudă)
        values = shap_values[:, :, 1]
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(values[0], show=False)
        st.pyplot(fig)