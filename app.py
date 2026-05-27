import streamlit as st
import pandas as pd
import os
import shap
import random
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# PATCH pentru compatibilitate SHAP + NumPy
np.int = int

# Configurare pagină
st.set_page_config(
    page_title="FinTech Fraud XAI",
    layout="wide"
)

# Titlu aplicație
st.title("Sistem de Detecție a Fraudelor Bancare")

st.markdown(
    "**Demonstrator XAI (eXplainable AI) pentru conformitatea cu EU AI Act**"
)

# Încărcare model și date
@st.cache_resource
def load_assets():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        base_dir,
        'data',
        'models',
        'xgboost.json'
    )

    data_path = os.path.join(
        base_dir,
        'data',
        'processed',
        'model_ready_data.pkl'
    )

    # Încărcare model XGBoost
    model = xgb.XGBClassifier()

    model.load_model(model_path)

    # Încărcare set test
    _, X_test, _, y_test = pd.read_pickle(data_path)

    return model, X_test, y_test


# Load assets
model, X_test, y_test = load_assets()

# Sidebar
st.sidebar.header("Panou de Control")

st.sidebar.write(
    f"Tranzacții disponibile pentru simulare: {X_test.shape[0]}"
)

# Buton simulare
if st.sidebar.button(
    "Simulează o Tranzacție Aleatoare",
    type="primary"
):

    # Selectare tranzacție random
    idx = random.choice(range(len(X_test)))

    transaction = X_test.iloc[[idx]]

    actual_label = y_test.iloc[idx]

    # Predicție model
    prediction = model.predict(transaction)[0]

    probability = model.predict_proba(transaction)[0][1]

    # Layout rezultate
    col1, col2 = st.columns(2)

    # Coloana stânga
    with col1:

        st.subheader("Profil Tranzacție")

        st.dataframe(
            transaction.T.head(10),
            use_container_width=True
        )

    # Coloana dreapta
    with col2:

        st.subheader("Decizie Algoritmică (XGBoost)")

        if prediction == 1:

            st.error(
                f"⚠️ TRANZACȚIE BLOCATĂ "
                f"(Risc Fraudă: {probability * 100:.2f}%)"
            )

        else:

            st.success(
                f"✅ TRANZACȚIE AUTORIZATĂ "
                f"(Risc Fraudă: {probability * 100:.2f}%)"
            )

        st.write(
            f"**Eticheta reală:** "
            f"{'Fraudă' if actual_label == 1 else 'Legitimă'}"
        )

    # Separator
    st.divider()

    # SHAP Section
    st.subheader("Justificarea Deciziei (SHAP XAI)")

    st.markdown(
        """
        Parametrii roșii cresc probabilitatea de fraudă.
        Parametrii albaștri reduc probabilitatea de fraudă.
        """
    )

    with st.spinner(
        'Se calculează explicația SHAP...'
    ):

        # Explainer SHAP
        explainer = shap.Explainer(
            model.predict_proba,
            X_test.iloc[:100]
        )

        # Valori SHAP
        shap_values = explainer(transaction)

        # Selectăm clasa fraudă = 1
        values = shap_values[:, :, 1]

        # Creare figură
        plt.figure()

        shap.plots.waterfall(
            values[0],
            max_display=10,
            show=False
        )

        # Afișare în Streamlit
        st.pyplot(
            plt.gcf(),
            clear_figure=True
        )