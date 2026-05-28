import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import gc

def load_processed_data():
    """Incarca setul de date preprocesat din memoria persistenta."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'data', 'processed', 'train_merged_optimized.pkl')
    print(f"1. Incarcare date din: {input_path}")
    return pd.read_pickle(input_path)

def time_based_split(df, target_col='isFraud', time_col='TransactionDT', test_size=0.20):
    """Segrega cronologic setul de date pentru a preveni data leakage."""
    print("2. Executare separare temporala (Time-Based Split)...")
    df_sorted = df.sort_values(by=time_col)

    split_idx = int(len(df_sorted) * (1 - test_size))

    y = df_sorted[target_col]
    X = df_sorted.drop(columns=[target_col])

    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    del df, df_sorted, X, y
    gc.collect()

    return X_train, X_test, y_train, y_test

def engineer_features(X_train, X_test, null_threshold=0.75):
    """
    Curata matricea de caracteristici si pregateste datele pentru algoritmii matematici.
    """
    print(f"3. Inginerie caracteristici. Dimensiune initiala: {X_train.shape[1]} variabile.")

    # Pasul A: Eliminarea vectorilor cu lipsa masiva de date (peste 75% NaN)
    null_fractions = X_train.isnull().mean()
    cols_to_drop = null_fractions[null_fractions > null_threshold].index.tolist()

    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    print(f"   -> S-au eliminat {len(cols_to_drop)} variabile (zgomot statistic).")

    # Identificarea tipologiilor de date ramase
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Pasul B: Imputarea datelor (FARA data leakage)
    print("   -> Imputare valori reziduale...")
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    for col in cat_cols:
        mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Missing'
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    # Pasul C: Codificarea matematica a textelor
    print("   -> Codificare categorii (Label Encoding)...")
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

        le = LabelEncoder()
        # Calibrare pe intreg spectrul de unice pentru a evita erori la valori noi in test
        le.fit(list(X_train[col].unique()) + list(X_test[col].unique()))
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    print(f"   -> Dimensiune finala matrice antrenare: {X_train.shape[1]} variabile.")
    return X_train, X_test

if __name__ == "__main__":
    df_raw = load_processed_data()
    X_train_raw, X_test_raw, y_train, y_test = time_based_split(df_raw)

    X_train_clean, X_test_clean = engineer_features(X_train_raw, X_test_raw)

    # Persistenta matricelor pregatite pentru modelare
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'data', 'processed')

    pd.to_pickle((X_train_clean, X_test_clean, y_train, y_test),
                 os.path.join(out_dir, 'model_ready_data.pkl'))
    print("4. Matricele finale au fost salvate. Sistem pregatit pentru antrenarea modelelor.")
