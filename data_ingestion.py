import pandas as pd
import numpy as np
import gc
import os

def reduce_mem_usage(df, verbose=True):
    """
    Scaneaza tipurile de date si le compreseaza la cel mai mic format posibil
    pentru a elibera memoria RAM.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Memorie redusă la: {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reducere)')
    return df

if __name__ == "__main__":
    print("Începem ingestia și optimizarea datelor...")

    # Determinam automat folderul in care se afla acest script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construim caile absolute corecte
    trans_path = os.path.join(base_dir, 'data', 'raw', 'train_transaction.csv')
    id_path = os.path.join(base_dir, 'data', 'raw', 'train_identity.csv')

    print(f"Căutăm tranzacțiile la: {trans_path}")
    print("1. Încărcare tranzacții...")
    df_trans = pd.read_csv(trans_path)
    df_trans = reduce_mem_usage(df_trans)

    print("2. Încărcare identități...")
    df_id = pd.read_csv(id_path)
    df_id = reduce_mem_usage(df_id)

    print("3. Unificarea tabelelor (Merge pe TransactionID)...")
    df_train = pd.merge(df_trans, df_id, on='TransactionID', how='left')

    # Curățăm memoria RAM de fișierele inițiale
    del df_trans, df_id
    gc.collect()

    print("4. Optimizare finală a memoriei...")
    df_train = reduce_mem_usage(df_train)

    print(f"Dimensiunea finală a setului de date: {df_train.shape[0]} rânduri și {df_train.shape[1]} coloane.")

    # Crearea directorului pentru date procesate (daca nu exista)
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Salvarea in format binar pentru conservarea tipurilor de date comprimate
    output_path = os.path.join(processed_dir, 'train_merged_optimized.pkl')
    df_train.to_pickle(output_path)
    print(f"Setul de date a fost salvat cu succes la: {output_path}")
