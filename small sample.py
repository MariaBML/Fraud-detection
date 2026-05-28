"""
Generare esantion redus pentru deploy (sampling stratificat).
"""
import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'data', 'processed', 'model_ready_data.pkl')

print("Incarcare date complete...")
_, X_test, _, y_test = pd.read_pickle(file_path)

print(f"Set complet: {len(X_test)} randuri | Frauda: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# Sampling stratificat — pastreaza proportia frauda/legitim
N_DEPLOY = 5000

fraud_idx = y_test[y_test == 1].index
legit_idx  = y_test[y_test == 0].index

# Calculam cate cazuri din fiecare clasa sa luam (proportional)
n_fraud = min(len(fraud_idx), round(N_DEPLOY * y_test.mean()))
n_legit = min(len(legit_idx), N_DEPLOY - n_fraud)

fraud_sample = fraud_idx.to_series().sample(n=n_fraud, random_state=42)
legit_sample = legit_idx.to_series().sample(n=n_legit, random_state=42)

all_idx = pd.Index(list(fraud_sample.index) + list(legit_sample.index))

X_deploy = X_test.loc[all_idx]
y_deploy = y_test.loc[all_idx]

print(f"\nEsantion deploy: {len(X_deploy)} randuri")
print(f"  Frauda:   {y_deploy.sum()} ({y_deploy.mean()*100:.2f}%)")
print(f"  Legitime: {(y_deploy==0).sum()} ({(1-y_deploy.mean())*100:.2f}%)")

# Salveaza ambele fisiere din radacina (folosite de app.py si Streamlit Cloud)
out1 = os.path.join(base_dir, 'model_ready_data_deploy.pkl')
out2 = os.path.join(base_dir, 'model_ready_data.pkl')

pd.to_pickle((None, X_deploy, None, y_deploy), out1)
pd.to_pickle((None, X_deploy, None, y_deploy), out2)

size_kb = os.path.getsize(out1) / 1024
print(f"\nSalvat: model_ready_data_deploy.pkl ({size_kb:.0f} KB)")
print(f"Salvat: model_ready_data.pkl ({size_kb:.0f} KB)")
