import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, classification_report, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

df_raw = pd.read_csv('combined_sepsis_data.csv')

varmeans = [84.58, 97.19, 36.97, 123.75, 82.40, 63.83, 18.72, 32.95, -0.68, 24.07, 0.55, 7.37, 41.02, 92.65, 260.22, 23.91, 102.48, 7.55, 105.82, 1.51, 1.83, 136.93, 2.64, 2.05, 3.54, 4.13, 2.11, 8.29, 30.79, 10.43, 41.23, 11.44, 287.38, 196.01, 62.00, 0.55, 0.49, 0.50, -56.12, 26.99]
varstds = [17.32, 2.93, 0.77, 23.23, 16.34, 13.95, 5.09, 7.95, 4.29, 4.37, 11.12, 0.07, 9.26, 10.89, 855.74, 19.99, 120.12, 2.43, 5.88, 1.80, 3.69, 51.31, 2.52, 0.39, 1.42, 0.64, 4.31, 24.80, 5.49, 1.96, 26.21, 7.73, 153.00, 103.63, 16.38, 0.49, 0.5, 0.5, 162.25, 29.00]
varlogmeans = [4.41, 4.57, 3.61, 4.80, 4.39, 4.13, 2.89, 3.46, 0.1, 0.1, 0.1, 1.99, 3.69, 4.52, 4.10, 2.92, 4.38, 1.90, 4.66, 0.10, -0.55, 4.86, 0.70, 0.70, 1.19, 1.40, 0.03, -0.86, 3.41, 2.32, 3.60, 2.30, 5.53, 5.14, 4.08, 0.1, 0.1, 0.1, 0.1, 2.88]
varlogstds = [0.20, 0.03, 0.02, 0.18, 0.19, 0.21, 0.28, 0.26, 0.1, 0.1, 0.1, 0.01, 0.21, 0.14, 1.37, 0.69, 0.61, 0.61, 0.05, 0.68, 1.48, 0.31, 0.67, 0.18, 0.38, 0.14, 1.01, 2.72, 0.17, 0.18, 0.42, 0.51, 0.50, 0.55, 0.31, 0.1, 0.1, 0.1, 0.1, 0.97]
log_indices = [14, 15, 16, 19, 20, 22, 25, 26, 27, 30, 31, 32]

clinical_cols = [c for c in df_raw.columns if c not in ['Patient_ID', 'SepsisLabel', 'Hour', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']][:40]

for col in clinical_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

mask_df = df_raw[clinical_cols].notnull().astype(int)
mask_df.columns = [f"{c}_mask" for c in clinical_cols]

df_filled = df_raw.copy()
for i, col in enumerate(clinical_cols):
    df_filled[col] = pd.to_numeric(df_filled[col].fillna(varmeans[i]), errors='coerce')
    
df_filled[clinical_cols] = df_filled[clinical_cols].apply(pd.to_numeric, errors='coerce')
df_filled[clinical_cols] = df_filled.groupby('Patient_ID')[clinical_cols].ffill()

df_delta = df_filled[clinical_cols] - df_filled.groupby('Patient_ID')[clinical_cols].shift(1)
df_delta = df_delta.fillna(0)
df_delta.columns = [f"{c}_delta" for c in clinical_cols]

for i, col in enumerate(clinical_cols):
    if i in log_indices:
        df_filled[col] = 10 * (np.log(df_filled[col].clip(lower=1e-5)) - varlogmeans[i]) / varlogstds[i]
    else:
        df_filled[col] = 10 * (df_filled[col] - varmeans[i]) / varstds[i]

df_level1 = pd.concat([df_filled[clinical_cols], df_delta, mask_df], axis=1)

shifts = [df_level1]
for i in range(1, 6):
    shifted = df_level1.groupby(df_filled['Patient_ID']).shift(i).fillna(0)
    shifted.columns = [f"{c}_Tminus{i}" for c in shifted.columns]
    shifts.append(shifted)

X_mega = pd.concat(shifts, axis=1).astype('float32')
y_mega = pd.to_numeric(df_filled['SepsisLabel'], errors='coerce').fillna(0).astype('int32')
X_mega['Patient_ID'] = df_filled['Patient_ID'].astype(str)

patients = X_mega['Patient_ID'].unique()
train_idx, test_idx = train_test_split(patients, test_size=0.25, random_state=42)

X_train = X_mega[X_mega['Patient_ID'].isin(train_idx)].drop(columns=['Patient_ID'])
y_train = y_mega[X_mega['Patient_ID'].isin(train_idx)]

X_test = X_mega[X_mega['Patient_ID'].isin(test_idx)].drop(columns=['Patient_ID'])
y_test = y_mega[X_mega['Patient_ID'].isin(test_idx)]

params = {'objective': 'binary:logistic', 'n_estimators': 30, 'learning_rate': 0.1, 'max_depth': 4, 'scale_pos_weight': 40, 'random_state': 42}
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f_meas = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(recall, precision)

# Emulate Table 4 output structure
print(f"Accuracy | {acc:.3f}")
print(f"F measure | {f_meas:.3f}")
print(f"AUROC | {auroc:.3f}")
print(f"AUPRC | {auprc:.3f}")
