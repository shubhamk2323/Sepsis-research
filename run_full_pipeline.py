import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, auc,
    roc_auc_score, f1_score, confusion_matrix,
    balanced_accuracy_score, precision_score, recall_score,
    fbeta_score, matthews_corrcoef, brier_score_loss, log_loss
)

# ── Load FULL dataset (no row limit) ──────────────────────────────────────
print("Loading FULL combined CSV...")
df_raw = pd.read_csv('combined_sepsis_data.csv', low_memory=False)
print(f"Loaded {len(df_raw)} rows.")

# ── Sepsyd constants (from their submission code) ──────────────────────────
varmeans    = [84.58, 97.19, 36.97, 123.75, 82.40, 63.83, 18.72, 32.95,
               -0.68, 24.07, 0.55, 7.37, 41.02, 92.65, 260.22, 23.91,
               102.48, 7.55, 105.82, 1.51, 1.83, 136.93, 2.64, 2.05,
               3.54, 4.13, 2.11, 8.29, 30.79, 10.43, 41.23, 11.44,
               287.38, 196.01, 62.00, 0.55, 0.49, 0.50, -56.12, 26.99]

varstds     = [17.32, 2.93, 0.77, 23.23, 16.34, 13.95, 5.09, 7.95,
               4.29, 4.37, 11.12, 0.07, 9.26, 10.89, 855.74, 19.99,
               120.12, 2.43, 5.88, 1.80, 3.69, 51.31, 2.52, 0.39,
               1.42, 0.64, 4.31, 24.80, 5.49, 1.96, 26.21, 7.73,
               153.00, 103.63, 16.38, 0.49, 0.5, 0.5, 162.25, 29.00]

varlogmeans = [4.41, 4.57, 3.61, 4.80, 4.39, 4.13, 2.89, 3.46,
               0.1, 0.1, 0.1, 1.99, 3.69, 4.52, 4.10, 2.92,
               4.38, 1.90, 4.66, 0.10, -0.55, 4.86, 0.70, 0.70,
               1.19, 1.40, 0.03, -0.86, 3.41, 2.32, 3.60, 2.30,
               5.53, 5.14, 4.08, 0.1, 0.1, 0.1, 0.1, 2.88]

varlogstds  = [0.20, 0.03, 0.02, 0.18, 0.19, 0.21, 0.28, 0.26,
               0.1, 0.1, 0.1, 0.01, 0.21, 0.14, 1.37, 0.69,
               0.61, 0.61, 0.05, 0.68, 1.48, 0.31, 0.67, 0.18,
               0.38, 0.14, 1.01, 2.72, 0.17, 0.18, 0.42, 0.51,
               0.50, 0.55, 0.31, 0.1, 0.1, 0.1, 0.1, 0.97]

log_indices = [14, 15, 16, 19, 20, 22, 25, 26, 27, 30, 31, 32]

# ── ALL 40 features in exact Sepsyd order ──────────────────────────────────
# 34 clinical + 6 demographic (Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS)
# This is what Sepsyd's normalization constants correspond to
clinical_cols = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
    'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb',
    'PTT', 'WBC', 'Fibrinogen', 'Platelets',
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]
print(f"Using {len(clinical_cols)} features (Sepsyd-matched)")

# Add missing columns as NaN if not present
for col in clinical_cols:
    if col not in df_raw.columns:
        print(f"  [WARN] Column '{col}' missing in dataset — filling with NaN")
        df_raw[col] = np.nan
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# ── Feature Engineering ────────────────────────────────────────────────────
print("Step 1: Building missingness mask...")
mask_df = df_raw[clinical_cols].notnull().astype(int)
mask_df.columns = [f"{c}_mask" for c in clinical_cols]

print("Step 2: Imputing missing values with population means...")
df_filled = df_raw.copy()
for i, col in enumerate(clinical_cols):
    df_filled[col] = df_filled[col].fillna(varmeans[i])
df_filled[clinical_cols] = df_filled[clinical_cols].apply(pd.to_numeric, errors='coerce')
df_filled[clinical_cols] = df_filled.groupby('Patient_ID')[clinical_cols].ffill()

print("Step 3: Computing deltas...")
df_delta = df_filled[clinical_cols] - df_filled.groupby('Patient_ID')[clinical_cols].shift(1)
df_delta = df_delta.fillna(0)
df_delta.columns = [f"{c}_delta" for c in clinical_cols]

print("Step 4: Z-score / log normalization...")
for i, col in enumerate(clinical_cols):
    if i in log_indices:
        df_filled[col] = 10 * (np.log(df_filled[col].clip(lower=1e-5)) - varlogmeans[i]) / varlogstds[i]
    else:
        df_filled[col] = 10 * (df_filled[col] - varmeans[i]) / varstds[i]

print("Step 5: Building 6-hour sliding window (720 features)...")
df_level1 = pd.concat([df_filled[clinical_cols], df_delta, mask_df], axis=1)

shifts = [df_level1]
for i in range(1, 6):
    shifted = df_level1.groupby(df_filled['Patient_ID']).shift(i).fillna(0)
    shifted.columns = [f"{c}_Tminus{i}" for c in shifted.columns]
    shifts.append(shifted)

X_mega = pd.concat(shifts, axis=1).astype('float32')
y_mega = pd.to_numeric(df_filled['SepsisLabel'], errors='coerce').fillna(0).astype('int32')
X_mega['Patient_ID'] = df_filled['Patient_ID'].astype(str)

print(f"Feature matrix shape: {X_mega.shape}  (rows x features)")
print("Extraction Complete!\n")

# ── Train/Test Split (patient-level) ──────────────────────────────────────
patients = X_mega['Patient_ID'].unique()
train_idx, test_idx = train_test_split(patients, test_size=0.25, random_state=42)

X_train = X_mega[X_mega['Patient_ID'].isin(train_idx)].drop(columns=['Patient_ID'])
y_train = y_mega[X_mega['Patient_ID'].isin(train_idx)]
X_test  = X_mega[X_mega['Patient_ID'].isin(test_idx)].drop(columns=['Patient_ID'])
y_test  = y_mega[X_mega['Patient_ID'].isin(test_idx)]

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"Sepsis prevalence — Train: {y_train.mean():.3%} | Test: {y_test.mean():.3%}\n")

# ── XGBoost Training (Sepsyd params) ──────────────────────────────────────
params = {
    'objective': 'binary:logistic',
    'n_estimators': 30,
    'learning_rate': 0.1,
    'max_depth': 4,
    'scale_pos_weight': 40,
    'random_state': 42,
    'eval_metric': 'aucpr',
    'tree_method': 'hist',   # faster on large data
    'verbosity': 1,
}
model = xgb.XGBClassifier(**params)
print("Training XGBoost Model (Sepsyd config)...")
model.fit(X_train, y_train)
print("Model Trained!\n")

# ── Evaluation ─────────────────────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc         = accuracy_score(y_test, y_pred)
bal_acc     = balanced_accuracy_score(y_test, y_pred)
precision   = precision_score(y_test, y_pred)
recall      = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
f1          = f1_score(y_test, y_pred)
f2          = fbeta_score(y_test, y_pred, beta=2)
auroc       = roc_auc_score(y_test, y_pred_proba)
pr_p, pr_r, _ = precision_recall_curve(y_test, y_pred_proba)
auprc       = auc(pr_r, pr_p)
mcc         = matthews_corrcoef(y_test, y_pred)
brier       = brier_score_loss(y_test, y_pred_proba)
logloss_val = log_loss(y_test, y_pred_proba)

print("===========================================================")
print("     *** SEPSIS EVALUATION REPORT ***")
print("===========================================================")
print(f"   TN : {tn}   TP : {tp}")
print(f"   FP : {fp}   FN : {fn}")
print("-----------------------------------------------------------")
print(f"   RECALL (Sensitivity)  : {recall:.4f}")
print(f"   Specificity          : {specificity:.4f}")
print(f"   Precision            : {precision:.4f}")
print(f"   F1-Score             : {f1:.4f}")
print(f"   F2-Score             : {f2:.4f}")
print(f"   AUROC                : {auroc:.4f}")
print(f"   AUPRC                : {auprc:.4f}")
print(f"   MCC                  : {mcc:.4f}")
print(f"   Brier Score          : {brier:.4f}")
print("===========================================================\n")

# ── Official PhysioNet Utility Score ──────────────────────────────────────
def compute_prediction_utility(labels, predictions,
                                dt_early=-12, dt_optimal=-6, dt_late=3.0,
                                max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')
    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return np.sum(u)

print("Evaluating per-patient Utility Score...")
test_df = X_mega[X_mega['Patient_ID'].isin(test_idx)][['Patient_ID']].copy()
test_df['Label'] = y_test.values
test_df['Pred']  = y_pred

observed_utilities  = []
best_utilities      = []
inaction_utilities  = []

for pid, group in test_df.groupby('Patient_ID'):
    labels = group['Label'].values
    preds  = group['Pred'].values
    n      = len(labels)

    best_preds = np.zeros(n)
    if np.any(labels):
        t_sep = np.argmax(labels) + 6   # dt_optimal = -6
        best_preds[max(0, t_sep - 12) : min(t_sep + 3 + 1, n)] = 1

    inaction_preds = np.zeros(n)

    observed_utilities.append(compute_prediction_utility(labels, preds))
    best_utilities.append(compute_prediction_utility(labels, best_preds))
    inaction_utilities.append(compute_prediction_utility(labels, inaction_preds))

u_obs     = np.sum(observed_utilities)
u_best    = np.sum(best_utilities)
u_inaction = np.sum(inaction_utilities)
normalized = (u_obs - u_inaction) / (u_best - u_inaction)

print("===========================================================")
print(f"  ** OFFICIAL PHYSIONET UTILITY SCORE : {normalized:.4f} **")
print(f"     (Sepsyd benchmark: 0.345)")
print(f"     Unnormalized — Observed: {u_obs:.1f} | Best: {u_best:.1f} | Inaction: {u_inaction:.1f}")
print("===========================================================")
