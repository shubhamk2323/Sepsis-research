import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, classification_report, roc_auc_score, confusion_matrix, accuracy_score
import re

class SepsisFeatureEngineer:
    def __init__(self, time_col='Hour', id_col='Patient_ID'):
        self.time_col = time_col
        self.id_col = id_col
        
    def add_informative_missingness(self, df, clinical_cols):
        df = df.copy()
        for col in clinical_cols:
            if col not in df.columns: df[col] = np.nan
            df[f'{col}_is_measured'] = df[col].notnull().astype(int)
            temp_time = df[self.time_col].where(df[col].notnull())
            last_measured_time = temp_time.groupby(df[self.id_col]).ffill()
            df[f'{col}_time_since_measured'] = df[self.time_col] - last_measured_time
            df[f'{col}_time_since_measured'].fillna(999, inplace=True)
            df[f'{col}_freq_last_6h'] = df.groupby(self.id_col)[f'{col}_is_measured'].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True)
        return df

    def add_clinical_indices(self, df):
        df = df.copy()
        if 'HeartRate' in df.columns and 'SBP' in df.columns: df['Shock_Index'] = df['HeartRate'] / (df['SBP'] + 1e-5)
        if 'BUN' in df.columns and 'Creatinine' in df.columns: df['BUN_Creatinine_Ratio'] = df['BUN'] / (df['Creatinine'] + 1e-5)
        if 'SBP' in df.columns and 'DBP' in df.columns: df['Calculated_MAP'] = (df['SBP'] + 2 * df['DBP']) / 3
        return df

    def add_baseline_deterioration(self, df, clinical_cols):
        df = df.copy()
        for col in clinical_cols:
            admission_baseline = df.groupby(self.id_col)[col].transform('first')
            df[f'{col}_change_from_baseline'] = df[col] - admission_baseline
        return df

    def add_temporal_aggregations(self, df, clinical_cols):
        df = df.copy()
        for col in clinical_cols:
            group_col = df.groupby(self.id_col)[col]
            ffill_col = group_col.ffill()
            df[f'{col}_mean_6h'] = ffill_col.groupby(df[self.id_col]).rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_std_6h'] = ffill_col.groupby(df[self.id_col]).rolling(window=6, min_periods=1).std().reset_index(level=0, drop=True)
            df[f'{col}_velocity_1h'] = ffill_col.groupby(df[self.id_col]).diff(1)
            df[f'{col}_mean_6h'] = df[f'{col}_mean_6h'].fillna(-999)
            df[f'{col}_velocity_1h'] = df[f'{col}_velocity_1h'].fillna(0)
        return df

    def transform(self, df, clinical_cols):
        df = self.add_informative_missingness(df, clinical_cols)
        df = self.add_clinical_indices(df)
        df = self.add_baseline_deterioration(df, clinical_cols)
        df = self.add_temporal_aggregations(df, clinical_cols)
        df[clinical_cols] = df.groupby(self.id_col)[clinical_cols].ffill()
        return df

clinical_cols = ['HeartRate', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 
                 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
                 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
                 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
                 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
                 'Fibrinogen', 'Platelets']

print("Generating dataset of 1000 patients...")
np.random.seed(42)
n_patients = 1000
hours = 48
data = []
for pid in range(n_patients):
    is_sepsis = np.random.rand() < 0.15
    for h in range(hours):
        hr = np.random.normal(80, 10) + (15 * (h/48) if is_sepsis else 0)
        sbp = np.random.normal(120, 15) - (15 * (h/48) if is_sepsis else 0)
        row = [pid, h, hr, 98, 37.0, sbp, 90, 80, 18, 35, 0, 24, 1.0, 7.35, 40, 98, 25, 15, 80, 9.0, 100, 1.0, 0.5, 100, 1.5, 2.0, 3.5, 4.0, 1.0, 0.1, 40, 12, 30, 8.0, 200, 250]
        row.append(1 if is_sepsis and h > 36 else 0)
        data.append(row)
        
cols = ['Patient_ID', 'Hour'] + clinical_cols + ['SepsisLabel']
df_train = pd.DataFrame(data, columns=cols)

print("Running Top Team Feature Engineering Pipelines...")
eng = SepsisFeatureEngineer(time_col='Hour', id_col='Patient_ID')
df_train_transformed = eng.transform(df_train, clinical_cols)

features = [c for c in df_train_transformed.columns if c not in ['Patient_ID', 'SepsisLabel']]
X = df_train_transformed[features].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', str(x)))
y = df_train_transformed['SepsisLabel']

# Need to split by actual patients, NOT row randomly, otherwise there's severe clinical data leakage
unique_patients = df_train_transformed['Patient_ID'].unique()
train_patients, test_patients = train_test_split(unique_patients, test_size=0.3, random_state=42)

X_train = X[df_train_transformed['Patient_ID'].isin(train_patients)]
y_train = y[df_train_transformed['Patient_ID'].isin(train_patients)]
X_test = X[df_train_transformed['Patient_ID'].isin(test_patients)]
y_test = y[df_train_transformed['Patient_ID'].isin(test_patients)]

print(f"Training LightGBM on {len(train_patients)} patients...")
scale_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, scale_pos_weight=scale_weight, random_state=42)
model.fit(X_train, y_train)

print("Validating & Extracting Metrics...")
y_scores = model.predict_proba(X_test)[:, 1]
y_preds = model.predict(X_test)

acc = accuracy_score(y_test, y_preds)
roc_auc = roc_auc_score(y_test, y_scores)
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

print("\n" + "*"*50)
print("CLINICAL MODEL METRICS")
print(f"Accuracy:      {acc:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"PR-AUC Score:  {pr_auc:.4f} (Focus for Clinical Sepsis Imbalance)")
print("*"*50)
print("\nClassification Report (Focus on Class 1 - Sepsis):")
print(classification_report(y_test, y_preds))
