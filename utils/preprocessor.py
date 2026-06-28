# utils/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
TARGET_COL = 'Churn'

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0})
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).astype(int)
    feature_df = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df.copy()
    cat_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
    for col in feature_df.columns:
        if feature_df[col].dtype == bool:
            feature_df[col] = feature_df[col].astype(int)
    scaler = StandardScaler()
    num_cols_present = [c for c in NUMERICAL_COLS if c in feature_df.columns]
    if num_cols_present:
        feature_df[num_cols_present] = scaler.fit_transform(feature_df[num_cols_present])
    feature_df = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(list(feature_df.columns), os.path.join(MODEL_DIR, 'feature_names.pkl'))
    joblib.dump(num_cols_present, os.path.join(MODEL_DIR, 'num_cols.pkl'))
    processed_df = feature_df.copy()
    if TARGET_COL in df.columns:
        processed_df[TARGET_COL] = df[TARGET_COL].values
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return processed_df, feature_df, y, list(feature_df.columns)

def preprocess_single(input_dict):
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    num_cols_path = os.path.join(MODEL_DIR, 'num_cols.pkl')
    if not os.path.exists(feature_names_path):
        raise ValueError("Model not trained yet. Please upload data and train first.")
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    num_cols_present = joblib.load(num_cols_path)
    df = pd.DataFrame([input_dict])
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    df = df.reindex(columns=feature_names, fill_value=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    cols_to_scale = [c for c in num_cols_present if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df
