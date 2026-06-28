# utils/trainer.py
# Handles ML model training, evaluation, and persistence

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_auc_score)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')


def train_model(X, y, model_type='random_forest', test_size=0.2):
    """
    Train selected model, return metrics and save model to disk.
    model_type: 'logistic_regression' or 'random_forest'
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Select model
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42,
                                        class_weight='balanced', n_jobs=-1)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
        'model_type': model_type,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Feature importances (Random Forest only)
    if model_type == 'random_forest':
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        importances = model.feature_importances_
        feat_imp = sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)[:15]
        metrics['feature_importance'] = [
            {'feature': f, 'importance': round(float(i) * 100, 2)}
            for f, i in feat_imp
        ]
    else:
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        coefs = np.abs(model.coef_[0])
        feat_imp = sorted(zip(feature_names, coefs),
                          key=lambda x: x[1], reverse=True)[:15]
        metrics['feature_importance'] = [
            {'feature': f, 'importance': round(float(i) * 100, 2)}
            for f, i in feat_imp
        ]

    # Save model
    model_path = os.path.join(MODEL_DIR, f'{model_type}.pkl')
    joblib.dump(model, model_path)
    joblib.dump(model_type, os.path.join(MODEL_DIR, 'active_model.pkl'))

    return metrics


def predict_single(X_processed):
    """
    Predict churn for a single preprocessed input.
    Returns prediction (0/1) and probability.
    """
    active_model_path = os.path.join(MODEL_DIR, 'active_model.pkl')
    if not os.path.exists(active_model_path):
        raise ValueError("No trained model found. Please train a model first.")

    model_type = joblib.load(active_model_path)
    model = joblib.load(os.path.join(MODEL_DIR, f'{model_type}.pkl'))

    pred = model.predict(X_processed)[0]
    prob = model.predict_proba(X_processed)[0][1]

    return int(pred), round(float(prob) * 100, 1)
