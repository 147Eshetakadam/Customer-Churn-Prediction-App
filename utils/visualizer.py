# utils/visualizer.py
# Generates chart data as JSON for Chart.js frontend rendering

# import json
# import numpy as np


# def get_churn_distribution(df):
#     """Returns churn distribution counts."""
#     if 'Churn' not in df.columns:
#         return {}
#     counts = df['Churn'].value_counts()
#     return {
#         'labels': ['No Churn', 'Churn'],
#         'values': [int(counts.get(0, 0)), int(counts.get(1, 0))],
#         'colors': ['#22c55e', '#ef4444']
#     }


# def get_confusion_matrix_data(cm_list):
#     """Formats confusion matrix for heatmap display."""
#     return {
#         'tn': cm_list[0][0],
#         'fp': cm_list[0][1],
#         'fn': cm_list[1][0],
#         'tp': cm_list[1][1]
#     }


# def get_feature_importance_chart(feat_imp_list):
#     """Returns top 10 features for bar chart."""
#     top10 = feat_imp_list[:10]
#     return {
#         'labels': [f['feature'] for f in top10],
#         'values': [f['importance'] for f in top10],
#     }


# def get_tenure_churn_chart(df):
#     """Bins tenure and shows churn rate per bin."""
#     if 'tenure' not in df.columns or 'Churn' not in df.columns:
#         return {}
#     bins = [0, 12, 24, 36, 48, 60, 100]
#     labels = ['0-12m', '13-24m', '25-36m', '37-48m', '49-60m', '60m+']
#     df = df.copy()
#     df['tenure_bin'] = np.digitize(df['tenure'], bins) - 1
#     grouped = df.groupby('tenure_bin')['Churn'].mean() * 100
#     return {
#         'labels': labels,
#         'values': [round(grouped.get(i, 0), 1) for i in range(len(labels))]
#     }


# def get_monthly_charges_chart(df):
#     """Returns average monthly charges for churned vs non-churned."""
#     if 'MonthlyCharges' not in df.columns or 'Churn' not in df.columns:
#         return {}
#     grouped = df.groupby('Churn')['MonthlyCharges'].mean()
#     return {
#         'labels': ['No Churn', 'Churned'],
#         'values': [round(float(grouped.get(0, 0)), 2),
#                    round(float(grouped.get(1, 0)), 2)]
#     }





# utils/visualizer.py
# Generates chart data as JSON for Chart.js frontend rendering

import json
import numpy as np
import pandas as pd


def get_churn_distribution(df):
    """Returns churn distribution counts."""
    if 'Churn' not in df.columns:
        return {}
    
    # Ensure Churn is numeric
    churn_series = df['Churn']
    if churn_series.dtype == 'object':
        churn_series = churn_series.map({'Yes': 1, 'No': 0}).fillna(churn_series)
        churn_series = pd.to_numeric(churn_series, errors='coerce')
    
    counts = churn_series.value_counts()
    return {
        'labels': ['No Churn', 'Churn'],
        'values': [int(counts.get(0, 0)), int(counts.get(1, 0))],
        'colors': ['#22c55e', '#ef4444']
    }


def get_confusion_matrix_data(cm_list):
    """Formats confusion matrix for heatmap display."""
    return {
        'tn': cm_list[0][0],
        'fp': cm_list[0][1],
        'fn': cm_list[1][0],
        'tp': cm_list[1][1]
    }


def get_feature_importance_chart(feat_imp_list):
    """Returns top 10 features for bar chart."""
    top10 = feat_imp_list[:10]
    return {
        'labels': [f['feature'] for f in top10],
        'values': [f['importance'] for f in top10],
    }


def get_tenure_churn_chart(df):
    """Bins tenure and shows churn rate per bin."""
    if 'tenure' not in df.columns or 'Churn' not in df.columns:
        return {}
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure Churn is numeric
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(df['Churn'])
        df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
    
    # Create tenure bins
    bins = [0, 12, 24, 36, 48, 60, 100]
    labels = ['0-12m', '13-24m', '25-36m', '37-48m', '49-60m', '60m+']
    df['tenure_bin'] = np.digitize(df['tenure'], bins) - 1
    
    # Calculate mean churn rate per bin, handling any issues
    grouped = df.groupby('tenure_bin')['Churn'].mean()
    # Fill NaN values with 0
    grouped = grouped.fillna(0) * 100
    
    return {
        'labels': labels,
        'values': [round(float(grouped.get(i, 0)), 1) for i in range(len(labels))]
    }


def get_monthly_charges_chart(df):
    """Returns average monthly charges for churned vs non-churned."""
    if 'MonthlyCharges' not in df.columns or 'Churn' not in df.columns:
        return {}
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure Churn is numeric
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(df['Churn'])
        df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
    
    # Ensure MonthlyCharges is numeric
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    # Group and calculate mean
    grouped = df.groupby('Churn')['MonthlyCharges'].mean().fillna(0)
    
    return {
        'labels': ['No Churn', 'Churned'],
        'values': [round(float(grouped.get(0, 0)), 2),
                   round(float(grouped.get(1, 0)), 2)]
    }