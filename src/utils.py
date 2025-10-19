"""
Utility functions for Instagram Fake Account Detection
"""

import pandas as pd
import numpy as np
import json
import pickle

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, filepath):
    """Save data to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def format_percentage(value):
    """Format value as percentage"""
    return f"{value * 100:.2f}%"

def format_number(value):
    """Format large numbers with commas"""
    return f"{value:,.0f}"

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=1),
        'recall': recall_score(y_true, y_pred, pos_label=1),
        'f1_score': f1_score(y_true, y_pred, pos_label=1)
    }

def print_metrics(metrics, model_name="Model"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*60}")

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low", "green"
    elif probability < 0.7:
        return "Medium", "orange"
    else:
        return "High", "red"

def validate_features(features_df, required_features):
    """Validate that all required features are present"""
    missing = set(required_features) - set(features_df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return True
