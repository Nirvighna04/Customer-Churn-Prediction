"""
predict.py
----------
Inference utilities used by the Streamlit app and any external scripts.

Usage (standalone):
    python src/predict.py

Usage (imported by app.py):
    from src.predict import load_model, predict_single, predict_batch
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import load_encoders, transform_single_input, transform_batch


# ─────────────────────────────────────────────
# Default paths (relative to project root)
# ─────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


# ─────────────────────────────────────────────
# Load Model & Encoders
# ─────────────────────────────────────────────

def load_model(model_dir: str = None):
    """Load the saved model from disk."""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_path = os.path.join(model_dir, 'model.pkl')
    model = joblib.load(model_path)
    print(f'[predict] Model loaded from: {model_path}')
    return model


def load_all_artifacts(model_dir: str = None):
    """
    Load model + all encoders in one call.
    Returns: (model, encoders)
    """
    if model_dir is None:
        model_dir = MODEL_DIR
    model = load_model(model_dir)
    encoders = load_encoders(model_dir)
    return model, encoders


# ─────────────────────────────────────────────
# Single Prediction
# ─────────────────────────────────────────────

def predict_single(input_dict: dict, model=None, encoders: dict = None, model_dir: str = None):
    """
    Predict churn for a single customer.

    Args:
        input_dict: dict of raw feature values (before any encoding)
        model: loaded model (optional — will load from disk if None)
        encoders: dict of fitted encoders (optional — will load from disk if None)
        model_dir: path to models directory

    Returns:
        dict with keys:
            'prediction'  → 0 or 1 (0=Retained, 1=Churned)
            'probability' → float, probability of churn (class 1)
            'label'       → human-readable string
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    # Load from disk if not provided
    if model is None:
        model = load_model(model_dir)
    if encoders is None:
        encoders = load_encoders(model_dir)

    # Preprocess input
    X = transform_single_input(input_dict, encoders)

    # Predict — threshold lowered to 0.3 to catch more churners (improves Recall)
    probability = float(model.predict_proba(X)[0][1])
    prediction  = 1 if probability >= 0.3 else 0

    label = 'Will Churn 🔴' if prediction == 1 else 'Will Stay 🟢'

    return {
        'prediction':  prediction,
        'probability': round(probability, 4),
        'label':       label
    }


# ─────────────────────────────────────────────
# Batch Prediction
# ─────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, model=None, encoders: dict = None, model_dir: str = None) -> pd.DataFrame:
    """
    Predict churn for a batch of customers.

    Args:
        df: raw DataFrame (same columns as training data, customer_id and churn optional)
        model, encoders: if None, loaded from disk

    Returns:
        Original DataFrame with two new columns:
            'churn_prediction' → 0 or 1
            'churn_probability' → float
    """
    if model_dir is None:
        model_dir = MODEL_DIR
    if model is None:
        model = load_model(model_dir)
    if encoders is None:
        encoders = load_encoders(model_dir)

    df_out = df.copy()

    X = transform_batch(df_out, encoders)

    df_out['churn_prediction']  = model.predict(X)
    df_out['churn_probability'] = model.predict_proba(X)[:, 1].round(4)

    return df_out


# ─────────────────────────────────────────────
# Quick Test (run directly)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Example: predict for a single customer
    sample_customer = {
        'credit_score':    650,
        'country':         'France',
        'gender':          'Male',
        'age':             35,
        'tenure':          5,
        'balance':         80000.0,
        'products_number': 2,
        'credit_card':     1,
        'active_member':   1,
        'estimated_salary': 55000.0
    }

    result = predict_single(sample_customer)
    print('\n[predict] Sample Customer Prediction:')
    print(f'  → Prediction:  {result["prediction"]} ({result["label"]})')
    print(f'  → Churn Probability: {result["probability"] * 100:.1f}%')
