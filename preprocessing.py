"""
preprocessing.py
----------------
Reusable preprocessing utilities used by both:
  - train.py  (fit + transform during training)
  - predict.py (transform-only during inference)

Ensures preprocessing during prediction EXACTLY matches training.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Numerical features to scale
NUMERICAL_FEATURES = [
    'credit_score', 'age', 'tenure', 'balance',
    'products_number', 'estimated_salary'
]

# Categorical features
CATEGORICAL_BINARY = 'gender'
CATEGORICAL_OHE = 'country'

# Column to drop (ID — not a feature)
ID_COLUMN = 'customer_id'

# Target column
TARGET = 'churn'


# ─────────────────────────────────────────────
# Training-time: Fit + Transform
# ─────────────────────────────────────────────

def fit_transform_preprocessing(df: pd.DataFrame):
    """
    Fit all encoders and scaler on the full training DataFrame.
    Returns:
      - X_processed: preprocessed feature DataFrame
      - y: target Series
      - encoders: dict with fitted scaler, label_encoder, ohe
    """
    df = df.copy()

    # 1. Drop ID column
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # 2. Separate features and target
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # 3. Label Encode gender (binary)
    le = LabelEncoder()
    X[CATEGORICAL_BINARY] = le.fit_transform(X[CATEGORICAL_BINARY])

    # 4. OneHot Encode country (multi-class)
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    country_encoded = ohe.fit_transform(X[[CATEGORICAL_OHE]])
    country_cols = ohe.get_feature_names_out([CATEGORICAL_OHE])
    country_df = pd.DataFrame(country_encoded, columns=country_cols, index=X.index)
    X = X.drop(columns=[CATEGORICAL_OHE])
    X = pd.concat([X, country_df], axis=1)

    # 5. Scale numerical features
    scaler = StandardScaler()
    X[NUMERICAL_FEATURES] = scaler.fit_transform(X[NUMERICAL_FEATURES])

    encoders = {
        'label_encoder': le,
        'ohe': ohe,
        'scaler': scaler,
        'feature_columns': X.columns.tolist()
    }

    return X, y, encoders


def save_encoders(encoders: dict, model_dir: str = '../models'):
    """Save all fitted encoders and feature columns to disk."""
    joblib.dump(encoders['label_encoder'], f'{model_dir}/label_encoder_gender.pkl')
    joblib.dump(encoders['ohe'],           f'{model_dir}/onehot_encoder_country.pkl')
    joblib.dump(encoders['scaler'],        f'{model_dir}/scaler.pkl')
    joblib.dump(encoders['feature_columns'], f'{model_dir}/feature_columns.pkl')
    print(f'[preprocessing] Encoders saved to {model_dir}/')


# ─────────────────────────────────────────────
# Inference-time: Transform Only (no fit)
# ─────────────────────────────────────────────

def load_encoders(model_dir: str = '../models') -> dict:
    """Load all saved encoders from disk."""
    return {
        'label_encoder':   joblib.load(f'{model_dir}/label_encoder_gender.pkl'),
        'ohe':             joblib.load(f'{model_dir}/onehot_encoder_country.pkl'),
        'scaler':          joblib.load(f'{model_dir}/scaler.pkl'),
        'feature_columns': joblib.load(f'{model_dir}/feature_columns.pkl')
    }


def transform_single_input(input_dict: dict, encoders: dict) -> pd.DataFrame:
    """
    Transform a single new customer's input dict into a model-ready DataFrame.
    This mirrors the exact steps used during training.

    Args:
        input_dict: raw feature values as a dictionary
            Example: {
                'credit_score': 650,
                'country': 'France',
                'gender': 'Male',
                'age': 35,
                'tenure': 5,
                'balance': 80000.0,
                'products_number': 2,
                'credit_card': 1,
                'active_member': 1,
                'estimated_salary': 55000.0
            }
        encoders: dict with 'label_encoder', 'ohe', 'scaler', 'feature_columns'

    Returns:
        pd.DataFrame with shape (1, n_features) — ready for model.predict()
    """
    row = pd.DataFrame([input_dict])

    # 1. Label encode gender
    le: LabelEncoder = encoders['label_encoder']
    row['gender'] = le.transform(row['gender'])

    # 2. OneHot encode country
    ohe: OneHotEncoder = encoders['ohe']
    country_encoded = ohe.transform(row[['country']])
    country_cols = ohe.get_feature_names_out(['country'])
    country_df = pd.DataFrame(country_encoded, columns=country_cols, index=row.index)
    row = row.drop(columns=['country'])
    row = pd.concat([row, country_df], axis=1)

    # 3. Scale numerical features
    scaler: StandardScaler = encoders['scaler']
    row[NUMERICAL_FEATURES] = scaler.transform(row[NUMERICAL_FEATURES])

    # 4. Reorder columns to match training feature order
    feature_columns = encoders['feature_columns']
    row = row[feature_columns]

    return row


def transform_batch(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Transform a DataFrame of new customers using fitted encoders.
    Useful for batch predictions.
    """
    df = df.copy()

    # Drop ID and target if present
    for col in [ID_COLUMN, TARGET]:
        if col in df.columns:
            df = df.drop(columns=[col])

    le = encoders['label_encoder']
    ohe = encoders['ohe']
    scaler = encoders['scaler']
    feature_columns = encoders['feature_columns']

    df['gender'] = le.transform(df['gender'])

    country_encoded = ohe.transform(df[['country']])
    country_cols = ohe.get_feature_names_out(['country'])
    country_df = pd.DataFrame(country_encoded, columns=country_cols, index=df.index)
    df = df.drop(columns=['country'])
    df = pd.concat([df, country_df], axis=1)

    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    return df[feature_columns]
