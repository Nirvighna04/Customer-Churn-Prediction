"""
train.py
--------
End-to-end training script that:
  1. Loads the dataset
  2. Preprocesses it using preprocessing.py
  3. Trains 4 models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
  4. Evaluates all models and prints a comparison table
  5. Saves the best model (by ROC-AUC) and all encoders

Run from the project root:
    python src/train.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import fit_transform_preprocessing, save_encoders


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'churn.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
TEST_SIZE  = 0.2
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """Compute all evaluation metrics for a trained model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1':        f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, y_proba)
    }


# ─────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load Data ──
    print(f'\n[train] Loading data from: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    print(f'[train] Dataset shape: {df.shape}')
    print(f'[train] Churn distribution:\n{df["churn"].value_counts().to_string()}\n')

    # ── 2. Preprocess ──
    print('[train] Running preprocessing...')
    X, y, encoders = fit_transform_preprocessing(df)
    save_encoders(encoders, model_dir=MODEL_DIR)

    # ── 3. Train-Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f'[train] Train: {X_train.shape} | Test: {X_test.shape}')

    # ── 4. SMOTE — Handle Class Imbalance ──
    print('[train] Applying SMOTE...')
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f'[train] After SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}')

    # ── 5. Define Models ──
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost':             XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                              random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0)
    }

    # ── 6. Train & Evaluate ──
    print('\n[train] Training models...\n')
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f'  → {name}')
        model.fit(X_train_sm, y_train_sm)
        trained_models[name] = model
        metrics = evaluate_model(model, X_test, y_test)
        results.append({'Model': name, **metrics})

    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    results_df = results_df.set_index('Model').round(4)

    print('\n[train] === Model Comparison ===')
    print(results_df.to_string())

    # Save results CSV
    results_df.reset_index().to_csv(os.path.join(MODEL_DIR, 'model_results.csv'), index=False)

    # ── 7. Select Best Model ──
    best_name = results_df['ROC-AUC'].idxmax()
    best_model = trained_models[best_name]

    print(f'\n[train] Best model selected: {best_name}')
    print(f'        ROC-AUC: {results_df.loc[best_name, "ROC-AUC"]:.4f}')

    # ── 8. Feature Importance ──
    feature_names = encoders['feature_columns']
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False)
    feat_df.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)
    print('\n[train] Top 5 features:')
    print(feat_df.head().to_string(index=False))

    # ── 9. Save Best Model ──
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'model.pkl'))
    print(f'\n[train] model.pkl saved → {MODEL_DIR}/model.pkl')
    print('[train] Training complete!\n')


if __name__ == '__main__':
    main()
