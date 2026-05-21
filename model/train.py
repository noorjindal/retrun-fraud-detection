"""
train.py
--------
Trains a Random Forest classifier on the engineered feature matrix.

Key decisions:
  - class_weight='balanced'  →  handles 85/15 imbalance without oversampling
  - GridSearchCV             →  tunes n_estimators, max_depth, min_samples_leaf
  - Saves: fraud_model.pkl, scaler.pkl (for the backend API to use)

Run:
    python train.py
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

BASE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(BASE, "../features")
OUT      = BASE

print("Loading features...")
X_train = pd.read_csv(os.path.join(FEATURES, "X_train.csv"))
y_train = pd.read_csv(os.path.join(FEATURES, "y_train.csv")).squeeze()

print(f"X_train shape : {X_train.shape}")
print(f"Fraud rate    : {y_train.mean():.1%}\n")

# ── Pipeline: scaler + classifier ────────────────────────────────────────────
# Random Forest doesn't strictly need scaling, but we include it
# so the saved pipeline works cleanly with the Flask backend.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1
    ))
])

# ── Hyperparameter search ─────────────────────────────────────────────────────
param_grid = {
    "clf__n_estimators":      [100, 200],
    "clf__max_depth":         [8, 12, None],
    "clf__min_samples_leaf":  [1, 3],
    "clf__max_features":      ["sqrt"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running GridSearchCV (5-fold, scoring=f1)...")
print("This takes ~1–2 minutes...\n")

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="f1",          # optimise for F1 — right metric for imbalanced fraud data
    n_jobs=-1,
    verbose=1,
    refit=True
)
grid.fit(X_train, y_train)

print(f"\n✓  Best params : {grid.best_params_}")
print(f"   Best CV F1  : {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# ── Feature importances ───────────────────────────────────────────────────────
rf         = best_model.named_steps["clf"]
feat_names = X_train.columns.tolist()
importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

print("\nTop 15 feature importances:")
for feat, imp in importances.head(15).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<40} {imp:.4f}  {bar}")

# ── Save ──────────────────────────────────────────────────────────────────────
model_path = os.path.join(OUT, "fraud_model.pkl")
joblib.dump(best_model, model_path)
print(f"\n✓  Saved model → {model_path}")

# Also save importances for the dashboard
importances.to_csv(os.path.join(OUT, "feature_importances.csv"), header=True)
print(f"✓  Saved feature_importances.csv")
