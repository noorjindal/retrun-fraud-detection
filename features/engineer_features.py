"""
engineer_features.py
--------------------
Transforms raw_returns.csv into a model-ready feature matrix.

What we do here:
  - Encode categoricals (return_reason, item_condition, product_category)
  - Build derived risk signals (refund_ratio, velocity_flag, etc.)
  - Drop columns that leak the label or are not useful for inference
  - Save: features.csv (full, for inspection) and X_train/X_test/y_train/y_test split

Run:
    python engineer_features.py
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "../data/raw_returns.csv")
OUT  = BASE  # save engineered files here

print("Loading raw data...")
df = pd.read_csv(DATA, parse_dates=["purchase_date","return_request_date"])

# ── Drop columns that leak info or are not features ───────────────────────────
# customer_id, order_id  → identifiers
# fraud_type             → derived label, would leak during training
# purchase_date, return_request_date → already captured as days_to_return_request
DROP = ["customer_id","order_id","purchase_date","return_request_date","fraud_type"]
df.drop(columns=DROP, inplace=True)

# ── Derived features ──────────────────────────────────────────────────────────

# 1. How much of the order value is being refunded? >1.0 is suspicious
df["refund_ratio"] = (df["refund_amount_inr"] / df["order_value_inr"].replace(0, np.nan)).round(4)
df["refund_ratio"].fillna(0, inplace=True)

# 2. High return rate customer + high order value = elevated risk
df["high_value_high_returner"] = (
    (df["order_value_inr"] > 3000) & (df["customer_return_rate"] > 0.4)
).astype(int)

# 3. Account is very new (< 30 days) — common in account abuse fraud
df["new_account_flag"] = (df["customer_account_age_days"] < 30).astype(int)

# 4. Multiple accounts at same address
df["multi_account_flag"] = (df["same_address_accounts"] >= 3).astype(int)

# 5. Claimed non-delivery but delivery was confirmed = strong signal
df["false_non_delivery"] = (
    (df["claimed_non_delivery"] == 1) & (df["delivery_confirmed"] == 1)
).astype(int)

# 6. Return requested but pickup failed (item switching attempt)
df["pickup_failed_flag"] = (
    (df["return_pickup_attempted"] == 1) & (df["return_pickup_successful"] == 0)
).astype(int)

# 7. Heavy discount + high return rate (price abuse signal)
df["discount_abuse_flag"] = (
    (df["discount_percent"] >= 30) & (df["customer_return_rate"] > 0.3)
).astype(int)

# 8. Returned on Monday or Tuesday after weekend (wardrobing signal)
df["weekend_return_flag"] = df["return_after_weekend"]  # already binary, rename for clarity

# 9. Contacted support multiple times before return (orchestrated fraud)
df["pre_return_support_flag"] = (df["support_contacts_pre_return"] >= 2).astype(int)

# ── Encode categoricals ───────────────────────────────────────────────────────

# return_reason
reason_map = {
    "size_issue":          0,
    "defective_product":   1,
    "wrong_item_sent":     2,
    "not_as_described":    3,
    "changed_mind":        4,
}
df["return_reason_enc"] = df["return_reason"].map(reason_map).fillna(0).astype(int)

# item_condition_on_return (ordinal: new → used → damaged)
condition_map = {
    "new_with_tags":    0,
    "new_without_tags": 1,
    "lightly_used":     2,
    "damaged":          3,
}
df["item_condition_enc"] = df["item_condition_on_return"].map(condition_map).fillna(1).astype(int)

# product_category (one-hot)
df = pd.get_dummies(df, columns=["product_category"], prefix="cat", dtype=int)

# ── Drop original string columns (already encoded) ───────────────────────────
df.drop(columns=["return_reason","item_condition_on_return"], inplace=True)

# ── Separate label ────────────────────────────────────────────────────────────
y = df.pop("is_fraud")
X = df

print(f"Feature matrix shape : {X.shape}")
print(f"Label distribution   : {y.value_counts().to_dict()}")
print(f"\nFeature list ({len(X.columns)} features):")
for col in X.columns:
    print(f"  {col}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")
print(f"Train fraud rate : {y_train.mean():.1%}")
print(f"Test  fraud rate : {y_test.mean():.1%}")

# ── Save ──────────────────────────────────────────────────────────────────────
X_train.to_csv(os.path.join(OUT, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUT,  "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(OUT, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUT,  "y_test.csv"),  index=False)

# Save feature names for the backend to use
joblib.dump(list(X.columns), os.path.join(OUT, "feature_names.pkl"))

print(f"\n✓  Saved X_train, X_test, y_train, y_test, feature_names.pkl → {OUT}")
