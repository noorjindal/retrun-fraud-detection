"""
generate_dataset.py
-------------------
Builds a synthetic return fraud dataset from scratch.

Why synthetic? There is no public real-time dataset for e-commerce return fraud.
This script encodes real fraud behavioral patterns as statistical distributions,
producing a dataset that reflects how fraud actually looks in the wild.

Fraud patterns modeled:
  1. Wardrobing       — buy, use, return (high return rate, post-weekend timing)
  2. Item switching   — return a different/damaged item than what was bought
  3. Refund fishing   — claim non-delivery on delivered orders
  4. Account abuse    — burst returns, multiple accounts at same address
  5. Price abuse      — buy on sale, return within full-price window

Run:
    python generate_dataset.py
    python generate_dataset.py --rows 20000 --fraud-rate 0.18 --seed 99
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("--rows",       type=int,   default=10000)
parser.add_argument("--fraud-rate", type=float, default=0.15)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--out",        type=str,   default="raw_returns.csv")
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

N          = args.rows
FRAUD_RATE = args.fraud_rate
N_FRAUD    = int(N * FRAUD_RATE)
N_LEGIT    = N - N_FRAUD

print(f"Generating {N} records  ({N_FRAUD} fraud / {N_LEGIT} legitimate)...")

CATEGORIES   = ["clothing", "electronics", "footwear", "accessories", "home_decor"]
CAT_WEIGHTS  = [0.40, 0.20, 0.20, 0.10, 0.10]

def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)

def random_dates(start_str, end_str, n):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    delta = (end - start).days
    return [start + timedelta(days=int(d)) for d in np.random.randint(0, delta, n)]


# ── LEGITIMATE ────────────────────────────────────────────────────────────────
def build_legitimate(n):
    d = {}
    d["customer_id"]             = [f"CUST{random.randint(10000,99999)}" for _ in range(n)]
    d["order_id"]                = [f"ORD{random.randint(100000,999999)}" for _ in range(n)]
    d["product_category"]        = np.random.choice(CATEGORIES, n, p=CAT_WEIGHTS)

    purchase_dates               = random_dates("2022-01-01", "2024-06-01", n)
    d["purchase_date"]           = purchase_dates
    days_to_return               = np.random.randint(3, 21, n)
    d["return_request_date"]     = [dt + timedelta(days=int(x)) for dt, x in zip(purchase_dates, days_to_return)]
    d["days_to_return_request"]  = days_to_return
    d["return_window_days"]      = np.random.choice([7, 14, 30], n, p=[0.2, 0.5, 0.3])
    d["within_return_window"]    = (d["days_to_return_request"] <= d["return_window_days"]).astype(int)

    d["order_value_inr"]         = clip(np.random.normal(1500, 800, n), 200, 15000).round(2)
    d["discount_applied"]        = np.random.choice([0, 1], n, p=[0.6, 0.4])
    d["discount_percent"]        = np.where(d["discount_applied"] == 1, np.random.randint(5, 25, n), 0)

    d["customer_total_orders"]       = np.random.randint(1, 30, n)
    d["customer_past_returns"]       = np.random.randint(0, 4, n)
    d["customer_return_rate"]        = clip(
        d["customer_past_returns"] / np.maximum(d["customer_total_orders"], 1), 0, 1).round(3)
    d["customer_account_age_days"]   = np.random.randint(90, 1800, n)
    d["same_address_accounts"]       = np.random.choice([1, 2], n, p=[0.92, 0.08])

    legit_reasons  = ["size_issue","defective_product","wrong_item_sent","not_as_described","changed_mind"]
    reason_weights = [0.35, 0.25, 0.20, 0.12, 0.08]
    d["return_reason"]               = np.random.choice(legit_reasons, n, p=reason_weights)

    d["item_returned_matches_order"] = np.random.choice([1, 0], n, p=[0.97, 0.03])
    d["item_condition_on_return"]    = np.random.choice(
        ["new_with_tags","new_without_tags","lightly_used","damaged"], n,
        p=[0.45, 0.30, 0.22, 0.03])

    d["delivery_confirmed"]          = np.random.choice([1, 0], n, p=[0.98, 0.02])
    d["claimed_non_delivery"]        = np.where(
        d["delivery_confirmed"] == 0, np.random.choice([0,1], n, p=[0.5,0.5]), np.zeros(n, dtype=int))
    d["return_pickup_attempted"]     = np.ones(n, dtype=int)
    d["return_pickup_successful"]    = np.random.choice([1, 0], n, p=[0.95, 0.05])

    d["return_request_weekday"]      = [dt.weekday() for dt in d["return_request_date"]]
    d["return_after_weekend"]        = [1 if dt.weekday() in [0,1] else 0 for dt in d["return_request_date"]]

    d["refund_amount_inr"]           = (d["order_value_inr"] * 0.95).round(2)
    d["previous_fraud_flag"]         = np.zeros(n, dtype=int)
    d["support_contacts_pre_return"] = np.random.choice([0,1,2], n, p=[0.70, 0.25, 0.05])
    d["is_fraud"]                    = np.zeros(n, dtype=int)
    d["fraud_type"]                  = "none"
    return pd.DataFrame(d)


# ── FRAUDULENT ────────────────────────────────────────────────────────────────
def build_fraudulent(n):
    fraud_types   = ["wardrobing","item_switching","refund_fishing","account_abuse","price_abuse"]
    fraud_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    ftype = np.random.choice(fraud_types, n, p=fraud_weights)

    d = {}
    d["customer_id"]             = [f"CUST{random.randint(10000,99999)}" for _ in range(n)]
    d["order_id"]                = [f"ORD{random.randint(100000,999999)}" for _ in range(n)]
    d["product_category"]        = np.where(
        np.isin(ftype, ["wardrobing"]),
        np.random.choice(["clothing","footwear"], n),
        np.random.choice(CATEGORIES, n, p=CAT_WEIGHTS))

    purchase_dates = random_dates("2022-01-01", "2024-06-01", n)
    d["purchase_date"] = purchase_dates
    days_to_return = np.where(ftype=="wardrobing", np.random.randint(2,8,n),
                     np.where(ftype=="refund_fishing", np.random.randint(1,5,n),
                              np.random.randint(1,29,n)))
    d["return_request_date"]     = [dt + timedelta(days=int(x)) for dt, x in zip(purchase_dates, days_to_return)]
    d["days_to_return_request"]  = days_to_return
    d["return_window_days"]      = np.random.choice([7,14,30], n, p=[0.2,0.5,0.3])
    d["within_return_window"]    = (d["days_to_return_request"] <= d["return_window_days"]).astype(int)

    d["order_value_inr"]         = clip(np.random.normal(3500, 1200, n), 500, 15000).round(2)
    d["discount_applied"]        = np.where(ftype=="price_abuse", np.ones(n,dtype=int),
                                            np.random.choice([0,1], n, p=[0.4,0.6]))
    d["discount_percent"]        = np.where(d["discount_applied"]==1,
                                   np.where(ftype=="price_abuse", np.random.randint(30,70,n),
                                            np.random.randint(10,50,n)), 0)

    d["customer_total_orders"]       = np.random.randint(2, 20, n)
    d["customer_past_returns"]       = np.where(ftype=="account_abuse",
                                                np.random.randint(5,15,n), np.random.randint(3,10,n))
    d["customer_return_rate"]        = clip(
        d["customer_past_returns"] / np.maximum(d["customer_total_orders"], 1), 0, 1).round(3)
    d["customer_account_age_days"]   = np.where(ftype=="account_abuse",
                                                np.random.randint(1,60,n), np.random.randint(10,400,n))
    d["same_address_accounts"]       = np.where(ftype=="account_abuse",
                                                np.random.randint(3,8,n),
                                                np.random.choice([1,2,3], n, p=[0.6,0.3,0.1]))

    d["return_reason"]               = np.where(ftype=="item_switching",
                                                np.random.choice(["not_as_described","defective_product"], n),
                                                np.random.choice(["not_as_described","changed_mind",
                                                                  "defective_product","size_issue",
                                                                  "wrong_item_sent"], n))
    d["item_returned_matches_order"] = np.where(ftype=="item_switching", np.zeros(n,dtype=int),
                                                np.random.choice([1,0], n, p=[0.60,0.40]))
    d["item_condition_on_return"]    = np.where(ftype=="wardrobing",
                                                np.random.choice(["lightly_used","new_without_tags"], n, p=[0.7,0.3]),
                                                np.random.choice(["new_with_tags","new_without_tags",
                                                                  "lightly_used","damaged"], n,
                                                                 p=[0.15,0.25,0.40,0.20]))

    d["delivery_confirmed"]          = np.where(ftype=="refund_fishing", np.ones(n,dtype=int),
                                                np.random.choice([1,0], n, p=[0.97,0.03]))
    d["claimed_non_delivery"]        = np.where(ftype=="refund_fishing", np.ones(n,dtype=int),
                                                np.random.choice([0,1], n, p=[0.85,0.15]))
    d["return_pickup_attempted"]     = np.random.choice([1,0], n, p=[0.85,0.15])
    d["return_pickup_successful"]    = np.where(d["return_pickup_attempted"]==1,
                                                np.random.choice([1,0], n, p=[0.70,0.30]),
                                                np.zeros(n, dtype=int))

    d["return_request_weekday"]      = np.where(ftype=="wardrobing",
                                                np.random.choice([0,1], n),
                                                [dt.weekday() for dt in d["return_request_date"]])
    d["return_after_weekend"]        = np.isin(d["return_request_weekday"], [0,1]).astype(int)

    d["refund_amount_inr"]           = (d["order_value_inr"] * np.random.uniform(0.9,1.0,n)).round(2)
    d["previous_fraud_flag"]         = np.random.choice([0,1], n, p=[0.55,0.45])
    d["support_contacts_pre_return"] = np.random.choice([0,1,2,3], n, p=[0.30,0.30,0.25,0.15])
    d["is_fraud"]                    = np.ones(n, dtype=int)
    d["fraud_type"]                  = ftype
    return pd.DataFrame(d)


# ── Combine ───────────────────────────────────────────────────────────────────
legit = build_legitimate(N_LEGIT)
fraud = build_fraudulent(N_FRAUD)
df    = pd.concat([legit, fraud], ignore_index=True)
df    = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
df.to_csv(out_path, index=False)

print(f"\n✓  {len(df)} rows → {out_path}")
print(f"   Fraud rate : {df['is_fraud'].mean():.1%}")
print(f"   Columns    : {len(df.columns)}")
print(f"\n   Fraud type breakdown:")
for ft, cnt in df[df.is_fraud==1]["fraud_type"].value_counts().items():
    print(f"     {ft:<20} {cnt}")
print(f"\n   Return rate  legit={df[df.is_fraud==0]['customer_return_rate'].mean():.2f}  "
      f"fraud={df[df.is_fraud==1]['customer_return_rate'].mean():.2f}")
