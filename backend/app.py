"""
app.py
------
Flask REST API for the Return Fraud Detection System.

Endpoints:
  POST /api/predict          →  score a single return request
  POST /api/predict/batch    →  score multiple return requests
  GET  /api/health           →  health check
  GET  /api/stats            →  aggregate stats from DB (if connected)

The API loads the trained model once at startup and keeps it in memory.
PostgreSQL logging is optional — if DB env vars are not set, the API
runs in stateless mode (still scores, just doesn't persist).

Run:
    python app.py
    # or: flask run --port 5001
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify

# Add parent to path so we can import risk_scoring
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_scoring import compute_risk

# ── Load model & feature list at startup ──────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "../model/fraud_model.pkl")
FEAT_PATH   = os.path.join(BASE, "../features/feature_names.pkl")

print("Loading model...")
model         = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEAT_PATH)
print(f"✓  Model loaded. Expecting {len(feature_names)} features.")

app = Flask(__name__)


# ── Optional DB connection (graceful fallback if not configured) ──────────────
db_conn = None
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, "../.env"))

    import psycopg2
    db_conn = psycopg2.connect(
        host     = os.getenv("DB_HOST", "localhost"),
        port     = os.getenv("DB_PORT", 5432),
        dbname   = os.getenv("DB_NAME", "fraud_detection"),
        user     = os.getenv("DB_USER", "postgres"),
        password = os.getenv("DB_PASSWORD", ""),
    )
    print("✓  PostgreSQL connected.")
except Exception as e:
    print(f"⚠  DB not connected (stateless mode): {e}")


# ── Feature builder ───────────────────────────────────────────────────────────
CATEGORY_COLS = ["cat_accessories","cat_clothing","cat_electronics",
                 "cat_footwear","cat_home_decor"]

CONDITION_MAP = {
    "new_with_tags": 0, "new_without_tags": 1, "lightly_used": 2, "damaged": 3
}
REASON_MAP = {
    "size_issue": 0, "defective_product": 1, "wrong_item_sent": 2,
    "not_as_described": 3, "changed_mind": 4
}

def build_feature_vector(data: dict) -> pd.DataFrame:
    """
    Takes a raw return request (dict) and builds the exact feature
    vector the model was trained on.
    """
    f = {}

    # Raw pass-through features
    f["days_to_return_request"]      = int(data.get("days_to_return_request", 7))
    f["return_window_days"]          = int(data.get("return_window_days", 14))
    f["within_return_window"]        = int(f["days_to_return_request"] <= f["return_window_days"])
    f["order_value_inr"]             = float(data.get("order_value_inr", 1000))
    f["discount_applied"]            = int(data.get("discount_percent", 0) > 0)
    f["discount_percent"]            = int(data.get("discount_percent", 0))
    f["customer_total_orders"]       = int(data.get("customer_total_orders", 5))
    f["customer_past_returns"]       = int(data.get("customer_past_returns", 1))
    f["customer_return_rate"]        = float(data.get("customer_return_rate",
                                            f["customer_past_returns"] / max(f["customer_total_orders"], 1)))
    f["customer_account_age_days"]   = int(data.get("customer_account_age_days", 365))
    f["same_address_accounts"]       = int(data.get("same_address_accounts", 1))
    f["item_returned_matches_order"] = int(data.get("item_returned_matches_order", 1))
    f["delivery_confirmed"]          = int(data.get("delivery_confirmed", 1))
    f["claimed_non_delivery"]        = int(data.get("claimed_non_delivery", 0))
    f["return_pickup_attempted"]     = int(data.get("return_pickup_attempted", 1))
    f["return_pickup_successful"]    = int(data.get("return_pickup_successful", 1))
    f["return_request_weekday"]      = int(data.get("return_request_weekday", 2))
    f["return_after_weekend"]        = int(data.get("return_after_weekend", 0))
    f["refund_amount_inr"]           = float(data.get("refund_amount_inr", f["order_value_inr"] * 0.95))
    f["previous_fraud_flag"]         = int(data.get("previous_fraud_flag", 0))
    f["support_contacts_pre_return"] = int(data.get("support_contacts_pre_return", 0))

    # Derived features (must mirror engineer_features.py exactly)
    f["refund_ratio"]               = round(f["refund_amount_inr"] / max(f["order_value_inr"], 1), 4)
    f["high_value_high_returner"]   = int(f["order_value_inr"] > 3000 and f["customer_return_rate"] > 0.4)
    f["new_account_flag"]           = int(f["customer_account_age_days"] < 30)
    f["multi_account_flag"]         = int(f["same_address_accounts"] >= 3)
    f["false_non_delivery"]         = int(f["claimed_non_delivery"] == 1 and f["delivery_confirmed"] == 1)
    f["pickup_failed_flag"]         = int(f["return_pickup_attempted"] == 1 and f["return_pickup_successful"] == 0)
    f["discount_abuse_flag"]        = int(f["discount_percent"] >= 30 and f["customer_return_rate"] > 0.3)
    f["weekend_return_flag"]        = f["return_after_weekend"]
    f["pre_return_support_flag"]    = int(f["support_contacts_pre_return"] >= 2)

    # Encoded categoricals
    f["return_reason_enc"]   = REASON_MAP.get(data.get("return_reason", "size_issue"), 0)
    f["item_condition_enc"]  = CONDITION_MAP.get(data.get("item_condition_on_return", "new_with_tags"), 0)

    # One-hot product category
    cat = data.get("product_category", "clothing")
    for col in CATEGORY_COLS:
        f[col] = int(f"cat_{cat}" == col)

    # Build DataFrame in exact feature order
    row = pd.DataFrame([{feat: f.get(feat, 0) for feat in feature_names}])
    return row, f


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
def index():
    """Home page — lists available endpoints (avoids 404 on /)."""
    return jsonify({
        "service": "Return Fraud Detection API",
        "status":  "running",
        "endpoints": {
            "GET  /api/health":        "Health check",
            "POST /api/predict":       "Score a single return (JSON body)",
            "POST /api/predict/batch": "Score multiple returns (JSON array)",
            "GET  /api/stats":         "Aggregate stats (requires PostgreSQL)",
        },
        "docs": "See README.md for example curl requests",
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "model":      "fraud_model.pkl",
        "features":   len(feature_names),
        "db_connected": db_conn is not None,
        "timestamp":  datetime.utcnow().isoformat()
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        X, raw_features = build_feature_vector(data)
        probability     = float(model.predict_proba(X)[0][1])
        result          = compute_risk(probability, raw_features)

        response = {
            "order_id":     data.get("order_id", "unknown"),
            "risk_score":   result.risk_score,
            "risk_tier":    result.risk_tier,
            "probability":  result.probability,
            "action":       result.action,
            "signals":      result.signals,
            "scored_at":    datetime.utcnow().isoformat(),
        }

        # Persist to DB if connected
        if db_conn:
            _log_to_db(data, response)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    if not data or not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of return requests"}), 400

    results = []
    for item in data:
        try:
            X, raw_features = build_feature_vector(item)
            probability     = float(model.predict_proba(X)[0][1])
            result          = compute_risk(probability, raw_features)
            results.append({
                "order_id":    item.get("order_id", "unknown"),
                "risk_score":  result.risk_score,
                "risk_tier":   result.risk_tier,
                "probability": result.probability,
                "action":      result.action,
                "signals":     result.signals,
            })
        except Exception as e:
            results.append({"order_id": item.get("order_id","?"), "error": str(e)})

    return jsonify({"count": len(results), "results": results})


@app.route("/api/stats", methods=["GET"])
def stats():
    if not db_conn:
        return jsonify({"error": "Database not connected"}), 503
    try:
        cur = db_conn.cursor()
        cur.execute("""
            SELECT risk_tier, COUNT(*) as count, AVG(fraud_probability) as avg_prob
            FROM fraud_decisions
            GROUP BY risk_tier
        """)
        rows = cur.fetchall()
        return jsonify({
            "tiers": [{"tier": r[0], "count": r[1], "avg_probability": round(float(r[2]),3)} for r in rows]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _log_to_db(request_data: dict, response: dict):
    try:
        cur = db_conn.cursor()
        cur.execute("""
            INSERT INTO return_requests
            (order_id, customer_id, product_category, order_value_inr, days_to_return_request)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (
            request_data.get("order_id","?"),
            request_data.get("customer_id","?"),
            request_data.get("product_category","unknown"),
            request_data.get("order_value_inr", 0),
            request_data.get("days_to_return_request", 0),
        ))
        req_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO fraud_decisions
            (return_request_id, risk_score, risk_tier, fraud_probability, action_recommended, signals)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            req_id,
            response["risk_score"],
            response["risk_tier"],
            response["probability"],
            response["action"],
            json.dumps(response["signals"]),
        ))
        db_conn.commit()
    except Exception as e:
        db_conn.rollback()
        print(f"DB log error: {e}")


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    print(f"\n🚀  Starting Flask API on http://localhost:{port}")
    print(f"    POST /api/predict       →  score a return request")
    print(f"    POST /api/predict/batch →  score multiple requests")
    print(f"    GET  /api/health        →  health check\n")
    app.run(debug=True, port=port)
