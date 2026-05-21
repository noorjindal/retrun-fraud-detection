# Return Fraud Detection System

End-to-end ML system that scores e-commerce return requests for fraud risk. Built from scratch — synthetic dataset, custom features, imbalanced classification, Flask API, and Streamlit dashboard.

## Architecture

```
return-fraud/
├── data/                    # Layer 1 — Dataset
│   ├── generate_dataset.py
│   └── raw_returns.csv
├── features/                # Layer 2 — Feature Engineering
│   ├── engineer_features.py
│   ├── X_train.csv, X_test.csv
│   └── feature_names.pkl
├── model/                   # Layer 3–4 — Train & Evaluate
│   ├── train.py
│   ├── evaluate.py
│   ├── fraud_model.pkl
│   └── evaluation_plots.png
├── backend/                 # Layer 5 — Risk Engine + Flask API
│   ├── risk_scoring.py
│   └── app.py
├── dashboard/               # Layer 6 — Streamlit UI
│   └── streamlit_app.py
├── schema.sql               # PostgreSQL (optional)
├── run_pipeline.py
└── requirements.txt
```

## Fraud patterns modeled

| Pattern | Signal |
|---------|--------|
| Wardrobing | Post-weekend returns, high return rate |
| Item switching | Item doesn't match order, pickup failed |
| Refund fishing | Non-delivery claim on confirmed delivery |
| Account abuse | New account, multiple accounts at address |
| Price abuse | Heavy discount + frequent returns |

## Deploy (simplest)

**Free live dashboard:** see **[DEPLOY.md](DEPLOY.md)** — push to GitHub → Streamlit Community Cloud → done in ~5 min.

## Quick start

```bash
cd ~/Desktop/return-fraud
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-full.txt   # API + dashboard
# or: pip install -r requirements.txt  # dashboard only

# Run all ML layers (dataset → features → train → evaluate)
python run_pipeline.py --skip-data   # use existing CSV
# or full rebuild:
python run_pipeline.py

# Start API
python backend/app.py

# Start dashboard (new terminal)
streamlit run streamlit_app.py
```

## API

```bash
# Health check
curl http://localhost:5001/api/health

# Score a return
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD123456",
    "product_category": "clothing",
    "order_value_inr": 4500,
    "discount_percent": 40,
    "return_reason": "not_as_described",
    "days_to_return_request": 3,
    "return_window_days": 14,
    "item_condition_on_return": "lightly_used",
    "item_returned_matches_order": 0,
    "customer_total_orders": 5,
    "customer_past_returns": 4,
    "customer_return_rate": 0.8,
    "customer_account_age_days": 20,
    "same_address_accounts": 4,
    "claimed_non_delivery": 1,
    "delivery_confirmed": 1,
    "previous_fraud_flag": 1,
    "support_contacts_pre_return": 2
  }'
```

**Response:**
```json
{
  "risk_score": 78,
  "risk_tier": "HIGH",
  "probability": 0.78,
  "action": "Flag for manual review — hold refund",
  "signals": ["Claimed non-delivery on a confirmed-delivered order", "..."]
}
```

## Risk tiers

| Tier | Probability | Action |
|------|-------------|--------|
| LOW | < 0.30 | Auto-approve return |
| MEDIUM | 0.30 – 0.65 | Request verification |
| HIGH | ≥ 0.65 | Manual review, hold refund |

## Why not accuracy?

With ~85% legitimate returns, a model predicting "not fraud" always gets 85% accuracy. This project optimizes **F1**, and reports **precision**, **recall**, and **ROC-AUC**.

## PostgreSQL (optional)

```bash
createdb fraud_detection
psql -d fraud_detection -f schema.sql
cp .env.example .env   # fill in credentials
```

The API runs in stateless mode without a database.

## Interview talking points

1. **No public dataset** — built synthetic data encoding real fraud behaviors
2. **Class imbalance** — `class_weight='balanced'` + F1 scoring (not accuracy)
3. **Explainability** — risk signals list tells ops *why* a return was flagged
4. **Production shape** — REST API, optional DB persistence, dashboard demo

## Author

Noor Jindal
