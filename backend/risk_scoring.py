"""
risk_scoring.py
---------------
Takes a fraud probability from the model and converts it into:
  - A risk tier  : LOW / MEDIUM / HIGH
  - A risk score : 0–100
  - A recommended action
  - A breakdown of which signals contributed most

Thresholds are configurable and should be tuned based on
business cost trade-offs (cost of false positive vs false negative).
"""

from dataclasses import dataclass
from typing import Literal

# ── Thresholds ────────────────────────────────────────────────────────────────
# Tune these based on your acceptable false-positive rate
THRESHOLD_LOW    = 0.30   # below this → LOW risk, auto-approve
THRESHOLD_HIGH   = 0.65   # above this → HIGH risk, flag for manual review


RiskTier = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class RiskResult:
    risk_score:  int          # 0–100
    risk_tier:   RiskTier
    probability: float        # raw model output
    action:      str
    signals:     list[str]    # human-readable explanation


def score_to_tier(probability: float) -> RiskTier:
    if probability < THRESHOLD_LOW:
        return "LOW"
    elif probability < THRESHOLD_HIGH:
        return "MEDIUM"
    return "HIGH"


def tier_to_action(tier: RiskTier) -> str:
    return {
        "LOW":    "Auto-approve return request",
        "MEDIUM": "Request additional verification (photo proof / OTP)",
        "HIGH":   "Flag for manual review — hold refund",
    }[tier]


def extract_signals(features: dict) -> list[str]:
    """
    Converts raw feature values into human-readable risk signals.
    These appear in the dashboard and API response so ops teams
    understand WHY a return was flagged.
    """
    signals = []

    if features.get("false_non_delivery", 0):
        signals.append("Claimed non-delivery on a confirmed-delivered order")

    if features.get("item_returned_matches_order", 1) == 0:
        signals.append("Returned item does not match original order")

    if features.get("customer_return_rate", 0) > 0.5:
        rate = features["customer_return_rate"]
        signals.append(f"High customer return rate ({rate:.0%})")

    if features.get("multi_account_flag", 0):
        n = features.get("same_address_accounts", "multiple")
        signals.append(f"Multiple accounts at same address ({n})")

    if features.get("new_account_flag", 0):
        age = features.get("customer_account_age_days", "?")
        signals.append(f"Very new account ({age} days old)")

    if features.get("previous_fraud_flag", 0):
        signals.append("Account has previous fraud history")

    if features.get("pickup_failed_flag", 0):
        signals.append("Return pickup was attempted but failed")

    if features.get("discount_abuse_flag", 0):
        signals.append("Heavy discount purchase with high return rate")

    if features.get("return_after_weekend", 0) and features.get("customer_return_rate", 0) > 0.3:
        signals.append("Post-weekend return pattern (possible wardrobing)")

    if features.get("pre_return_support_flag", 0):
        n = features.get("support_contacts_pre_return", "multiple")
        signals.append(f"Multiple support contacts before return ({n})")

    if features.get("high_value_high_returner", 0):
        val = features.get("order_value_inr", "?")
        signals.append(f"High-value order (₹{val}) from frequent returner")

    if not signals:
        signals.append("No strong individual signals — low composite risk")

    return signals


def compute_risk(probability: float, features: dict) -> RiskResult:
    tier    = score_to_tier(probability)
    score   = int(round(probability * 100))
    action  = tier_to_action(tier)
    signals = extract_signals(features)

    return RiskResult(
        risk_score  = score,
        risk_tier   = tier,
        probability = round(probability, 4),
        action      = action,
        signals     = signals,
    )
