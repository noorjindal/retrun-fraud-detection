"""
streamlit_app.py
----------------
Interactive fraud risk dashboard.

Features:
  - Live scoring: enter a return request and get a risk score instantly
  - Batch upload: upload a CSV to score many returns at once
  - Model insights: feature importances, score distribution
  - Dataset explorer: browse the generated dataset with filters

Run:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# Add paths
BASE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.join(BASE, "..")
MODEL_DIR = os.path.join(ROOT, "model")
FEAT_DIR  = os.path.join(ROOT, "features")
DATA_DIR  = os.path.join(ROOT, "data")

sys.path.insert(0, os.path.join(ROOT, "backend"))
from risk_scoring import compute_risk

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
    feature_names = joblib.load(os.path.join(FEAT_DIR,  "feature_names.pkl"))
    return model, feature_names

@st.cache_data
def load_dataset():
    path = os.path.join(DATA_DIR, "raw_returns.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_importances():
    path = os.path.join(MODEL_DIR, "feature_importances.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, header=0).squeeze()
    return None

# ── Feature builder (mirrors backend/app.py) ──────────────────────────────────
CATEGORY_COLS = ["cat_accessories","cat_clothing","cat_electronics",
                 "cat_footwear","cat_home_decor"]
CONDITION_MAP = {"new_with_tags":0,"new_without_tags":1,"lightly_used":2,"damaged":3}
REASON_MAP    = {"size_issue":0,"defective_product":1,"wrong_item_sent":2,
                 "not_as_described":3,"changed_mind":4}

def build_vector(data, feature_names):
    f = {}
    f["days_to_return_request"]      = int(data.get("days_to_return_request", 7))
    f["return_window_days"]          = int(data.get("return_window_days", 14))
    f["within_return_window"]        = int(f["days_to_return_request"] <= f["return_window_days"])
    f["order_value_inr"]             = float(data.get("order_value_inr", 1000))
    f["discount_applied"]            = int(data.get("discount_percent", 0) > 0)
    f["discount_percent"]            = int(data.get("discount_percent", 0))
    f["customer_total_orders"]       = int(data.get("customer_total_orders", 5))
    f["customer_past_returns"]       = int(data.get("customer_past_returns", 1))
    f["customer_return_rate"]        = float(data.get("customer_return_rate",
                                            f["customer_past_returns"] / max(f["customer_total_orders"],1)))
    f["customer_account_age_days"]   = int(data.get("customer_account_age_days", 365))
    f["same_address_accounts"]       = int(data.get("same_address_accounts", 1))
    f["item_returned_matches_order"] = int(data.get("item_returned_matches_order", 1))
    f["delivery_confirmed"]          = int(data.get("delivery_confirmed", 1))
    f["claimed_non_delivery"]        = int(data.get("claimed_non_delivery", 0))
    f["return_pickup_attempted"]     = int(data.get("return_pickup_attempted", 1))
    f["return_pickup_successful"]    = int(data.get("return_pickup_successful", 1))
    f["return_request_weekday"]      = int(data.get("return_request_weekday", 2))
    f["return_after_weekend"]        = int(data.get("return_after_weekend", 0))
    f["refund_amount_inr"]           = float(data.get("refund_amount_inr", f["order_value_inr"]*0.95))
    f["previous_fraud_flag"]         = int(data.get("previous_fraud_flag", 0))
    f["support_contacts_pre_return"] = int(data.get("support_contacts_pre_return", 0))
    f["refund_ratio"]                = round(f["refund_amount_inr"] / max(f["order_value_inr"],1), 4)
    f["high_value_high_returner"]    = int(f["order_value_inr"]>3000 and f["customer_return_rate"]>0.4)
    f["new_account_flag"]            = int(f["customer_account_age_days"]<30)
    f["multi_account_flag"]          = int(f["same_address_accounts"]>=3)
    f["false_non_delivery"]          = int(f["claimed_non_delivery"]==1 and f["delivery_confirmed"]==1)
    f["pickup_failed_flag"]          = int(f["return_pickup_attempted"]==1 and f["return_pickup_successful"]==0)
    f["discount_abuse_flag"]         = int(f["discount_percent"]>=30 and f["customer_return_rate"]>0.3)
    f["weekend_return_flag"]         = f["return_after_weekend"]
    f["pre_return_support_flag"]     = int(f["support_contacts_pre_return"]>=2)
    f["return_reason_enc"]           = REASON_MAP.get(data.get("return_reason","size_issue"), 0)
    f["item_condition_enc"]          = CONDITION_MAP.get(data.get("item_condition_on_return","new_with_tags"), 0)
    cat = data.get("product_category","clothing")
    for col in CATEGORY_COLS:
        f[col] = int(f"cat_{cat}" == col)
    row = pd.DataFrame([{feat: f.get(feat, 0) for feat in feature_names}])
    return row, f

def tier_color(tier):
    return {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}.get(tier, "#888")

def score_gauge(score, tier):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    color = tier_color(tier)
    ax.barh(0, score,   height=0.4, color=color,   alpha=0.85)
    ax.barh(0, 100-score, height=0.4, left=score, color="#e9ecef")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Risk Score (0–100)", fontsize=9)
    ax.axvline(30, color="#2ecc71", lw=1, linestyle="--", alpha=0.6)
    ax.axvline(65, color="#e74c3c", lw=1, linestyle="--", alpha=0.6)
    ax.text(score/2, 0, f"{score}", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.set_title(f"Risk Tier: {tier}", fontsize=11, fontweight="bold", color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Return Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.risk-low    { background:#d5f5e3; border-left:5px solid #2ecc71; padding:12px; border-radius:6px; }
.risk-medium { background:#fef9e7; border-left:5px solid #f39c12; padding:12px; border-radius:6px; }
.risk-high   { background:#fde8e8; border-left:5px solid #e74c3c; padding:12px; border-radius:6px; }
.signal-item { background:#f8f9fa; padding:6px 10px; border-radius:4px; margin:3px 0; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Return Fraud Detection")
    st.markdown("---")
    page = st.radio("Navigate", ["Live Scoring", "Batch Scoring", "Model Insights", "Dataset Explorer"])
    st.markdown("---")
    st.caption("Built by Noor Jindal\nModel: Random Forest\nFeatures: 30+")

try:
    model, feature_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found. Run `python model/train.py` first.\n\n{e}")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE SCORING
# ════════════════════════════════════════════════════════════════════════════════
if page == "Live Scoring":
    st.title("Score a Return Request")
    st.caption("Fill in the return details below and get an instant fraud risk assessment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Order Details")
        order_id         = st.text_input("Order ID", value="ORD123456")
        product_category = st.selectbox("Product Category",
                            ["clothing","electronics","footwear","accessories","home_decor"])
        order_value      = st.number_input("Order Value (₹)", min_value=100, max_value=50000, value=2500)
        discount_percent = st.slider("Discount (%)", 0, 80, 0)
        return_reason    = st.selectbox("Return Reason",
                            ["size_issue","defective_product","wrong_item_sent",
                             "not_as_described","changed_mind"])

    with col2:
        st.subheader("Return Behaviour")
        days_to_return        = st.slider("Days to Return Request", 1, 60, 7)
        return_window         = st.selectbox("Return Window (days)", [7, 14, 30], index=1)
        item_condition        = st.selectbox("Item Condition on Return",
                                ["new_with_tags","new_without_tags","lightly_used","damaged"])
        item_matches          = st.checkbox("Item matches original order", value=True)
        return_after_weekend  = st.checkbox("Return requested Mon/Tue (post-weekend)", value=False)
        pickup_successful     = st.checkbox("Return pickup successful", value=True)

    with col3:
        st.subheader("Customer History")
        total_orders         = st.number_input("Total Orders",       min_value=1,  max_value=200, value=8)
        past_returns         = st.number_input("Past Returns",        min_value=0,  max_value=100, value=1)
        account_age          = st.number_input("Account Age (days)",  min_value=1,  max_value=3650, value=365)
        same_addr_accounts   = st.number_input("Accounts at Address", min_value=1,  max_value=10, value=1)
        claimed_non_delivery = st.checkbox("Claimed non-delivery",    value=False)
        delivery_confirmed   = st.checkbox("Delivery confirmed",      value=True)
        previous_fraud       = st.checkbox("Previous fraud flag",     value=False)
        support_contacts     = st.number_input("Support contacts pre-return", min_value=0, max_value=10, value=0)

    st.markdown("---")
    if st.button("🔍  Score This Return", use_container_width=True, type="primary"):
        return_rate = past_returns / max(total_orders, 1)

        inp = {
            "order_id":                    order_id,
            "product_category":            product_category,
            "order_value_inr":             order_value,
            "discount_percent":            discount_percent,
            "return_reason":               return_reason,
            "days_to_return_request":      days_to_return,
            "return_window_days":          return_window,
            "item_condition_on_return":    item_condition,
            "item_returned_matches_order": int(item_matches),
            "return_after_weekend":        int(return_after_weekend),
            "return_pickup_attempted":     1,
            "return_pickup_successful":    int(pickup_successful),
            "customer_total_orders":       total_orders,
            "customer_past_returns":       past_returns,
            "customer_return_rate":        return_rate,
            "customer_account_age_days":   account_age,
            "same_address_accounts":       same_addr_accounts,
            "claimed_non_delivery":        int(claimed_non_delivery),
            "delivery_confirmed":          int(delivery_confirmed),
            "previous_fraud_flag":         int(previous_fraud),
            "support_contacts_pre_return": support_contacts,
            "refund_amount_inr":           order_value * 0.95,
            "return_request_weekday":      0 if return_after_weekend else 2,
        }

        X, raw_f     = build_vector(inp, feature_names)
        probability  = float(model.predict_proba(X)[0][1])
        result       = compute_risk(probability, raw_f)

        tier_class = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[result.risk_tier]

        r1, r2 = st.columns([1, 2])

        with r1:
            fig = score_gauge(result.risk_score, result.risk_tier)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.metric("Fraud Probability", f"{result.probability:.1%}")

        with r2:
            st.markdown(f'<div class="{tier_class}"><strong>Recommended Action</strong><br>{result.action}</div>',
                        unsafe_allow_html=True)
            st.markdown("**Risk Signals Detected:**")
            for signal in result.signals:
                st.markdown(f'<div class="signal-item">⚠ {signal}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH SCORING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Batch Scoring":
    st.title("Batch Return Scoring")
    st.caption("Upload a CSV of return requests. The model will score each one.")

    with st.expander("Required CSV columns"):
        st.code("""order_id, product_category, order_value_inr, discount_percent,
return_reason, days_to_return_request, return_window_days,
item_condition_on_return, item_returned_matches_order,
customer_total_orders, customer_past_returns, customer_account_age_days,
same_address_accounts, claimed_non_delivery, delivery_confirmed,
previous_fraud_flag, support_contacts_pre_return""")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.info(f"Loaded {len(df_upload)} rows")

        results = []
        bar = st.progress(0)
        for i, row in df_upload.iterrows():
            try:
                X, raw_f    = build_vector(row.to_dict(), feature_names)
                prob        = float(model.predict_proba(X)[0][1])
                result      = compute_risk(prob, raw_f)
                results.append({
                    "order_id":   row.get("order_id","?"),
                    "risk_score": result.risk_score,
                    "risk_tier":  result.risk_tier,
                    "probability":result.probability,
                    "action":     result.action,
                    "signals":    " | ".join(result.signals),
                })
            except Exception as e:
                results.append({"order_id": row.get("order_id","?"), "error": str(e)})
            bar.progress((i+1) / len(df_upload))

        df_results = pd.DataFrame(results)
        bar.empty()

        # Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",        len(df_results))
        c2.metric("🟢 LOW",       (df_results["risk_tier"]=="LOW").sum())
        c3.metric("🟡 MEDIUM",    (df_results["risk_tier"]=="MEDIUM").sum())
        c4.metric("🔴 HIGH",      (df_results["risk_tier"]=="HIGH").sum())

        # Color the tier column
        def color_tier(val):
            colors = {"LOW":"#d5f5e3","MEDIUM":"#fef9e7","HIGH":"#fde8e8"}
            return f"background-color: {colors.get(val,'')}"

        st.dataframe(
            df_results.style.map(color_tier, subset=["risk_tier"]),
            use_container_width=True
        )

        csv = df_results.to_csv(index=False)
        st.download_button("⬇ Download Results CSV", csv, "fraud_scores.csv", "text/csv")

    else:
        # Let them test with the generated dataset
        dataset = load_dataset()
        if dataset is not None:
            if st.button("Score sample from generated dataset (50 rows)"):
                sample = dataset.sample(50, random_state=7)
                results = []
                for _, row in sample.iterrows():
                    try:
                        X, raw_f = build_vector(row.to_dict(), feature_names)
                        prob     = float(model.predict_proba(X)[0][1])
                        result   = compute_risk(prob, raw_f)
                        results.append({
                            "order_id":     row.get("order_id","?"),
                            "actual_fraud": int(row.get("is_fraud",0)),
                            "risk_tier":    result.risk_tier,
                            "risk_score":   result.risk_score,
                            "probability":  result.probability,
                        })
                    except:
                        pass
                df_r = pd.DataFrame(results)
                st.dataframe(df_r, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":
    st.title("Model Insights")

    importances = load_importances()

    if importances is not None:
        st.subheader("Feature Importances (Top 20)")
        top20 = importances.head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#1A7A8A"] * len(top20)
        ax.barh(top20.index[::-1], top20.values[::-1], color=colors, alpha=0.85)
        ax.set_xlabel("Importance Score")
        ax.set_title("What drives the fraud prediction?", fontsize=12, fontweight="bold")
        for i, (val, name) in enumerate(zip(top20.values[::-1], top20.index[::-1])):
            ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.warning("Feature importances not found. Run `python model/train.py` first.")

    # Show eval plots if they exist
    eval_plot = os.path.join(ROOT, "model", "evaluation_plots.png")
    pr_plot   = os.path.join(ROOT, "model", "pr_curve.png")

    if os.path.exists(eval_plot):
        st.subheader("Evaluation Plots")
        st.image(eval_plot, use_container_width=True)

    if os.path.exists(pr_plot):
        st.subheader("Precision-Recall Curve")
        st.image(pr_plot, use_column_width=True)

    # Show text report
    report_path = os.path.join(ROOT, "model", "evaluation_report.txt")
    if os.path.exists(report_path):
        st.subheader("Classification Report")
        with open(report_path) as f:
            st.code(f.read())


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.caption("Browse and filter the generated dataset.")

    dataset = load_dataset()
    if dataset is None:
        st.error("Dataset not found. Run `python data/generate_dataset.py` first.")
        st.stop()

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        fraud_filter = st.selectbox("Label", ["All","Fraud only","Legitimate only"])
    with f2:
        cat_filter   = st.multiselect("Category", dataset["product_category"].unique().tolist(),
                                       default=dataset["product_category"].unique().tolist())
    with f3:
        tier_filter  = st.selectbox("Fraud type", ["All"] + dataset["fraud_type"].unique().tolist())

    df_view = dataset.copy()
    if fraud_filter == "Fraud only":
        df_view = df_view[df_view["is_fraud"] == 1]
    elif fraud_filter == "Legitimate only":
        df_view = df_view[df_view["is_fraud"] == 0]
    if cat_filter:
        df_view = df_view[df_view["product_category"].isin(cat_filter)]
    if tier_filter != "All":
        df_view = df_view[df_view["fraud_type"] == tier_filter]

    st.info(f"Showing {len(df_view)} rows")

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           len(df_view))
    c2.metric("Fraud Rate",     f"{df_view['is_fraud'].mean():.1%}")
    c3.metric("Avg Order ₹",    f"₹{df_view['order_value_inr'].mean():,.0f}")
    c4.metric("Avg Return Rate",f"{df_view['customer_return_rate'].mean():.2f}")

    st.dataframe(df_view.head(200), use_container_width=True)

    # Return rate distribution
    st.subheader("Return Rate: Fraud vs Legitimate")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    legit = df_view[df_view["is_fraud"]==0]["customer_return_rate"]
    fraud = df_view[df_view["is_fraud"]==1]["customer_return_rate"]
    ax2.hist(legit, bins=30, alpha=0.6, color="#1A7A8A", label="Legitimate")
    ax2.hist(fraud, bins=30, alpha=0.6, color="#e74c3c", label="Fraud")
    ax2.set_xlabel("Customer Return Rate"); ax2.set_ylabel("Count")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
