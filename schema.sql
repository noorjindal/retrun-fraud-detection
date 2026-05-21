-- schema.sql
-- PostgreSQL schema for the Return Fraud Detection System
--
-- Run with:
--   psql -U postgres -d fraud_detection -f schema.sql

-- ── Return requests ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS return_requests (
    id                          SERIAL PRIMARY KEY,
    order_id                    VARCHAR(20)  NOT NULL,
    customer_id                 VARCHAR(20)  NOT NULL,
    product_category            VARCHAR(50),
    order_value_inr             NUMERIC(10,2),
    discount_percent            INT          DEFAULT 0,
    days_to_return_request      INT,
    return_reason               VARCHAR(50),
    item_condition_on_return    VARCHAR(30),
    item_returned_matches_order BOOLEAN,
    delivery_confirmed          BOOLEAN,
    claimed_non_delivery        BOOLEAN,
    return_pickup_attempted     BOOLEAN,
    return_pickup_successful    BOOLEAN,
    customer_return_rate        NUMERIC(5,3),
    customer_account_age_days   INT,
    same_address_accounts       INT          DEFAULT 1,
    previous_fraud_flag         BOOLEAN      DEFAULT FALSE,
    created_at                  TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Fraud decisions (model output) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fraud_decisions (
    id                  SERIAL PRIMARY KEY,
    return_request_id   INT          REFERENCES return_requests(id) ON DELETE CASCADE,
    risk_score          INT          CHECK (risk_score BETWEEN 0 AND 100),
    risk_tier           VARCHAR(10)  CHECK (risk_tier IN ('LOW','MEDIUM','HIGH')),
    fraud_probability   NUMERIC(6,4),
    action_recommended  TEXT,
    signals             TEXT,        -- JSON array stored as text
    model_version       VARCHAR(20)  DEFAULT '1.0',
    decided_at          TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Audit log ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          SERIAL PRIMARY KEY,
    event_type  VARCHAR(50),    -- e.g. 'RETURN_SUBMITTED', 'FRAUD_FLAGGED', 'APPROVED'
    order_id    VARCHAR(20),
    details     TEXT,
    logged_at   TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_return_customer   ON return_requests(customer_id);
CREATE INDEX IF NOT EXISTS idx_return_order      ON return_requests(order_id);
CREATE INDEX IF NOT EXISTS idx_decision_tier     ON fraud_decisions(risk_tier);
CREATE INDEX IF NOT EXISTS idx_decision_score    ON fraud_decisions(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event       ON audit_log(event_type);
