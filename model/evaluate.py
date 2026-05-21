"""
evaluate.py
-----------
Evaluates the trained model on the held-out test set.

Outputs:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix
  - ROC-AUC score
  - Saves: evaluation_report.txt, confusion_matrix.png, roc_curve.png

Why not just accuracy?
  With 85% legitimate / 15% fraud, a model that predicts "not fraud" every time
  gets 85% accuracy. Useless. We care about:
    Precision  →  of all flagged returns, how many are actually fraud?
    Recall     →  of all actual fraud, how many did we catch?
    ROC-AUC    →  overall discriminative ability across all thresholds

Run:
    python evaluate.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

BASE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(BASE, "../features")

print("Loading test data and model...")
X_test  = pd.read_csv(os.path.join(FEATURES, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(FEATURES, "y_test.csv")).squeeze()
model   = joblib.load(os.path.join(BASE, "fraud_model.pkl"))

y_pred      = model.predict(X_test)
y_proba     = model.predict_proba(X_test)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────
report  = classification_report(y_test, y_pred, target_names=["Legitimate","Fraud"])
roc_auc = roc_auc_score(y_test, y_proba)
ap      = average_precision_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n" + "="*55)
print("  EVALUATION REPORT")
print("="*55)
print(report)
print(f"  ROC-AUC Score          : {roc_auc:.4f}")
print(f"  Average Precision (AP) : {ap:.4f}")
print(f"\n  Confusion matrix:")
print(f"    True Negatives  (TN) : {tn}  (legit → legit ✓)")
print(f"    False Positives (FP) : {fp}  (legit → fraud ✗)")
print(f"    False Negatives (FN) : {fn}  (fraud → legit ✗)")
print(f"    True Positives  (TP) : {tp}  (fraud → fraud ✓)")
print("="*55)

# ── Save text report ──────────────────────────────────────────────────────────
report_path = os.path.join(BASE, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write("RETURN FRAUD DETECTION — EVALUATION REPORT\n")
    f.write("="*55 + "\n")
    f.write(report + "\n")
    f.write(f"ROC-AUC Score          : {roc_auc:.4f}\n")
    f.write(f"Average Precision (AP) : {ap:.4f}\n")
    f.write(f"\nConfusion matrix:\n")
    f.write(f"  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}\n")
print(f"\n✓  Saved evaluation_report.txt")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Legitimate","Fraud"]); ax.set_yticklabels(["Legitimate","Fraud"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
plt.colorbar(im, ax=ax)

# ── ROC curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2 = axes[1]
ax2.plot(fpr, tpr, color="#1A7A8A", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
ax2.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random classifier")
ax2.fill_between(fpr, tpr, alpha=0.08, color="#1A7A8A")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve", fontsize=13, fontweight="bold")
ax2.legend(loc="lower right")
ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])

plt.tight_layout()
plot_path = os.path.join(BASE, "evaluation_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓  Saved evaluation_plots.png")

# ── Precision-Recall curve ────────────────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(7, 5))
prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
ax3.plot(rec, prec, color="#1A7A8A", lw=2, label=f"AP = {ap:.3f}")
ax3.fill_between(rec, prec, alpha=0.08, color="#1A7A8A")
ax3.axhline(y=y_test.mean(), color="gray", linestyle="--", lw=1, label=f"Baseline ({y_test.mean():.2f})")
ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
ax3.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
ax3.legend()
plt.tight_layout()
pr_path = os.path.join(BASE, "pr_curve.png")
plt.savefig(pr_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓  Saved pr_curve.png")

print("\n  Done. Update README.md with the metrics above.")
