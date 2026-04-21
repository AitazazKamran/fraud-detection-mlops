# pipeline/cost_sensitive_learning.py
"""
Task 4: Cost-Sensitive Learning
- Assign higher penalty to false negatives (missed fraud)
- Compare standard vs cost-sensitive training
- Analyze business impact: fraud loss vs false alarms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & Preprocess ────────────────────────────────────────────────────
print("Loading data...")
trans = pd.read_csv("data/train_transaction.csv")
ident = pd.read_csv("data/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")

# Drop high-missing columns (>90%)
missing_pct = df.isnull().mean()
drop_cols = missing_pct[missing_pct > 0.9].index.tolist()
df.drop(columns=drop_cols, inplace=True)

# Separate features
X = df.drop(columns=["isFraud", "TransactionID"])
y = df["isFraud"]

# Identify column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Impute
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Encode categoricals
for col in cat_cols:
    if X[col].nunique() > 50:
        means = df.groupby(col)["isFraud"].mean()
        X[col] = X[col].map(means).fillna(0)
    else:
        X[col] = pd.factorize(X[col])[0]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Undersample (winner from Task 2)
sampler = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
print(f"Training shape after resampling: {X_train_res.shape}")

# ── 2. Business Cost Parameters ─────────────────────────────────────────────
# These are simplified assumptions for analysis:
# - Average fraud transaction: $500 loss if missed (false negative)
# - Cost of investigating a false alarm: $10 per case
AVG_FRAUD_LOSS = 500   # $ lost per missed fraud
FALSE_ALARM_COST = 10  # $ spent investigating per false positive

def compute_business_cost(cm, fraud_loss=AVG_FRAUD_LOSS, fa_cost=FALSE_ALARM_COST):
    """
    cm: confusion matrix [[TN, FP], [FN, TP]]
    Returns total estimated business cost
    """
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fn * fraud_loss) + (fp * fa_cost)
    return total_cost, fn, fp

# ── 3. Train & Evaluate Function ─────────────────────────────────────────────
def train_evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    auc   = roc_auc_score(y_te, y_prob)
    cm    = confusion_matrix(y_te, y_pred)
    rep   = classification_report(y_te, y_pred, output_dict=True)
    recall    = rep["1"]["recall"]
    precision = rep["1"]["precision"]
    f1        = rep["1"]["f1-score"]
    cost, fn, fp = compute_business_cost(cm)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_te, y_pred))
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n  💰 Business Cost Analysis:")
    print(f"     False Negatives (missed fraud) : {fn:,}  →  ${fn * AVG_FRAUD_LOSS:,.0f} loss")
    print(f"     False Positives (false alarms) : {fp:,}  →  ${fp * FALSE_ALARM_COST:,.0f} cost")
    print(f"     Total Estimated Cost           : ${cost:,.0f}")

    return {
        "name": name, "model": model,
        "auc": auc, "recall": recall,
        "precision": precision, "f1": f1,
        "cost": cost, "fn": fn, "fp": fp,
        "y_prob": y_prob, "cm": cm
    }

# ── 4. Experiments: Standard vs Cost-Sensitive ───────────────────────────────
results = []

# --- XGBoost: Standard (scale_pos_weight=1) ---
print("\n\n>>> XGBoost — Standard Training (no cost weighting)")
xgb_std = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=1,          # no extra penalty
    use_label_encoder=False, eval_metric="auc",
    random_state=42, n_jobs=-1
)
results.append(train_evaluate(
    "XGBoost — Standard",
    xgb_std, X_train_res, y_train_res, X_test, y_test
))

# --- XGBoost: Cost-Sensitive (scale_pos_weight=5) ---
# scale_pos_weight = ratio of negatives to positives * extra weight
# After undersampling ratio ≈ 1:1, so spw=5 means 5× penalty for FN
print("\n\n>>> XGBoost — Cost-Sensitive (scale_pos_weight=5)")
xgb_cs5 = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=5,
    use_label_encoder=False, eval_metric="auc",
    random_state=42, n_jobs=-1
)
results.append(train_evaluate(
    "XGBoost — Cost-Sensitive (spw=5)",
    xgb_cs5, X_train_res, y_train_res, X_test, y_test
))

# --- XGBoost: Cost-Sensitive (scale_pos_weight=10) ---
print("\n\n>>> XGBoost — Cost-Sensitive (scale_pos_weight=10)")
xgb_cs10 = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=10,
    use_label_encoder=False, eval_metric="auc",
    random_state=42, n_jobs=-1
)
results.append(train_evaluate(
    "XGBoost — Cost-Sensitive (spw=10)",
    xgb_cs10, X_train_res, y_train_res, X_test, y_test
))

# --- LightGBM: Standard ---
print("\n\n>>> LightGBM — Standard Training")
lgb_std = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    class_weight=None,           # no extra penalty
    random_state=42, n_jobs=-1, verbose=-1
)
results.append(train_evaluate(
    "LightGBM — Standard",
    lgb_std, X_train_res, y_train_res, X_test, y_test
))

# --- LightGBM: Cost-Sensitive (class_weight balanced) ---
print("\n\n>>> LightGBM — Cost-Sensitive (class_weight='balanced')")
lgb_cs = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    class_weight="balanced",     # auto-weighs by inverse frequency
    random_state=42, n_jobs=-1, verbose=-1
)
results.append(train_evaluate(
    "LightGBM — Cost-Sensitive (balanced)",
    lgb_cs, X_train_res, y_train_res, X_test, y_test
))

# --- LightGBM: Cost-Sensitive (manual scale_pos_weight=10) ---
print("\n\n>>> LightGBM — Cost-Sensitive (scale_pos_weight=10)")
lgb_cs10 = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=10,
    random_state=42, n_jobs=-1, verbose=-1
)
results.append(train_evaluate(
    "LightGBM — Cost-Sensitive (spw=10)",
    lgb_cs10, X_train_res, y_train_res, X_test, y_test
))

# ── 5. Save Best Cost-Sensitive Model ────────────────────────────────────────
# Pick the one with lowest business cost (most important for fraud system)
best = min(results, key=lambda r: r["cost"])
print(f"\n\n✅ Best model by business cost: {best['name']}")
print(f"   Business Cost: ${best['cost']:,.0f}  |  Recall: {best['recall']:.4f}  |  AUC: {best['auc']:.4f}")
joblib.dump(best["model"], "models/cost_sensitive_best_model.pkl")
print("Saved: models/cost_sensitive_best_model.pkl")

# ── 6. Visualization ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

names      = [r["name"] for r in results]
short_names = [n.replace("XGBoost — ", "XGB\n").replace("LightGBM — ", "LGB\n") for n in names]
aucs       = [r["auc"]       for r in results]
recalls    = [r["recall"]    for r in results]
precisions = [r["precision"] for r in results]
costs      = [r["cost"]      for r in results]
fns        = [r["fn"]        for r in results]
fps        = [r["fp"]        for r in results]

colors_std = ["#5B9BD5", "#5B9BD5", "#5B9BD5", "#ED7D31", "#ED7D31", "#ED7D31"]
colors_cs  = ["#A9D18E" if "Cost" in n else "#C9C9C9" for n in names]

# — Plot 1: AUC Comparison —
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(short_names, aucs, color=colors_std, edgecolor="white", linewidth=0.8)
ax1.set_title("AUC-ROC Comparison", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("AUC-ROC")
ax1.set_ylim(0.88, 0.96)
ax1.axhline(0.85, color="red", linestyle="--", linewidth=1, label="Deploy threshold (0.85)")
ax1.legend(fontsize=8)
for bar, val in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax1.tick_params(axis="x", labelsize=7)

# — Plot 2: Recall Comparison —
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(short_names, recalls, color=colors_std, edgecolor="white", linewidth=0.8)
ax2.set_title("Fraud Recall Comparison\n(Higher = fewer missed frauds)", fontsize=13, fontweight="bold", pad=10)
ax2.set_ylabel("Recall (Fraud Class)")
ax2.set_ylim(0.75, 1.02)
for bar, val in zip(bars, recalls):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax2.tick_params(axis="x", labelsize=7)

# — Plot 3: Business Cost Comparison —
ax3 = fig.add_subplot(gs[0, 2])
bar_colors = ["#2E7D32" if c == min(costs) else "#C62828" for c in costs]
bars = ax3.bar(short_names, [c/1e6 for c in costs], color=bar_colors, edgecolor="white", linewidth=0.8)
ax3.set_title("Estimated Business Cost\n(Lower = better)", fontsize=13, fontweight="bold", pad=10)
ax3.set_ylabel("Cost ($ Millions)")
for bar, val in zip(bars, costs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"${val/1e6:.2f}M", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax3.tick_params(axis="x", labelsize=7)

# — Plot 4: Precision-Recall Tradeoff —
ax4 = fig.add_subplot(gs[1, 0])
for r in results:
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, r["y_prob"])
    ax4.plot(rec_curve, prec_curve, label=r["name"].replace(" — ", "\n"), linewidth=1.5)
ax4.set_title("Precision–Recall Curves", fontsize=13, fontweight="bold", pad=10)
ax4.set_xlabel("Recall")
ax4.set_ylabel("Precision")
ax4.legend(fontsize=6.5, loc="upper right")
ax4.grid(True, alpha=0.3)

# — Plot 5: FN vs FP Tradeoff —
ax5 = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(names))
width = 0.35
ax5.bar(x_pos - width/2, fns, width, label="False Negatives\n(missed fraud)", color="#C62828", alpha=0.85)
ax5.bar(x_pos + width/2, fps, width, label="False Positives\n(false alarms)", color="#1565C0", alpha=0.85)
ax5.set_title("FN vs FP per Model\n(FN is more costly)", fontsize=13, fontweight="bold", pad=10)
ax5.set_ylabel("Count")
ax5.set_xticks(x_pos)
ax5.set_xticklabels(short_names, fontsize=7)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")

# — Plot 6: Summary Table —
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
table_data = [
    [r["name"].replace(" — ", "\n"), f"{r['auc']:.4f}",
     f"{r['recall']:.4f}", f"{r['precision']:.4f}",
     f"${r['cost']/1e6:.2f}M"]
    for r in results
]
col_labels = ["Model", "AUC", "Recall", "Prec.", "Cost"]
tbl = ax6.table(
    cellText=table_data, colLabels=col_labels,
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.1, 1.8)
ax6.set_title("Summary Table", fontsize=13, fontweight="bold", pad=10)

fig.suptitle(
    "Task 4: Cost-Sensitive Learning — Standard vs Cost-Sensitive Training\n"
    f"Assumptions: Fraud loss=${AVG_FRAUD_LOSS}/case | False alarm cost=${FALSE_ALARM_COST}/case",
    fontsize=14, fontweight="bold", y=1.01
)

plt.savefig("models/cost_sensitive_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: models/cost_sensitive_comparison.png")
plt.close()

# ── 7. Final Summary ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TASK 4 FINAL COMPARISON")
print("="*65)
print(f"{'Model':<38} {'AUC':>6}  {'Recall':>7}  {'Prec':>6}  {'Cost':>10}")
print("-"*65)
for r in results:
    marker = " ✅" if r["name"] == best["name"] else ""
    print(f"{r['name']:<38} {r['auc']:.4f}  {r['recall']:.4f}  {r['precision']:.4f}  ${r['cost']/1e6:.2f}M{marker}")

print(f"\n💡 Business Insight:")
print(f"   With ${AVG_FRAUD_LOSS} fraud loss and ${FALSE_ALARM_COST} false alarm cost,")
print(f"   reducing false negatives is 50× more valuable than reducing false positives.")
print(f"   Cost-sensitive training shifts the model to catch more fraud at the")
print(f"   expense of more false alarms — a worthwhile trade in financial fraud detection.")
print(f"\n✅ Task 4 Complete!")