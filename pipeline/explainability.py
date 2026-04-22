# pipeline/explainability.py
"""
Task 9: Explainability Requirement
- Feature importance analysis
- SHAP values: why is the model predicting fraud?
- Global + local explanations
- Fraud vs legitimate comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & Preprocess ─────────────────────────────────────────────────────
print("Loading data...")
trans = pd.read_csv("data/train_transaction.csv")
ident = pd.read_csv("data/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")

missing_pct = df.isnull().mean()
drop_cols = missing_pct[missing_pct > 0.9].index.tolist()
df.drop(columns=drop_cols, inplace=True)

X = df.drop(columns=["isFraud", "TransactionID"])
y = df["isFraud"]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
for col in cat_cols:
    if X[col].nunique() > 50:
        means = df.groupby(col)["isFraud"].mean()
        X[col] = X[col].map(means).fillna(0)
    else:
        X[col] = pd.factorize(X[col])[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
sampler = RandomUnderSampler(random_state=42)
X_res, y_res = sampler.fit_resample(X_train, y_train)
print(f"Training shape: {X_res.shape}")

# ── 2. Train Model ────────────────────────────────────────────────────────────
print("Training XGBoost for SHAP analysis...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    use_label_encoder=False, eval_metric="auc",
    random_state=42, n_jobs=-1, verbosity=0
)
model.fit(X_res, y_res)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# ── 3. SHAP Explainer ─────────────────────────────────────────────────────────
# Use a sample for speed — SHAP on full 118k test set takes too long
print("\nComputing SHAP values (sample of 2,000 rows)...")
sample_size = 2000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=sample_size, replace=False)
X_sample   = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample   = y_test.iloc[sample_idx].reset_index(drop=True)

# TreeExplainer is fast and exact for XGBoost
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For binary classification XGBoost returns 2D array (one value per sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # take fraud class

print(f"SHAP values shape: {shap_values.shape}")
print(f"Sample fraud rate: {y_sample.mean():.2%}")

# ── 4. Global Feature Importance ─────────────────────────────────────────────
print("\nTop 15 features by mean |SHAP|:")
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_sample.columns
).sort_values(ascending=False)

print(f"  {'Rank':<5} {'Feature':<20} {'Mean |SHAP|':>12}")
print(f"  {'-'*40}")
for i, (feat, val) in enumerate(mean_shap.head(15).items(), 1):
    print(f"  {i:<5} {feat:<20} {val:>12.4f}")

# ── 5. Separate Fraud vs Legitimate SHAP ─────────────────────────────────────
fraud_idx = np.where(y_sample == 1)[0]
legit_idx = np.where(y_sample == 0)[0]

shap_fraud = shap_values[fraud_idx]
shap_legit = shap_values[legit_idx]

mean_shap_fraud = pd.Series(
    np.abs(shap_fraud).mean(axis=0),
    index=X_sample.columns
).sort_values(ascending=False)

mean_shap_legit = pd.Series(
    np.abs(shap_legit).mean(axis=0),
    index=X_sample.columns
).sort_values(ascending=False)

print(f"\nTop 10 features driving FRAUD predictions:")
for i, (f, v) in enumerate(mean_shap_fraud.head(10).items(), 1):
    print(f"  {i}. {f:<20} {v:.4f}")

print(f"\nTop 10 features driving LEGITIMATE predictions:")
for i, (f, v) in enumerate(mean_shap_legit.head(10).items(), 1):
    print(f"  {i}. {f:<20} {v:.4f}")

# ── 6. Local Explanation — single transaction ─────────────────────────────────
print("\nLocal explanation — most confident fraud prediction:")
y_pred_prob = model.predict_proba(X_sample)[:, 1]
most_fraud_idx = np.argmax(y_pred_prob * (y_sample == 1).values)
fraud_shap     = shap_values[most_fraud_idx]
fraud_features = X_sample.iloc[most_fraud_idx]

top_fraud_drivers = pd.Series(fraud_shap, index=X_sample.columns)\
    .reindex(pd.Series(np.abs(fraud_shap), index=X_sample.columns)
    .sort_values(ascending=False).index).head(10)

print(f"  Transaction fraud probability: {y_pred_prob[most_fraud_idx]:.4f}")
print(f"  Top 10 SHAP drivers for this transaction:")
for feat, shap_val in top_fraud_drivers.items():
    direction = "→ FRAUD" if shap_val > 0 else "→ legit"
    print(f"    {feat:<20} SHAP={shap_val:+.4f}  val={fraud_features[feat]:.2f}  {direction}")

# ── 7. Visualization ──────────────────────────────────────────────────────────
print("\nGenerating SHAP plots...")
fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

# — Plot 1: Global feature importance (mean |SHAP|) —
ax1 = fig.add_subplot(gs[0, 0])
top15 = mean_shap.head(15)
colors_bar = ["#C62828" if i < 5 else "#1565C0" if i < 10 else "#4CAF50"
              for i in range(15)]
bars = ax1.barh(range(15), top15.values[::-1], color=colors_bar[::-1], alpha=0.85)
ax1.set_yticks(range(15))
ax1.set_yticklabels(top15.index[::-1], fontsize=9)
ax1.set_title("Global Feature Importance\n(Mean |SHAP| across all predictions)",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Mean |SHAP value|")
ax1.grid(True, alpha=0.3, axis="x")
for bar, val in zip(bars, top15.values[::-1]):
    ax1.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=8)

# — Plot 2: Fraud vs Legitimate top features comparison —
ax2 = fig.add_subplot(gs[0, 1])
top_features = list(dict.fromkeys(
    mean_shap_fraud.head(8).index.tolist() +
    mean_shap_legit.head(8).index.tolist()
))[:12]
fraud_vals = [mean_shap_fraud.get(f, 0) for f in top_features]
legit_vals = [mean_shap_legit.get(f, 0) for f in top_features]
x = np.arange(len(top_features))
w = 0.35
ax2.barh(x - w/2, fraud_vals, w, label="Fraud cases",      color="#C62828", alpha=0.85)
ax2.barh(x + w/2, legit_vals, w, label="Legitimate cases", color="#1565C0", alpha=0.85)
ax2.set_yticks(x)
ax2.set_yticklabels(top_features, fontsize=9)
ax2.set_title("Feature Importance: Fraud vs Legitimate\n(Mean |SHAP| per class)",
              fontsize=12, fontweight="bold")
ax2.set_xlabel("Mean |SHAP value|")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="x")

# — Plot 3: SHAP beeswarm (manual implementation) —
ax3 = fig.add_subplot(gs[1, 0])
top10_feats = mean_shap.head(10).index.tolist()
top10_idx   = [X_sample.columns.get_loc(f) for f in top10_feats]
shap_top10  = shap_values[:, top10_idx]
feat_vals_top10 = X_sample[top10_feats].values

for i, feat in enumerate(top10_feats[::-1]):
    fi   = top10_feats.index(feat)
    sv   = shap_top10[:, fi]
    fv   = feat_vals_top10[:, fi]
    # Normalize feature values for color
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
    y_jitter = np.random.uniform(-0.3, 0.3, size=len(sv))
    sc = ax3.scatter(sv, np.full_like(sv, i) + y_jitter,
                     c=fv_norm, cmap="RdBu_r", alpha=0.4, s=8)
ax3.set_yticks(range(10))
ax3.set_yticklabels(top10_feats[::-1], fontsize=9)
ax3.axvline(0, color="black", linewidth=1)
ax3.set_title("SHAP Beeswarm Plot\n(Red=high feature value, Blue=low)",
              fontsize=12, fontweight="bold")
ax3.set_xlabel("SHAP value (impact on fraud prediction)")
ax3.grid(True, alpha=0.2, axis="x")
plt.colorbar(sc, ax=ax3, label="Feature value (normalized)", shrink=0.6)

# — Plot 4: Local explanation waterfall (single fraud transaction) —
ax4 = fig.add_subplot(gs[1, 1])
top10_local = pd.Series(fraud_shap, index=X_sample.columns)\
    .abs().sort_values(ascending=False).head(10)
local_shap_vals = pd.Series(fraud_shap, index=X_sample.columns)[top10_local.index]

bar_colors = ["#C62828" if v > 0 else "#1565C0" for v in local_shap_vals.values[::-1]]
ax4.barh(range(10), local_shap_vals.values[::-1],
         color=bar_colors, alpha=0.85)
ax4.set_yticks(range(10))
ax4.set_yticklabels(top10_local.index[::-1], fontsize=9)
ax4.axvline(0, color="black", linewidth=1)
ax4.set_title(f"Local Explanation — Single Fraud Transaction\n"
              f"(Fraud probability: {y_pred_prob[most_fraud_idx]:.1%})\n"
              f"Red=pushes toward fraud, Blue=pushes toward legitimate",
              fontsize=11, fontweight="bold")
ax4.set_xlabel("SHAP value")
ax4.grid(True, alpha=0.3, axis="x")

# — Plot 5: SHAP value distribution for top 5 features —
ax5 = fig.add_subplot(gs[2, 0])
top5 = mean_shap.head(5).index.tolist()
for i, feat in enumerate(top5):
    fi = X_sample.columns.get_loc(feat)
    sv = shap_values[:, fi]
    ax5.hist(sv, bins=40, alpha=0.5, label=feat, density=True)
ax5.axvline(0, color="black", linewidth=1.5, linestyle="--")
ax5.set_title("SHAP Value Distribution — Top 5 Features\n"
              "(Right of 0 = pushes toward fraud prediction)",
              fontsize=12, fontweight="bold")
ax5.set_xlabel("SHAP value")
ax5.set_ylabel("Density")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# — Plot 6: Feature value vs SHAP (dependence plot for top feature) —
ax6 = fig.add_subplot(gs[2, 1])
top_feat     = mean_shap.index[0]
top_feat_idx = X_sample.columns.get_loc(top_feat)
feat_values  = X_sample[top_feat].values
shap_vals_tf = shap_values[:, top_feat_idx]

# Color by fraud label
sc = ax6.scatter(
    feat_values, shap_vals_tf,
    c=y_sample.values, cmap="RdYlGn_r",
    alpha=0.4, s=10
)
ax6.axhline(0, color="black", linewidth=1, linestyle="--")
ax6.set_title(f"SHAP Dependence Plot — {top_feat}\n"
              f"(Red=fraud, Green=legitimate)",
              fontsize=12, fontweight="bold")
ax6.set_xlabel(f"{top_feat} value")
ax6.set_ylabel("SHAP value")
plt.colorbar(sc, ax=ax6, label="True label (1=fraud)", shrink=0.6)
ax6.grid(True, alpha=0.3)

fig.suptitle(
    "Task 9: Model Explainability — SHAP Analysis\n"
    "Why is the model predicting fraud?",
    fontsize=14, fontweight="bold"
)
plt.savefig("models/shap_explainability.png", dpi=150, bbox_inches="tight")
print("Saved: models/shap_explainability.png")
plt.close()

# ── 8. Save SHAP summary ──────────────────────────────────────────────────────
shap_summary = pd.DataFrame({
    "feature":         mean_shap.index,
    "mean_abs_shap":   mean_shap.values,
    "mean_shap_fraud": [mean_shap_fraud.get(f, 0) for f in mean_shap.index],
    "mean_shap_legit": [mean_shap_legit.get(f, 0) for f in mean_shap.index],
}).head(30)
shap_summary.to_csv("models/shap_feature_importance.csv", index=False)
print("Saved: models/shap_feature_importance.csv")

print("\n" + "="*60)
print("  TASK 9 COMPLETE — SHAP EXPLAINABILITY SUMMARY")
print("="*60)
print(f"\n  Top 5 features driving fraud predictions:")
for i, (feat, val) in enumerate(mean_shap_fraud.head(5).items(), 1):
    print(f"    {i}. {feat:<20} mean|SHAP|={val:.4f}")
print(f"\n  Key insight: Features with high SHAP values")
print(f"  consistently separate fraud from legitimate")
print(f"  transactions, giving investigators a clear")
print(f"  audit trail for every model decision.")
print(f"\n✅ Task 9 Complete!")