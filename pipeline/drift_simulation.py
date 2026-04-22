# pipeline/drift_simulation.py
"""
Task 7: Realistic Drift Simulation
- Time-based drift: train on earlier data, test on later distribution
- New fraud patterns introduced over time
- Feature importance shifts tracked
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & Preprocess ─────────────────────────────────────────────────────
print("Loading data...")
trans = pd.read_csv("data/train_transaction.csv")
ident = pd.read_csv("data/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")

# Drop high-missing columns
missing_pct = df.isnull().mean()
drop_cols = missing_pct[missing_pct > 0.9].index.tolist()
df.drop(columns=drop_cols, inplace=True)

# Sort by TransactionDT to simulate time-based ordering
df.sort_values("TransactionDT", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Total transactions: {len(df):,}")
print(f"Time range: {df['TransactionDT'].min():,} → {df['TransactionDT'].max():,} seconds")

# ── 2. Time-Based Split ──────────────────────────────────────────────────────
# Split into 3 temporal windows:
#   Early  (0-50%)   → training data
#   Middle (50-75%)  → validation / baseline test
#   Late   (75-100%) → drifted test (later distribution)

n = len(df)
early_end  = int(n * 0.50)
middle_end = int(n * 0.75)

df_early  = df.iloc[:early_end].copy()     # train here
df_middle = df.iloc[early_end:middle_end].copy()   # baseline test
df_late   = df.iloc[middle_end:].copy()    # drifted test

print(f"\nTemporal splits:")
print(f"  Early  (train)     : {len(df_early):,} rows  | fraud rate: {df_early['isFraud'].mean():.2%}")
print(f"  Middle (baseline)  : {len(df_middle):,} rows  | fraud rate: {df_middle['isFraud'].mean():.2%}")
print(f"  Late   (drifted)   : {len(df_late):,} rows  | fraud rate: {df_late['isFraud'].mean():.2%}")

# ── 3. Preprocessing Function ────────────────────────────────────────────────
def preprocess(df_input, fit_stats=None):
    """
    Preprocess a dataframe. If fit_stats provided, use training stats
    (simulates real deployment where you can't peek at test data).
    """
    X = df_input.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")
    y = df_input["isFraud"]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if fit_stats is None:
        # Fit on this data (training set)
        medians = X[num_cols].median()
        fit_stats = {"medians": medians, "num_cols": num_cols, "cat_cols": cat_cols}
    else:
        medians   = fit_stats["medians"]
        num_cols  = fit_stats["num_cols"]
        cat_cols  = fit_stats["cat_cols"]

    # Align columns
    for col in num_cols:
        if col not in X.columns:
            X[col] = 0
    for col in cat_cols:
        if col not in X.columns:
            X[col] = -1

    X[num_cols] = X[num_cols].fillna(medians)

    # Encode categoricals using training fraud means
    for col in cat_cols:
        if col in X.columns:
            X[col] = pd.factorize(X[col])[0]

    # Keep only columns seen during training
    all_cols = num_cols + cat_cols
    X = X[[c for c in all_cols if c in X.columns]]

    return X, y, fit_stats

# ── 4. Train on Early Data ───────────────────────────────────────────────────
print("\nPreprocessing early (training) data...")
X_early, y_early, fit_stats = preprocess(df_early)

sampler = RandomUnderSampler(random_state=42)
X_res, y_res = sampler.fit_resample(X_early, y_early)
print(f"After resampling: {X_res.shape}")

print("Training XGBoost on early data...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=1, use_label_encoder=False,
    eval_metric="auc", random_state=42, n_jobs=-1
)
model.fit(X_res, y_res)
print("Training complete.")

# ── 5. Evaluate on Middle (Baseline) ────────────────────────────────────────
print("\nEvaluating on middle (baseline) data...")
X_mid, y_mid, _ = preprocess(df_middle, fit_stats)

# Align columns to training
for col in X_res.columns:
    if col not in X_mid.columns:
        X_mid[col] = 0
X_mid = X_mid[X_res.columns]

y_pred_mid  = model.predict(X_mid)
y_prob_mid  = model.predict_proba(X_mid)[:, 1]
auc_mid     = roc_auc_score(y_mid, y_prob_mid)
recall_mid  = recall_score(y_mid, y_pred_mid)
prec_mid    = precision_score(y_mid, y_pred_mid, zero_division=0)
f1_mid      = f1_score(y_mid, y_pred_mid)

print(f"  Baseline — AUC: {auc_mid:.4f} | Recall: {recall_mid:.4f} | Precision: {prec_mid:.4f} | F1: {f1_mid:.4f}")

# ── 6. Introduce Drift in Late Data ─────────────────────────────────────────
print("\nIntroducing drift in late data...")

df_late_drifted = df_late.copy()

# Drift Type 1: Transaction amount inflation (new fraud pattern — larger amounts)
if "TransactionAmt" in df_late_drifted.columns:
    fraud_mask = df_late_drifted["isFraud"] == 1
    df_late_drifted.loc[fraud_mask, "TransactionAmt"] *= np.random.uniform(
        3.0, 8.0, size=fraud_mask.sum()
    )
    print(f"  ✅ Drift 1: Fraud transaction amounts inflated 3-8x")

# Drift Type 2: New fraud pattern — introduce high-value card fraud cluster
# Drift Type 2: New fraud pattern — introduce high-value card fraud cluster
if "card1" in df_late_drifted.columns:
    fraud_mask = df_late_drifted["isFraud"] == 1
    n_new_pattern = int(fraud_mask.sum() * 0.3)
    new_pattern_idx = df_late_drifted[fraud_mask].sample(
        n=n_new_pattern, random_state=42
    ).index
    # Cast to int to match card1 dtype
    df_late_drifted.loc[new_pattern_idx, "card1"] = np.random.randint(
        25000, 35000, size=n_new_pattern
    )
    print(f"  ✅ Drift 2: New fraud pattern — {n_new_pattern:,} cases with unusual card1 range")
# Drift Type 3: Feature importance shift — addr1 becomes more predictive
if "addr1" in df_late_drifted.columns:
    fraud_mask = df_late_drifted["isFraud"] == 1
    df_late_drifted.loc[fraud_mask, "addr1"] = np.random.randint(
        400, 500, size=fraud_mask.sum()
    )
    print(f"  ✅ Drift 3: addr1 distribution shifted for fraud cases")

# ── 7. Evaluate on Late (Drifted) ───────────────────────────────────────────
print("\nEvaluating on late (drifted) data...")
X_late, y_late, _ = preprocess(df_late_drifted, fit_stats)

for col in X_res.columns:
    if col not in X_late.columns:
        X_late[col] = 0
X_late = X_late[X_res.columns]

y_pred_late  = model.predict(X_late)
y_prob_late  = model.predict_proba(X_late)[:, 1]
auc_late     = roc_auc_score(y_late, y_prob_late)
recall_late  = recall_score(y_late, y_pred_late)
prec_late    = precision_score(y_late, y_pred_late, zero_division=0)
f1_late      = f1_score(y_late, y_pred_late)

print(f"  Drifted  — AUC: {auc_late:.4f} | Recall: {recall_late:.4f} | Precision: {prec_late:.4f} | F1: {f1_late:.4f}")

# ── 8. Feature Importance Shift Analysis ─────────────────────────────────────
print("\nAnalyzing feature importance shift...")

importances_train = pd.Series(
    model.feature_importances_,
    index=X_res.columns
).sort_values(ascending=False)

# Top 10 features at training time
top10_train = importances_train.head(10)

# REPLACE this slow block:
# perm = permutation_importance(
#     model, X_late, y_late,
#     n_repeats=5, random_state=42, n_jobs=-1,
#     scoring="roc_auc"
# )

# WITH this fast version (sample 5k rows, 2 repeats):
sample_idx = np.random.choice(len(X_late), size=min(5000, len(X_late)), replace=False)
X_late_sample = X_late.iloc[sample_idx]
y_late_sample = y_late.iloc[sample_idx]

perm = permutation_importance(
    model, X_late_sample, y_late_sample,
    n_repeats=2, random_state=42, n_jobs=-1,
    scoring="roc_auc"
)
importances_drifted = pd.Series(
    perm.importances_mean,
    index=X_res.columns
).sort_values(ascending=False)
importances_drifted = pd.Series(
    perm.importances_mean,
    index=X_res.columns
).sort_values(ascending=False)

top10_drifted = importances_drifted.head(10)

print("\n  Top 10 features — Training time vs Drifted data:")
print(f"  {'Rank':<5} {'Train Feature':<20} {'Train Score':>12}  |  {'Drifted Feature':<20} {'Drift Score':>12}")
print(f"  {'-'*75}")
for i, ((tf, tv), (df_feat, dv)) in enumerate(
    zip(top10_train.items(), top10_drifted.items()), 1
):
    changed = " ← SHIFTED" if tf != df_feat else ""
    print(f"  {i:<5} {tf:<20} {tv:>12.4f}  |  {df_feat:<20} {dv:>12.4f}{changed}")

# ── 9. Simulate Rolling Performance Degradation ──────────────────────────────
print("\nSimulating rolling performance over time windows...")

# Split late data into 10 time windows to show gradual degradation
window_results = []
window_size = len(df_late_drifted) // 10

for i in range(10):
    start = i * window_size
    end   = start + window_size
    window_df = df_late_drifted.iloc[start:end].copy()

    # Gradually increase drift intensity over windows
    drift_intensity = 1.0 + (i * 0.3)
    if "TransactionAmt" in window_df.columns:
        fraud_mask = window_df["isFraud"] == 1
        if fraud_mask.sum() > 0:
            window_df.loc[fraud_mask, "TransactionAmt"] *= drift_intensity

    X_w, y_w, _ = preprocess(window_df, fit_stats)
    for col in X_res.columns:
        if col not in X_w.columns:
            X_w[col] = 0
    X_w = X_w[X_res.columns]

    if y_w.sum() < 5:
        continue

    yp    = model.predict(X_w)
    yprob = model.predict_proba(X_w)[:, 1]
    window_results.append({
        "window": i + 1,
        "auc":    roc_auc_score(y_w, yprob),
        "recall": recall_score(y_w, yp),
        "f1":     f1_score(y_w, yp)
    })

windows_df = pd.DataFrame(window_results)

# ── 10. Visualization ────────────────────────────────────────────────────────
print("\nGenerating plots...")
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# — Plot 1: Baseline vs Drifted performance —
ax1 = fig.add_subplot(gs[0, 0])
metrics   = ["AUC-ROC", "Recall", "Precision", "F1"]
baseline  = [auc_mid,  recall_mid,  prec_mid,  f1_mid]
drifted_v = [auc_late, recall_late, prec_late, f1_late]
x = np.arange(len(metrics))
w = 0.35
ax1.bar(x - w/2, baseline,  w, label="Baseline (Middle)",  color="#2196F3", alpha=0.85)
ax1.bar(x + w/2, drifted_v, w, label="Drifted (Late)",     color="#F44336", alpha=0.85)
ax1.set_title("Baseline vs Drifted Performance", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.axhline(0.75, color="orange", linestyle="--", linewidth=1, label="Recall threshold")
for bar in ax1.patches:
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{bar.get_height():.3f}",
             ha="center", va="bottom", fontsize=8)

# — Plot 2: Rolling degradation over time windows —
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(windows_df["window"], windows_df["auc"],    "b-o", label="AUC",    linewidth=2)
ax2.plot(windows_df["window"], windows_df["recall"], "r-o", label="Recall", linewidth=2)
ax2.plot(windows_df["window"], windows_df["f1"],     "g-o", label="F1",     linewidth=2)
ax2.axhline(0.75, color="orange", linestyle="--", linewidth=1.5, label="Recall alert threshold")
ax2.fill_between(windows_df["window"],
                 windows_df["recall"], 0.75,
                 where=windows_df["recall"] < 0.75,
                 alpha=0.2, color="red", label="Below threshold")
ax2.set_title("Performance Degradation Over Time\n(Increasing drift intensity)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Time Window (→ more drift)")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# — Plot 3: Feature importance shift —
ax3 = fig.add_subplot(gs[0, 2])
top_features = list(set(top10_train.index.tolist() + top10_drifted.index.tolist()))[:10]
train_vals   = [importances_train.get(f, 0) for f in top_features]
drift_vals   = [importances_drifted.get(f, 0) / (max(importances_drifted) + 1e-9) for f in top_features]
x = np.arange(len(top_features))
ax3.barh(x - 0.2, train_vals,  0.35, label="Train importance",   color="#1565C0", alpha=0.85)
ax3.barh(x + 0.2, drift_vals,  0.35, label="Drifted importance",  color="#B71C1C", alpha=0.85)
ax3.set_yticks(x)
ax3.set_yticklabels(top_features, fontsize=8)
ax3.set_title("Feature Importance Shift\n(Train vs Drifted)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Importance Score (normalized)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="x")

# — Plot 4: Transaction amount distribution shift —
ax4 = fig.add_subplot(gs[1, 0])
if "TransactionAmt" in df_early.columns and "TransactionAmt" in df_late_drifted.columns:
    early_fraud_amt = df_early[df_early["isFraud"]==1]["TransactionAmt"].clip(0, 2000)
    late_fraud_amt  = df_late_drifted[df_late_drifted["isFraud"]==1]["TransactionAmt"].clip(0, 2000)
    ax4.hist(early_fraud_amt, bins=50, alpha=0.6, color="#1565C0", label="Early (train) fraud", density=True)
    ax4.hist(late_fraud_amt,  bins=50, alpha=0.6, color="#B71C1C", label="Late (drifted) fraud", density=True)
    ax4.set_title("Fraud Transaction Amount Distribution\n(Train vs Drifted)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Transaction Amount ($)")
    ax4.set_ylabel("Density")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# — Plot 5: Fraud rate over time —
ax5 = fig.add_subplot(gs[1, 1])
df["time_bin"] = pd.cut(df["TransactionDT"], bins=20, labels=False)
fraud_over_time = df.groupby("time_bin")["isFraud"].mean()
ax5.plot(fraud_over_time.index, fraud_over_time.values, "r-o", linewidth=2, markersize=5)
ax5.axvline(10, color="blue",   linestyle="--", linewidth=1.5, label="Train/Test split (50%)")
ax5.axvline(15, color="orange", linestyle="--", linewidth=1.5, label="Middle/Late split (75%)")
ax5.set_title("Fraud Rate Over Time\n(Time-based drift)", fontsize=12, fontweight="bold")
ax5.set_xlabel("Time Bin (→ later transactions)")
ax5.set_ylabel("Fraud Rate")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# — Plot 6: Summary table —
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
table_data = [
    ["Baseline (Middle)", f"{auc_mid:.4f}",  f"{recall_mid:.4f}",  f"{prec_mid:.4f}",  f"{f1_mid:.4f}"],
    ["Drifted (Late)",    f"{auc_late:.4f}", f"{recall_late:.4f}", f"{prec_late:.4f}", f"{f1_late:.4f}"],
    ["Δ Change",
     f"{auc_late-auc_mid:+.4f}",
     f"{recall_late-recall_mid:+.4f}",
     f"{prec_late-prec_mid:+.4f}",
     f"{f1_late-f1_mid:+.4f}"]
]
tbl = ax6.table(
    cellText=table_data,
    colLabels=["Period", "AUC", "Recall", "Precision", "F1"],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 2.0)
# Color the delta row red/green
for j in range(5):
    tbl[2, j].set_facecolor("#FFEBEE")   # light red for baseline
    tbl[3, j].set_facecolor("#FFF3E0")   # light orange for drifted
ax6.set_title("Drift Impact Summary", fontsize=12, fontweight="bold", pad=60)

fig.suptitle(
    "Task 7: Realistic Drift Simulation\n"
    "Time-based split | New fraud patterns | Feature importance shifts",
    fontsize=14, fontweight="bold"
)
plt.savefig("models/drift_simulation.png", dpi=150, bbox_inches="tight")
print("Saved: models/drift_simulation.png")
plt.close()

# ── 11. Final Summary ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TASK 7 DRIFT SIMULATION SUMMARY")
print("="*60)
print(f"  {'Metric':<12} {'Baseline':>10}  {'Drifted':>10}  {'Change':>10}")
print(f"  {'-'*50}")
for metric, base, drift in [
    ("AUC",       auc_mid,    auc_late),
    ("Recall",    recall_mid, recall_late),
    ("Precision", prec_mid,   prec_late),
    ("F1",        f1_mid,     f1_late),
]:
    delta  = drift - base
    symbol = "↓" if delta < 0 else "↑"
    print(f"  {metric:<12} {base:>10.4f}  {drift:>10.4f}  {symbol} {abs(delta):.4f}")

print(f"\n  Drift types introduced:")
print(f"    1. Transaction amount inflation (3-8x for fraud cases)")
print(f"    2. New fraud pattern — unusual card1 range (25k-35k)")
print(f"    3. addr1 distribution shift for fraud cases")
print(f"\n  Recall {'BELOW' if recall_late < 0.75 else 'above'} retraining threshold (0.75)")
print(f"  → {'Retraining triggered!' if recall_late < 0.75 else 'No retraining needed yet'}")
print(f"\n✅ Task 7 Complete!")