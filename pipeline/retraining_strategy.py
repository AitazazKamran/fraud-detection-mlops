# pipeline/retraining_strategy.py
"""
Task 8: Intelligent Retraining Strategy
Implements and compares 3 strategies:
  1. Threshold-based  — retrain when recall drops below 0.75
  2. Periodic         — retrain every N time windows regardless of performance
  3. Hybrid           — periodic checks + performance gate
Compares: stability, compute cost, performance improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import joblib, time, warnings
warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
RECALL_THRESHOLD   = 0.75   # trigger retraining if recall drops below this
PERIODIC_INTERVAL  = 3      # retrain every 3 windows in periodic strategy
N_WINDOWS          = 12     # simulate 12 time windows total
COMPUTE_COST_PER_TRAIN = 1.0  # normalized compute unit per retraining run

# ── 1. Load & Preprocess ─────────────────────────────────────────────────────
print("Loading data...")
trans = pd.read_csv("data/train_transaction.csv")
ident = pd.read_csv("data/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")

missing_pct = df.isnull().mean()
drop_cols = missing_pct[missing_pct > 0.9].index.tolist()
df.drop(columns=drop_cols, inplace=True)
df.sort_values("TransactionDT", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Total rows: {len(df):,}")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
def preprocess(df_input, fit_stats=None):
    X = df_input.drop(
        columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore"
    )
    y = df_input["isFraud"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if fit_stats is None:
        medians   = X[num_cols].median()
        fit_stats = {"medians": medians, "num_cols": num_cols,
                     "cat_cols": cat_cols, "train_cols": num_cols + cat_cols}
    else:
        medians  = fit_stats["medians"]
        num_cols = fit_stats["num_cols"]
        cat_cols = fit_stats["cat_cols"]

    for col in num_cols:
        if col not in X.columns:
            X[col] = 0
    X[num_cols] = X[num_cols].fillna(medians)
    for col in cat_cols:
        if col not in X.columns:
            X[col] = -1
        else:
            X[col] = pd.factorize(X[col])[0]

    all_cols = [c for c in (num_cols + cat_cols) if c in X.columns]
    return X[all_cols], y, fit_stats

def train_model(X_train, y_train):
    """Train XGBoost with undersampling. Returns model + compute time."""
    t0 = time.time()
    sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric="auc",
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_res, y_res)
    return model, time.time() - t0

def evaluate(model, X_test, y_test, train_cols):
    """Evaluate model, aligning columns to training set."""
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_cols]
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "auc":       roc_auc_score(y_test, y_prob),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

def introduce_gradual_drift(df_window, window_idx, total_windows):
    """Gradually increase drift intensity over time windows."""
    df_w = df_window.copy()
    intensity = 1.0 + (window_idx / total_windows) * 4.0  # ramps 1x → 5x
    if "TransactionAmt" in df_w.columns:
        fraud_mask = df_w["isFraud"] == 1
        if fraud_mask.sum() > 0:
            df_w.loc[fraud_mask, "TransactionAmt"] *= intensity
    if "card1" in df_w.columns and window_idx > total_windows // 2:
        # New fraud pattern appears halfway through
        fraud_mask = df_w["isFraud"] == 1
        n = int(fraud_mask.sum() * 0.4)
        if n > 0:
            idx = df_w[fraud_mask].sample(n=n, random_state=window_idx).index
            df_w.loc[idx, "card1"] = np.random.randint(25000, 35000, size=n)
    return df_w

# ── 3. Create Time Windows ────────────────────────────────────────────────────
print(f"Creating {N_WINDOWS} time windows...")
window_size = len(df) // (N_WINDOWS + 2)   # +2 for initial training window

# Initial training window (first 2 chunks)
df_init = df.iloc[:window_size * 2].copy()
windows = [
    df.iloc[(i + 2) * window_size: (i + 3) * window_size].copy()
    for i in range(N_WINDOWS)
]
print(f"  Initial train size : {len(df_init):,}")
print(f"  Each window size   : {len(windows[0]):,}")

# ── 4. Initial Training ───────────────────────────────────────────────────────
print("\nTraining initial model...")
X_init, y_init, fit_stats = preprocess(df_init)
init_model, init_time = train_model(X_init, y_init)
train_cols = X_init.columns.tolist()
print(f"  Initial training time: {init_time:.1f}s")

# ── 5. Simulate All 3 Strategies ─────────────────────────────────────────────

def simulate_strategy(strategy_name, windows, init_model, fit_stats, train_cols):
    """
    Run a full simulation of a retraining strategy over N_WINDOWS.
    Returns per-window metrics + retraining events.
    """
    print(f"\n{'='*55}")
    print(f"  Strategy: {strategy_name}")
    print(f"{'='*55}")

    current_model  = init_model
    results        = []
    retrain_events = []
    total_compute  = 0.0
    cumulative_data = df_init.copy()

    for i, window_df in enumerate(windows):
        # Apply gradual drift
        window_drifted = introduce_gradual_drift(window_df, i, N_WINDOWS)

        # Evaluate current model on this window
        X_w, y_w, _ = preprocess(window_drifted, fit_stats)
        if y_w.sum() < 5:
            continue
        metrics = evaluate(current_model, X_w.copy(), y_w, train_cols)

        # Decide whether to retrain based on strategy
        should_retrain = False

        if strategy_name == "Threshold-Based":
            # Retrain only when recall drops below threshold
            if metrics["recall"] < RECALL_THRESHOLD:
                should_retrain = True
                reason = f"recall={metrics['recall']:.3f} < {RECALL_THRESHOLD}"

        elif strategy_name == "Periodic":
            # Retrain every PERIODIC_INTERVAL windows regardless
            if (i + 1) % PERIODIC_INTERVAL == 0:
                should_retrain = True
                reason = f"scheduled (every {PERIODIC_INTERVAL} windows)"

        elif strategy_name == "Hybrid":
            # Retrain if recall drops OR it's been too long since last retrain
            windows_since_retrain = i - (retrain_events[-1]["window"] if retrain_events else -1)
            if metrics["recall"] < RECALL_THRESHOLD:
                should_retrain = True
                reason = f"recall drop ({metrics['recall']:.3f})"
            elif windows_since_retrain >= PERIODIC_INTERVAL * 2:
                should_retrain = True
                reason = f"max interval reached ({windows_since_retrain} windows)"

        # Retrain if needed
        if should_retrain:
            # Accumulate data up to this point (drift-aware: use recent data)
            cumulative_data = pd.concat([cumulative_data, window_drifted], ignore_index=True)
            # Keep only last 3 windows worth of data to stay current
            keep_rows = window_size * 5
            if len(cumulative_data) > keep_rows:
                cumulative_data = cumulative_data.iloc[-keep_rows:]

            X_retrain, y_retrain, _ = preprocess(cumulative_data, fit_stats)
            current_model, retrain_time = train_model(X_retrain, y_retrain)
            total_compute += COMPUTE_COST_PER_TRAIN

            # Re-evaluate after retraining
            metrics = evaluate(current_model, X_w.copy(), y_w, train_cols)
            retrain_events.append({
                "window": i, "reason": reason,
                "recall_after": metrics["recall"],
                "compute_time": retrain_time
            })
            print(f"  Window {i+1:>2}: RETRAINED ({reason}) → recall={metrics['recall']:.3f}")
        else:
            print(f"  Window {i+1:>2}: no retrain  | recall={metrics['recall']:.3f} | AUC={metrics['auc']:.3f}")

        results.append({
            "window":    i + 1,
            "auc":       metrics["auc"],
            "recall":    metrics["recall"],
            "precision": metrics["precision"],
            "f1":        metrics["f1"],
            "retrained": should_retrain
        })

    results_df = pd.DataFrame(results)
    below_threshold = (results_df["recall"] < RECALL_THRESHOLD).sum()

    print(f"\n  Summary:")
    print(f"    Retraining events     : {len(retrain_events)}")
    print(f"    Total compute cost    : {total_compute:.1f} units")
    print(f"    Windows below recall  : {below_threshold}/{N_WINDOWS}")
    print(f"    Mean recall           : {results_df['recall'].mean():.4f}")
    print(f"    Min recall            : {results_df['recall'].min():.4f}")
    print(f"    Recall stability (std): {results_df['recall'].std():.4f}")

    return results_df, retrain_events, total_compute

# Run all 3 strategies
results_threshold, events_threshold, cost_threshold = simulate_strategy(
    "Threshold-Based", windows, init_model, fit_stats, train_cols
)
results_periodic, events_periodic, cost_periodic = simulate_strategy(
    "Periodic", windows, init_model, fit_stats, train_cols
)
results_hybrid, events_hybrid, cost_hybrid = simulate_strategy(
    "Hybrid", windows, init_model, fit_stats, train_cols
)

# ── 6. Comparison Summary ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TASK 8 STRATEGY COMPARISON")
print("="*65)
print(f"  {'Strategy':<20} {'Retrains':>8} {'Compute':>8} {'MeanRecall':>11} {'MinRecall':>10} {'Stability':>10}")
print(f"  {'-'*65}")

comparison = []
for name, results, events, cost in [
    ("Threshold-Based", results_threshold, events_threshold, cost_threshold),
    ("Periodic",        results_periodic,  events_periodic,  cost_periodic),
    ("Hybrid",          results_hybrid,    events_hybrid,    cost_hybrid),
]:
    mean_r = results["recall"].mean()
    min_r  = results["recall"].min()
    std_r  = results["recall"].std()
    print(f"  {name:<20} {len(events):>8} {cost:>8.1f} {mean_r:>11.4f} {min_r:>10.4f} {std_r:>10.4f}")
    comparison.append({
        "name": name, "retrains": len(events), "compute": cost,
        "mean_recall": mean_r, "min_recall": min_r, "stability": std_r
    })

# ── 7. Visualization ──────────────────────────────────────────────────────────
print("\nGenerating comparison plots...")
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = {"Threshold-Based": "#1565C0", "Periodic": "#2E7D32", "Hybrid": "#E65100"}
all_results = [
    ("Threshold-Based", results_threshold, events_threshold),
    ("Periodic",        results_periodic,  events_periodic),
    ("Hybrid",          results_hybrid,    events_hybrid),
]

# — Plot 1: Recall over time for all strategies —
ax1 = fig.add_subplot(gs[0, 0:2])
for name, results, events in all_results:
    ax1.plot(results["window"], results["recall"],
             "-o", label=name, color=colors[name], linewidth=2, markersize=5)
    # Mark retraining events
    for ev in events:
        w = ev["window"] + 1
        row = results[results["window"] == w]
        if not row.empty:
            ax1.axvline(w, color=colors[name], linestyle=":", alpha=0.4)
            ax1.scatter(w, row["recall"].values[0],
                        marker="^", s=120, color=colors[name], zorder=5)

ax1.axhline(RECALL_THRESHOLD, color="red", linestyle="--",
            linewidth=2, label=f"Recall threshold ({RECALL_THRESHOLD})")
ax1.fill_between(range(1, N_WINDOWS + 1), 0, RECALL_THRESHOLD,
                 alpha=0.05, color="red")
ax1.set_title("Fraud Recall Over Time — All Strategies\n(▲ = retraining event)",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Time Window")
ax1.set_ylabel("Recall")
ax1.set_ylim(0.4, 1.05)
ax1.set_xlim(0.5, N_WINDOWS + 0.5)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# — Plot 2: Compute cost vs performance —
ax2 = fig.add_subplot(gs[0, 2])
comp_df = pd.DataFrame(comparison)
scatter_colors = [colors[n] for n in comp_df["name"]]
sc = ax2.scatter(comp_df["compute"], comp_df["mean_recall"],
                 s=300, c=scatter_colors, zorder=5, edgecolors="white", linewidths=2)
for _, row in comp_df.iterrows():
    ax2.annotate(row["name"],
                 (row["compute"], row["mean_recall"]),
                 textcoords="offset points", xytext=(8, 4), fontsize=9)
ax2.set_title("Compute Cost vs Mean Recall\n(Top-right = best)",
              fontsize=12, fontweight="bold")
ax2.set_xlabel("Total Compute Cost (units)")
ax2.set_ylabel("Mean Recall")
ax2.grid(True, alpha=0.3)

# — Plot 3: AUC over time —
ax3 = fig.add_subplot(gs[1, 0])
for name, results, _ in all_results:
    ax3.plot(results["window"], results["auc"],
             "-o", label=name, color=colors[name], linewidth=2, markersize=4)
ax3.set_title("AUC-ROC Over Time", fontsize=12, fontweight="bold")
ax3.set_xlabel("Time Window")
ax3.set_ylabel("AUC-ROC")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# — Plot 4: Retraining frequency bar —
ax4 = fig.add_subplot(gs[1, 1])
names   = [c["name"]     for c in comparison]
retrains = [c["retrains"] for c in comparison]
costs   = [c["compute"]  for c in comparison]
x = np.arange(len(names))
w = 0.35
b1 = ax4.bar(x - w/2, retrains, w, label="# Retrains",   color=[colors[n] for n in names], alpha=0.85)
b2 = ax4.bar(x + w/2, costs,    w, label="Compute Cost", color=[colors[n] for n in names], alpha=0.4)
ax4.set_title("Retraining Events & Compute Cost", fontsize=12, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(names, fontsize=9)
ax4.set_ylabel("Count / Cost Units")
ax4.legend()
for bar in b1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{int(bar.get_height())}", ha="center", fontsize=9, fontweight="bold")

# — Plot 5: Stability comparison (box-like bar of std) —
ax5 = fig.add_subplot(gs[1, 2])
stabilities  = [c["stability"]   for c in comparison]
mean_recalls = [c["mean_recall"] for c in comparison]
min_recalls  = [c["min_recall"]  for c in comparison]

x = np.arange(len(names))
ax5.bar(x - 0.25, mean_recalls, 0.25, label="Mean Recall", color=[colors[n] for n in names], alpha=0.9)
ax5.bar(x,        min_recalls,  0.25, label="Min Recall",  color=[colors[n] for n in names], alpha=0.5)
ax5.bar(x + 0.25, stabilities,  0.25, label="Std (lower=stable)", color="gray", alpha=0.6)
ax5.axhline(RECALL_THRESHOLD, color="red", linestyle="--", linewidth=1)
ax5.set_title("Stability Analysis\n(Higher recall + lower std = better)",
              fontsize=12, fontweight="bold")
ax5.set_xticks(x)
ax5.set_xticklabels(names, fontsize=9)
ax5.set_ylabel("Score")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "Task 8: Intelligent Retraining Strategy Comparison\n"
    "Threshold-Based vs Periodic vs Hybrid",
    fontsize=14, fontweight="bold"
)
plt.savefig("models/retraining_strategy.png", dpi=150, bbox_inches="tight")
print("Saved: models/retraining_strategy.png")
plt.close()

# ── 8. Business Recommendation ────────────────────────────────────────────────
best = min(comparison, key=lambda x: (x["compute"] / (x["mean_recall"] + 1e-9)))
print(f"\n💡 Recommendation: {best['name']}")
print(f"   Best compute-to-recall efficiency")
print(f"\n   Strategy trade-offs:")
print(f"   • Threshold-Based : reacts to problems, cheapest — but may lag behind gradual drift")
print(f"   • Periodic        : predictable schedule, easy to operate — wastes compute when stable")
print(f"   • Hybrid          : best of both — catches drops AND prevents staleness")
print(f"\n✅ Task 8 Complete!")