import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score, recall_score, precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────
print("Loading data...")
trans = pd.read_csv("data/train_transaction.csv")
ident = pd.read_csv("data/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")

# Drop high missing columns
missing_pct = df.isnull().mean()
drop_cols = missing_pct[missing_pct > 0.9].index.tolist()
df.drop(columns=drop_cols, inplace=True)

# Impute
num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
for c in ["isFraud","TransactionID"]:
    if c in num_cols:
        num_cols.remove(c)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    if df[col].nunique() > 50:
        means = df.groupby(col)["isFraud"].mean()
        df[col] = df[col].map(means)
    else:
        df[col] = pd.factorize(df[col])[0]

# Features
X = df.drop(columns=["isFraud","TransactionID"], errors="ignore")
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance with undersampling (winner from Task 2)
sampler = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
print(f"Training shape after resampling: {X_train_res.shape}")

# Scale pos weight for cost-sensitive XGBoost
scale = len(y_train_res[y_train_res==0]) / len(y_train_res[y_train_res==1])

# ─────────────────────────────────────────
# Helper: Evaluate Model
# ─────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc    = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "name": name, "auc": auc, "recall": recall,
        "precision": prec, "f1": f1,
        "y_pred": y_pred, "y_prob": y_prob, "cm": cm
    }

results = {}

# ─────────────────────────────────────────
# 2. Model 1 — XGBoost
# ─────────────────────────────────────────
# WHY XGBoost? Gradient boosting builds trees sequentially,
# each tree correcting errors of previous ones.
# scale_pos_weight adds cost-sensitive learning —
# penalizes missing fraud more than false alarms.
print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale,   # cost-sensitive
    random_state=42,
    eval_metric="auc",
    verbosity=0
)
xgb.fit(X_train_res, y_train_res)
results["XGBoost"] = evaluate_model("XGBoost", xgb, X_test, y_test)
joblib.dump(xgb, "models/xgboost_model.pkl")
print("Saved: models/xgboost_model.pkl")

# ─────────────────────────────────────────
# 3. Model 2 — LightGBM
# ─────────────────────────────────────────
# WHY LightGBM? Faster than XGBoost on large datasets.
# Uses leaf-wise tree growth instead of level-wise.
# is_unbalance=True internally handles class imbalance.
print("\nTraining LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    is_unbalance=True,        # handles imbalance internally
    random_state=42,
    verbosity=-1
)
lgbm.fit(X_train_res, y_train_res)
results["LightGBM"] = evaluate_model("LightGBM", lgbm, X_test, y_test)
joblib.dump(lgbm, "models/lightgbm_model.pkl")
print("Saved: models/lightgbm_model.pkl")

# ─────────────────────────────────────────
# 4. Model 3 — Hybrid (RF + Feature Selection + XGBoost)
# ─────────────────────────────────────────
# WHY Hybrid? Random Forest finds the most important features.
# We then train XGBoost only on those features.
# This reduces noise and speeds up training.
print("\nTraining Hybrid Model (RF feature selection + XGBoost)...")

# Step 1: Use RF to select important features
print("  Step 1: Random Forest feature selection...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train_res, y_train_res)

# Step 2: Select top features
selector = SelectFromModel(rf, prefit=True, threshold="mean")
X_train_selected = selector.transform(X_train_res)
X_test_selected  = selector.transform(X_test)
print(f"  Selected {X_train_selected.shape[1]} features from {X_train_res.shape[1]}")

# Step 3: Train XGBoost on selected features
print("  Step 2: XGBoost on selected features...")
hybrid = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric="auc",
    verbosity=0
)
hybrid.fit(X_train_selected, y_train_res)
results["Hybrid"] = evaluate_model("Hybrid (RF+XGB)", hybrid, X_test_selected, y_test)
joblib.dump(hybrid, "models/hybrid_model.pkl")
joblib.dump(selector, "models/feature_selector.pkl")
print("Saved: models/hybrid_model.pkl")

# ─────────────────────────────────────────
# 5. Compare All Models
# ─────────────────────────────────────────
print("\n\n=== FINAL MODEL COMPARISON ===")
print(f"{'Model':<20} {'AUC':>8} {'Recall':>8} {'Precision':>10} {'F1':>8}")
print("-" * 56)
for name, res in results.items():
    print(f"{name:<20} {res['auc']:>8.4f} {res['recall']:>8.4f} {res['precision']:>10.4f} {res['f1']:>8.4f}")

# ─────────────────────────────────────────
# 6. Plot Confusion Matrices
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, res) in enumerate(results.items()):
    sns.heatmap(
        res["cm"], annot=True, fmt="d",
        ax=axes[i], cmap="Blues",
        xticklabels=["Legit","Fraud"],
        yticklabels=["Legit","Fraud"]
    )
    axes[i].set_title(f"{name}\nAUC: {res['auc']:.4f} | Recall: {res['recall']:.4f}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("models/model_comparison.png")
print("\nSaved: models/model_comparison.png")

# ─────────────────────────────────────────
# 7. Plot AUC-ROC Curves
# ─────────────────────────────────────────
from sklearn.metrics import roc_curve

plt.figure(figsize=(10, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})")

plt.plot([0,1],[0,1],"k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend()
plt.tight_layout()
plt.savefig("models/roc_curves.png")
print("Saved: models/roc_curves.png")

print("\n✅ Task 3 Complete!")