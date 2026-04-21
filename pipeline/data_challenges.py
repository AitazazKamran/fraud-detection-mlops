import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────
print("Loading data...")
trans = pd.read_csv("data/train_transaction.csv")
ident = pd.read_csv("data/train_identity.csv")
df = trans.merge(ident, on="TransactionID", how="left")
print(f"Dataset shape: {df.shape}")

# ─────────────────────────────────────────
# 1. Missing Values — Advanced Strategy
# ─────────────────────────────────────────
print("\n--- Handling Missing Values ---")

# Drop columns with >90% missing
missing_pct = df.isnull().mean()
drop_cols = missing_pct[missing_pct > 0.9].index.tolist()
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped {len(drop_cols)} columns with >90% missing")

# Numerical: median imputation
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if "isFraud" in num_cols:
    num_cols.remove("isFraud")
if "TransactionID" in num_cols:
    num_cols.remove("TransactionID")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical: mode imputation
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Missing values remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 2. High Cardinality + Feature Encoding
# ─────────────────────────────────────────
print("\n--- Encoding Features ---")

for col in cat_cols:
    if df[col].nunique() > 50:
        # Target encoding for high cardinality
        means = df.groupby(col)["isFraud"].mean()
        df[col] = df[col].map(means)
        print(f"Target encoded: {col} ({df[col].nunique()} unique)")
    else:
        # Label encoding for low cardinality
        df[col] = pd.factorize(df[col])[0]

# ─────────────────────────────────────────
# 3. Prepare Features
# ─────────────────────────────────────────
X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
y = df["isFraud"]

print(f"\nClass distribution:")
print(y.value_counts())
print(f"Fraud ratio: {y.mean():.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 4. Compare Imbalance Strategies
# ─────────────────────────────────────────
print("\n--- Comparing Imbalance Strategies ---")

results = {}

strategies = {
    "SMOTE": SMOTE(random_state=42),
    "Undersampling": RandomUnderSampler(random_state=42)
}

for name, sampler in strategies.items():
    print(f"\nTraining with {name}...")
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"Resampled shape: {X_res.shape}, Fraud ratio: {y_res.mean():.4f}")

    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {auc:.4f}")

    results[name] = {
        "model": model,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# ─────────────────────────────────────────
# 5. Plot Comparison
# ─────────────────────────────────────────
print("\n--- Saving Comparison Plot ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (name, res) in enumerate(results.items()):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[i], cmap="Blues")
    axes[i].set_title(f"{name} (AUC: {res['auc']:.4f})")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("models/imbalance_comparison.png")
print("Saved: models/imbalance_comparison.png")

# Summary
print("\n=== STRATEGY COMPARISON SUMMARY ===")
for name, res in results.items():
    print(f"{name}: AUC = {res['auc']:.4f}")

winner = max(results, key=lambda x: results[x]["auc"])
print(f"\nBest strategy: {winner}")