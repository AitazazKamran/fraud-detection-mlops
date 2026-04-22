# Fraud Detection MLOps System

[![CI/CD](https://github.com/AitazazKamran/fraud-detection-mlops/actions/workflows/fraud_detection_cicd.yml/badge.svg)](https://github.com/AitazazKamran/fraud-detection-mlops/actions)

A production-grade MLOps pipeline for financial fraud detection built on Kubeflow, Prometheus, Grafana, and GitHub Actions. The system maintains high fraud recall, scales under transaction volume, and responds automatically to performance degradation.

---

## Table of Contents
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Task Summary](#task-summary)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Monitoring](#monitoring)
- [CI/CD](#cicd)

---

## Architecture

```
IEEE CIS Dataset (CSV)
        ↓
Kubeflow Pipeline (7 steps)
        ↓
Data Ingestion → Validation → Preprocessing →
Feature Engineering → Model Training →
Evaluation → Conditional Deployment (AUC > 0.85)
        ↓
Models saved as .pkl in models/
        ↓
Inference API (Flask + Prometheus metrics)
        ↓
Prometheus scrapes → Grafana dashboards → Alert rules
        ↓
GitHub Actions CI/CD ← Monitoring alerts trigger retraining
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13.4 |
| ML Pipeline | Kubeflow Pipelines (KFP) v2.5.0 |
| Kubernetes | Minikube v1.38.1 on Windows 11 |
| Container Runtime | Docker Desktop v4.61.0 |
| Models | XGBoost, LightGBM, RF+XGB Hybrid |
| Imbalance Handling | SMOTE, RandomUnderSampler |
| Explainability | SHAP (TreeExplainer) |
| Monitoring | Prometheus v2.51.0 + Grafana v10.4.0 |
| CI/CD | GitHub Actions |
| Version Control | Git + GitHub |

---

## Project Structure

```
fraud-detection-mlops/
├── pipeline/
│   ├── fraud_detection_pipeline.py   # Kubeflow 7-step pipeline
│   ├── data_challenges.py            # Task 2: imbalance handling
│   ├── model_training.py             # Task 3: XGBoost, LightGBM, Hybrid
│   ├── cost_sensitive_learning.py    # Task 4: cost-sensitive training
│   ├── drift_simulation.py           # Task 7: time-based drift
│   ├── retraining_strategy.py        # Task 8: threshold/periodic/hybrid
│   └── explainability.py             # Task 9: SHAP analysis
├── monitoring/
│   ├── inference_api.py              # Flask API with Prometheus metrics
│   ├── prometheus.yml                # Prometheus scrape config
│   ├── alert_rules.yml               # Alert rule definitions
│   ├── grafana_dashboard.json        # Grafana dashboard export
│   ├── docker-compose.yml            # Runs API + Prometheus + Grafana
│   ├── Dockerfile.api                # Inference API container
│   └── grafana_provisioning/         # Auto-provisioned datasource + dashboard
├── .github/
│   └── workflows/
│       └── fraud_detection_cicd.yml  # 4-stage CI/CD pipeline
├── models/                           # Saved models + plots
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── hybrid_model.pkl
│   ├── cost_sensitive_best_model.pkl
│   ├── imbalance_comparison.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── cost_sensitive_comparison.png
│   ├── drift_simulation.png
│   ├── retraining_strategy.png
│   ├── shap_explainability.png
│   └── shap_feature_importance.csv
├── data/                             # IEEE CIS dataset (not in repo — large)
│   ├── train_transaction.csv
│   └── train_identity.csv
├── cicd/
│   └── logs/                         # Automated retraining logs
├── requirements.txt
└── README.md
```

---

## Task Summary

### Task 1 — Kubeflow Environment Setup ✅
- Minikube deployed with `--cpus=4 --memory=8192`
- All 14 Kubeflow pods running (`1/1 Running`)
- 7-component pipeline: ingestion → validation → preprocessing → feature engineering → training → evaluation → conditional deployment
- Conditional deployment gate: AUC > 0.85
- Retry mechanism: `set_retry(num_retries=3)` on training step

### Task 2 — Data Challenges ✅
- Dropped 12 columns with >90% missing values
- Median imputation (numerical), mode (categorical)
- Target encoding for high-cardinality features (>50 unique values)
- **SMOTE vs Undersampling comparison:**

| Strategy | AUC | Fraud Recall |
|---|---|---|
| SMOTE | 0.9265 | ~76% |
| **Undersampling** | **0.9406** | **84%** |

**Winner: Undersampling** — used for all subsequent tasks.

### Task 3 — Model Complexity ✅

| Model | AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| XGBoost | 0.9330 | 0.8243 | 0.2178 | 0.3446 |
| LightGBM | 0.9325 | 0.8275 | 0.2146 | 0.3408 |
| **Hybrid (RF+XGB)** | **0.9350** | **0.8289** | **0.2182** | **0.3455** |

**Winner: Hybrid** — RF selects 118/420 features, XGBoost trains on reduced set.

### Task 4 — Cost-Sensitive Learning ✅
Assumptions: $500 fraud loss per missed case | $10 false alarm cost per case.

| Model | Recall | Business Cost |
|---|---|---|
| **XGBoost Standard** | 0.8168 | **$503,320** ✅ |
| XGBoost spw=5 | 0.9594 | $569,030 |
| XGBoost spw=10 | 0.9867 | $725,440 |
| LightGBM Standard | 0.8159 | $506,960 |
| LightGBM spw=10 | 0.9865 | $750,000 |

Key insight: At $500/$10 cost ratio, standard training wins. At $5,000/$10 (more realistic for large transactions), cost-sensitive spw=5 would win. LightGBM `class_weight='balanced'` had no effect after undersampling since classes were already 1:1.

### Task 5 — CI/CD Pipeline ✅
4-stage GitHub Actions workflow:
- **Stage 1 (CI):** Linting (flake8) + unit tests + data schema validation — triggers on push/PR
- **Stage 2 (Build):** Docker images for training pipeline + inference API
- **Stage 3 (Deploy):** Kubeflow pipeline trigger + model deployment gate (recall > 0.75)
- **Stage 4 (Intelligent):** Manual dispatch simulating Prometheus alert webhook for recall drop or drift detection

### Task 6 — Observability & Monitoring ✅
Full monitoring stack via Docker Compose:

**Prometheus metrics exposed:**
- System: request rate, latency (p50/p95/p99), error rate
- Model: fraud recall, false positive rate, precision (rolling 100-prediction window)
- Data: feature drift score (z-score vs training distribution), missing value rate, input anomalies

**Grafana dashboards:**
- System Health: latency percentiles, throughput, error rate
- Model Performance: recall/precision trends, fraud detection rate
- Data Drift: per-feature drift scores, missing value trends

**Alert rules (6 total):**
- `FraudRecallDrop` — recall < 0.75 for 2 min → CRITICAL
- `FeatureDriftDetected` — drift score > 0.7 → CRITICAL
- `APIDown` — API unreachable 1 min → CRITICAL
- `APILatencySpike` — p95 > 1s → CRITICAL
- `HighFalsePositiveRate` — FPR > 25% → WARNING
- `HighMissingValueRate` — missing > 20% → WARNING

### Task 7 — Drift Simulation ✅
Time-based split: train on earliest 50%, test on middle 25% (baseline) and late 25% (drifted).

**3 drift types introduced:**
1. Fraud transaction amounts inflated 3–8× (new high-value fraud pattern)
2. New fraud ring — card1 values shifted to 25,000–35,000 range
3. addr1 distribution shift for fraud cases

**Results:**

| Period | AUC | Recall | Change |
|---|---|---|---|
| Baseline (Middle) | 0.8613 | 0.7165 | — |
| Drifted (Late) | 0.8755 | 0.7224 | ↑ slight |

Feature importance completely reshuffled — `TransactionAmt` jumped to #1 on drifted data (was outside top 10 at training time), confirming the amount inflation drift was detected.

### Task 8 — Intelligent Retraining Strategy ✅
Simulated 12 time windows with gradually increasing drift intensity.

| Strategy | Retrains | Compute | Mean Recall | Min Recall | Stability (std) |
|---|---|---|---|---|---|
| **Threshold-Based** | 2 | 2.0 | 0.8131 | 0.7551 | **0.037** |
| Periodic | 4 | 4.0 | **0.8603** | 0.6867 | 0.090 |
| Hybrid | 3 | 3.0 | 0.8472 | **0.7611** | 0.063 |

**Recommendation: Hybrid** for production — zero windows below threshold, safety net prevents staleness, moderate compute cost. Threshold-based wins on pure efficiency but Hybrid's max-interval trigger boosted recall from 0.83 → 0.93 in later windows.

### Task 9 — Explainability ✅
SHAP TreeExplainer on 2,000-sample test set.

**Top 5 fraud-driving features:**
1. `DeviceInfo` — mean|SHAP|=0.3988
2. `V258` — mean|SHAP|=0.3687
3. `C1` — mean|SHAP|=0.2982
4. `R_emaildomain` — mean|SHAP|=0.2596
5. `C14` — mean|SHAP|=0.2459

**Local explanation highlight:** Most confident fraud transaction (probability=99.91%) had all top 10 SHAP drivers pushing toward fraud, with `DeviceInfo` (SHAP=+2.00) as the dominant signal — indicating device fingerprinting is the strongest fraud indicator in this dataset.

---

## Results Summary

| Metric | Value |
|---|---|
| Best Model | Hybrid (RF + XGBoost) |
| AUC-ROC | 0.9350 |
| Fraud Recall | 82.89% |
| False Positive Rate | ~10.7% |
| Deployment Threshold | AUC > 0.85 ✅ |
| Retraining Trigger | Recall < 0.75 |
| Monitoring Stack | Prometheus + Grafana (6 alert rules) |
| CI/CD | 4-stage GitHub Actions |

---

## Setup & Installation

### Prerequisites
- Windows 11 with Docker Desktop v4.61.0+
- Minikube v1.38.1+
- Python 3.10+
- Git

### 1. Clone the repository
```powershell
git clone https://github.com/AitazazKamran/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### 2. Create virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add dataset
Download from [Kaggle IEEE CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place in `data/`:
```
data/train_transaction.csv
data/train_identity.csv
```

### 4. Start Minikube + Kubeflow
```powershell
minikube start --driver=docker --cpus=4 --memory=8192
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
```

---

## Running the Pipeline

```powershell
# Task 2: Data challenges
python pipeline/data_challenges.py

# Task 3: Model training
python pipeline/model_training.py

# Task 4: Cost-sensitive learning
python pipeline/cost_sensitive_learning.py

# Task 7: Drift simulation
python pipeline/drift_simulation.py

# Task 8: Retraining strategy
python pipeline/retraining_strategy.py

# Task 9: SHAP explainability
python pipeline/explainability.py
```

---

## Monitoring

```powershell
cd monitoring
docker-compose up -d
```

| Service | URL | Credentials |
|---|---|---|
| Inference API | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / fraud123 |

**Simulate traffic:**
```python
import urllib.request, json

def post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        'http://localhost:5000/simulate', data=data,
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(req) as r:
        print(json.loads(r.read()))

post({'n': 500, 'drift': False, 'recall_drop': False})  # normal
post({'n': 200, 'drift': True,  'recall_drop': False})  # drift
post({'n': 300, 'drift': False, 'recall_drop': True})   # recall drop
```

---

## CI/CD

The GitHub Actions workflow triggers automatically on push to `main`. To simulate an intelligent retraining trigger:

1. Go to **Actions** → "Fraud Detection MLOps Pipeline" → **Run workflow**
2. Set `trigger_reason` = `recall_drop`, `recall_value` = `0.70`
3. Click **Run workflow**

This simulates a Prometheus alert firing Stage 4 (intelligent retraining).

---

## Dataset

IEEE CIS Fraud Detection — [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- 590,540 transactions × 434 features after merge
- 3.5% fraud rate (highly imbalanced)
- Features: transaction amount, card details, email domains, device info, Vesta engineered features (V1–V339)