# monitoring/inference_api.py
"""
Task 6: Inference API with Prometheus metrics instrumentation
Exposes:
  - System metrics: request rate, latency, error rate
  - Model metrics: fraud recall, false positive rate, prediction confidence
  - Data metrics: feature drift indicators, missing value trends
"""

from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST
)
import joblib, numpy as np, pandas as pd
import time, random, os

app = Flask(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────

# A) System-Level Metrics
REQUEST_COUNT = Counter(
    'fraud_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)
REQUEST_LATENCY = Histogram(
    'fraud_api_request_latency_seconds',
    'API response latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
ERROR_RATE = Counter(
    'fraud_api_errors_total',
    'Total API errors',
    ['error_type']
)

# B) Model-Level Metrics
FRAUD_PREDICTIONS = Counter(
    'fraud_predictions_total',
    'Total predictions made',
    ['predicted_class']   # 'fraud' or 'legitimate'
)
FRAUD_RECALL_GAUGE = Gauge(
    'fraud_model_recall',
    'Current fraud recall (rolling window)'
)
FALSE_POSITIVE_RATE = Gauge(
    'fraud_false_positive_rate',
    'Current false positive rate (rolling window)'
)
PRECISION_GAUGE = Gauge(
    'fraud_model_precision',
    'Current fraud precision'
)
PREDICTION_CONFIDENCE = Histogram(
    'fraud_prediction_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
TRUE_LABELS_BUFFER = []   # rolling buffer for recall calculation
PRED_LABELS_BUFFER = []

# C) Data-Level Metrics
MISSING_VALUE_RATE = Gauge(
    'fraud_input_missing_value_rate',
    'Rate of missing values in incoming requests'
)
FEATURE_DRIFT_SCORE = Gauge(
    'fraud_feature_drift_score',
    'KS-test drift score for key features',
    ['feature_name']
)
INPUT_ANOMALY_COUNT = Counter(
    'fraud_input_anomalies_total',
    'Count of anomalous input values detected',
    ['anomaly_type']
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "models/hybrid_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️  Model not found at {MODEL_PATH} — running in simulation mode")

# Training distribution stats (from Task 3) for drift detection
TRAINING_STATS = {
    "TransactionAmt": {"mean": 134.0, "std": 395.0},
    "card1":          {"mean": 9906.0, "std": 5672.0},
    "card2":          {"mean": 361.0,  "std": 160.0},
    "addr1":          {"mean": 272.0,  "std": 163.0},
}

# ── Helper: rolling recall update ────────────────────────────────────────────
WINDOW_SIZE = 100

def update_rolling_metrics(y_true, y_pred):
    TRUE_LABELS_BUFFER.append(y_true)
    PRED_LABELS_BUFFER.append(y_pred)
    # Keep only last WINDOW_SIZE predictions
    if len(TRUE_LABELS_BUFFER) > WINDOW_SIZE:
        TRUE_LABELS_BUFFER.pop(0)
        PRED_LABELS_BUFFER.pop(0)
    if len(TRUE_LABELS_BUFFER) >= 10:
        yt = np.array(TRUE_LABELS_BUFFER)
        yp = np.array(PRED_LABELS_BUFFER)
        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        recall    = tp / (tp + fn + 1e-9)
        fpr       = fp / (fp + tn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        FRAUD_RECALL_GAUGE.set(recall)
        FALSE_POSITIVE_RATE.set(fpr)
        PRECISION_GAUGE.set(precision)

def detect_drift(feature_name, value):
    """Simple z-score drift detection against training distribution."""
    if feature_name not in TRAINING_STATS:
        return 0.0
    stats = TRAINING_STATS[feature_name]
    z = abs((value - stats["mean"]) / (stats["std"] + 1e-9))
    drift_score = min(z / 5.0, 1.0)   # normalize to [0,1]
    FEATURE_DRIFT_SCORE.labels(feature_name=feature_name).set(drift_score)
    return drift_score

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    status_code = 200

    try:
        data = request.get_json(force=True)
        features = data.get("features", {})
        true_label = data.get("true_label", None)   # optional, for recall tracking

        # ── Data quality checks ──────────────────────────────
        if not features:
            ERROR_RATE.labels(error_type="empty_payload").inc()
            return jsonify({"error": "No features provided"}), 400

        # Missing value tracking
        expected_features = ["TransactionAmt", "card1", "card2", "addr1"]
        missing = sum(1 for f in expected_features if f not in features or features[f] is None)
        miss_rate = missing / len(expected_features)
        MISSING_VALUE_RATE.set(miss_rate)
        if miss_rate > 0:
            INPUT_ANOMALY_COUNT.labels(anomaly_type="missing_values").inc()

        # ── Drift detection ──────────────────────────────────
        for feat in ["TransactionAmt", "card1", "card2", "addr1"]:
            if feat in features and features[feat] is not None:
                drift = detect_drift(feat, float(features[feat]))
                if drift > 0.7:
                    INPUT_ANOMALY_COUNT.labels(anomaly_type=f"drift_{feat}").inc()

        # ── Prediction ───────────────────────────────────────
        if model is not None:
            # Build feature vector (simplified — in production use full pipeline)
            feat_values = [features.get(f, 0) or 0 for f in expected_features]
            # Pad to model's expected input size with zeros
            n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 420
            full_vector = feat_values + [0] * (n_features - len(feat_values))
            X = np.array(full_vector[:n_features]).reshape(1, -1)
            proba = model.predict_proba(X)[0][1]
            prediction = int(proba >= 0.5)
        else:
            # Simulation mode — random prediction weighted toward legitimate
            proba = random.betavariate(1, 9) if random.random() > 0.035 else random.betavariate(5, 2)
            prediction = int(proba >= 0.5)

        # ── Update metrics ───────────────────────────────────
        label = "fraud" if prediction == 1 else "legitimate"
        FRAUD_PREDICTIONS.labels(predicted_class=label).inc()
        PREDICTION_CONFIDENCE.observe(proba)

        if true_label is not None:
            update_rolling_metrics(int(true_label), prediction)

        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(
            method="POST", endpoint="/predict", status_code="200"
        ).inc()

        return jsonify({
            "prediction": prediction,
            "label": label,
            "fraud_probability": round(float(proba), 4),
            "latency_ms": round(latency * 1000, 2)
        }), 200

    except Exception as e:
        status_code = 500
        ERROR_RATE.labels(error_type="internal_error").inc()
        REQUEST_COUNT.labels(
            method="POST", endpoint="/predict", status_code="500"
        ).inc()
        return jsonify({"error": str(e)}), 500

    finally:
        if status_code != 200:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus scrape endpoint — exposes all metrics."""
    REQUEST_COUNT.labels(
        method="GET", endpoint="/metrics", status_code="200"
    ).inc()
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/simulate", methods=["POST"])
def simulate_traffic():
    """
    Simulate realistic transaction traffic for dashboard demo.
    POST {"n": 200, "drift": false, "recall_drop": false}
    """
    body = request.get_json(force=True) or {}
    n          = int(body.get("n", 100))
    drift      = body.get("drift", False)
    recall_drop = body.get("recall_drop", False)

    results = {"processed": 0, "fraud_detected": 0, "errors": 0}

    for _ in range(n):
        # Simulate fraud rate ~3.5%
        is_fraud = 1 if random.random() < 0.035 else 0

        # Simulate drift: inflate TransactionAmt far from training distribution
        amt = random.uniform(5000, 20000) if drift else random.lognormvariate(4.5, 1.5)
        c1  = random.uniform(20000, 30000) if drift else random.uniform(1000, 18000)

        features = {
            "TransactionAmt": amt,
            "card1": c1,
            "card2": random.uniform(100, 600),
            "addr1": random.uniform(100, 500)
        }

        # Simulate recall drop: model misses more fraud
        if recall_drop and is_fraud:
            proba = random.uniform(0.1, 0.45)   # model underestimates fraud
        else:
            proba = random.betavariate(5, 2) if is_fraud else random.betavariate(1, 9)

        prediction = int(proba >= 0.5)
        label = "fraud" if prediction == 1 else "legitimate"

        FRAUD_PREDICTIONS.labels(predicted_class=label).inc()
        PREDICTION_CONFIDENCE.observe(proba)
        update_rolling_metrics(is_fraud, prediction)

        for feat, val in features.items():
            detect_drift(feat, val)

        miss_rate = random.uniform(0.05, 0.3) if drift else random.uniform(0, 0.05)
        MISSING_VALUE_RATE.set(miss_rate)

        results["processed"] += 1
        if prediction == 1:
            results["fraud_detected"] += 1

    return jsonify(results), 200

if __name__ == "__main__":
    # Initialize gauges with Task 3 best model values
    FRAUD_RECALL_GAUGE.set(0.8289)
    FALSE_POSITIVE_RATE.set(0.1077)
    PRECISION_GAUGE.set(0.2182)
    for feat in TRAINING_STATS:
        FEATURE_DRIFT_SCORE.labels(feature_name=feat).set(0.0)
    print("🚀 Fraud Detection Inference API starting on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)