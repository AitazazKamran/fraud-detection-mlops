import kfp
from kfp import dsl
from kfp.dsl import component, pipeline

# ─────────────────────────────────────────
# Component 1: Data Ingestion
# ─────────────────────────────────────────
@component(
    packages_to_install=["pandas"],
    base_image="python:3.10"
)
def data_ingestion(
    transaction_path: str,
    identity_path: str,
    output_path: dsl.Output[dsl.Dataset]
):
    import pandas as pd
    trans = pd.read_csv(transaction_path)
    ident = pd.read_csv(identity_path)
    df = trans.merge(ident, on="TransactionID", how="left")
    df.to_csv(output_path.path, index=False)
    print(f"Data ingested: {df.shape}")

# ─────────────────────────────────────────
# Component 2: Data Validation
# ─────────────────────────────────────────
@component(
    packages_to_install=["pandas"],
    base_image="python:3.10"
)
def data_validation(
    input_data: dsl.Input[dsl.Dataset],
    output_data: dsl.Output[dsl.Dataset]
):
    import pandas as pd
    df = pd.read_csv(input_data.path)

    # Check required columns
    assert "isFraud" in df.columns, "Target column missing!"
    assert "TransactionID" in df.columns, "ID column missing!"

    # Check missing value threshold
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.9].index.tolist()
    df.drop(columns=high_missing, inplace=True)

    print(f"Validation passed. Shape: {df.shape}")
    print(f"Dropped {len(high_missing)} high-missing columns")
    df.to_csv(output_data.path, index=False)

# ─────────────────────────────────────────
# Component 3: Data Preprocessing
# ─────────────────────────────────────────
@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.10"
)
def data_preprocessing(
    input_data: dsl.Input[dsl.Dataset],
    output_data: dsl.Output[dsl.Dataset]
):
    import pandas as pd
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(input_data.path)

    # Separate features
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target from features
    if "isFraud" in num_cols:
        num_cols.remove("isFraud")

    # Impute numerical
    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute categorical
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    print(f"Preprocessing done. Shape: {df.shape}")
    df.to_csv(output_data.path, index=False)

# ─────────────────────────────────────────
# Component 4: Feature Engineering
# ─────────────────────────────────────────
@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.10"
)
def feature_engineering(
    input_data: dsl.Input[dsl.Dataset],
    output_data: dsl.Output[dsl.Dataset]
):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(input_data.path)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Target encoding for high cardinality + Label encoding for rest
    for col in cat_cols:
        if df[col].nunique() > 50:
            # Target encoding
            means = df.groupby(col)["isFraud"].mean()
            df[col] = df[col].map(means)
        else:
            # Label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    print(f"Feature engineering done. Shape: {df.shape}")
    df.to_csv(output_data.path, index=False)

# ─────────────────────────────────────────
# Component 5: Model Training
# ─────────────────────────────────────────
@component(
    packages_to_install=["pandas", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "joblib"],
    base_image="python:3.10"
)
def model_training(
    input_data: dsl.Input[dsl.Dataset],
    model_output: dsl.Output[dsl.Model],
    strategy: str = "smote"
):
    import pandas as pd
    import joblib
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    if strategy == "smote":
        sampler = SMOTE(random_state=42)
    else:
        sampler = RandomUnderSampler(random_state=42)

    X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Train XGBoost with cost-sensitive learning
    model = XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        n_estimators=100,
        random_state=42,
        eval_metric="auc"
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_output.path)
    print(f"Model trained with strategy: {strategy}")

# ─────────────────────────────────────────
# Component 6: Model Evaluation
# ─────────────────────────────────────────
@component(
    packages_to_install=["pandas", "scikit-learn", "xgboost", "joblib"],
    base_image="python:3.10"
)
def model_evaluation(
    input_data: dsl.Input[dsl.Dataset],
    model_input: dsl.Input[dsl.Model],
    metrics_output: dsl.Output[dsl.Metrics]
) -> float:
    import pandas as pd
    import joblib
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        confusion_matrix, f1_score, recall_score
    )

    df = pd.read_csv(input_data.path)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    y = df["isFraud"]

    model = joblib.load(model_input.path)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_prob)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(classification_report(y, y_pred))
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    metrics_output.log_metric("auc", auc)
    metrics_output.log_metric("recall", recall)
    metrics_output.log_metric("f1", f1)

    return auc

# ─────────────────────────────────────────
# Component 7: Conditional Deployment
# ─────────────────────────────────────────
@component(
    packages_to_install=["joblib"],
    base_image="python:3.10"
)
def conditional_deployment(
    model_input: dsl.Input[dsl.Model],
    auc_score: float,
    threshold: float = 0.85
):
    if auc_score >= threshold:
        print(f"✅ Model deployed! AUC {auc_score:.4f} >= threshold {threshold}")
    else:
        print(f"❌ Deployment skipped. AUC {auc_score:.4f} < threshold {threshold}")
        raise ValueError("Model did not meet deployment threshold")

# ─────────────────────────────────────────
# Pipeline Definition
# ─────────────────────────────────────────
@pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection pipeline with KFP"
)
def fraud_detection_pipeline(
    transaction_path: str = "/data/train_transaction.csv",
    identity_path: str = "/data/train_identity.csv",
    imbalance_strategy: str = "smote",
    deploy_threshold: float = 0.85
):
    # Step 1
    ingest = data_ingestion(
        transaction_path=transaction_path,
        identity_path=identity_path
    )

    # Step 2
    validate = data_validation(
        input_data=ingest.outputs["output_path"]
    )

    # Step 3
    preprocess = data_preprocessing(
        input_data=validate.outputs["output_data"]
    )

    # Step 4
    features = feature_engineering(
        input_data=preprocess.outputs["output_data"]
    )

    # Step 5 - with retry
    train = model_training(
        input_data=features.outputs["output_data"],
        strategy=imbalance_strategy
    )
    train.set_retry(num_retries=3)

    # Step 6
    evaluate = model_evaluation(
        input_data=features.outputs["output_data"],
        model_input=train.outputs["model_output"]
    )

    # Step 7 - conditional
    deploy = conditional_deployment(
        model_input=train.outputs["model_output"],
        auc_score=evaluate.outputs["Output"],
        threshold=deploy_threshold
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="pipeline/fraud_detection_pipeline.yaml"
    )
    print("Pipeline compiled successfully!")