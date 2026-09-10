import os
import sys
import pandas as pd
import joblib


# ==========================================================
# Make ml folder available for imports
# ==========================================================

ML_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

if ML_FOLDER not in sys.path:
    sys.path.insert(0, ML_FOLDER)


# ==========================================================
# Project imports
# ==========================================================

from preprocessing import (
    load_data,
    remove_id_columns,
    remove_high_missing_columns,
    remove_constant_columns
)

from feature_engineering import (
    extract_time_features,
    extract_description_features,
    extract_location_features,
    extract_business_name_features
)


# ==========================================================
# Project paths
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    ML_FOLDER
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "fraud_dataset.csv"
)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "models"
)

XGBOOST_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "xgboost_model.pkl"
)

ISOLATION_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "isolation_forest_model.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_FOLDER,
    "xgboost_feature_columns.pkl"
)

PREPROCESSING_METADATA_PATH = os.path.join(
    MODEL_FOLDER,
    "preprocessing_metadata.pkl"
)


# ==========================================================
# Load models and metadata
# ==========================================================

def load_models_and_metadata():

    print("Loading models and preprocessing information...")

    if not os.path.exists(XGBOOST_MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found:\n"
            f"{XGBOOST_MODEL_PATH}"
        )

    if not os.path.exists(ISOLATION_MODEL_PATH):
        raise FileNotFoundError(
            f"Isolation Forest model not found:\n"
            f"{ISOLATION_MODEL_PATH}"
        )

    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(
            f"Feature columns file not found:\n"
            f"{FEATURE_COLUMNS_PATH}"
        )

    if not os.path.exists(PREPROCESSING_METADATA_PATH):
        raise FileNotFoundError(
            f"Preprocessing metadata not found:\n"
            f"{PREPROCESSING_METADATA_PATH}"
        )

    xgb_model = joblib.load(
        XGBOOST_MODEL_PATH
    )

    isolation_model = joblib.load(
        ISOLATION_MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    metadata = joblib.load(
        PREPROCESSING_METADATA_PATH
    )

    print(
        "XGBoost model loaded successfully."
    )

    print(
        "Isolation Forest model loaded successfully."
    )

    print(
        "Feature columns loaded successfully."
    )

    print(
        "Preprocessing metadata loaded successfully."
    )

    return (
        xgb_model,
        isolation_model,
        feature_columns,
        metadata
    )


# ==========================================================
# Encode low-cardinality categorical columns
# ==========================================================

def encode_low_cardinality_for_prediction(
    X,
    metadata
):

    X = X.copy()

    low_cardinality_columns = metadata.get(
        "low_cardinality_columns",
        []
    )

    categorical_categories = metadata.get(
        "categorical_categories",
        {}
    )

    for column in low_cardinality_columns:

        if column not in X.columns:
            continue

        categories = categorical_categories.get(
            column,
            []
        )

        X[column] = (
            X[column]
            .fillna("__MISSING__")
            .astype(str)
        )

        if "__MISSING__" not in categories:

            categories = (
                list(categories)
                + ["__MISSING__"]
            )

        X[column] = pd.Categorical(
            X[column],
            categories=categories
        )

    columns_present = [
        column
        for column in low_cardinality_columns
        if column in X.columns
    ]

    if columns_present:

        X = pd.get_dummies(
            X,
            columns=columns_present,
            drop_first=True,
            dtype=float
        )

    return X


# ==========================================================
# Encode remaining categorical columns
# ==========================================================

def encode_remaining_categorical_for_prediction(
    X,
    metadata
):

    X = X.copy()

    remaining_categories = metadata.get(
        "remaining_categories",
        {}
    )

    for column, categories in (
        remaining_categories.items()
    ):

        if column not in X.columns:
            continue

        X[column] = (
            X[column]
            .fillna("__MISSING__")
            .astype(str)
        )

        categories = list(categories)

        if "__MISSING__" not in categories:

            categories.append(
                "__MISSING__"
            )

        X[column] = pd.Categorical(
            X[column],
            categories=categories
        )

    columns_present = [
        column
        for column in remaining_categories
        if column in X.columns
    ]

    if columns_present:

        X = pd.get_dummies(
            X,
            columns=columns_present,
            drop_first=False,
            dtype=float
        )

    return X


# ==========================================================
# Prepare transaction
# ==========================================================

def prepare_transaction(
    transaction,
    metadata,
    feature_columns
):

    print(
        "\nPreparing transaction..."
    )

    X = transaction.copy()

    # ------------------------------------------------------
    # Remove target
    # ------------------------------------------------------

    if "is_fraud" in X.columns:

        X = X.drop(
            columns=["is_fraud"]
        )

    # ------------------------------------------------------
    # Remove ID columns
    # ------------------------------------------------------

    X = remove_id_columns(
        X
    )

    # ------------------------------------------------------
    # Remove highly missing columns
    # ------------------------------------------------------

    X = remove_high_missing_columns(
        X
    )

    # ------------------------------------------------------
    # Remove constant columns
    # ------------------------------------------------------

    constant_columns = metadata.get(
        "constant_columns",
        []
    )

    columns_to_remove = [
        column
        for column in constant_columns
        if column in X.columns
    ]

    if columns_to_remove:

        X = X.drop(
            columns=columns_to_remove
        )

    # ------------------------------------------------------
    # Remove leakage columns
    # ------------------------------------------------------

    leakage_columns = metadata.get(
        "leakage_columns",
        []
    )

    leakage_present = [
        column
        for column in leakage_columns
        if column in X.columns
    ]

    if leakage_present:

        X = X.drop(
            columns=leakage_present
        )

    # ------------------------------------------------------
    # Convert numeric-looking columns
    # ------------------------------------------------------

    numeric_columns = metadata.get(
        "numeric_columns",
        []
    )

    for column in numeric_columns:

        if column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    # ------------------------------------------------------
    # Encode low-cardinality columns
    # ------------------------------------------------------

    X = encode_low_cardinality_for_prediction(
        X,
        metadata
    )

    # ------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------

    X = extract_time_features(
        X
    )

    X = extract_description_features(
        X
    )

    X = extract_location_features(
        X
    )

    X = extract_business_name_features(
        X
    )

    # ------------------------------------------------------
    # Encode remaining categorical columns
    # ------------------------------------------------------

    X = encode_remaining_categorical_for_prediction(
        X,
        metadata
    )

    # ------------------------------------------------------
    # Convert boolean columns
    # ------------------------------------------------------

    bool_columns = X.select_dtypes(
        include=["bool"]
    ).columns

    if len(bool_columns) > 0:

        X[bool_columns] = (
            X[bool_columns].astype(int)
        )

    # ------------------------------------------------------
    # Clean feature names
    # ------------------------------------------------------

    from train_xgboost import clean_feature_names

    X = clean_feature_names(
        X
    )

    # ------------------------------------------------------
    # Convert everything to numeric
    # ------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Replace infinity
    # ------------------------------------------------------

    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # ------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------

    X = X.fillna(0)

    # ------------------------------------------------------
    # IMPORTANT:
    # Use EXACT training feature order
    # ------------------------------------------------------

    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # ------------------------------------------------------
    # Convert to float
    # ------------------------------------------------------

    X = X.astype(float)

    print(
        "Prepared transaction shape:",
        X.shape
    )

    print(
        "Feature count verified:",
        X.shape[1]
    )

    # ------------------------------------------------------
    # Safety check
    # ------------------------------------------------------

    if X.shape[1] != len(feature_columns):

        raise ValueError(
            f"Feature count mismatch! "
            f"Expected {len(feature_columns)}, "
            f"but got {X.shape[1]}"
        )

    return X


# ==========================================================
# Predict transaction
# ==========================================================

def predict_transaction(
    transaction
):

    (
        xgb_model,
        isolation_model,
        feature_columns,
        metadata
    ) = load_models_and_metadata()

    # ------------------------------------------------------
    # Prepare transaction
    # ------------------------------------------------------

    X = prepare_transaction(
        transaction,
        metadata,
        feature_columns
    )

    # ------------------------------------------------------
    # IMPORTANT FIX
    #
    # Convert DataFrame to NumPy.
    #
    # This prevents sklearn from comparing feature names
    # stored during Isolation Forest training.
    # ------------------------------------------------------

    X_numpy = X.to_numpy()

    # ------------------------------------------------------
    # XGBoost prediction
    # ------------------------------------------------------

    xgb_prediction = xgb_model.predict(
        X_numpy
    )[0]

    xgb_probability = (
        xgb_model
        .predict_proba(X_numpy)[0][1]
    )

    # ------------------------------------------------------
    # Isolation Forest prediction
    # ------------------------------------------------------

    isolation_raw_prediction = (
        isolation_model.predict(
            X_numpy
        )[0]
    )

    if isolation_raw_prediction == -1:

        anomaly = True

    else:

        anomaly = False

    # ------------------------------------------------------
    # Fraud probability
    # ------------------------------------------------------

    probability_percentage = (
        xgb_probability * 100
    )

    # ------------------------------------------------------
    # Risk level
    # ------------------------------------------------------

    if probability_percentage >= 70:

        risk_level = "High"

    elif probability_percentage >= 30:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    # ------------------------------------------------------
    # Prediction text
    # ------------------------------------------------------

    if xgb_prediction == 1:

        prediction_text = "Fraud"

    else:

        prediction_text = "Not Fraud"

    # ------------------------------------------------------
    # Display result
    # ------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "          TRANSACTION PREDICTION"
    )

    print(
        "========================================"
    )

    print(
        f"\nPrediction: {prediction_text}"
    )

    print(
        f"Fraud Probability: "
        f"{probability_percentage:.2f}%"
    )

    print(
        f"Risk Level: {risk_level}"
    )

    print(
        f"Isolation Forest Anomaly: {anomaly}"
    )

    print(
        "========================================"
    )

    return {

        "prediction":
            int(xgb_prediction),

        "prediction_text":
            prediction_text,

        "fraud_probability":
            float(xgb_probability),

        "risk_level":
            risk_level,

        "isolation_forest_anomaly":
            anomaly
    }


# ==========================================================
# Test multiple dataset transactions
# ==========================================================

def test_dataset_transactions():

    print(
        "\nLoading test transactions from dataset..."
    )

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "Dataset loaded successfully."
    )

    # ------------------------------------------------------
    # Transactions to test
    # ------------------------------------------------------

    test_indices = [
        0,
        1,
        2,
        4,
        17,
        23
    ]

    results = []

    print(
        "\n========================================"
    )

    print(
        "       MULTIPLE TRANSACTION TEST"
    )

    print(
        "========================================"
    )

    for index in test_indices:

        if index >= len(df):

            continue

        transaction = df.iloc[
            [index]
        ]

        if "is_fraud" in transaction.columns:

            actual = int(
                transaction[
                    "is_fraud"
                ].iloc[0]
            )

        else:

            actual = None

        print(
            f"\nTesting transaction index: {index}"
        )

        if actual == 1:

            print(
                "Actual: Fraud"
            )

        else:

            print(
                "Actual: Not Fraud"
            )

        result = predict_transaction(
            transaction
        )

        results.append({

            "index":
                index,

            "actual":
                actual,

            "predicted":
                result[
                    "prediction"
                ],

            "probability":
                result[
                    "fraud_probability"
                ],

            "risk":
                result[
                    "risk_level"
                ],

            "anomaly":
                result[
                    "isolation_forest_anomaly"
                ]
        })

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    print(
        "\n========================================"
    )

    print(
        "          TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Calculate selected transaction accuracy
    # ------------------------------------------------------

    valid_results = result_df[
        result_df["actual"].notna()
    ]

    if len(valid_results) > 0:

        accuracy = (
            valid_results["actual"]
            ==
            valid_results["predicted"]
        ).mean()

        print(
            f"\nTest accuracy on selected "
            f"transactions: "
            f"{accuracy * 100:.2f}%"
        )

    print(
        "\nTesting completed successfully."
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test_dataset_transactions()