import os
import sys
import pandas as pd
import joblib

# --------------------------------------------------
# Make ml folder available for imports
# --------------------------------------------------

ML_FOLDER = os.path.dirname(os.path.abspath(__file__))

if ML_FOLDER not in sys.path:
    sys.path.insert(0, ML_FOLDER)


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from preprocessing import (
    load_data,
    separate_target,
    remove_id_columns,
    remove_high_missing_columns,
    remove_constant_columns,
    encode_categorical_columns
)

from feature_engineering import (
    extract_time_features,
    extract_description_features,
    extract_location_features,
    extract_business_name_features
)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(ML_FOLDER)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "fraud_dataset.csv"
)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "isolation_forest_model.pkl"
)


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

def prepare_data():

    print("Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = load_data(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Separate features and target
    X, y = separate_target(df)

    print(
        "Features shape after separating target:",
        X.shape
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    X = remove_id_columns(X)

    print(
        "Features shape after removing ID columns:",
        X.shape
    )

    X = remove_high_missing_columns(X)

    print(
        "Features shape after removing highly-missing columns:",
        X.shape
    )

    X = remove_constant_columns(X)

    print(
        "Features shape after removing constant columns:",
        X.shape
    )

    X = encode_categorical_columns(X)

    print(
        "Features shape after categorical encoding:",
        X.shape
    )

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------

    X = extract_time_features(X)

    print(
        "Features shape after time feature engineering:",
        X.shape
    )

    X = extract_description_features(X)

    print(
        "Features shape after description feature engineering:",
        X.shape
    )

    X = extract_location_features(X)

    print(
        "Features shape after location feature engineering:",
        X.shape
    )

    X = extract_business_name_features(X)

    print(
        "Features shape after business-name feature engineering:",
        X.shape
    )

    # --------------------------------------------------
    # Handle remaining categorical columns
    # --------------------------------------------------

    remaining_categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if remaining_categorical:

        print(
            "\nRemaining categorical columns:",
            remaining_categorical
        )

        X = pd.get_dummies(
            X,
            columns=remaining_categorical,
            drop_first=False,
            dtype=float
        )

        print(
            "Features shape after final encoding:",
            X.shape
        )

    # --------------------------------------------------
    # Convert boolean columns to integers
    # --------------------------------------------------

    bool_columns = X.select_dtypes(
        include=["bool"]
    ).columns

    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)

    # --------------------------------------------------
    # Make sure everything is numeric
    # --------------------------------------------------

    non_numeric = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric:

        print(
            "\nNon-numeric columns found:",
            non_numeric
        )

        X = pd.get_dummies(
            X,
            columns=non_numeric,
            drop_first=False,
            dtype=float
        )

    # Replace infinite values
    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Fill missing values
    X = X.fillna(0)

    print(
        "\nFinal features shape:",
        X.shape
    )

    print(
        "All features numeric:",
        all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in X.dtypes
        )
    )

    return X, y


# --------------------------------------------------
# Train Isolation Forest
# --------------------------------------------------

def train_isolation_forest():

    X, y = prepare_data()

    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    # IMPORTANT:
    # Isolation Forest is unsupervised.
    # We train using X only, NOT y.

    model.fit(X)

    print(
        "Isolation Forest training completed!"
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    predictions = model.predict(X)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly

    anomaly_predictions = (
        predictions == -1
    ).astype(int)

    print(
        "\nNumber of predicted anomalies:",
        anomaly_predictions.sum()
    )

    print(
        "Total transactions:",
        len(anomaly_predictions)
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "ISOLATION FOREST EVALUATION"
    )

    print(
        "================================"
    )

    print("\nAccuracy:")

    print(
        accuracy_score(
            y,
            anomaly_predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            anomaly_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            anomaly_predictions
        )
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "\nModel saved successfully at:"
    )

    print(MODEL_PATH)

    return model


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    train_isolation_forest()