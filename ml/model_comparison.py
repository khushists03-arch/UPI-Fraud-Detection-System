import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# Add ml folder to Python path
# --------------------------------------------------

ML_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

if ML_FOLDER not in sys.path:
    sys.path.insert(0, ML_FOLDER)


# --------------------------------------------------
# Import the same data preparation used for XGBoost
# --------------------------------------------------

from train_xgboost import prepare_data

from preprocessing import split_data


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(ML_FOLDER)

XGBOOST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_model.pkl"
)

ISOLATION_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "isolation_forest_model.pkl"
)

RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "model_comparison.csv"
)


# --------------------------------------------------
# Main comparison function
# --------------------------------------------------

def compare_models():

    print("==========================================")
    print("       STARTING MODEL COMPARISON")
    print("==========================================")

    # --------------------------------------------------
    # Check model files
    # --------------------------------------------------

    if not os.path.exists(XGBOOST_MODEL_PATH):
        raise FileNotFoundError(
            "XGBoost model not found:\n"
            + XGBOOST_MODEL_PATH
        )

    if not os.path.exists(ISOLATION_MODEL_PATH):
        raise FileNotFoundError(
            "Isolation Forest model not found:\n"
            + ISOLATION_MODEL_PATH
        )

    print("\nBoth trained model files found.")

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    print("\nPreparing data...")

    X, y = prepare_data()

    print("\nFinal feature shape:")
    print(X.shape)

    # --------------------------------------------------
    # Same train/test split
    # --------------------------------------------------

    print("\nCreating test data...")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print(
        "Training data shape:",
        X_train.shape
    )

    print(
        "Testing data shape:",
        X_test.shape
    )

    # --------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------
    #
    # This prevents feature-name conflicts between
    # the saved models and pandas column names.
    #

    X_test_numpy = X_test.to_numpy()

    y_test_numpy = y_test.to_numpy()

    # --------------------------------------------------
    # Load XGBoost
    # --------------------------------------------------

    print("\nLoading XGBoost model...")

    xgb_model = joblib.load(
        XGBOOST_MODEL_PATH
    )

    print("XGBoost model loaded.")

    # --------------------------------------------------
    # Load Isolation Forest
    # --------------------------------------------------

    print("\nLoading Isolation Forest model...")

    isolation_model = joblib.load(
        ISOLATION_MODEL_PATH
    )

    print("Isolation Forest model loaded.")

    # --------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------

    print("\nMaking XGBoost predictions...")

    xgb_predictions = xgb_model.predict(
        X_test_numpy
    )

    # Convert to integer
    xgb_predictions = xgb_predictions.astype(int)

    # --------------------------------------------------
    # Isolation Forest prediction
    # --------------------------------------------------

    print(
        "Making Isolation Forest predictions..."
    )

    isolation_raw_predictions = (
        isolation_model.predict(
            X_test_numpy
        )
    )

    # Isolation Forest:
    #
    #  1  = normal
    # -1  = anomaly
    #
    # Our project:
    #
    #  0  = non-fraud
    #  1  = fraud
    #
    # Therefore:
    #
    # -1 -> 1
    #  1 -> 0

    isolation_predictions = (
        isolation_raw_predictions == -1
    ).astype(int)

    # --------------------------------------------------
    # XGBoost metrics
    # --------------------------------------------------

    xgb_accuracy = accuracy_score(
        y_test_numpy,
        xgb_predictions
    )

    xgb_precision = precision_score(
        y_test_numpy,
        xgb_predictions,
        zero_division=0
    )

    xgb_recall = recall_score(
        y_test_numpy,
        xgb_predictions,
        zero_division=0
    )

    xgb_f1 = f1_score(
        y_test_numpy,
        xgb_predictions,
        zero_division=0
    )

    # --------------------------------------------------
    # Isolation Forest metrics
    # --------------------------------------------------

    iso_accuracy = accuracy_score(
        y_test_numpy,
        isolation_predictions
    )

    iso_precision = precision_score(
        y_test_numpy,
        isolation_predictions,
        zero_division=0
    )

    iso_recall = recall_score(
        y_test_numpy,
        isolation_predictions,
        zero_division=0
    )

    iso_f1 = f1_score(
        y_test_numpy,
        isolation_predictions,
        zero_division=0
    )

    # --------------------------------------------------
    # Create comparison table
    # --------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Model": [
                "XGBoost",
                "Isolation Forest"
            ],
            "Accuracy": [
                xgb_accuracy,
                iso_accuracy
            ],
            "Precision": [
                xgb_precision,
                iso_precision
            ],
            "Recall": [
                xgb_recall,
                iso_recall
            ],
            "F1 Score": [
                xgb_f1,
                iso_f1
            ]
        }
    )

    # --------------------------------------------------
    # Print comparison
    # --------------------------------------------------

    print("\n")
    print("==========================================")
    print("            MODEL COMPARISON")
    print("==========================================")

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # XGBoost confusion matrix
    # --------------------------------------------------

    print("\n")
    print("==========================================")
    print("       XGBOOST CONFUSION MATRIX")
    print("==========================================")

    xgb_matrix = confusion_matrix(
        y_test_numpy,
        xgb_predictions
    )

    print(xgb_matrix)

    # --------------------------------------------------
    # Isolation Forest confusion matrix
    # --------------------------------------------------

    print("\n")
    print("==========================================")
    print("   ISOLATION FOREST CONFUSION MATRIX")
    print("==========================================")

    isolation_matrix = confusion_matrix(
        y_test_numpy,
        isolation_predictions
    )

    print(isolation_matrix)

    # --------------------------------------------------
    # Save comparison results
    # --------------------------------------------------

    comparison.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n")
    print("==========================================")
    print("       COMPARISON SAVED SUCCESSFULLY")
    print("==========================================")

    print(
        RESULTS_PATH
    )


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    compare_models()