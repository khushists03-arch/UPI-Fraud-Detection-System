import os
import sys
import numpy as np
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
# Import the SAME data preparation pipeline
# ==========================================================

from train_xgboost import prepare_data

from sklearn.ensemble import IsolationForest


# ==========================================================
# Project paths
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    ML_FOLDER
)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "isolation_forest_model.pkl"
)


# ==========================================================
# Train Isolation Forest
# ==========================================================

def train_isolation_forest():

    print("\n========================================")
    print("   ISOLATION FOREST TRAINING")
    print("========================================")

    print("\nPreparing data...")

    # ------------------------------------------------------
    # IMPORTANT
    #
    # prepare_data() may currently return:
    #
    #     X, y
    #
    # OR:
    #
    #     X, y, constant_columns
    #
    # We only need X and y here.
    # ------------------------------------------------------

    prepared_data = prepare_data()

    X = prepared_data[0]
    y = prepared_data[1]

    print("\nData preparation completed.")

    print(
        "Final feature shape:",
        X.shape
    )

    print(
        "Number of transactions:",
        X.shape[0]
    )

    print(
        "Number of features:",
        X.shape[1]
    )


    # ======================================================
    # Verify that all features are numeric
    # ======================================================

    if not all(
        np.issubdtype(
            dtype,
            np.number
        )
        for dtype in X.dtypes
    ):

        raise ValueError(
            "Some features are not numeric."
        )

    print(
        "All features numeric: True"
    )


    # ======================================================
    # Convert DataFrame to NumPy
    # ======================================================

    X_numpy = X.to_numpy(
        dtype=float
    )


    # ======================================================
    # Check feature count
    # ======================================================

    feature_count = X_numpy.shape[1]

    print(
        "\nFeature count:",
        feature_count
    )


    if feature_count <= 0:

        raise ValueError(
            "No features available for training."
        )


    # ======================================================
    # Create Isolation Forest
    # ======================================================

    print(
        "\nCreating Isolation Forest model..."
    )

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )


    # ======================================================
    # Train model
    # ======================================================

    print(
        "\nTraining Isolation Forest model..."
    )

    model.fit(
        X_numpy
    )

    print(
        "\nIsolation Forest training completed successfully!"
    )


    # ======================================================
    # Make predictions on training data
    # ======================================================

    print(
        "\nChecking anomaly predictions..."
    )

    predictions = model.predict(
        X_numpy
    )


    # Isolation Forest:
    #
    #  1  = normal
    # -1  = anomaly
    #

    normal_count = np.sum(
        predictions == 1
    )

    anomaly_count = np.sum(
        predictions == -1
    )


    print(
        "\n========================================"
    )

    print(
        "ISOLATION FOREST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        "Normal transactions:",
        normal_count
    )

    print(
        "Anomalous transactions:",
        anomaly_count
    )


    # ======================================================
    # Convert predictions to project format
    #
    # 0 = Normal
    # 1 = Fraud / Anomaly
    # ======================================================

    fraud_predictions = np.where(
        predictions == -1,
        1,
        0
    )


    print(
        "\nProject-format anomaly count:",
        np.sum(fraud_predictions == 1)
    )


    # ======================================================
    # Create models folder
    # ======================================================

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )


    # ======================================================
    # Save model
    # ======================================================

    joblib.dump(
        model,
        MODEL_PATH
    )


    print(
        "\n========================================"
    )

    print(
        "MODEL SAVED SUCCESSFULLY"
    )

    print(
        "========================================"
    )

    print(
        MODEL_PATH
    )


    # ======================================================
    # Verify model feature count
    # ======================================================

    print(
        "\n========================================"
    )

    print(
        "FEATURE COUNT VERIFICATION"
    )

    print(
        "========================================"
    )

    print(
        "Training feature count:",
        feature_count
    )

    print(
        "Model expected feature count:",
        model.n_features_in_
    )


    if (
        feature_count
        ==
        model.n_features_in_
    ):

        print(
            "Feature count verification: PASSED"
        )

    else:

        raise ValueError(
            "Feature count verification FAILED."
        )


    print(
        "\n========================================"
    )

    print(
        "ISOLATION FOREST COMPLETE"
    )

    print(
        "========================================"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_isolation_forest()
