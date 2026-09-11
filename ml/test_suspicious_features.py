import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# Make ml folder available for imports
# --------------------------------------------------

ML_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

if ML_FOLDER not in sys.path:
    sys.path.insert(0, ML_FOLDER)


# --------------------------------------------------
# Project import
# --------------------------------------------------

from train_xgboost import prepare_data


# --------------------------------------------------
# Main
# --------------------------------------------------

def test_suspicious_features():

    print("Preparing data...")

    X, y = prepare_data()


    # --------------------------------------------------
    # Suspicious features
    # --------------------------------------------------

    suspicious_features = [
        "unusual_transaction_amount_flag",
        "merchant_category_code_unknown",
        "time_pressure_indicators",
        "handle_registration_pattern_recent",
        "receiver_transaction_history",
        "unusual_device_flag",
        "unusual_ip_flag",
        "unusual_location_flag"
    ]


    # Keep only features that actually exist
    available_features = [
        feature
        for feature in suspicious_features
        if feature in X.columns
    ]


    print("\nSuspicious features being tested:")

    for feature in available_features:
        print("-", feature)


    X_suspicious = X[
        available_features
    ]


    print(
        "\nSuspicious feature shape:",
        X_suspicious.shape
    )


    # --------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_suspicious,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # ==================================================
    # TEST 1 - Decision Tree depth 1
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "TEST 1 - DECISION TREE DEPTH 1"
    )

    print(
        "========================================"
    )


    model_depth_1 = DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    )


    model_depth_1.fit(
        X_train,
        y_train
    )


    prediction_1 = model_depth_1.predict(
        X_test
    )


    accuracy_1 = accuracy_score(
        y_test,
        prediction_1
    )


    print(
        "\nAccuracy:",
        accuracy_1
    )


    # ==================================================
    # TEST 2 - Decision Tree depth 3
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "TEST 2 - DECISION TREE DEPTH 3"
    )

    print(
        "========================================"
    )


    model_depth_3 = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )


    model_depth_3.fit(
        X_train,
        y_train
    )


    prediction_3 = model_depth_3.predict(
        X_test
    )


    accuracy_3 = accuracy_score(
        y_test,
        prediction_3
    )


    print(
        "\nAccuracy:",
        accuracy_3
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            prediction_3
        )
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            prediction_3
        )
    )


    # ==================================================
    # TEST 3 - Decision Tree depth 5
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "TEST 3 - DECISION TREE DEPTH 5"
    )

    print(
        "========================================"
    )


    model_depth_5 = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )


    model_depth_5.fit(
        X_train,
        y_train
    )


    prediction_5 = model_depth_5.predict(
        X_test
    )


    accuracy_5 = accuracy_score(
        y_test,
        prediction_5
    )


    print(
        "\nAccuracy:",
        accuracy_5
    )


    # ==================================================
    # Final result
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL RESULT"
    )

    print(
        "========================================"
    )


    print(
        "\nDecision Tree depth 1:",
        accuracy_1
    )

    print(
        "Decision Tree depth 3:",
        accuracy_3
    )

    print(
        "Decision Tree depth 5:",
        accuracy_5
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    test_suspicious_features()