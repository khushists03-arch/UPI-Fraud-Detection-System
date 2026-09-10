import os
import sys
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from xgboost import XGBClassifier


# --------------------------------------------------
# Make ml folder available for imports
# --------------------------------------------------

ML_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

if ML_FOLDER not in sys.path:
    sys.path.insert(0, ML_FOLDER)


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from preprocessing import split_data
from train_xgboost import prepare_data


# --------------------------------------------------
# Main validation
# --------------------------------------------------

def validate_xgboost():

    print("Preparing data...")

    X, y = prepare_data()

    print(
        "\nOriginal target distribution:"
    )

    print(
        y.value_counts()
    )


    # --------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )


    print(
        "\nTraining data:",
        X_train.shape
    )

    print(
        "Testing data:",
        X_test.shape
    )


    # --------------------------------------------------
    # Test 1: Normal labels
    # --------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "TEST 1: NORMAL LABELS"
    )

    print(
        "========================================"
    )


    normal_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )


    normal_model.fit(
        X_train,
        y_train
    )


    normal_prediction = normal_model.predict(
        X_test
    )


    normal_accuracy = accuracy_score(
        y_test,
        normal_prediction
    )


    print(
        "\nNormal-label Accuracy:",
        normal_accuracy
    )

    print(
        "\nNormal-label Classification Report:"
    )

    print(
        classification_report(
            y_test,
            normal_prediction
        )
    )


    # --------------------------------------------------
    # Test 2: Randomly shuffled labels
    # --------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "TEST 2: SHUFFLED LABELS"
    )

    print(
        "========================================"
    )


    # Convert labels to numpy
    shuffled_y_train = y_train.to_numpy().copy()

    # Shuffle the labels randomly
    rng = np.random.default_rng(42)

    rng.shuffle(
        shuffled_y_train
    )


    print(
        "\nOriginal training labels:"
    )

    print(
        y_train.value_counts()
    )


    print(
        "\nShuffled training labels:"
    )

    unique, counts = np.unique(
        shuffled_y_train,
        return_counts=True
    )

    print(
        dict(zip(unique, counts))
    )


    # --------------------------------------------------
    # Train model with random labels
    # --------------------------------------------------

    shuffled_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )


    print(
        "\nTraining XGBoost with shuffled labels..."
    )


    shuffled_model.fit(
        X_train,
        shuffled_y_train
    )


    shuffled_prediction = shuffled_model.predict(
        X_test
    )


    shuffled_accuracy = accuracy_score(
        y_test,
        shuffled_prediction
    )


    print(
        "\nShuffled-label Accuracy:",
        shuffled_accuracy
    )


    print(
        "\nShuffled-label Classification Report:"
    )

    print(
        classification_report(
            y_test,
            shuffled_prediction
        )
    )


    # --------------------------------------------------
    # Final conclusion
    # --------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "VALIDATION RESULT"
    )

    print(
        "========================================"
    )


    print(
        "\nNormal-label Accuracy:",
        normal_accuracy
    )

    print(
        "Shuffled-label Accuracy:",
        shuffled_accuracy
    )


    if (
        normal_accuracy > 0.95
        and shuffled_accuracy < 0.70
    ):

        print(
            "\nResult: Model learns the real labels."
        )

        print(
            "The perfect score may be caused by"
            " strong relationships in the dataset."
        )

    else:

        print(
            "\nResult: Further investigation is required."
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    validate_xgboost()