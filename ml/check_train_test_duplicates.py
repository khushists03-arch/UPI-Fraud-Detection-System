import os
import sys
import pandas as pd


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

from train_xgboost import prepare_data
from preprocessing import split_data


# --------------------------------------------------
# Main
# --------------------------------------------------

def check_train_test_duplicates():

    print("Preparing data...")

    X, y = prepare_data()


    # --------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------

    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )


    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Testing shape:",
        X_test.shape
    )


    # --------------------------------------------------
    # Check duplicate rows inside complete dataset
    # --------------------------------------------------

    print("\n========================================")
    print("DUPLICATE ROW CHECK")
    print("========================================")

    duplicate_rows = X.duplicated().sum()

    print(
        "Duplicate feature rows in complete dataset:",
        duplicate_rows
    )


    # --------------------------------------------------
    # Check duplicates inside training data
    # --------------------------------------------------

    train_duplicates = X_train.duplicated().sum()

    print(
        "Duplicate feature rows in training data:",
        train_duplicates
    )


    # --------------------------------------------------
    # Check duplicates inside testing data
    # --------------------------------------------------

    test_duplicates = X_test.duplicated().sum()

    print(
        "Duplicate feature rows in testing data:",
        test_duplicates
    )


    # --------------------------------------------------
    # Check overlap between train and test
    # --------------------------------------------------

    print("\n========================================")
    print("TRAIN/TEST OVERLAP CHECK")
    print("========================================")


    # Add an identifier to each row
    train_check = X_train.copy()
    test_check = X_test.copy()

    train_check["_row_source"] = "train"
    test_check["_row_source"] = "test"


    # Find rows appearing in both datasets
    combined = pd.concat(
        [
            train_check,
            test_check
        ],
        ignore_index=True
    )


    feature_columns = [
        col for col in combined.columns
        if col != "_row_source"
    ]


    duplicate_mask = combined.duplicated(
        subset=feature_columns,
        keep=False
    )


    overlapping_rows = combined[
        duplicate_mask
    ]


    # Count unique overlapping feature rows
    unique_overlap = overlapping_rows[
        feature_columns
    ].drop_duplicates()


    print(
        "Unique feature rows appearing in both train and test:",
        len(unique_overlap)
    )


    # --------------------------------------------------
    # If overlap exists, inspect target consistency
    # --------------------------------------------------

    if len(unique_overlap) > 0:

        print(
            "\nWARNING: Train/test feature overlap detected!"
        )

        print(
            "This can cause artificially high model performance."
        )

    else:

        print(
            "\nNo exact feature-row overlap detected."
        )


    # --------------------------------------------------
    # Check target distribution
    # --------------------------------------------------

    print("\n========================================")
    print("TARGET DISTRIBUTION")
    print("========================================")

    print(
        "\nTraining target:"
    )

    print(
        y_train.value_counts()
    )


    print(
        "\nTesting target:"
    )

    print(
        y_test.value_counts()
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    check_train_test_duplicates()