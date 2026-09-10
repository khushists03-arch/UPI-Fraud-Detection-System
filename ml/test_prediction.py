import os
import sys
import pandas as pd


# --------------------------------------------------
# Add ml folder to Python path
# --------------------------------------------------

ML_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

if ML_FOLDER not in sys.path:
    sys.path.insert(0, ML_FOLDER)


# --------------------------------------------------
# Import prediction function
# --------------------------------------------------

from predict import predict_transaction


# --------------------------------------------------
# Dataset path
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    ML_FOLDER
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "fraud_dataset.csv"
)


# --------------------------------------------------
# Test multiple transactions
# --------------------------------------------------

def test_multiple_transactions():

    print(
        "Loading dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )


    print(
        "Dataset loaded successfully."
    )


    print(
        "\nTesting multiple transactions..."
    )


    # --------------------------------------------------
    # Select transactions
    # --------------------------------------------------

    test_transactions = pd.concat(
        [
            df[df["is_fraud"] == 0].head(3),
            df[df["is_fraud"] == 1].head(3)
        ]
    )


    # --------------------------------------------------
    # Test each transaction
    # --------------------------------------------------

    results = []


    for index, row in test_transactions.iterrows():

        actual_value = int(
            row["is_fraud"]
        )


        transaction = row.drop(
            labels=["is_fraud"]
        )


        print(
            "\n"
            + "=" * 50
        )


        print(
            f"Testing transaction index: {index}"
        )


        print(
            "Actual:",
            "Fraud" if actual_value == 1
            else "Not Fraud"
        )


        result = predict_transaction(
            transaction
        )


        predicted_value = (
            1
            if result["prediction"] == "Fraud"
            else 0
        )


        results.append(
            {
                "index": index,
                "actual": actual_value,
                "predicted": predicted_value,
                "probability":
                    result["fraud_probability"],
                "risk":
                    result["risk_level"],
                "anomaly":
                    result["anomaly"]
            }
        )


    # --------------------------------------------------
    # Create results table
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------
    # Calculate test accuracy
    # --------------------------------------------------

    correct = (
        results_df["actual"]
        ==
        results_df["predicted"]
    ).sum()


    total = len(
        results_df
    )


    accuracy = correct / total


    # --------------------------------------------------
    # Display summary
    # --------------------------------------------------

    print(
        "\n\n"
        + "=" * 60
    )

    print(
        "        MULTIPLE TRANSACTION TEST"
    )

    print(
        "=" * 60
    )


    print(
        results_df.to_string(
            index=False
        )
    )


    print(
        "\nTest accuracy:",
        f"{accuracy * 100:.2f}%"
    )


    print(
        "\nTesting completed successfully." 
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    test_multiple_transactions()