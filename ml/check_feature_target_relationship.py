import os
import sys
import pandas as pd
import numpy as np


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

def check_feature_target_relationship():

    print("Preparing data...")

    X, y = prepare_data()

    print(
        "\nFeatures shape:",
        X.shape
    )


    # --------------------------------------------------
    # Separate numeric and categorical features
    # --------------------------------------------------

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


    print(
        "\nNumber of numeric features:",
        len(numeric_columns)
    )

    print(
        "Number of categorical features:",
        len(categorical_columns)
    )


    # ==================================================
    # NUMERIC FEATURE ANALYSIS
    # ==================================================

    print(
        "\n=============================================="
    )

    print(
        "NUMERIC FEATURES - TARGET CORRELATION"
    )

    print(
        "=============================================="
    )


    correlations = X[numeric_columns].corrwith(y)

    numeric_results = pd.DataFrame({
        "feature": correlations.index,
        "correlation": correlations.values,
        "absolute_correlation": correlations.abs().values,
        "unique_values": [
            X[col].nunique()
            for col in correlations.index
        ]
    })


    numeric_results = numeric_results.sort_values(
        by="absolute_correlation",
        ascending=False
    )


    print(
        numeric_results.head(20).to_string(
            index=False
        )
    )


    # ==================================================
    # CATEGORICAL FEATURE ANALYSIS
    # ==================================================

    print(
        "\n=============================================="
    )

    print(
        "CATEGORICAL FEATURES - TARGET RELATIONSHIP"
    )

    print(
        "=============================================="
    )


    categorical_results = []


    for column in categorical_columns:

        # Number of unique categories
        unique_count = X[column].nunique()


        # Skip extremely high-cardinality columns
        # because they are not useful for this test
        if unique_count > 50:
            continue


        table = pd.crosstab(
            X[column],
            y,
            normalize="index"
        )


        # Maximum fraud/non-fraud proportion
        max_target_probability = table.max(
            axis=1
        ).max()


        categorical_results.append({
            "feature": column,
            "unique_values": unique_count,
            "max_target_probability":
                max_target_probability
        })


    categorical_results_df = pd.DataFrame(
        categorical_results
    )


    if len(categorical_results_df) > 0:

        categorical_results_df = (
            categorical_results_df.sort_values(
                by="max_target_probability",
                ascending=False
            )
        )


        print(
            categorical_results_df.to_string(
                index=False
            )
        )

    else:

        print(
            "No low-cardinality categorical "
            "features found."
        )


    # ==================================================
    # FLAG FEATURES
    # ==================================================

    print(
        "\n=============================================="
    )

    print(
        "FRAUD FLAG FEATURE ANALYSIS"
    )

    print(
        "=============================================="
    )


    flag_features = [
        col
        for col in X.columns
        if (
            "flag" in col.lower()
            or "fraud" in col.lower()
            or "suspicious" in col.lower()
            or "unusual" in col.lower()
        )
    ]


    if flag_features:

        for column in flag_features:

            print(
                f"\n--- {column} ---"
            )

            print(
                pd.crosstab(
                    X[column],
                    y,
                    margins=True
                )
            )

    else:

        print(
            "No obvious flag features found."
        )


    # ==================================================
    # Save results
    # ==================================================

    output_path = os.path.join(
        os.path.dirname(ML_FOLDER),
        "models",
        "feature_target_analysis.csv"
    )


    numeric_results.to_csv(
        output_path,
        index=False
    )


    print(
        "\nNumeric analysis saved at:"
    )

    print(
        output_path
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    check_feature_target_relationship()