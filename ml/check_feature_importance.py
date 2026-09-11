import os
import sys
import pandas as pd
import joblib


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


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(ML_FOLDER)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_tuned_model.pkl"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def check_feature_importance():

    print("Loading prepared data...")

    X, y = prepare_data()

    print(
        "\nLoading tuned XGBoost model..."
    )

    model = joblib.load(MODEL_PATH)


    # --------------------------------------------------
    # Get feature importance
    # --------------------------------------------------

    importance = model.feature_importances_


    # Create dataframe
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })


    # Sort by importance
    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )


    # --------------------------------------------------
    # Display top features
    # --------------------------------------------------

    print("\n========================================")
    print("       TOP 20 FEATURE IMPORTANCE")
    print("========================================")

    print(
        feature_importance.head(20).to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "feature_importance.csv"
    )

    feature_importance.to_csv(
        output_path,
        index=False
    )


    print(
        "\nFeature importance saved at:"
    )

    print(output_path)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    check_feature_importance()