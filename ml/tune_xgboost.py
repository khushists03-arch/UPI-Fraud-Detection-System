import os
import sys
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(ML_FOLDER)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "models"
)

TUNED_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "xgboost_tuned_model.pkl"
)


# --------------------------------------------------
# Main tuning function
# --------------------------------------------------

def tune_xgboost():

    print("Preparing data for XGBoost tuning...")

    X, y = prepare_data()


    # --------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------

    print("\nPreparing train/test split...")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print(
        "Training features shape:",
        X_train.shape
    )

    print(
        "Testing features shape:",
        X_test.shape
    )


    # --------------------------------------------------
    # Base XGBoost model
    # --------------------------------------------------

    print("\nCreating base XGBoost model...")

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )


    # --------------------------------------------------
    # Parameters for tuning
    # --------------------------------------------------

    param_grid = {

        "n_estimators": [
            100,
            200,
            300
        ],

        "max_depth": [
            3,
            4,
            5,
            6
        ],

        "learning_rate": [
            0.03,
            0.05,
            0.1
        ],

        "subsample": [
            0.8,
            1.0
        ],

        "colsample_bytree": [
            0.8,
            1.0
        ],

        "min_child_weight": [
            1,
            3,
            5
        ]
    }


    # --------------------------------------------------
    # Randomized Search
    # --------------------------------------------------

    print("\nStarting XGBoost hyperparameter tuning...")

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=15,
        scoring="f1",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )


    # --------------------------------------------------
    # Train tuning process
    # --------------------------------------------------

    print("\nTraining and searching for best parameters...")

    random_search.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # Best parameters
    # --------------------------------------------------

    print("\n================================")
    print("       BEST PARAMETERS")
    print("================================")

    print(
        random_search.best_params_
    )


    print("\nBest Cross-Validation F1 Score:")

    print(
        random_search.best_score_
    )


    # --------------------------------------------------
    # Best model
    # --------------------------------------------------

    best_model = random_search.best_estimator_


    # --------------------------------------------------
    # Test Set Prediction
    # --------------------------------------------------

    print("\nMaking predictions using tuned model...")

    y_pred = best_model.predict(
        X_test
    )


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print("\n================================")
    print("       TUNED MODEL EVALUATION")
    print("================================")


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nAccuracy:")
    print(accuracy)


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    # --------------------------------------------------
    # Save tuned model
    # --------------------------------------------------

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )


    joblib.dump(
        best_model,
        TUNED_MODEL_PATH
    )


    print(
        "\nTuned model saved successfully at:"
    )

    print(
        TUNED_MODEL_PATH
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    tune_xgboost()