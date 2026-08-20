import os
import sys
import re
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
    encode_categorical_columns,
    split_data
)

from feature_engineering import (
    extract_time_features,
    extract_description_features,
    extract_location_features,
    extract_business_name_features
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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
    "xgboost_model.pkl"
)


# --------------------------------------------------
# Clean feature names
# --------------------------------------------------

def clean_feature_names(X):
    """
    Make feature names safe for XGBoost.
    """

    new_columns = []

    for column in X.columns:

        # Convert column name to string
        name = str(column)

        # Replace characters that XGBoost does not allow
        name = re.sub(r"[\[\]<>]", "_", name)

        # Replace other unusual characters
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name)

        # Remove repeated underscores
        name = re.sub(r"_+", "_", name)

        # Remove underscores from beginning/end
        name = name.strip("_")

        # Prevent empty column names
        if name == "":
            name = "feature"

        new_columns.append(name)

    # Make duplicate column names unique
    used_names = {}
    unique_columns = []

    for name in new_columns:

        if name not in used_names:
            used_names[name] = 0
            unique_columns.append(name)

        else:
            used_names[name] += 1
            unique_columns.append(
                f"{name}_{used_names[name]}"
            )

    X.columns = unique_columns

    return X


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

def prepare_data():

    print("Loading dataset...")

    # Check dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = load_data(DATA_PATH)

    print("Dataset shape:", df.shape)

    # --------------------------------------------------
    # Separate target
    # --------------------------------------------------

    X, y = separate_target(df)

    print(
        "Features shape after separating target:",
        X.shape
    )

    # --------------------------------------------------
    # Remove ID columns
    # --------------------------------------------------

    X = remove_id_columns(X)

    print(
        "Features shape after removing ID columns:",
        X.shape
    )

    # --------------------------------------------------
    # Remove highly missing columns
    # --------------------------------------------------

    X = remove_high_missing_columns(X)

    print(
        "Features shape after removing highly-missing columns:",
        X.shape
    )

    # --------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------

    X = remove_constant_columns(X)

    print(
        "Features shape after removing constant columns:",
        X.shape
    )

    # --------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------

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
    # Handle any remaining categorical columns
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
    # Convert boolean columns to numeric
    # --------------------------------------------------

    bool_columns = X.select_dtypes(
        include=["bool"]
    ).columns

    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)

    # --------------------------------------------------
    # Clean feature names for XGBoost
    # --------------------------------------------------

    X = clean_feature_names(X)

    print(
        "Feature names cleaned for XGBoost."
    )

    # --------------------------------------------------
    # Make sure all data is numeric
    # --------------------------------------------------

    non_numeric = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric:

        print(
            "\nWarning: non-numeric columns found:",
            non_numeric
        )

        X = pd.get_dummies(
            X,
            columns=non_numeric,
            drop_first=False,
            dtype=float
        )

        X = clean_feature_names(X)

    # Replace infinite values
    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Fill missing numerical values
    X = X.fillna(0)

    # Final check
    print(
        "\nFinal features shape:",
        X.shape
    )

    print(
        "All features numeric:",
        all(pd.api.types.is_numeric_dtype(dtype)
            for dtype in X.dtypes)
    )

    return X, y


# --------------------------------------------------
# Train XGBoost
# --------------------------------------------------

def train_xgboost():

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
    # Create XGBoost model
    # --------------------------------------------------

    print("\nCreating XGBoost model...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    print("\nTraining XGBoost model...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nXGBoost training completed successfully!"
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    print("\nMaking predictions...")

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print("\n================================")
    print("       MODEL EVALUATION")
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

    train_xgboost()