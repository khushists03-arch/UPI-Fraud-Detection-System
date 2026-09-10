import os
import sys
import re
import pandas as pd
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
# Project imports
# ==========================================================

from preprocessing import (
    load_data,
    separate_target,
    remove_id_columns,
    remove_high_missing_columns,
    remove_constant_columns,
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


# ==========================================================
# Project paths
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    ML_FOLDER
)

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

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_FOLDER,
    "xgboost_feature_columns.pkl"
)

PREPROCESSING_METADATA_PATH = os.path.join(
    MODEL_FOLDER,
    "preprocessing_metadata.pkl"
)


# ==========================================================
# Clean feature names
# ==========================================================

def clean_feature_names(X):
    """
    Make feature names safe for XGBoost.
    """

    X = X.copy()

    new_columns = []

    for column in X.columns:

        name = str(column)

        # Remove characters that can cause
        # XGBoost feature-name errors
        name = re.sub(
            r"[\[\]<>]",
            "_",
            name
        )

        name = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            name
        )

        name = re.sub(
            r"_+",
            "_",
            name
        )

        name = name.strip("_")

        if name == "":
            name = "feature"

        new_columns.append(name)


    # ======================================================
    # Make duplicate feature names unique
    # ======================================================

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


# ==========================================================
# Encode low-cardinality categorical columns
# ==========================================================

def encode_low_cardinality_columns(
    X,
    categorical_columns
):
    """
    Encode categorical columns having 10 or fewer
    unique values.

    The category information is saved so that the
    exact same encoding can be used during prediction.
    """

    X = X.copy()

    low_cardinality_columns = []

    categorical_categories = {}


    # ======================================================
    # Find low-cardinality columns
    # ======================================================

    for column in categorical_columns:

        if column not in X.columns:
            continue

        unique_values = (
            X[column]
            .dropna()
            .unique()
            .tolist()
        )

        if len(unique_values) <= 10:

            low_cardinality_columns.append(
                column
            )

            categorical_categories[column] = [
                str(value)
                for value in unique_values
            ]


    # ======================================================
    # Prepare categorical columns
    # ======================================================

    for column in low_cardinality_columns:

        categories = (
            categorical_categories[column]
        )

        X[column] = (
            X[column]
            .fillna("__MISSING__")
            .astype(str)
        )

        if "__MISSING__" not in categories:

            categories.append(
                "__MISSING__"
            )

        X[column] = pd.Categorical(
            X[column],
            categories=categories
        )


    # ======================================================
    # One-hot encoding
    # ======================================================

    if low_cardinality_columns:

        X = pd.get_dummies(
            X,
            columns=low_cardinality_columns,
            drop_first=True,
            dtype=float
        )


    return (
        X,
        low_cardinality_columns,
        categorical_categories
    )


# ==========================================================
# Prepare training data
# ==========================================================

def prepare_data():

    print(
        "Loading dataset..."
    )


    # ======================================================
    # Check dataset
    # ======================================================

    if not os.path.exists(
        DATA_PATH
    ):

        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )


    df = load_data(
        DATA_PATH
    )


    print(
        "Dataset shape:",
        df.shape
    )


    # ======================================================
    # Separate target
    # ======================================================

    X, y = separate_target(
        df
    )


    print(
        "Features shape after separating target:",
        X.shape
    )


    # ======================================================
    # Remove ID columns
    # ======================================================

    X = remove_id_columns(
        X
    )


    print(
        "Features shape after removing ID columns:",
        X.shape
    )


    # ======================================================
    # Remove highly missing columns
    # ======================================================

    X = remove_high_missing_columns(
        X
    )


    print(
        "Features shape after removing highly-missing columns:",
        X.shape
    )


    # ======================================================
    # Remove constant columns
    # ======================================================

    constant_result = remove_constant_columns(
        X
    )


    # Your preprocessing.py may return:
    #
    # X
    #
    # OR
    #
    # X, constant_columns
    #
    # This code supports both.

    if isinstance(
        constant_result,
        tuple
    ):

        X = constant_result[0]

        constant_columns = constant_result[1]

    else:

        X = constant_result

        constant_columns = []


    print(
        "Features shape after removing constant columns:",
        X.shape
    )

    print(
        "Constant columns removed:",
        constant_columns
    )


    # ======================================================
    # Remove suspicious leakage-related feature
    # ======================================================

    leakage_columns = [
        "handle_verification_status"
    ]


    for column in leakage_columns:

        if column in X.columns:

            print(
                f"\nRemoving leakage feature: {column}"
            )

            X = X.drop(
                columns=[column]
            )


    # ======================================================
    # Convert numeric-looking columns
    # ======================================================

    numeric_columns = [

        "upi_handle_age",

        "handle_contains_official_terms"
    ]


    for column in numeric_columns:

        if column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )


    # ======================================================
    # Identify categorical columns
    # ======================================================

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )


    print(
        "\nCategorical columns found:",
        len(categorical_columns)
    )


    print(
        categorical_columns
    )


    # ======================================================
    # Encode low-cardinality categorical columns
    # ======================================================

    (
        X,
        low_cardinality_columns,
        categorical_categories
    ) = encode_low_cardinality_columns(
        X,
        categorical_columns
    )


    print(
        "\nLow-cardinality columns encoded:"
    )

    print(
        low_cardinality_columns
    )


    print(
        "\nFeatures shape after categorical encoding:",
        X.shape
    )


    # ======================================================
    # Feature engineering
    # ======================================================

    print(
        "\nApplying feature engineering..."
    )


    X = extract_time_features(
        X
    )

    X = extract_description_features(
        X
    )

    X = extract_location_features(
        X
    )

    X = extract_business_name_features(
        X
    )


    print(
        "Features shape after feature engineering:",
        X.shape
    )


    # ======================================================
    # Handle remaining categorical columns
    # ======================================================

    remaining_categorical = (
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )


    remaining_categories = {}


    if remaining_categorical:

        print(
            "\nRemaining categorical columns:"
        )

        print(
            remaining_categorical
        )


        for column in remaining_categorical:

            values = (
                X[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            categories = (
                values
                .unique()
                .tolist()
            )

            remaining_categories[column] = (
                categories
            )

            X[column] = pd.Categorical(
                values,
                categories=categories
            )


        X = pd.get_dummies(
            X,
            columns=remaining_categorical,
            drop_first=False,
            dtype=float
        )


    # ======================================================
    # Convert boolean columns
    # ======================================================

    bool_columns = X.select_dtypes(
        include=["bool"]
    ).columns


    if len(bool_columns) > 0:

        X[bool_columns] = (
            X[bool_columns].astype(int)
        )


    # ======================================================
    # Clean feature names
    # ======================================================

    X = clean_feature_names(
        X
    )


    # ======================================================
    # Convert everything to numeric
    # ======================================================

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    # ======================================================
    # Replace infinite values
    # ======================================================

    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )


    # ======================================================
    # Fill missing values
    # ======================================================

    X = X.fillna(0)


    # ======================================================
    # Final information
    # ======================================================

    print(
        "\nFinal features shape:",
        X.shape
    )


    print(
        "Number of final features:",
        len(X.columns)
    )


    print(
        "All features numeric:",
        all(
            pd.api.types.is_numeric_dtype(
                dtype
            )
            for dtype in X.dtypes
        )
    )


    # ======================================================
    # Return everything required for training + metadata
    # ======================================================

    return (
        X,
        y,
        constant_columns,
        categorical_columns,
        low_cardinality_columns,
        categorical_categories,
        remaining_categories
    )


# ==========================================================
# Train XGBoost
# ==========================================================

def train_xgboost():

    (
        X,
        y,
        constant_columns,
        categorical_columns,
        low_cardinality_columns,
        categorical_categories,
        remaining_categories
    ) = prepare_data()


    # ======================================================
    # Leakage check
    # ======================================================

    print(
        "\n========== LEAKAGE CHECK =========="
    )


    print(
        "Is 'is_fraud' present in X?",
        "is_fraud" in X.columns
    )


    matching_columns = []


    for column in X.columns:

        try:

            if X[column].equals(y):

                matching_columns.append(
                    column
                )

        except Exception:

            pass


    print(
        "Features exactly matching target:",
        matching_columns
    )


    # ======================================================
    # Correlation check
    # ======================================================

    correlations = X.corrwith(
        y
    )


    high_corr = correlations[
        correlations.abs() > 0.9
    ]


    print(
        "\nFeatures with correlation > 0.9 with target:"
    )


    if len(high_corr) == 0:

        print(
            "No features found with correlation > 0.9"
        )

    else:

        print(
            high_corr.sort_values(
                ascending=False
            )
        )


    # ======================================================
    # Train/Test split
    # ======================================================

    print(
        "\nPreparing train/test split..."
    )


    X_train, X_test, y_train, y_test = (
        split_data(
            X,
            y
        )
    )


    print(
        "Training features shape:",
        X_train.shape
    )


    print(
        "Testing features shape:",
        X_test.shape
    )


    # ======================================================
    # Create XGBoost model
    # ======================================================

    print(
        "\nCreating XGBoost model..."
    )


    model = XGBClassifier(

        n_estimators=100,

        max_depth=6,

        learning_rate=0.1,

        random_state=42,

        eval_metric="logloss"
    )


    # ======================================================
    # Train model
    # ======================================================

    print(
        "\nTraining XGBoost model..."
    )


    model.fit(
        X_train,
        y_train
    )


    print(
        "\nXGBoost training completed successfully!"
    )


    # ======================================================
    # Make predictions
    # ======================================================

    print(
        "\nMaking predictions..."
    )


    y_pred = model.predict(
        X_test
    )


    # ======================================================
    # Model evaluation
    # ======================================================

    print(
        "\n================================"
    )

    print(
        "       MODEL EVALUATION"
    )

    print(
        "================================"
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print(
        "\nAccuracy:"
    )

    print(
        accuracy
    )


    print(
        "\nClassification Report:"
    )


    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    # ======================================================
    # Create models folder
    # ======================================================

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )


    # ======================================================
    # Save XGBoost model
    # ======================================================

    joblib.dump(
        model,
        MODEL_PATH
    )


    print(
        "\nModel saved successfully at:"
    )

    print(
        MODEL_PATH
    )


    # ======================================================
    # Save feature columns
    # ======================================================

    feature_columns = (
        X.columns.tolist()
    )


    joblib.dump(
        feature_columns,
        FEATURE_COLUMNS_PATH
    )


    print(
        "\nFeature columns saved successfully at:"
    )

    print(
        FEATURE_COLUMNS_PATH
    )


    print(
        "Number of features saved:",
        len(feature_columns)
    )


    # ======================================================
    # Save preprocessing metadata
    # ======================================================

    preprocessing_metadata = {

        # --------------------------------------------------
        # Columns removed
        # --------------------------------------------------

        "constant_columns":
            constant_columns,

        "id_columns": [

            "transaction_id",

            "user_id",

            "merchant_id",

            "device_id",

            "ip_address"
        ],

        "high_missing_columns": [

            "url_referrer",

            "request_description"
        ],

        "leakage_columns": [

            "handle_verification_status"
        ],


        # --------------------------------------------------
        # Numeric conversion information
        # --------------------------------------------------

        "numeric_columns": [

            "upi_handle_age",

            "handle_contains_official_terms"
        ],


        # --------------------------------------------------
        # Categorical information
        # --------------------------------------------------

        "categorical_columns":
            categorical_columns,

        "low_cardinality_columns":
            low_cardinality_columns,

        "categorical_categories":
            categorical_categories,

        "remaining_categorical_columns":
            list(
                remaining_categories.keys()
            ),

        "remaining_categories":
            remaining_categories,


        # --------------------------------------------------
        # Final feature information
        # --------------------------------------------------

        "feature_columns":
            feature_columns,

        "feature_count":
            len(feature_columns)
    }


    # ======================================================
    # Save metadata
    # ======================================================

    joblib.dump(
        preprocessing_metadata,
        PREPROCESSING_METADATA_PATH
    )


    print(
        "\nPreprocessing metadata saved successfully at:"
    )

    print(
        PREPROCESSING_METADATA_PATH
    )


    print(
        "\n================================"
    )

    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "================================"
    )

    print(
        f"Final feature count: {len(feature_columns)}"
    )


    return model


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_xgboost()