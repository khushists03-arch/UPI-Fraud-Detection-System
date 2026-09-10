import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data(file_path):
    """
    Load dataset from CSV.
    """
    return pd.read_csv(file_path)


# --------------------------------------------------
# Inspect dataset
# --------------------------------------------------

def inspect_data(df):
    """
    Display basic information about the dataset.
    """

    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# Separate target
# --------------------------------------------------

def separate_target(df):
    """
    Separate features and target variable.
    """

    X = df.drop(
        "is_fraud",
        axis=1
    )

    y = df["is_fraud"]

    return X, y


# --------------------------------------------------
# Remove ID columns
# --------------------------------------------------

def remove_id_columns(X):
    """
    Remove identifier columns from the feature set.
    """

    id_columns = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "device_id",
        "ip_address"
    ]

    X = X.drop(
        columns=id_columns,
        errors="ignore"
    )

    return X


# --------------------------------------------------
# Remove highly missing columns
# --------------------------------------------------

def remove_high_missing_columns(X):
    """
    Remove columns with very high missing values.
    """

    columns_to_remove = [
        "url_referrer",
        "request_description"
    ]

    X = X.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    return X


# --------------------------------------------------
# Identify categorical columns
# --------------------------------------------------

def identify_categorical_columns(X):
    """
    Identify categorical/text columns.
    """

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return categorical_columns


# --------------------------------------------------
# Find constant columns
# --------------------------------------------------

def get_constant_columns(X):
    """
    Find columns containing only one unique value.

    This function is used during training.
    The resulting list can later be saved and
    reused when predicting new transactions.
    """

    constant_columns = [
        col
        for col in X.columns
        if X[col].nunique() <= 1
    ]

    return constant_columns


# --------------------------------------------------
# Remove constant columns
# --------------------------------------------------

def remove_constant_columns(
    X,
    constant_columns=None
):
    """
    Remove constant columns.

    During training:
        If constant_columns is None,
        they are detected automatically.

    During prediction:
        The constant columns identified during
        training can be passed to this function.

    Returns:
        X
        constant_columns
    """

    if constant_columns is None:

        constant_columns = get_constant_columns(X)

    X = X.drop(
        columns=constant_columns,
        errors="ignore"
    )

    return X, constant_columns


# --------------------------------------------------
# Encode low-cardinality categorical columns
# --------------------------------------------------

def encode_categorical_columns(X):
    """
    One-hot encode low-cardinality categorical columns.

    Only categorical columns with 10 or fewer
    unique values are encoded here.

    High-cardinality text columns are left for
    feature engineering.
    """

    categorical_columns = identify_categorical_columns(X)

    low_cardinality_columns = [
        col
        for col in categorical_columns
        if X[col].nunique() <= 10
    ]

    X = pd.get_dummies(
        X,
        columns=low_cardinality_columns,
        drop_first=True,
        dtype=int
    )

    return X


# --------------------------------------------------
# Convert numeric-looking columns
# --------------------------------------------------

def convert_numeric_columns(X):
    """
    Convert numeric-looking text columns into
    numeric values.
    """

    numeric_columns = [
        "upi_handle_age",
        "handle_contains_official_terms"
    ]

    for col in numeric_columns:

        if col in X.columns:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

    return X


# --------------------------------------------------
# Align features with training columns
# --------------------------------------------------

def align_features(
    X,
    feature_columns
):
    """
    Make prediction features match the exact
    feature columns used during training.

    Missing training features are filled with 0.

    Extra features that were not present during
    training are removed.

    Finally, columns are arranged in the exact
    same order as during training.
    """

    # Add missing columns
    for column in feature_columns:

        if column not in X.columns:

            X[column] = 0

    # Remove extra columns
    X = X[
        [
            column
            for column in feature_columns
        ]
    ]

    return X


# --------------------------------------------------
# Prepare new transaction
# --------------------------------------------------

def prepare_new_transaction(
    X,
    feature_columns,
    constant_columns=None
):
    """
    Prepare a new transaction for prediction.

    This function performs the preprocessing
    required before the transaction is sent
    to the trained ML model.

    Important:
    Constant columns are NOT detected from the
    single new transaction. Instead, the list
    from training should be supplied.
    """

    # --------------------------------------------------
    # Remove ID columns
    # --------------------------------------------------

    X = remove_id_columns(X)


    # --------------------------------------------------
    # Remove highly missing columns
    # --------------------------------------------------

    X = remove_high_missing_columns(X)


    # --------------------------------------------------
    # Remove the same constant columns used in training
    # --------------------------------------------------

    if constant_columns is not None:

        X = X.drop(
            columns=constant_columns,
            errors="ignore"
        )


    # --------------------------------------------------
    # Remove leakage feature
    # --------------------------------------------------

    leakage_feature = "handle_verification_status"

    if leakage_feature in X.columns:

        X = X.drop(
            columns=[leakage_feature]
        )


    # --------------------------------------------------
    # Convert numeric-looking columns
    # --------------------------------------------------

    X = convert_numeric_columns(X)


    # --------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------

    X = encode_categorical_columns(X)


    # --------------------------------------------------
    # Handle missing values
    # --------------------------------------------------

    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    X = X.fillna(0)


    # --------------------------------------------------
    # Align with training features
    # --------------------------------------------------

    X = align_features(
        X,
        feature_columns
    )


    return X


# --------------------------------------------------
# Train/Test split
# --------------------------------------------------

def split_data(X, y):
    """
    Split data into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test