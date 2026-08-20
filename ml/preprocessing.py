import pandas as pd


def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)
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

def separate_target(df):
     X = df.drop("is_fraud", axis=1)
     y = df["is_fraud"]
     return X, y
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

    X = X.drop(columns=id_columns, errors="ignore")

    return X
def remove_high_missing_columns(X):
    """
    Remove columns with a very high percentage of missing values.
    """
    columns_to_remove = [
        "url_referrer",
        "request_description"
    ]

    X = X.drop(columns=columns_to_remove, errors="ignore")

    return X
def identify_categorical_columns(X):
    """
    Identify columns containing categorical/text data.
    """
    categorical_columns = X.select_dtypes(include="object").columns.tolist()

    return categorical_columns
def remove_constant_columns(X):
    """
    Remove columns that contain only one unique value.
    """
    constant_columns = [
        col for col in X.columns
        if X[col].nunique() <= 1
    ]

    X = X.drop(columns=constant_columns, errors="ignore")

    return X
def encode_categorical_columns(X):
    """
    Encode low-cardinality categorical columns using one-hot encoding.
    High-cardinality columns are left for feature engineering.
    """

    categorical_columns = X.select_dtypes(include="object").columns.tolist()

    low_cardinality_columns = [
        col for col in categorical_columns
        if X[col].nunique() <= 10
    ]

    X = pd.get_dummies(
        X,
        columns=low_cardinality_columns,
        drop_first=True,
        dtype=int
    )

    return X
def convert_numeric_columns(X):
    """
    Convert numeric-looking columns stored as text into numeric values.
    """
    numeric_columns = [
        "upi_handle_age",
        "handle_contains_official_terms"
    ]

    for col in numeric_columns:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X
from sklearn.model_selection import train_test_split


def split_data(X, y):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test