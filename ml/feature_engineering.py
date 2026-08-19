import pandas as pd


def extract_time_features(X):
    """
    Extract useful time-based features from the timestamp column.
    """

    if "timestamp" in X.columns:
        time = pd.to_datetime(
            X["timestamp"],
            format="%M:%S.%f",
            errors="coerce"
        )

        X["transaction_minute"] = time.dt.minute
        X["transaction_second"] = time.dt.second

        X = X.drop(columns=["timestamp"])

    return X
def extract_description_features(X):
    """
    Extract numerical features from transaction descriptions.
    """

    if "description" in X.columns:
        X["description_length"] = (
            X["description"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        X["description_word_count"] = (
            X["description"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

        X = X.drop(columns=["description"])

    return X