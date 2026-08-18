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