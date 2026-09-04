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
def extract_location_features(X):
    """
    Extract latitude and longitude from location coordinates.
    """

    if "location" in X.columns:

        location = X["location"].fillna("")

        X["latitude"] = location.str.extract(
            r"\(\s*(-?\d+(?:\.\d+)?)"
        )[0].astype(float)

        X["longitude"] = location.str.extract(
            r",\s*(-?\d+(?:\.\d+)?)\s*\)"
        )[0].astype(float)

        X = X.drop(columns=["location"])

    return X
def extract_business_name_features(X):
    """
    Convert business_name_match into a binary feature
    indicating whether a business name match is present.
    """

    if "business_name_match" in X.columns:
        X["business_name_match_present"] = (
            X["business_name_match"]
            .fillna("none")
            .astype(str)
            .str.lower()
            .ne("none")
            .astype(int)
        )

        X = X.drop(columns=["business_name_match"])

    return X