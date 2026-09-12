from functools import lru_cache
from datetime import datetime

import pandas as pd

from backend.config import Config

from ml.preprocessing import (
    load_data,
    separate_target,
    remove_id_columns,
    remove_high_missing_columns,
    remove_constant_columns,
    encode_categorical_columns,
)

from ml.feature_engineering import (
    extract_time_features,
    extract_description_features,
    extract_location_features,
    extract_business_name_features,
)

from ml.train_xgboost import clean_feature_names


class FeatureBuilder:
    """
    Build the exact 60-feature input expected by the trained ML models.

    The existing training pipeline is reused so that inference follows
    the same feature transformations used during model training.
    """

    def __init__(self):
        self.dataset = self._load_reference_dataset()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_reference_dataset():
        """Load the training dataset used as the inference reference."""

        return load_data(Config.DATASET_PATH)

    def build_features(self, transaction):
        """
        Convert a frontend transaction request into the model's
        expected feature matrix.
        """

        reference_df = self.dataset.copy()

        # ---------------------------------------------------------
        # Remove target from reference data
        # ---------------------------------------------------------

        reference_X, _ = separate_target(reference_df)

        # ---------------------------------------------------------
        # Create a default transaction using dataset statistics
        # ---------------------------------------------------------

        transaction_row = self._create_default_transaction(reference_X)

        # ---------------------------------------------------------
        # Apply values supplied by the frontend
        # ---------------------------------------------------------

        self._apply_transaction_values(
            transaction_row,
            transaction
        )

        # ---------------------------------------------------------
        # Combine reference data + new transaction
        #
        # This allows categorical encoding to use the same
        # categories that existed during training.
        # ---------------------------------------------------------

        combined = pd.concat(
            [
                reference_X,
                transaction_row
            ],
            ignore_index=True
        )

        # ---------------------------------------------------------
        # Apply the same preprocessing pipeline used during training
        # ---------------------------------------------------------

        combined = remove_id_columns(combined)

        combined = remove_high_missing_columns(combined)

        combined = remove_constant_columns(combined)

        combined = encode_categorical_columns(combined)

        # ---------------------------------------------------------
        # Feature engineering
        # ---------------------------------------------------------

        combined = extract_time_features(combined)

        combined = extract_description_features(combined)

        combined = extract_location_features(combined)

        combined = extract_business_name_features(combined)

        # ---------------------------------------------------------
        # Encode any remaining categorical columns
        # ---------------------------------------------------------

        remaining_categorical = combined.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if remaining_categorical:
            combined = pd.get_dummies(
                combined,
                columns=remaining_categorical,
                drop_first=False,
                dtype=float
            )

        # ---------------------------------------------------------
        # Convert boolean columns
        # ---------------------------------------------------------

        bool_columns = combined.select_dtypes(
            include=["bool"]
        ).columns

        if len(bool_columns) > 0:
            combined[bool_columns] = combined[bool_columns].astype(int)

        # ---------------------------------------------------------
        # Clean feature names exactly like training
        # ---------------------------------------------------------

        combined = clean_feature_names(combined)

        # ---------------------------------------------------------
        # Make sure everything is numeric
        # ---------------------------------------------------------

        non_numeric = combined.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        if non_numeric:
            combined = pd.get_dummies(
                combined,
                columns=non_numeric,
                drop_first=False,
                dtype=float
            )

            combined = clean_feature_names(combined)

        # ---------------------------------------------------------
        # Handle invalid/missing values
        # ---------------------------------------------------------

        combined = combined.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )

        combined = combined.fillna(0)

        # ---------------------------------------------------------
        # Extract the newly-created transaction row
        # ---------------------------------------------------------

        transaction_features = combined.iloc[[-1]].copy()

        return transaction_features

    def _create_default_transaction(self, reference_X):
        """
        Create a transaction row using representative values from
        the training dataset.

        Numeric columns use the median.
        Categorical columns use the mode.
        """

        row = {}

        for column in reference_X.columns:

            series = reference_X[column]

            if pd.api.types.is_numeric_dtype(series):
                row[column] = series.median()

            else:
                mode = series.mode(dropna=True)

                if not mode.empty:
                    row[column] = mode.iloc[0]
                else:
                    row[column] = ""

        return pd.DataFrame([row])

    def _apply_transaction_values(self, row, transaction):
        """Apply values supplied by the frontend."""

        # ---------------------------------------------------------
        # Amount
        # ---------------------------------------------------------

        if "amount" in transaction:
            row.loc[0, "amount"] = float(transaction["amount"])

        # ---------------------------------------------------------
        # Description
        # ---------------------------------------------------------

        if "description" in transaction:
            row.loc[0, "description"] = str(
                transaction["description"]
            )

        # ---------------------------------------------------------
        # Location
        # ---------------------------------------------------------

        if "location" in transaction:
            row.loc[0, "location"] = str(
                transaction["location"]
            )

        # ---------------------------------------------------------
        # Transaction type
        # ---------------------------------------------------------

        if "transactionType" in transaction:
                 transaction_type = str(
                 transaction["transactionType"]
             ).strip().lower()

        transaction_type_mapping = {
    "upi": "payment",
    "payment": "payment",
    "transfer": "payment",
    "request money": "collection_request",
    "collection_request": "collection_request",
}

        row.loc[0, "transaction_type"] = (
             transaction_type_mapping.get(
             transaction_type,
             "payment"
        )
    )

        # ---------------------------------------------------------
        # Merchant
        #
        # merchant itself is not a model feature because merchant_id
        # is removed during preprocessing.
        #
        # We use the merchant name to infer a merchant category.
        # ---------------------------------------------------------

        if "merchant" in transaction:
            merchant = str(
                transaction["merchant"]
            ).lower()

            row.loc[0, "merchant_category_code"] = (
                self._infer_merchant_category(merchant)
            )

        # ---------------------------------------------------------
        # Generate a current timestamp.
        #
        # The training feature engineering expects timestamp in
        # %M:%S.%f format.
        # ---------------------------------------------------------

        row.loc[0, "timestamp"] = datetime.now().strftime(
            "%M:%S.%f"
        )

    @staticmethod
    def _infer_merchant_category(merchant):
        """Infer a basic merchant category from the merchant name."""

        food_keywords = [
            "food",
            "restaurant",
            "cafe",
            "coffee",
            "pizza",
            "swiggy",
            "zomato",
            "dominos",
            "mcdonald"
        ]

        retail_keywords = [
            "amazon",
            "flipkart",
            "walmart",
            "shop",
            "store",
            "mart",
            "mall"
        ]

        utility_keywords = [
            "electricity",
            "water",
            "gas",
            "recharge",
            "broadband",
            "mobile",
            "airtel",
            "jio",
            "vi"
        ]

        for keyword in food_keywords:
            if keyword in merchant:
                return "food"

        for keyword in retail_keywords:
            if keyword in merchant:
                return "retail"

        for keyword in utility_keywords:
            if keyword in merchant:
                return "utilities"

        return "unknown"