import ast
import re
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TransactionFeaturePreprocessor:
    """
    Converts raw UPI transaction data into numeric node features
    suitable for the Graph Neural Network.

    Important:
    - Transaction/user/merchant/device IDs are NOT used as numeric features.
    - They are used later for graph construction.
    - is_fraud is the target and is NEVER included as an input feature.
    """

    ID_COLUMNS = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "device_id",
        "ip_address",
    ]

    TARGET_COLUMNS = [
        "is_fraud"
    ]

    LIST_COLUMNS = [
        "recent_app_installs",
        "permissions_granted",
        "recognized_screen_sharing_apps",
        "request_description_keywords",
    ]

    DROP_COLUMNS = [
        "url_referrer",
        "request_description",
        "relationship_to_requester",
        "social_media_presence",
    ]

    def __init__(self, max_one_hot_categories: int = 20):

        self.max_one_hot_categories = max_one_hot_categories

        self.one_hot_columns_ = []
        self.one_hot_values_: Dict[str, list] = {}

        self.frequency_columns_ = []
        self.frequency_maps_: Dict[str, Dict[str, float]] = {}

        self.feature_columns_ = []

        self.scaler_ = StandardScaler()

        self.fitted_ = False

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------

    @staticmethod
    def _list_length(value) -> float:

        if pd.isna(value):
            return 0.0

        text = str(value).strip()

        if not text or text == "[]":
            return 0.0

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                return float(len(parsed))

        except (ValueError, SyntaxError):
            pass

        return 1.0

    @staticmethod
    def _clean_name(name: str) -> str:

        name = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            str(name)
        )

        name = re.sub(
            r"_+",
            "_",
            name
        )

        name = name.strip("_")

        return name or "feature"

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:

        x = df.copy()

        # -----------------------------------------------------
        # Remove IDs
        # -----------------------------------------------------

        x = x.drop(
            columns=[
                c
                for c in self.ID_COLUMNS
                if c in x.columns
            ],
            errors="ignore",
        )

        # -----------------------------------------------------
        # Remove target
        # -----------------------------------------------------

        x = x.drop(
            columns=[
                c
                for c in self.TARGET_COLUMNS
                if c in x.columns
            ],
            errors="ignore",
        )

        # -----------------------------------------------------
        # Remove unnecessary high-cardinality/text columns
        # -----------------------------------------------------

        x = x.drop(
            columns=[
                c
                for c in self.DROP_COLUMNS
                if c in x.columns
            ],
            errors="ignore",
        )

        # -----------------------------------------------------
        # Timestamp
        # -----------------------------------------------------

        if "timestamp" in x.columns:

            ts = (
                x["timestamp"]
                .fillna("")
                .astype(str)
            )

            parts = ts.str.extract(
                r"^\s*(\d+):(\d+)(?:\.(\d+))?\s*$"
            )

            x["transaction_minute"] = (
                pd.to_numeric(
                    parts[0],
                    errors="coerce"
                )
                .fillna(0)
            )

            x["transaction_second"] = (
                pd.to_numeric(
                    parts[1],
                    errors="coerce"
                )
                .fillna(0)
            )

            fraction = (
                "0."
                + parts[2]
                .fillna("0")
                .astype(str)
            )

            x["transaction_fraction"] = (
                pd.to_numeric(
                    fraction,
                    errors="coerce"
                )
                .fillna(0)
            )

            x["transaction_seconds_from_start"] = (
                x["transaction_minute"] * 60
                + x["transaction_second"]
                + x["transaction_fraction"]
            )

            x = x.drop(
                columns=["timestamp"]
            )

        # -----------------------------------------------------
        # Description
        # -----------------------------------------------------

        if "description" in x.columns:

            text = (
                x["description"]
                .fillna("")
                .astype(str)
            )

            x["description_length"] = (
                text.str.len()
                .astype(float)
            )

            x["description_word_count"] = (
                text.str.split()
                .str.len()
                .astype(float)
            )

            x["description_digit_count"] = (
                text.str.count(r"\d")
                .astype(float)
            )

            x = x.drop(
                columns=["description"]
            )

        # -----------------------------------------------------
        # Location
        # -----------------------------------------------------

        if "location" in x.columns:

            location = (
                x["location"]
                .fillna("")
                .astype(str)
            )

            x["latitude"] = pd.to_numeric(
                location.str.extract(
                    r"\(\s*(-?\d+(?:\.\d+)?)"
                )[0],
                errors="coerce",
            )

            x["longitude"] = pd.to_numeric(
                location.str.extract(
                    r",\s*(-?\d+(?:\.\d+)?)\s*\)"
                )[0],
                errors="coerce",
            )

            x = x.drop(
                columns=["location"]
            )

        # -----------------------------------------------------
        # List-like columns
        # -----------------------------------------------------

        for column in self.LIST_COLUMNS:

            if column in x.columns:

                x[f"{column}_count"] = (
                    x[column]
                    .apply(self._list_length)
                )

                x = x.drop(
                    columns=[column]
                )

        # -----------------------------------------------------
        # Business name match
        # -----------------------------------------------------

        if "business_name_match" in x.columns:

            x["business_name_match_present"] = (
                x["business_name_match"]
                .fillna("none")
                .astype(str)
                .str.strip()
                .str.lower()
                .ne("none")
                .astype(float)
            )

            x = x.drop(
                columns=["business_name_match"]
            )

        # -----------------------------------------------------
        # Boolean → integer
        # -----------------------------------------------------

        for column in x.columns:

            if pd.api.types.is_bool_dtype(
                x[column]
            ):

                x[column] = (
                    x[column]
                    .astype(int)
                )

        return x

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(self, df: pd.DataFrame):

        x = self._engineer(df)

        categorical_columns = (
            x.select_dtypes(
                include=[
                    "object",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        # Small categorical columns → one-hot encoding
        self.one_hot_columns_ = [
            column
            for column in categorical_columns
            if x[column].nunique(
                dropna=False
            ) <= self.max_one_hot_categories
        ]

        # Large categorical columns → frequency encoding
        self.frequency_columns_ = [
            column
            for column in categorical_columns
            if column not in self.one_hot_columns_
        ]

        # -----------------------------------------------------
        # Save one-hot categories
        # -----------------------------------------------------

        self.one_hot_values_ = {}

        for column in self.one_hot_columns_:

            values = (
                x[column]
                .fillna("__MISSING__")
                .astype(str)
                .unique()
                .tolist()
            )

            self.one_hot_values_[column] = (
                sorted(values)
            )

        # -----------------------------------------------------
        # Save frequency maps
        # -----------------------------------------------------

        self.frequency_maps_ = {}

        for column in self.frequency_columns_:

            values = (
                x[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            frequencies = (
                values
                .value_counts(
                    normalize=True
                )
                .to_dict()
            )

            self.frequency_maps_[column] = {
                str(key): float(value)
                for key, value in frequencies.items()
            }

        encoded = self._encode(x)

        # -----------------------------------------------------
        # Remove constant columns
        # -----------------------------------------------------

        constant_columns = [
            column
            for column in encoded.columns
            if encoded[column]
            .nunique(dropna=False) <= 1
        ]

        encoded = encoded.drop(
            columns=constant_columns,
            errors="ignore",
        )

        self.feature_columns_ = [
            self._clean_name(column)
            for column in encoded.columns
        ]

        encoded.columns = self.feature_columns_

        values = (
            encoded
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0.0)
        )

        self.scaler_.fit(
            values.astype(np.float32)
        )

        self.fitted_ = True

        return self

    # ---------------------------------------------------------
    # Encode
    # ---------------------------------------------------------

    def _encode(
        self,
        x: pd.DataFrame
    ) -> pd.DataFrame:

        out = x.copy()

        # -----------------------------------------------------
        # One-hot encoding
        # -----------------------------------------------------

        for column in self.one_hot_columns_:

            values = (
                out[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            for category in (
                self.one_hot_values_[column]
            ):

                safe_category = (
                    self._clean_name(category)
                )

                out[
                    f"{column}__{safe_category}"
                ] = (
                    values == category
                ).astype(float)

            out = out.drop(
                columns=[column]
            )

        # -----------------------------------------------------
        # Frequency encoding
        # -----------------------------------------------------

        for column in self.frequency_columns_:

            values = (
                out[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            mapping = (
                self.frequency_maps_[column]
            )

            out[
                f"{column}__frequency"
            ] = (
                values
                .map(mapping)
                .fillna(0.0)
                .astype(float)
            )

            out = out.drop(
                columns=[column]
            )

        # -----------------------------------------------------
        # Convert everything possible to numeric
        # -----------------------------------------------------

        for column in out.columns:

            if not pd.api.types.is_numeric_dtype(
                out[column]
            ):

                out[column] = pd.to_numeric(
                    out[column],
                    errors="coerce"
                )

        return out

    # ---------------------------------------------------------
    # Transform
    # ---------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:

        if not self.fitted_:

            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        x = self._engineer(df)

        x = self._encode(x)

        cleaned = {}

        for column in x.columns:

            cleaned[
                self._clean_name(column)
            ] = x[column]

        x = pd.DataFrame(
            cleaned,
            index=x.index
        )

        # Make sure inference uses exactly
        # the same feature order as training.

        x = x.reindex(
            columns=self.feature_columns_,
            fill_value=0.0,
        )

        x = (
            x
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0.0)
        )

        return (
            self.scaler_
            .transform(
                x.astype(np.float32)
            )
            .astype(np.float32)
        )

    # ---------------------------------------------------------
    # Fit + Transform
    # ---------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:

        self.fit(df)

        return self.transform(df)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(
        self,
        path: str
    ):

        joblib.dump(
            self,
            path
        )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str
    ):

        return joblib.load(
            path
        )

    # ---------------------------------------------------------
    # Number of features
    # ---------------------------------------------------------

    @property
    def input_dim(self) -> int:

        if not self.fitted_:

            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        return len(
            self.feature_columns_
        )