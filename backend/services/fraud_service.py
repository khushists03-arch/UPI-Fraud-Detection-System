import pandas as pd

from backend.services.model_service import ModelService
from backend.utils.feature_builder import FeatureBuilder


class FraudService:
    """Handle fraud prediction using the trained ML models."""

    def __init__(self):
        self.model_service = ModelService()
        self.feature_builder = FeatureBuilder()

        # Load the trained models once.
        self.model_service.load_models()

    def predict(self, transaction):
        """
        Predict whether a transaction is fraudulent.

        Returns a dictionary containing:
        - final prediction
        - confidence
        - fraud score
        - XGBoost result
        - Isolation Forest result
        """

        # ---------------------------------------------------------
        # Build the 60 features expected by the models
        # ---------------------------------------------------------

        features = self.feature_builder.build_features(
            transaction
        )

        # ---------------------------------------------------------
        # Get trained models
        # ---------------------------------------------------------

        xgboost_model = (
            self.model_service.get_xgboost_model()
        )

        isolation_forest_model = (
            self.model_service.get_isolation_forest_model()
        )

        # ---------------------------------------------------------
        # XGBoost prediction
        # ---------------------------------------------------------

        xgboost_prediction = int(
            xgboost_model.predict(features)[0]
        )

        xgboost_probability = float(
            xgboost_model.predict_proba(features)[0][1]
        )

        # ---------------------------------------------------------
        # Isolation Forest prediction
        #
        # Use the exact feature names stored by the trained
        # Isolation Forest model.
        # ---------------------------------------------------------

        isolation_feature_names = (
            isolation_forest_model.feature_names_in_
        )

        isolation_input = pd.DataFrame(
            features.to_numpy(),
            columns=isolation_feature_names
        )

        isolation_prediction = int(
            isolation_forest_model.predict(
                isolation_input
            )[0]
        )

        is_anomaly = isolation_prediction == -1

        # ---------------------------------------------------------
        # Convert Isolation Forest result into anomaly score
        #
        # -1 = anomaly
        #  1 = normal
        # ---------------------------------------------------------

        anomaly_score = 1.0 if is_anomaly else 0.0

        # ---------------------------------------------------------
        # Combine model results
        #
        # XGBoost       = 70%
        # Isolation     = 30%
        # ---------------------------------------------------------

        raw_fraud_score = (
            0.70 * xgboost_probability
            + 0.30 * anomaly_score
        )

        # ---------------------------------------------------------
        # Final classification
        # ---------------------------------------------------------

        prediction = (
            "Fraud"
            if raw_fraud_score >= 0.50
            else "Safe"
        )

        # ---------------------------------------------------------
        # Calculate confidence BEFORE rounding.
        #
        # For Safe:
        # confidence = 1 - fraud score
        #
        # For Fraud:
        # confidence = fraud score
        # ---------------------------------------------------------

        if prediction == "Fraud":
            raw_confidence = raw_fraud_score
        else:
            raw_confidence = 1.0 - raw_fraud_score

        # ---------------------------------------------------------
        # Return complete prediction response
        # ---------------------------------------------------------

        return {
            "prediction": prediction,

            "confidence": round(
                raw_confidence,
                6
            ),

            "fraud_score": round(
                raw_fraud_score,
                6
            ),

            "xgboost": {
                "prediction": xgboost_prediction,

                "fraud_probability": round(
                    xgboost_probability,
                    6
                )
            },

            "isolation_forest": {
                "is_anomaly": is_anomaly,

                "prediction": isolation_prediction
            }
        }