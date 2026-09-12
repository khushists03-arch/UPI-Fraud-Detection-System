import joblib

from backend.config import Config


class ModelService:
    """Loads and provides access to the trained ML models."""

    def __init__(self):
        self.xgboost_model = None
        self.isolation_forest_model = None

    def load_models(self):
        """Load all trained models from disk."""

        self.xgboost_model = joblib.load(
            Config.XGBOOST_MODEL_PATH
        )

        self.isolation_forest_model = joblib.load(
            Config.ISOLATION_FOREST_MODEL_PATH
        )

    def get_xgboost_model(self):
        """Return the loaded XGBoost model."""

        if self.xgboost_model is None:
            raise RuntimeError("XGBoost model has not been loaded.")

        return self.xgboost_model

    def get_isolation_forest_model(self):
        """Return the loaded Isolation Forest model."""

        if self.isolation_forest_model is None:
            raise RuntimeError(
                "Isolation Forest model has not been loaded."
            )

        return self.isolation_forest_model