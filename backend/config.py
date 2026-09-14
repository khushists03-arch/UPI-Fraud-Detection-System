from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """
    Application configuration.
    """

    DEBUG = True

    DATA_DIR = BASE_DIR / "data"

    MODEL_DIR = BASE_DIR / "models"

    DATASET_PATH = (
        DATA_DIR
        / "raw"
        / "fraud_dataset.csv"
    )

    XGBOOST_MODEL_PATH = (
        MODEL_DIR
        / "xgboost_model.pkl"
    )

    ISOLATION_FOREST_MODEL_PATH = (
        MODEL_DIR
        / "isolation_forest_model.pkl"
    )

    DATABASE_PATH = (
        DATA_DIR
        / "upi_fraud.db"
    )

    SECRET_KEY = "upi-fraud-detection-development-secret-key"

    JWT_EXPIRATION_MINUTES = 60