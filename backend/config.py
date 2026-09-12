from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Application configuration."""

    DEBUG = True

    # Project directories
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"

    # Dataset
    DATASET_PATH = DATA_DIR / "raw" / "fraud_dataset.csv"

    # ML models
    XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
    ISOLATION_FOREST_MODEL_PATH = MODEL_DIR / "isolation_forest_model.pkl"