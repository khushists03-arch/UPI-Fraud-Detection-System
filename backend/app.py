from flask import Flask
from flask_cors import CORS

from backend.config import Config
from backend.routes.prediction_routes import prediction_bp


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)

    app.config.from_object(Config)

    CORS(app)

    @app.get("/api/health")
    def health_check():
        return {
            "status": "ok",
            "service": "upi-fraud-detection-api"
        }

    app.register_blueprint(prediction_bp)

    return app