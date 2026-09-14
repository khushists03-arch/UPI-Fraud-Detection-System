from flask import Flask
from flask_cors import CORS

from backend.auth import auth_service
from backend.config import Config
from backend.payment import PaymentRepository
from backend.routes.auth_routes import auth_bp
from backend.routes.payment_routes import payment_bp
from backend.routes.prediction_routes import prediction_bp


def create_app():
    """
    Create and configure the Flask application.
    """

    app = Flask(__name__)

    app.config.from_object(Config)

    CORS(app)

    # Initialize application databases.
    with app.app_context():
        auth_service.initialize_database()

        PaymentRepository().initialize_database()

    @app.get("/api/health")
    def health_check():
        return {
            "status": "ok",
            "service": "upi-fraud-detection-api"
        }

    # Authentication routes.
    app.register_blueprint(auth_bp)

    # Fraud prediction route.
    app.register_blueprint(prediction_bp)

    # Payment routes.
    app.register_blueprint(payment_bp)

    return app


app = create_app()