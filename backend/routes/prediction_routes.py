from flask import Blueprint, jsonify, request

from backend.services.fraud_service import FraudService
from backend.utils.validators import validate_transaction


prediction_bp = Blueprint(
    "prediction",
    __name__,
    url_prefix="/api"
)


fraud_service = FraudService()


@prediction_bp.post("/predict")
def predict_transaction():
    """
    Predict whether a UPI transaction is fraudulent.
    """

    # ---------------------------------------------------------
    # Read JSON request
    # ---------------------------------------------------------

    data = request.get_json(silent=True)

    # ---------------------------------------------------------
    # Validate transaction data
    # ---------------------------------------------------------

    is_valid, errors = validate_transaction(data)

    if not is_valid:
        return jsonify({
            "success": False,
            "error": "Invalid transaction data.",
            "details": errors
        }), 400

    # ---------------------------------------------------------
    # Run fraud prediction
    # ---------------------------------------------------------

    try:
        result = fraud_service.predict(data)

        return jsonify({
            "success": True,
            **result
        }), 200

    except (ValueError, TypeError) as exc:
        return jsonify({
            "success": False,
            "error": "Invalid transaction data.",
            "details": [str(exc)]
        }), 400

    except Exception:
        return jsonify({
            "success": False,
            "error": "Prediction failed."
        }), 500