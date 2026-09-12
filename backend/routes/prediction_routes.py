from flask import Blueprint, jsonify, request

from backend.services.fraud_service import FraudService


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

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Request body must contain JSON data."
        }), 400

    required_fields = [
        "amount",
        "merchant",
        "transactionType",
        "location",
        "description"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing_fields
        }), 400

    try:
        result = fraud_service.predict(data)

        return jsonify(result), 200

    except (ValueError, TypeError) as exc:
        return jsonify({
            "error": "Invalid transaction data.",
            "details": str(exc)
        }), 400

    except Exception:
        return jsonify({
            "error": "Prediction failed."
        }), 500