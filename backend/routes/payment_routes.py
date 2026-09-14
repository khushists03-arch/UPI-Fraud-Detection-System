from flask import Blueprint, g, jsonify, request

from backend.auth import require_authentication
from backend.services.payment_service import PaymentService
from backend.utils.validators import validate_transaction


payment_bp = Blueprint(
    "payments",
    __name__,
    url_prefix="/api/payments"
)


payment_service = PaymentService()


@payment_bp.post("")
@require_authentication
def create_payment():
    """
    Create and process a payment.

    The authenticated user's ID is taken from the JWT-authenticated
    Flask context. The client cannot choose another user's ID.
    """

    data = request.get_json(silent=True)

    is_valid, errors = validate_transaction(data)

    if not is_valid:
        return jsonify({
            "success": False,
            "error": "Invalid payment data.",
            "details": errors
        }), 400

    try:
        result = payment_service.process_payment(
            user_id=g.user["id"],
            transaction=data
        )

        payment = result["payment"]
        fraud = result["fraud"]

        return jsonify({
            "success": True,
            "message": (
                "Payment completed successfully."
                if payment["status"] == "COMPLETED"
                else "Payment blocked because it was identified as fraudulent."
            ),
            "payment": payment,
            "fraud": fraud
        }), 201

    except (ValueError, TypeError) as exc:
        return jsonify({
            "success": False,
            "error": "Invalid payment data.",
            "details": [str(exc)]
        }), 400

    except Exception:
        return jsonify({
            "success": False,
            "error": "Payment processing failed."
        }), 500


@payment_bp.get("")
@require_authentication
def get_payments():
    """
    Return all payments belonging to the authenticated user.
    """

    payments = payment_service.get_user_payments(
        user_id=g.user["id"]
    )

    return jsonify({
        "success": True,
        "payments": payments,
        "count": len(payments)
    }), 200


@payment_bp.get("/<int:payment_id>")
@require_authentication
def get_payment(payment_id):
    """
    Return one payment belonging to the authenticated user.
    """

    payment = payment_service.get_payment(
        payment_id=payment_id,
        user_id=g.user["id"]
    )

    if payment is None:
        return jsonify({
            "success": False,
            "error": "Payment not found."
        }), 404

    return jsonify({
        "success": True,
        "payment": payment
    }), 200