from backend.payment import PaymentRepository
from backend.services.fraud_service import FraudService


class PaymentService:
    """
    Coordinate payment processing and fraud detection.

    Responsibilities:
    - Run fraud detection on the transaction.
    - Decide whether the payment is completed or blocked.
    - Persist the processed payment.
    """

    def __init__(
        self,
        payment_repository=None,
        fraud_service=None
    ):
        self.payment_repository = (
            payment_repository
            or PaymentRepository()
        )

        self.fraud_service = (
            fraud_service
            or FraudService()
        )

    def process_payment(
        self,
        user_id,
        transaction
    ):
        """
        Process a payment through the fraud detection system.

        A fraudulent transaction is BLOCKED.
        A safe transaction is COMPLETED.
        """

        fraud_result = self.fraud_service.predict(
            transaction
        )

        prediction = fraud_result["prediction"]

        if prediction == "Fraud":
            status = "BLOCKED"
        else:
            status = "COMPLETED"

        payment = self.payment_repository.create_payment(
            user_id=user_id,
            amount=transaction["amount"],
            merchant=transaction["merchant"],
            transaction_type=transaction["transactionType"],
            location=transaction["location"],
            description=transaction["description"],
            fraud_score=fraud_result["fraud_score"],
            confidence=fraud_result["confidence"],
            prediction=prediction,
            status=status
        )

        return {
            "payment": payment,
            "fraud": fraud_result
        }

    def get_payment(
        self,
        payment_id,
        user_id=None
    ):
        """
        Retrieve a single payment.
        """

        return self.payment_repository.get_payment_by_id(
            payment_id=payment_id,
            user_id=user_id
        )

    def get_user_payments(
        self,
        user_id
    ):
        """
        Retrieve all payments belonging to a user.
        """

        return self.payment_repository.get_payments_by_user(
            user_id=user_id
        )