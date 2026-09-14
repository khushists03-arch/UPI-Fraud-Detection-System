import pytest

from backend.app import create_app


@pytest.fixture
def client():
    """
    Create a Flask test client.
    """

    app = create_app()

    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_health_check(client):
    """
    Verify that the health endpoint is working.
    """

    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"
    assert data["service"] == "upi-fraud-detection-api"


def test_prediction_with_valid_transaction(client):
    """
    Verify that a valid transaction reaches the ML pipeline
    and returns a prediction.
    """

    transaction = {
        "amount": 1500,
        "merchant": "Amazon",
        "transactionType": "Payment",
        "location": "India",
        "description": "payment for order"
    }

    response = client.post(
        "/api/predict",
        json=transaction
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["success"] is True

    assert data["prediction"] in [
        "Fraud",
        "Safe"
    ]

    assert "confidence" in data

    assert "fraud_score" in data

    assert "xgboost" in data

    assert "isolation_forest" in data


def test_prediction_rejects_missing_amount(client):
    """
    Verify that a transaction without an amount
    is rejected.
    """

    transaction = {
        "merchant": "Amazon",
        "transactionType": "Payment",
        "location": "India",
        "description": "payment for order"
    }

    response = client.post(
        "/api/predict",
        json=transaction
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False

    assert "error" in data

    assert "details" in data


def test_prediction_rejects_invalid_amount(client):
    """
    Verify that a negative transaction amount
    is rejected.
    """

    transaction = {
        "amount": -500,
        "merchant": "Amazon",
        "transactionType": "Payment",
        "location": "India",
        "description": "payment for order"
    }

    response = client.post(
        "/api/predict",
        json=transaction
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False


def test_prediction_rejects_missing_merchant(client):
    """
    Verify that a transaction without a merchant
    is rejected.
    """

    transaction = {
        "amount": 1500,
        "transactionType": "Payment",
        "location": "India",
        "description": "payment for order"
    }

    response = client.post(
        "/api/predict",
        json=transaction
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False


def test_prediction_rejects_invalid_transaction_type(client):
    """
    Verify that an unsupported transaction type
    is rejected.
    """

    transaction = {
        "amount": 1500,
        "merchant": "Amazon",
        "transactionType": "INVALID_TYPE",
        "location": "India",
        "description": "payment for order"
    }

    response = client.post(
        "/api/predict",
        json=transaction
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False


def test_prediction_rejects_missing_location(client):
    """
    Verify that a transaction without a location
    is rejected.
    """

    transaction = {
        "amount": 1500,
        "merchant": "Amazon",
        "transactionType": "Payment",
        "description": "payment for order"
    }

    response = client.post(
        "/api/predict",
        json=transaction
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False


def test_prediction_rejects_missing_description(client):
    """
    Verify that a transaction without a description
    is rejected.
    """

    transaction = {
        "amount": 1500,
        "merchant": "Amazon",
        "transactionType": "Payment",
        "location": "India"
    }

    response = client.post(
        "/api/predict",
        json=transaction
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False


def test_prediction_rejects_empty_json(client):
    """
    Verify that an empty JSON object is rejected.
    """

    response = client.post(
        "/api/predict",
        json={}
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False


def test_prediction_rejects_invalid_json(client):
    """
    Verify that a request without valid JSON
    is rejected.
    """

    response = client.post(
        "/api/predict",
        data="this is not json",
        content_type="application/json"
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["success"] is False