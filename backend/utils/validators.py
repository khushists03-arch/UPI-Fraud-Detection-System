from typing import Any, Dict, List, Tuple


ALLOWED_TRANSACTION_TYPES = {
    "Payment",
    "Transfer",
    "Request Money",
}


def validate_transaction(
    data: Any,
) -> Tuple[bool, List[str]]:
    """
    Validate transaction data before sending it
    to the machine-learning prediction pipeline.

    Returns:
        Tuple containing:
        - True/False indicating whether the data is valid.
        - A list of validation error messages.
    """

    errors: List[str] = []

    # ---------------------------------------------------------
    # Check request body
    # ---------------------------------------------------------

    if not isinstance(data, dict):
        return False, [
            "Request body must be a JSON object."
        ]

    # ---------------------------------------------------------
    # Required fields
    # ---------------------------------------------------------

    required_fields = [
        "amount",
        "merchant",
        "transactionType",
        "location",
        "description",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing_fields:
        errors.append(
            "Missing required fields: "
            + ", ".join(missing_fields)
        )

    # Stop here if required fields are missing.
    if errors:
        return False, errors

    # ---------------------------------------------------------
    # Validate amount
    # ---------------------------------------------------------

    amount = data["amount"]

    if isinstance(amount, bool):
        errors.append(
            "Amount must be a positive number."
        )

    else:
        try:
            amount = float(amount)

            if amount <= 0:
                errors.append(
                    "Amount must be greater than 0."
                )

        except (TypeError, ValueError):
            errors.append(
                "Amount must be a valid number."
            )

    # ---------------------------------------------------------
    # Validate merchant
    # ---------------------------------------------------------

    merchant = data["merchant"]

    if not isinstance(merchant, str):
        errors.append(
            "Merchant must be a string."
        )

    elif not merchant.strip():
        errors.append(
            "Merchant cannot be empty."
        )

    # ---------------------------------------------------------
    # Validate transaction type
    # ---------------------------------------------------------

    transaction_type = data["transactionType"]

    if transaction_type not in ALLOWED_TRANSACTION_TYPES:
        errors.append(
            "Invalid transaction type. "
            "Allowed values are: "
            + ", ".join(
                sorted(ALLOWED_TRANSACTION_TYPES)
            )
        )

    # ---------------------------------------------------------
    # Validate location
    # ---------------------------------------------------------

    location = data["location"]

    if not isinstance(location, str):
        errors.append(
            "Location must be a string."
        )

    elif not location.strip():
        errors.append(
            "Location cannot be empty."
        )

    # ---------------------------------------------------------
    # Validate description
    # ---------------------------------------------------------

    description = data["description"]

    if not isinstance(description, str):
        errors.append(
            "Description must be a string."
        )

    elif not description.strip():
        errors.append(
            "Description cannot be empty."
        )

    # ---------------------------------------------------------
    # Return validation result
    # ---------------------------------------------------------

    return len(errors) == 0, errors