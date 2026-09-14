from flask import Blueprint, g, jsonify, request

from backend.auth import (
    auth_service,
    require_authentication
)


auth_bp = Blueprint(
    "auth",
    __name__,
    url_prefix="/api/auth"
)


def _validate_registration_data(data):
    """
    Validate registration request data.
    """

    errors = []

    if not isinstance(data, dict):
        return [
            "Request body must contain a JSON object."
        ]

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not isinstance(username, str):
        errors.append(
            "Username must be a string."
        )
    elif not username.strip():
        errors.append(
            "Username is required."
        )
    elif len(username.strip()) < 3:
        errors.append(
            "Username must contain at least 3 characters."
        )
    elif len(username.strip()) > 50:
        errors.append(
            "Username must not exceed 50 characters."
        )

    if not isinstance(email, str):
        errors.append(
            "Email must be a string."
        )
    elif not email.strip():
        errors.append(
            "Email is required."
        )
    elif "@" not in email or "." not in email.split("@")[-1]:
        errors.append(
            "A valid email address is required."
        )

    if not isinstance(password, str):
        errors.append(
            "Password must be a string."
        )
    elif not password:
        errors.append(
            "Password is required."
        )
    elif len(password) < 8:
        errors.append(
            "Password must contain at least 8 characters."
        )

    return errors


def _validate_login_data(data):
    """
    Validate login request data.
    """

    errors = []

    if not isinstance(data, dict):
        return [
            "Request body must contain a JSON object."
        ]

    email = data.get("email")
    password = data.get("password")

    if not isinstance(email, str) or not email.strip():
        errors.append(
            "Email is required."
        )

    if not isinstance(password, str) or not password:
        errors.append(
            "Password is required."
        )

    return errors


@auth_bp.post("/register")
def register():
    """
    Register a new user.
    """

    data = request.get_json(
        silent=True
    )

    errors = _validate_registration_data(
        data
    )

    if errors:
        return jsonify({
            "success": False,
            "error": "Invalid registration data.",
            "details": errors
        }), 400

    user, error = auth_service.register_user(
        username=data["username"],
        email=data["email"],
        password=data["password"]
    )

    if error:
        return jsonify({
            "success": False,
            "error": error
        }), 409

    return jsonify({
        "success": True,
        "message": "User registered successfully.",
        "user": user
    }), 201


@auth_bp.post("/login")
def login():
    """
    Authenticate a user and return a JWT token.
    """

    data = request.get_json(
        silent=True
    )

    errors = _validate_login_data(
        data
    )

    if errors:
        return jsonify({
            "success": False,
            "error": "Invalid login data.",
            "details": errors
        }), 400

    user = auth_service.authenticate_user(
        email=data["email"],
        password=data["password"]
    )

    if user is None:
        return jsonify({
            "success": False,
            "error": "Invalid email or password."
        }), 401

    token = auth_service.generate_token(
        user
    )

    return jsonify({
        "success": True,
        "message": "Login successful.",
        "access_token": token,
        "token_type": "Bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "created_at": user["created_at"]
        }
    }), 200


@auth_bp.get("/me")
@require_authentication
def get_current_user():
    """
    Return information about the currently authenticated user.
    """

    return jsonify({
        "success": True,
        "user": g.user
    }), 200