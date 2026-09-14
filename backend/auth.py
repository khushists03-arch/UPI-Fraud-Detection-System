import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import jwt
from flask import current_app, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash


class AuthService:
    """
    Handle user authentication and authorization.

    Responsibilities:
    - Create the users database.
    - Register users.
    - Verify login credentials.
    - Generate JWT access tokens.
    - Authenticate requests using JWT.
    - Authorize users based on their role.
    """

    def __init__(self, database_path=None):
        self.database_path = database_path

    def _get_database_path(self):
        """
        Return the configured SQLite database path.
        """

        if self.database_path:
            return Path(self.database_path)

        configured_path = current_app.config.get(
            "DATABASE_PATH"
        )

        if configured_path:
            return Path(configured_path)

        return (
            Path(current_app.root_path).parent
            / "data"
            / "upi_fraud.db"
        )

    def _connect(self):
        """
        Create a connection to the SQLite database.
        """

        database_path = self._get_database_path()

        database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        connection = sqlite3.connect(
            database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def initialize_database(self):
        """
        Create the users table if it does not already exist.
        """

        connection = self._connect()

        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'USER',
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.commit()

        finally:
            connection.close()

    def register_user(
        self,
        username,
        email,
        password
    ):
        """
        Register a new user.

        New users always receive the USER role.
        ADMIN accounts must be created separately.
        """

        username = username.strip()
        email = email.strip().lower()

        self.initialize_database()

        password_hash = generate_password_hash(
            password
        )

        connection = self._connect()

        try:
            existing_user = connection.execute(
                """
                SELECT id
                FROM users
                WHERE username = ? OR email = ?
                """,
                (
                    username,
                    email
                )
            ).fetchone()

            if existing_user:
                return None, (
                    "A user with that username or "
                    "email already exists."
                )

            cursor = connection.execute(
                """
                INSERT INTO users (
                    username,
                    email,
                    password_hash,
                    role,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    password_hash,
                    "USER",
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )
            )

            connection.commit()

            user_id = cursor.lastrowid

            user = connection.execute(
                """
                SELECT
                    id,
                    username,
                    email,
                    role,
                    created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,)
            ).fetchone()

            return dict(user), None

        except sqlite3.IntegrityError:
            connection.rollback()

            return None, (
                "A user with that username or "
                "email already exists."
            )

        finally:
            connection.close()

    def authenticate_user(
        self,
        email,
        password
    ):
        """
        Verify user credentials.
        """

        email = email.strip().lower()

        self.initialize_database()

        connection = self._connect()

        try:
            user = connection.execute(
                """
                SELECT *
                FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

            if user is None:
                return None

            if not check_password_hash(
                user["password_hash"],
                password
            ):
                return None

            return dict(user)

        finally:
            connection.close()

    def generate_token(self, user):
        """
        Generate a JWT access token for an authenticated user.
        """

        secret_key = current_app.config.get(
            "SECRET_KEY"
        )

        if not secret_key:
            raise RuntimeError(
                "SECRET_KEY is not configured."
            )

        expires_in = current_app.config.get(
            "JWT_EXPIRATION_MINUTES",
            60
        )

        now = datetime.now(
            timezone.utc
        )

        payload = {
            "sub": str(user["id"]),
            "username": user["username"],
            "role": user["role"],
            "iat": now,
            "exp": now + timedelta(
                minutes=expires_in
            )
        }

        return jwt.encode(
            payload,
            secret_key,
            algorithm="HS256"
        )

    def decode_token(self, token):
        """
        Decode and validate a JWT token.
        """

        secret_key = current_app.config.get(
            "SECRET_KEY"
        )

        if not secret_key:
            raise RuntimeError(
                "SECRET_KEY is not configured."
            )

        try:
            return jwt.decode(
                token,
                secret_key,
                algorithms=["HS256"]
            )

        except jwt.ExpiredSignatureError:
            raise ValueError(
                "Authentication token has expired."
            )

        except jwt.InvalidTokenError:
            raise ValueError(
                "Invalid authentication token."
            )

    def get_user_by_id(self, user_id):
        """
        Retrieve a user by ID without returning
        the password hash.
        """

        self.initialize_database()

        connection = self._connect()

        try:
            user = connection.execute(
                """
                SELECT
                    id,
                    username,
                    email,
                    role,
                    created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,)
            ).fetchone()

            if user is None:
                return None

            return dict(user)

        finally:
            connection.close()


auth_service = AuthService()


def require_authentication(function):
    """
    Protect a route using JWT authentication.

    The authenticated user is stored in Flask's `g.user`.
    """

    @wraps(function)
    def decorated_function(*args, **kwargs):
        authorization_header = request.headers.get(
            "Authorization",
            ""
        )

        if not authorization_header:
            return jsonify({
                "success": False,
                "error": "Authentication required."
            }), 401

        parts = authorization_header.split(
            " ",
            1
        )

        if (
            len(parts) != 2
            or parts[0].lower() != "bearer"
            or not parts[1].strip()
        ):
            return jsonify({
                "success": False,
                "error": (
                    "Authorization header must use "
                    "Bearer token format."
                )
            }), 401

        token = parts[1].strip()

        try:
            payload = auth_service.decode_token(
                token
            )

        except ValueError as exc:
            return jsonify({
                "success": False,
                "error": str(exc)
            }), 401

        user = auth_service.get_user_by_id(
            payload["sub"]
        )

        if user is None:
            return jsonify({
                "success": False,
                "error": "User account not found."
            }), 401

        g.user = user
        g.token_payload = payload

        return function(*args, **kwargs)

    return decorated_function


def require_role(*allowed_roles):
    """
    Restrict a route to one or more user roles.

    Example:

        @require_authentication
        @require_role("ADMIN")
        def admin_route():
            ...
    """

    normalized_roles = {
        role.upper()
        for role in allowed_roles
    }

    def decorator(function):
        @wraps(function)
        def decorated_function(*args, **kwargs):
            user = getattr(
                g,
                "user",
                None
            )

            if user is None:
                return jsonify({
                    "success": False,
                    "error": "Authentication required."
                }), 401

            user_role = str(
                user.get("role", "")
            ).upper()

            if user_role not in normalized_roles:
                return jsonify({
                    "success": False,
                    "error": "You are not authorized to access this resource."
                }), 403

            return function(*args, **kwargs)

        return decorated_function

    return decorator