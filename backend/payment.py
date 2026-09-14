import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app


class PaymentRepository:
    """
    Handle payment persistence using SQLite.

    The repository is responsible only for database operations.
    Fraud detection and business logic belong in PaymentService.
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
        Create a SQLite database connection.
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

        # Enable foreign-key enforcement for this connection.
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    def initialize_database(self):
        """
        Create the payments table if it does not already exist.
        """

        connection = self._connect()

        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    user_id INTEGER NOT NULL,

                    amount REAL NOT NULL,
                    merchant TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    location TEXT NOT NULL,
                    description TEXT NOT NULL,

                    fraud_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    prediction TEXT NOT NULL,

                    status TEXT NOT NULL,

                    created_at TEXT NOT NULL,

                    FOREIGN KEY (user_id)
                        REFERENCES users(id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.commit()

        finally:
            connection.close()

    def create_payment(
        self,
        user_id,
        amount,
        merchant,
        transaction_type,
        location,
        description,
        fraud_score,
        confidence,
        prediction,
        status
    ):
        """
        Save a processed payment to the database.
        """

        self.initialize_database()

        connection = self._connect()

        try:
            created_at = datetime.now(
                timezone.utc
            ).isoformat()

            cursor = connection.execute(
                """
                INSERT INTO payments (
                    user_id,
                    amount,
                    merchant,
                    transaction_type,
                    location,
                    description,
                    fraud_score,
                    confidence,
                    prediction,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    amount,
                    merchant,
                    transaction_type,
                    location,
                    description,
                    fraud_score,
                    confidence,
                    prediction,
                    status,
                    created_at
                )
            )

            connection.commit()

            payment_id = cursor.lastrowid

            payment = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    amount,
                    merchant,
                    transaction_type,
                    location,
                    description,
                    fraud_score,
                    confidence,
                    prediction,
                    status,
                    created_at
                FROM payments
                WHERE id = ?
                """,
                (payment_id,)
            ).fetchone()

            return dict(payment)

        finally:
            connection.close()

    def get_payment_by_id(
        self,
        payment_id,
        user_id=None
    ):
        """
        Retrieve one payment.

        If user_id is provided, the payment must belong
        to that user.
        """

        self.initialize_database()

        connection = self._connect()

        try:
            if user_id is None:
                payment = connection.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        amount,
                        merchant,
                        transaction_type,
                        location,
                        description,
                        fraud_score,
                        confidence,
                        prediction,
                        status,
                        created_at
                    FROM payments
                    WHERE id = ?
                    """,
                    (payment_id,)
                ).fetchone()

            else:
                payment = connection.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        amount,
                        merchant,
                        transaction_type,
                        location,
                        description,
                        fraud_score,
                        confidence,
                        prediction,
                        status,
                        created_at
                    FROM payments
                    WHERE id = ?
                      AND user_id = ?
                    """,
                    (
                        payment_id,
                        user_id
                    )
                ).fetchone()

            if payment is None:
                return None

            return dict(payment)

        finally:
            connection.close()

    def get_payments_by_user(
        self,
        user_id
    ):
        """
        Retrieve all payments belonging to a user.
        """

        self.initialize_database()

        connection = self._connect()

        try:
            payments = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    amount,
                    merchant,
                    transaction_type,
                    location,
                    description,
                    fraud_score,
                    confidence,
                    prediction,
                    status,
                    created_at
                FROM payments
                WHERE user_id = ?
                ORDER BY id DESC
                """,
                (user_id,)
            ).fetchall()

            return [
                dict(payment)
                for payment in payments
            ]

        finally:
            connection.close()