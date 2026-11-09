
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "ledgerlens.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                type TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                description TEXT
            )
        """)


def add_transaction(date, type, category, amount, description):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO transactions
            (date, type, category, amount, description)
            VALUES (?, ?, ?, ?, ?)
        """, (date, type, category, amount, description))


def get_transactions():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM transactions
            ORDER BY date DESC, id DESC
        """).fetchall()
        return [dict(row) for row in rows]


def update_transaction(id, date, type, category, amount, description):
    with get_connection() as conn:
        conn.execute("""
            UPDATE transactions
            SET date = ?, type = ?, category = ?,
                amount = ?, description = ?
            WHERE id = ?
        """, (date, type, category, amount, description, id))


def delete_transaction(id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM transactions WHERE id = ?", (id,)
        )


def get_summary():
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN type = 'Income'
                    THEN amount ELSE 0 END), 0) AS income,
                COALESCE(SUM(CASE WHEN type = 'Expense'
                    THEN amount ELSE 0 END), 0) AS expenses
            FROM transactions
        """).fetchone()

        income = row["income"]
        expenses = row["expenses"]

        return {
            "income": income,
            "expenses": expenses,
            "balance": income - expenses
        }


create_database()
