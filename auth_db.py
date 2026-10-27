"""
StockSense AI - Authentication Database & User Management Module
Persistent SQLite storage for secure user authentication, role-based access,
salted password hashing (via Werkzeug), and session verification.
"""

import os
import sqlite3
import datetime
import secrets
from typing import Optional, Dict, Any, Tuple
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join("data", "auth.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Initializes the SQLite auth tables and seeds default enterprise accounts."""
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                company TEXT DEFAULT 'StockSense AI Enterprise',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE,
                token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                used INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resets_token ON password_resets (token)")

        # Seed default enterprise accounts if none exist
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM users")
        row = cursor.fetchone()
        if row and row["cnt"] == 0:
            demo_hash = generate_password_hash("StockSense2026!")
            admin_hash = generate_password_hash("AdminStock#2026")

            conn.execute("""
                INSERT INTO users (email, password_hash, full_name, role, company)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "demo@stocksense.ai",
                demo_hash,
                "Jaswin",
                "Lead Inventory Analyst",
                "StockSense Retail Labs"
            ))

            conn.execute("""
                INSERT INTO users (email, password_hash, full_name, role, company)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "kalpak@stocksense.ai",
                demo_hash,
                "Kalpak",
                "Senior Inventory Analyst",
                "StockSense Retail Labs"
            ))

            conn.execute("""
                INSERT INTO users (email, password_hash, full_name, role, company)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "admin@stocksense.ai",
                admin_hash,
                "Jaswin & Kalpak",
                "Supply Chain Analytics Directors",
                "StockSense Enterprise Operations"
            ))

    conn.close()


def verify_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Validates user credentials against stored password hash.
    Returns user dict on success, None on failure.
    """
    if not email or not password:
        return None

    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    user_dict = dict(row)
    if check_password_hash(user_dict["password_hash"], password):
        # Update last login timestamp
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with conn:
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_dict["id"]))
        conn.close()
        # Remove password hash from returned object for safety
        user_dict.pop("password_hash", None)
        return user_dict

    conn.close()
    return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves user profile by primary ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, full_name, role, company, created_at, last_login FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves user profile by email address."""
    if not email:
        return None
    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, full_name, role, company, created_at, last_login FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(
    email: str,
    password: str,
    full_name: str,
    role: str = "Inventory Analyst",
    company: str = "StockSense Enterprise"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user with securely hashed password.
    Returns (success, message, user_dict).
    """
    if not email or "@" not in email or "." not in email:
        return False, "Invalid email address format.", None

    email = email.strip().lower()
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters.", None

    if not full_name or len(full_name.strip()) < 2:
        return False, "Full name is required.", None

    conn = get_connection()
    try:
        pw_hash = generate_password_hash(password)
        with conn:
            cursor = conn.execute("""
                INSERT INTO users (email, password_hash, full_name, role, company)
                VALUES (?, ?, ?, ?, ?)
            """, (email, pw_hash, full_name.strip(), role.strip(), company.strip()))
            new_id = cursor.lastrowid
        conn.close()

        return True, "User registered successfully.", {
            "id": new_id,
            "email": email,
            "full_name": full_name.strip(),
            "role": role.strip(),
            "company": company.strip()
        }
    except sqlite3.IntegrityError:
        conn.close()
        return False, "An account with this email address already exists.", None
    except Exception as e:
        conn.close()
        return False, f"Failed to register account: {str(e)}", None


def create_password_reset_request(email: str) -> Tuple[bool, str, Optional[str]]:
    """
    Generates a secure password reset token for the given email.
    In a full production mail setup, this token would be emailed.
    Returns (success, message, token).
    """
    if not email:
        return False, "Email address is required.", None

    email = email.strip().lower()
    user = get_user_by_email(email)
    if not user:
        # Return generic success message to prevent user enumeration attacks
        return True, "If that email exists in our system, recovery instructions have been prepared.", None

    token = secrets.token_urlsafe(32)
    expires = (datetime.datetime.now() + datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    with conn:
        conn.execute("""
            INSERT INTO password_resets (email, token, expires_at)
            VALUES (?, ?, ?)
        """, (email, token, expires))
    conn.close()

    return True, "Password reset token created successfully.", token


def reset_password_with_token(token: str, new_password: str) -> Tuple[bool, str]:
    """Resets user password using a valid, unexpired token."""
    if not token:
        return False, "Invalid or missing reset token."
    if not new_password or len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM password_resets
        WHERE token = ? AND used = 0 AND expires_at > CURRENT_TIMESTAMP
    """, (token,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return False, "Reset link is invalid or has expired."

    reset_record = dict(row)
    email = reset_record["email"]
    new_hash = generate_password_hash(new_password)

    with conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, email))
        conn.execute("UPDATE password_resets SET used = 1 WHERE id = ?", (reset_record["id"],))
    conn.close()

    return True, "Password has been successfully updated. You may now sign in."
