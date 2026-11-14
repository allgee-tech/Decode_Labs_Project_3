
import sqlite3
import hashlib
import secrets
from pathlib import Path


# Store the database in the main project folder.
DB_PATH = Path(__file__).parent / "smartrecommend.db"


# ==========================================
# 1. DATABASE CONNECTION
# ==========================================

def get_connection():

    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# ==========================================
# 2. CREATE DATABASE TABLES
# ==========================================

def initialize_database():

    with get_connection() as connection:

        # Users table.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ratings table.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(user_id, title),

                FOREIGN KEY(user_id)
                REFERENCES users(username)
                ON DELETE CASCADE
            )
        """)


# ==========================================
# 3. PASSWORD HASHING
# ==========================================

def hash_password(password, salt):

    password_bytes = password.encode("utf-8")

    salt_bytes = salt.encode("utf-8")

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password_bytes,
        salt_bytes,
        310000
    )

    return hashed.hex()


# ==========================================
# 4. REGISTER A NEW USER
# ==========================================

def register_user(username, email, password):

    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:

        return False, "All fields are required."

    if len(username) < 3:

        return False, "Username must have at least 3 characters."

    if len(password) < 8:

        return False, "Password must have at least 8 characters."

    salt = secrets.token_hex(16)

    password_hash = hash_password(password, salt)

    try:

        with get_connection() as connection:

            connection.execute("""
                INSERT INTO users (
                    username,
                    email,
                    password_hash,
                    salt
                )
                VALUES (?, ?, ?, ?)
            """, (
                username,
                email,
                password_hash,
                salt
            ))

        return True, "Account created successfully."

    except sqlite3.IntegrityError:

        return False, "Username or email already exists."


# ==========================================
# 5. AUTHENTICATE A USER
# ==========================================

def authenticate_user(username, password):

    with get_connection() as connection:

        user = connection.execute("""
            SELECT username, password_hash, salt
            FROM users
            WHERE username = ?
        """, (username.strip(),)).fetchone()

    if user is None:

        return False

    stored_username, stored_hash, salt = user

    entered_hash = hash_password(password, salt)

    return secrets.compare_digest(
        stored_hash,
        entered_hash
    )


# ==========================================
# 6. SAVE A USER RATING
# ==========================================

def save_rating(username, title, rating):

    with get_connection() as connection:

        connection.execute("""
            INSERT INTO ratings (
                user_id,
                title,
                rating
            )
            VALUES (?, ?, ?)

            ON CONFLICT(user_id, title)

            DO UPDATE SET
                rating = excluded.rating,
                created_at = CURRENT_TIMESTAMP
        """, (
            username,
            title,
            int(rating)
        ))


# ==========================================
# 7. LOAD ALL RATINGS
# ==========================================

def load_ratings():

    with get_connection() as connection:

        ratings = connection.execute("""
            SELECT user_id, title, rating
            FROM ratings
        """).fetchall()

    import pandas as pd

    return pd.DataFrame(
        ratings,
        columns=["user_id", "title", "rating"]
    )


# ==========================================
# 8. LOAD ONE USER'S RATINGS
# ==========================================

def get_user_ratings(username):

    with get_connection() as connection:

        ratings = connection.execute("""
            SELECT title, rating, created_at
            FROM ratings
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (username,)).fetchall()

    import pandas as pd

    return pd.DataFrame(
        ratings,
        columns=["title", "rating", "created_at"]
    )
