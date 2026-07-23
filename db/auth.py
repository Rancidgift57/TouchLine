"""
db/auth.py — password hashing and user/team CRUD backing real login/signup.

Deliberately dependency-light (stdlib hashlib, no bcrypt/passlib) so it
installs cleanly on Render/Fly/Railway without extra native build steps.
PBKDF2-HMAC-SHA256 with a random per-user salt, stored as a single
self-describing string: "pbkdf2$<iterations>$<salt_hex>$<hash_hex>".

This replaces the frontend's old `ACCOUNTS` in-memory array — accounts now
live in Turso's `users`/`teams` tables and survive restarts/redeploys.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid

PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations_s, salt, hash_hex = stored.split("$")
        if scheme != "pbkdf2":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations_s))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


async def ensure_auth_schema(client):
    # 1. Ensure the base users table exists
    await client.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            manager_name TEXT,
            club_name TEXT
        )
    """)
    
    # 2. Ask SQLite for a list of all columns currently in the 'users' table
    res = await client.execute("PRAGMA table_info(users)")
    existing_columns = [row["name"] for row in res.rows]
    
    # 3. Only attempt to add the column if it doesn't exist yet
    if "password_hash" not in existing_columns:
        await client.execute("ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''")

async def create_user(client, email: str, password: str, manager_name: str, club_name: str) -> dict:
    email_norm = email.strip().lower()

    existing = await client.execute("SELECT id FROM users WHERE email = ?", [email_norm])
    if existing.rows:
        raise ValueError("An account with that email already exists.")

    user_id = str(uuid.uuid4())
    team_id = str(uuid.uuid4())
    await client.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
        [user_id, manager_name.strip() or "Manager", email_norm, hash_password(password)],
    )
    await client.execute(
        "INSERT INTO teams (id, manager_user_id, name) VALUES (?, ?, ?)",
        [team_id, user_id, club_name.strip() or f"{manager_name}'s Club"],
    )
    return {"user_id": user_id, "team_id": team_id, "manager_name": manager_name, "email": email_norm, "club_name": club_name}


async def authenticate_user(client, email: str, password: str) -> dict:
    email_norm = email.strip().lower()
    rs = await client.execute(
        "SELECT id, username, email, password_hash FROM users WHERE email = ?", [email_norm]
    )
    if not rs.rows:
        raise ValueError("No account matches that email and password.")

    row = rs.rows[0]
    if not verify_password(password, row["password_hash"]):
        raise ValueError("No account matches that email and password.")

    team_rs = await client.execute(
        "SELECT id, name, balance FROM teams WHERE manager_user_id = ? ORDER BY created_at LIMIT 1",
        [row["id"]],
    )
    team = team_rs.rows[0] if team_rs.rows else None

    return {
        "user_id": row["id"], "manager_name": row["username"], "email": row["email"],
        "team_id": team["id"] if team else None,
        "team_name": team["name"] if team else None,
        "balance": team["balance"] if team else None,
    }
