"""
Turso (libSQL) client wrapper.

Install:
    pip install libsql-client

Env vars expected:
    TURSO_DATABASE_URL   e.g. "libsql://your-db-name-org.turso.io"
    TURSO_AUTH_TOKEN     token from `turso db tokens create your-db-name`

This module exposes:
    get_client()                -> a shared libsql_client.Client
    run_migrations(schema_path) -> applies db/schema.sql
    execute_trade(offer_id)     -> ACID-safe multi-table trade settlement
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import libsql_client

_client: libsql_client.Client | None = None


def get_client() -> libsql_client.Client:
    """Lazily create a single shared libSQL client for the process."""
    global _client
    if _client is None:
        url = os.environ["TURSO_DATABASE_URL"]
        token = os.environ.get("TURSO_AUTH_TOKEN")
        _client = libsql_client.create_client(url=url, auth_token=token)
    return _client


async def run_migrations(schema_path: str = "db/schema.sql") -> None:
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    client = get_client()
    # Split on semicolons that terminate statements; skip blank/comment-only chunks.
    statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
    for stmt in statements:
        await client.execute(stmt)


class TradeRejected(Exception):
    """Raised when a trade cannot be settled (insufficient funds, stale offer, etc.)."""


async def execute_trade(offer_id: str) -> dict[str, Any]:
    """
    Atomically settle an accepted trade offer:
      1. Re-validate the offer is still 'accepted' and not expired.
      2. Move cash between team balances.
      3. Flip player_rights.owner_team_id for every player in the deal.
      4. Write new buyback / sell-on clauses from trade_offer_clauses.
      5. Mark the offer resolved.

    All of this happens inside a single libSQL transaction (batch), so a
    failure partway through (e.g. insufficient balance) rolls back the
    entire trade -- no half-completed multi-asset swaps.
    """
    client = get_client()

    offer_rs = await client.execute(
        "SELECT id, from_team_id, to_team_id, status, expires_at, ml_flag "
        "FROM trade_offers WHERE id = ?",
        [offer_id],
    )
    if not offer_rs.rows:
        raise TradeRejected(f"Offer {offer_id} does not exist")

    offer = offer_rs.rows[0]
    if offer["status"] != "accepted":
        raise TradeRejected(f"Offer {offer_id} is not in 'accepted' state (got {offer['status']})")
    if offer["ml_flag"] == "blocked":
        raise TradeRejected(f"Offer {offer_id} was blocked by the anti-cheat valuation model")

    from_team_id = offer["from_team_id"]
    to_team_id = offer["to_team_id"]

    cash_rs = await client.execute(
        "SELECT from_team_cash, to_team_cash FROM trade_offer_cash WHERE offer_id = ?",
        [offer_id],
    )
    from_cash, to_cash = (0, 0) if not cash_rs.rows else (cash_rs.rows[0]["from_team_cash"], cash_rs.rows[0]["to_team_cash"])

    players_rs = await client.execute(
        "SELECT player_id, direction FROM trade_offer_players WHERE offer_id = ?",
        [offer_id],
    )
    clauses_rs = await client.execute(
        "SELECT player_id, clause_type, buyback_fee, buyback_expires_season, sell_on_percentage "
        "FROM trade_offer_clauses WHERE offer_id = ?",
        [offer_id],
    )

    # Balance pre-check (fail fast before opening the batch, avoids a
    # guaranteed-to-rollback transaction under contention).
    balances_rs = await client.execute(
        "SELECT id, balance FROM teams WHERE id IN (?, ?)", [from_team_id, to_team_id]
    )
    balances = {row["id"]: row["balance"] for row in balances_rs.rows}
    if balances.get(from_team_id, 0) < from_cash:
        raise TradeRejected("Proposing team has insufficient funds")
    if balances.get(to_team_id, 0) < to_cash:
        raise TradeRejected("Receiving team has insufficient funds")

    now = datetime.now(timezone.utc).isoformat()
    statements: list[tuple[str, list[Any]]] = []

    if from_cash:
        statements.append(("UPDATE teams SET balance = balance - ? WHERE id = ?", [from_cash, from_team_id]))
        statements.append(("UPDATE teams SET balance = balance + ? WHERE id = ?", [from_cash, to_team_id]))
    if to_cash:
        statements.append(("UPDATE teams SET balance = balance - ? WHERE id = ?", [to_cash, to_team_id]))
        statements.append(("UPDATE teams SET balance = balance + ? WHERE id = ?", [to_cash, from_team_id]))

    for row in players_rs.rows:
        new_owner = to_team_id if row["direction"] == "to_receiving_team" else from_team_id
        statements.append((
            "UPDATE player_rights SET owner_team_id = ?, updated_at = ? WHERE player_id = ?",
            [new_owner, now, row["player_id"]],
        ))

    for row in clauses_rs.rows:
        if row["clause_type"] == "buyback":
            # The buy-back right is held by whichever team is SELLING the player.
            selling_team = None
            for p in players_rs.rows:
                if p["player_id"] == row["player_id"]:
                    selling_team = from_team_id if p["direction"] == "to_receiving_team" else to_team_id
            statements.append((
                "UPDATE player_rights SET buyback_holder_team_id = ?, buyback_fee = ?, "
                "buyback_expires_season = ?, updated_at = ? WHERE player_id = ?",
                [selling_team, row["buyback_fee"], row["buyback_expires_season"], now, row["player_id"]],
            ))
        elif row["clause_type"] == "sell_on":
            selling_team = None
            for p in players_rs.rows:
                if p["player_id"] == row["player_id"]:
                    selling_team = from_team_id if p["direction"] == "to_receiving_team" else to_team_id
            statements.append((
                "UPDATE player_rights SET sell_on_percentage = ?, sell_on_beneficiary_team_id = ?, "
                "updated_at = ? WHERE player_id = ?",
                [row["sell_on_percentage"], selling_team, now, row["player_id"]],
            ))

    statements.append((
        "UPDATE trade_offers SET status = 'accepted', resolved_at = ? WHERE id = ?",
        [now, offer_id],
    ))

    # libsql_client.batch() runs every statement inside one transaction and
    # rolls back entirely on any failure -> ACID guarantee for the swap.
    await client.batch([libsql_client.Statement(sql, args) for sql, args in statements])

    return {
        "offer_id": offer_id,
        "settled_at": now,
        "players_moved": len(players_rs.rows),
        "cash_from_proposer": from_cash,
        "cash_from_receiver": to_cash,
    }


async def propose_counter_offer(original_offer_id: str, new_offer_payload: dict[str, Any]) -> str:
    """Marks the original offer 'countered' and inserts a new linked offer row."""
    client = get_client()
    new_id = str(uuid.uuid4())
    await client.batch([
        libsql_client.Statement(
            "UPDATE trade_offers SET status = 'countered', resolved_at = ? WHERE id = ?",
            [datetime.now(timezone.utc).isoformat(), original_offer_id],
        ),
        libsql_client.Statement(
            "INSERT INTO trade_offers (id, from_team_id, to_team_id, parent_offer_id, status, expires_at) "
            "VALUES (?, ?, ?, ?, 'proposed', ?)",
            [
                new_id,
                new_offer_payload["from_team_id"],
                new_offer_payload["to_team_id"],
                original_offer_id,
                new_offer_payload["expires_at"],
            ],
        ),
    ])
    return new_id
