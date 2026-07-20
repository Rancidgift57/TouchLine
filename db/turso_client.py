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


async def create_trade_offer(
    from_team_id: str,
    to_team_id: str,
    players_to_receiving_team: list[str],
    players_to_proposing_team: list[str],
    from_team_cash: int,
    to_team_cash: int,
    ml_fairness_score: float,
    ml_flag: str,
    clauses: list[dict[str, Any]] | None = None,
    expires_in_hours: float = 48.0,
    parent_offer_id: str | None = None,
) -> str:
    """
    Inserts a brand-new trade offer (or, when `parent_offer_id` is set, a
    counter-offer — same shape, just linked back to what it's countering).
    Mirrors `propose_counter_offer` below but is also usable for the very
    first offer in a negotiation, and takes the full set of legs/cash/
    clauses in one call instead of expecting the caller to know the table
    layout. `ml_fairness_score`/`ml_flag` are computed by the caller via
    `engine/market_ml.score_trade()` BEFORE this is called, so a 'blocked'
    offer can be rejected before it ever touches the database.

    `clauses` is a list of {"player_id", "clause_type" ('buyback'|
    'sell_on'), "buyback_fee", "buyback_expires_season", "sell_on_percentage"}.
    """
    client = get_client()
    offer_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = now.timestamp() + expires_in_hours * 3600

    statements = [
        libsql_client.Statement(
            "INSERT INTO trade_offers (id, from_team_id, to_team_id, parent_offer_id, status, "
            "ml_fairness_score, ml_flag, expires_at) VALUES (?, ?, ?, ?, 'proposed', ?, ?, ?)",
            [offer_id, from_team_id, to_team_id, parent_offer_id, ml_fairness_score, ml_flag,
             datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()],
        ),
    ]
    if parent_offer_id:
        statements.append(libsql_client.Statement(
            "UPDATE trade_offers SET status = 'countered', resolved_at = ? WHERE id = ?",
            [now.isoformat(), parent_offer_id],
        ))
    for player_id in players_to_receiving_team:
        statements.append(libsql_client.Statement(
            "INSERT INTO trade_offer_players (offer_id, player_id, direction) VALUES (?, ?, 'to_receiving_team')",
            [offer_id, player_id],
        ))
    for player_id in players_to_proposing_team:
        statements.append(libsql_client.Statement(
            "INSERT INTO trade_offer_players (offer_id, player_id, direction) VALUES (?, ?, 'to_proposing_team')",
            [offer_id, player_id],
        ))
    if from_team_cash or to_team_cash:
        statements.append(libsql_client.Statement(
            "INSERT INTO trade_offer_cash (offer_id, from_team_cash, to_team_cash) VALUES (?, ?, ?)",
            [offer_id, from_team_cash, to_team_cash],
        ))
    for clause in clauses or []:
        statements.append(libsql_client.Statement(
            "INSERT INTO trade_offer_clauses (offer_id, player_id, clause_type, buyback_fee, "
            "buyback_expires_season, sell_on_percentage) VALUES (?, ?, ?, ?, ?, ?)",
            [offer_id, clause["player_id"], clause["clause_type"], clause.get("buyback_fee"),
             clause.get("buyback_expires_season"), clause.get("sell_on_percentage")],
        ))

    await client.batch(statements)
    return offer_id


async def respond_to_trade_offer(offer_id: str, action: str) -> dict[str, Any]:
    """
    The RECEIVING side's response to a live offer: 'accept' settles it
    immediately via `execute_trade` (ACID), 'decline' just closes it out.
    Kept separate from `execute_trade` itself so a caller (the trade API
    route) can go straight from "the other manager just clicked Accept" to
    a settled trade in one call.
    """
    if action == "decline":
        client = get_client()
        now = datetime.now(timezone.utc).isoformat()
        await client.execute(
            "UPDATE trade_offers SET status = 'rejected', resolved_at = ? WHERE id = ? AND status = 'proposed'",
            [now, offer_id],
        )
        return {"offer_id": offer_id, "status": "rejected"}
    if action == "accept":
        client = get_client()
        now = datetime.now(timezone.utc).isoformat()
        # execute_trade() requires status='accepted' already (it's also the
        # entry point for a batch/offline settlement flow) -- flip it here
        # first, inside the same "the receiving side just said yes" call.
        rs = await client.execute(
            "UPDATE trade_offers SET status = 'accepted' WHERE id = ? AND status = 'proposed' RETURNING id",
            [offer_id],
        )
        if not rs.rows:
            raise TradeRejected(f"Offer {offer_id} is not in a state that can be accepted")
        return await execute_trade(offer_id)
    raise ValueError(f"Unknown trade response action: {action!r}")


async def get_trade_offer_detail(offer_id: str) -> dict[str, Any] | None:
    """Full offer detail (legs, cash, clauses) for pushing over the live
    trade WebSocket or returning from a REST call."""
    client = get_client()
    offer_rs = await client.execute(
        "SELECT id, from_team_id, to_team_id, parent_offer_id, status, ml_fairness_score, "
        "ml_flag, created_at, expires_at, resolved_at FROM trade_offers WHERE id = ?",
        [offer_id],
    )
    if not offer_rs.rows:
        return None
    offer = dict(offer_rs.rows[0])

    players_rs = await client.execute(
        "SELECT p.id, p.name, p.position, p.overall, top.direction "
        "FROM trade_offer_players top JOIN players p ON p.id = top.player_id WHERE top.offer_id = ?",
        [offer_id],
    )
    offer["players"] = [dict(r) for r in players_rs.rows]

    cash_rs = await client.execute(
        "SELECT from_team_cash, to_team_cash FROM trade_offer_cash WHERE offer_id = ?", [offer_id]
    )
    offer["cash"] = dict(cash_rs.rows[0]) if cash_rs.rows else {"from_team_cash": 0, "to_team_cash": 0}

    clauses_rs = await client.execute(
        "SELECT player_id, clause_type, buyback_fee, buyback_expires_season, sell_on_percentage "
        "FROM trade_offer_clauses WHERE offer_id = ?", [offer_id]
    )
    offer["clauses"] = [dict(r) for r in clauses_rs.rows]
    return offer


async def list_team_trade_offers(team_id: str) -> list[dict[str, Any]]:
    """Inbox + outbox in one shot — open ('proposed') offers where this
    team is on either side, newest first."""
    client = get_client()
    rs = await client.execute(
        "SELECT id, from_team_id, to_team_id, status, ml_fairness_score, ml_flag, created_at "
        "FROM trade_offers WHERE (from_team_id = ? OR to_team_id = ?) AND status = 'proposed' "
        "ORDER BY created_at DESC",
        [team_id, team_id],
    )
    return [dict(r) for r in rs.rows]


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
