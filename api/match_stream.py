"""
WebSocket match streamer.

Compresses a 90-minute match into ~2 real minutes: 1 in-game minute ==
1.33 real seconds (90 * 1.33 ≈ 120s). The ML/rules engine still only
*decides* things at its own sparse pace (only "notable" minutes yield an
event — see engine/match_engine.py), but the on-screen clock does NOT
wait silently for those decisions: between events this module ticks the
clock every ~1 real second (`clock_tick` broadcasts, interpolated from
elapsed real time) so the minute counts up continuously instead of
freezing and then jumping the moment the next chance/goal is decided.

Two things this module owns that the pure `engine.match_engine` generator
deliberately does NOT (it has no I/O):
  1. The real half-time pause. `simulate_match` yields a `half_time_break`
     event carrying `break_seconds` (20s by default); THIS module is what
     actually `await asyncio.sleep()`s for that long before resuming the
     loop, while still listening on the tactics socket for a substitution.
  2. Live substitutions. `/ws/match/{id}/tactics` and `/ws/match/{id}` share
     a single `TeamSnapshot` instance per side (loaded once, before the
     loop starts). A substitution message mutates that same object via
     `apply_substitution`, in place — `simulate_match` re-reads
     `team.outfield()` every minute, so the change is live on the very
     next tick. This holds whether the match is being watched live OR was
     queued as a "simulate tonight's whole matchday" background job
     (see `replay_with_substitution` below): either way, the substitution
     is not cosmetic, it changes who gets rolled for every remaining duel.

Run with:
    uvicorn api.match_stream:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import random
import secrets
import string
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db.turso_client import get_client, ensure_match_schema
from db.auth import authenticate_user, create_user, ensure_auth_schema
from engine.match_engine import PlayerSnapshot, TeamSnapshot, apply_substitution, simulate_match
from engine.ml_bridge import MLBridge
from api import state_backend

app = FastAPI(title="Football Sim Match Streamer")

_schema_ensured = False


async def _ensure_schema(client) -> None:
    """Runs the auth + match schema migrations once per process (cheap
    no-ops on every call after the first, but calling every time would be
    wasted round-trips)."""
    global _schema_ensured
    if _schema_ensured:
        return
    await ensure_auth_schema(client)
    await ensure_match_schema(client)
    _schema_ensured = True

# CORS: the SPA (index.html) is deployed separately as a static site (see
# DEPLOY.md), so it calls this API from a different origin.
#
# Set FRONTEND_ORIGIN to the exact site URL(s), comma-separated if you have
# more than one (e.g. a Vercel prod URL + a preview URL) — no trailing
# slash, must match what the browser's address bar shows exactly:
#     FRONTEND_ORIGIN=https://touch-line-deploy.vercel.app,https://touchline.pages.dev
#
# FRONTEND_ORIGIN_REGEX additionally matches a pattern — handy for Vercel,
# which mints a new *.vercel.app URL per preview deploy:
#     FRONTEND_ORIGIN_REGEX=https://.*\.vercel\.app
#
# A "CORS error" in the browser console with net::ERR_FAILED is very often
# actually one of: (1) FRONTEND_ORIGIN unset/mismatched (check for a
# trailing slash or http vs https), (2) the backend host spun down/crashed
# and never answered the preflight at all (Render's free tier does this on
# every cold start — hit /health first to wake it up), or (3) the request
# hit an unhandled exception before CORSMiddleware could attach headers —
# see the try/except wrapping below for why that shouldn't happen here.
import os as _os
import re as _re
from fastapi.middleware.cors import CORSMiddleware

_frontend_origins = [
    o.strip().rstrip("/") for o in _os.environ.get("FRONTEND_ORIGIN", "").split(",") if o.strip()
] or ["*"]
_frontend_origin_regex = _os.environ.get("FRONTEND_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,
    allow_origin_regex=_frontend_origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Quick sanity check that the deploy is actually live and reachable."""
    return {"service": "touchline-api", "status": "ok", "cors_origins": _frontend_origins}


@app.get("/health")
async def health(db: int = 0):
    """Used by the host's health check (Fly.io/Render/Railway all probe this).
    Also useful to hit manually first if you suspect a Render cold start —
    the very first request after idle can take 30-60s, and THAT delay is
    what usually shows up in the browser as a CORS/network failure on the
    real request that follows it.

    GET /health?db=1 additionally round-trips a `SELECT 1` to Turso and
    reports a specific diagnosis instead of a stack trace — hit this
    directly (curl or browser) when auth/quick-match/etc. are 500ing, to
    tell a bad/expired TURSO_AUTH_TOKEN apart from a paused/deleted
    database apart from an actual outage, without digging through logs.
    """
    if not db:
        return {"status": "ok"}

    import time as _time
    started = _time.monotonic()
    try:
        client = get_client()
        await client.execute("SELECT 1")
        return {"status": "ok", "db": "reachable", "latency_ms": round((_time.monotonic() - started) * 1000)}
    except Exception as e:  # noqa: BLE001 - this endpoint's whole job is to explain any failure
        name = e.__class__.__name__
        message = str(e)
        diagnosis = "Unrecognized error talking to Turso — see 'detail' below."
        # NOTE: as of the migration off the archived `libsql_client` (see
        # db/turso_client.py), the client talks HTTP via the `libsql`
        # package instead of raw websockets, so a "WSServerHandshakeError"
        # can no longer happen here — that failure mode is retired along
        # with the old driver, not just less likely.
        if "401" in message or "unauthoriz" in message.lower() or "auth" in message.lower():
            diagnosis = ("Turso rejected the request as unauthorized — regenerate the token with "
                         "`turso db tokens create <db-name>` and update TURSO_AUTH_TOKEN.")
        elif "404" in message or "not found" in message.lower():
            diagnosis = ("Turso couldn't find the database — confirm it exists and isn't deleted with "
                         "`turso db show <db-name>`, and check TURSO_DATABASE_URL's hostname.")
        elif "TURSO_DATABASE_URL" in message or "TURSO_AUTH_TOKEN" in message:
            diagnosis = "TURSO_DATABASE_URL / TURSO_AUTH_TOKEN isn't set in this environment's variables."
        elif "timeout" in message.lower() or "connect" in message.lower():
            diagnosis = ("Couldn't reach Turso at all (timeout/connection error) — check network egress "
                         "from this host and that the database isn't paused.")
        return JSONResponse(status_code=503, content={
            "status": "error", "db": "unreachable",
            "diagnosis": diagnosis, "detail": f"{name}: {message}",
        })


def _resolve_cors_origin(origin: str | None) -> str | None:
    """Mirrors CORSMiddleware's own origin-matching logic (exact list,
    '*', or the regex) so we can attach the right Access-Control-Allow-
    Origin header by hand — see why that's necessary in the docstring
    below."""
    if not origin:
        return None
    if _frontend_origins == ["*"]:
        return "*"
    if origin in _frontend_origins:
        return origin
    if _frontend_origin_regex and _re.fullmatch(_frontend_origin_regex, origin):
        return origin
    return None


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """
    Belt-and-suspenders: CORSMiddleware normally attaches headers even to
    error responses — but Starlette special-cases a handler registered for
    the *bare* `Exception` class: instead of running it inside
    ExceptionMiddleware (which sits INSIDE/below CORSMiddleware), it's
    pulled out and used directly as ServerErrorMiddleware's handler, which
    sits OUTSIDE/above CORSMiddleware in the stack. That means whatever
    response this function returns skips CORSMiddleware entirely and never
    gets an Access-Control-Allow-Origin header attached — so a database
    outage, a missing env var, or any other truly unhandled exception
    (e.g. a Turso connectivity/auth failure, or TURSO_DATABASE_URL missing
    so get_client() throws a bare KeyError) surfaces to the browser as a
    bare CORS failure with no readable error at all, even though CORS was
    never actually the problem. Fix: compute and attach the header here,
    by hand, exactly as CORSMiddleware itself would have.
    """
    headers = {}
    allowed_origin = _resolve_cors_origin(request.headers.get("origin"))
    if allowed_origin:
        headers["Access-Control-Allow-Origin"] = allowed_origin
        headers["Vary"] = "Origin"

    message = str(exc) or exc.__class__.__name__
    # The DB client (db/turso_client.py, using the `libsql` package over
    # HTTP) can still fail on bad/expired TURSO_AUTH_TOKEN, a paused/
    # deleted database, or plain network unreachability — surface
    # something actionable instead of a raw exception repr. (There's no
    # websocket handshake or legacy v1/execute KeyError path anymore;
    # those were specific to the retired libsql_client driver.)
    if isinstance(exc, KeyError) and exc.args[:1] in (("TURSO_DATABASE_URL",),):
        message = "TURSO_DATABASE_URL isn't set in this environment's variables. See GET /health?db=1."
    elif "401" in message or "unauthoriz" in message.lower() or "connect" in message.lower() or "timeout" in message.lower():
        message = ("Could not reach the Turso database, or the credentials were rejected. "
                    "Check that TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set correctly "
                    "and that the database isn't paused or deleted. See GET /health?db=1.")

    return JSONResponse(status_code=500, content={"error": message}, headers=headers)


# ---------------------------------------------------------------------------
# Auth — real accounts backed by Turso (db/auth.py), replacing the old
# frontend-only `ACCOUNTS` in-memory array that reset on every page reload.
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    email: str
    password: str
    manager_name: str
    club_name: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/signup")
async def signup(req: SignupRequest) -> dict:
    if len(req.password) < 4:
        return JSONResponse(status_code=400, content={"error": "Password must be at least 4 characters."})
    client = get_client()
    await ensure_auth_schema(client)
    try:
        result = await create_user(client, req.email, req.password, req.manager_name, req.club_name)
    except ValueError as e:
        return JSONResponse(status_code=409, content={"error": str(e)})
    return result


@app.post("/auth/login")
async def login(req: LoginRequest) -> dict:
    client = get_client()
    await ensure_auth_schema(client)
    try:
        result = await authenticate_user(client, req.email, req.password)
    except ValueError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})
    return result


# ---------------------------------------------------------------------------
# Quick-match provisioning.
#
# index.html's squads live only in browser memory (synthetic clubs generated
# client-side) — they don't correspond to rows in `teams`/`players` yet. This
# endpoint is the bridge: given the two lightweight squads the frontend
# already has, it creates real DB rows for them (deriving the sub-attributes
# the frontend doesn't track — finishing/vision/positioning/tackling/
# gk_reflexes/pace — from position + overall) and returns a `match_id` the
# frontend can immediately open /ws/match/{match_id} against.
# ---------------------------------------------------------------------------

class SimplePlayer(BaseModel):
    id: str
    name: str
    pos: str
    overall: int
    consistency: int = 60


class QuickMatchRequest(BaseModel):
    home_name: str
    away_name: str
    home_squad: list[SimplePlayer]        # starting XI, 11 players
    home_bench: list[SimplePlayer] = []   # up to 7, subs are pulled from here
    away_squad: list[SimplePlayer]
    away_bench: list[SimplePlayer] = []
    # A logged-in manager's persistent `teams.id` (returned at signup/login
    # — see db/auth.py). When set, the matching side reuses that team's
    # existing player rows (upserted by client_ref_id) instead of
    # provisioning a brand-new team + players from scratch. Omitted for
    # anonymous/guest play, which keeps the old ephemeral behavior.
    home_team_id: str | None = None
    away_team_id: str | None = None


def _clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


def _derive_subattrs(pos: str, overall: int, stable_key: str) -> dict:
    """
    The frontend's player objects only carry {pos, overall, consistency, ...}
    — no finishing/vision/positioning/tackling/gk_reflexes/pace, since those
    only exist on the backend's PlayerSnapshot. Derive plausible values from
    position + overall, jittered by a stable hash of the player's own id so
    the same frontend player always maps to the same backend stat block
    (re-provisioning a rematch doesn't reshuffle anyone's attributes).
    """
    h = int(hashlib.sha256(stable_key.encode()).hexdigest(), 16)
    def jitter(n: int) -> int:
        return ((h >> (n * 4)) % 11) - 5  # stable pseudo-random -5..+5

    return {
        "finishing": _clamp(overall + (10 if pos in ("ST", "LW", "RW", "AM") else -5) + jitter(1), 1, 99),
        "vision": _clamp(overall + (10 if pos in ("AM", "CM", "DM") else -3) + jitter(2), 1, 99),
        "positioning": _clamp(overall + (8 if pos in ("CB", "DM", "LB", "RB") else -2) + jitter(3), 1, 99),
        "tackling": _clamp(overall + (12 if pos in ("CB", "DM", "LB", "RB") else -10) + jitter(4), 1, 99),
        "gk_reflexes": _clamp(overall + (15 if pos == "GK" else -30) + jitter(5), 1, 99),
        "pace": _clamp(overall + jitter(6), 1, 99),
    }


async def _provision_squad(client, squad: list[SimplePlayer], team_id: str, role: str) -> dict[str, str]:
    """Inserts each player + a player_rights row tagged with `role`
    ('starter'/'bench'). Returns {frontend_player_id: backend_player_id}.
    Used for ephemeral guest/demo teams — every kickoff gets brand-new
    rows. For a logged-in manager's persistent team, see `_upsert_squad`
    below instead."""
    id_map: dict[str, str] = {}
    for p in squad:
        attrs = _derive_subattrs(p.pos, p.overall, p.id)
        player_id = str(uuid.uuid4())
        await client.execute(
            """
            INSERT INTO players (id, name, age, position, overall, potential, consistency,
                                  pace, finishing, vision, positioning, tackling, gk_reflexes, stamina)
            VALUES (?, ?, 24, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 80)
            """,
            [player_id, p.name, p.pos, p.overall, p.overall, p.consistency,
             attrs["pace"], attrs["finishing"], attrs["vision"],
             attrs["positioning"], attrs["tackling"], attrs["gk_reflexes"]],
        )
        await client.execute(
            "INSERT INTO player_rights (player_id, owner_team_id, squad_role) VALUES (?, ?, ?)",
            [player_id, team_id, role],
        )
        id_map[p.id] = player_id
    return id_map


async def _upsert_squad(client, squad: list[SimplePlayer], team_id: str, role: str) -> dict[str, str]:
    """Persistent-team counterpart to `_provision_squad`: for a logged-in
    manager's real `team_id`, UPDATE the player row already linked to this
    frontend player id (via player_rights.client_ref_id) instead of
    INSERTing a fresh uuid'd row every kickoff — so stats/form/fatigue
    accumulated in previous matches stick around, and a rematch doesn't
    quietly duplicate the whole squad in the database. First time a given
    client_ref_id is seen for this team, it's created; every time after,
    it's reused."""
    id_map: dict[str, str] = {}
    for p in squad:
        attrs = _derive_subattrs(p.pos, p.overall, p.id)
        existing = await client.execute(
            "SELECT player_id FROM player_rights WHERE owner_team_id = ? AND client_ref_id = ?",
            [team_id, p.id],
        )
        if existing.rows:
            player_id = existing.rows[0]["player_id"]
            await client.execute(
                """
                UPDATE players SET name = ?, position = ?, overall = ?, potential = ?,
                       consistency = ?, pace = ?, finishing = ?, vision = ?,
                       positioning = ?, tackling = ?, gk_reflexes = ?
                WHERE id = ?
                """,
                [p.name, p.pos, p.overall, p.overall, p.consistency,
                 attrs["pace"], attrs["finishing"], attrs["vision"],
                 attrs["positioning"], attrs["tackling"], attrs["gk_reflexes"], player_id],
            )
            await client.execute(
                "UPDATE player_rights SET squad_role = ?, updated_at = datetime('now') WHERE player_id = ?",
                [role, player_id],
            )
        else:
            player_id = str(uuid.uuid4())
            await client.execute(
                """
                INSERT INTO players (id, name, age, position, overall, potential, consistency,
                                      pace, finishing, vision, positioning, tackling, gk_reflexes, stamina)
                VALUES (?, ?, 24, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 80)
                """,
                [player_id, p.name, p.pos, p.overall, p.overall, p.consistency,
                 attrs["pace"], attrs["finishing"], attrs["vision"],
                 attrs["positioning"], attrs["tackling"], attrs["gk_reflexes"]],
            )
            await client.execute(
                "INSERT INTO player_rights (player_id, owner_team_id, squad_role, client_ref_id) "
                "VALUES (?, ?, ?, ?)",
                [player_id, team_id, role, p.id],
            )
        id_map[p.id] = player_id
    return id_map


async def _resolve_persistent_team(client, team_id: str | None) -> str | None:
    """Returns `team_id` back if it's a real, existing team row, else None
    (stale/bad id from a client — the caller falls back to ephemeral
    provisioning rather than hard-failing the match)."""
    if not team_id:
        return None
    rs = await client.execute("SELECT id FROM teams WHERE id = ?", [team_id])
    return team_id if rs.rows else None


async def _provision_side(
    client, name: str, squad: list[SimplePlayer], bench: list[SimplePlayer],
    persistent_team_id: str | None, guest_user_id: str,
) -> tuple[str, dict[str, str]]:
    """One side (home or away) of a kickoff: either reuse+upsert a logged-in
    manager's persistent team, or provision a fresh ephemeral one. Returns
    (team_id, {frontend_player_id: backend_player_id})."""
    resolved_team_id = await _resolve_persistent_team(client, persistent_team_id)
    if resolved_team_id is not None:
        # Matchday squad selection can differ from last time (different XI/
        # bench) — reset every one of this team's tracked players to the
        # neutral 'squad' role first, so a player left out of *this*
        # kickoff doesn't linger as a stale 'starter'/'bench' from a
        # previous match when `_load_squad` reads roles back out.
        await client.execute(
            "UPDATE player_rights SET squad_role = 'squad' WHERE owner_team_id = ? AND client_ref_id IS NOT NULL",
            [resolved_team_id],
        )
        id_map = await _upsert_squad(client, squad, resolved_team_id, "starter")
        id_map.update(await _upsert_squad(client, bench, resolved_team_id, "bench"))
        return resolved_team_id, id_map

    team_id = str(uuid.uuid4())
    await client.execute(
        "INSERT INTO teams (id, manager_user_id, name) VALUES (?, ?, ?)",
        [team_id, guest_user_id, name],
    )
    id_map = await _provision_squad(client, squad, team_id, "starter")
    id_map.update(await _provision_squad(client, bench, team_id, "bench"))
    return team_id, id_map


async def _provision_and_create_match(
    client,
    home_name: str, home_squad: list[SimplePlayer], home_bench: list[SimplePlayer],
    away_name: str, away_squad: list[SimplePlayer], away_bench: list[SimplePlayer],
    home_persistent_team_id: str | None = None, away_persistent_team_id: str | None = None,
) -> dict:
    """Shared by the solo `/quick-match` endpoint (vs. a bot/your own two
    sides) and the two-real-player friend-match flow below: resolves both
    sides' teams (reusing a logged-in manager's persistent squad when a
    `*_team_id` is supplied, otherwise provisioning a fresh ephemeral team
    exactly as before), and inserts the `matches` row. Returns the same
    shape either caller can hand straight back to the frontend to open
    `/ws/match/{match_id}` against."""
    await _ensure_schema(client)
    guest_user_id = "guest-demo-user"
    if home_persistent_team_id is None or away_persistent_team_id is None:
        await client.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO NOTHING",
            [guest_user_id, "guest", "guest@touchline.local", "guest-account-no-password"],
        )

    home_team_id, home_map = await _provision_side(
        client, home_name, home_squad, home_bench, home_persistent_team_id, guest_user_id)
    away_team_id, away_map = await _provision_side(
        client, away_name, away_squad, away_bench, away_persistent_team_id, guest_user_id)

    match_id = str(uuid.uuid4())
    seed = random.randint(1, 2**31 - 1)
    await client.execute(
        "INSERT INTO matches (id, home_team_id, away_team_id, scheduled_at, rng_seed) "
        "VALUES (?, ?, ?, datetime('now'), ?)",
        [match_id, home_team_id, away_team_id, seed],
    )

    return {
        "match_id": match_id, "home_team_id": home_team_id, "away_team_id": away_team_id,
        "home_player_map": home_map, "away_player_map": away_map,
    }


@app.post("/quick-match")
async def create_quick_match(req: QuickMatchRequest) -> dict:
    client = get_client()
    return await _provision_and_create_match(
        client, req.home_name, req.home_squad, req.home_bench,
        req.away_name, req.away_squad, req.away_bench,
        req.home_team_id, req.away_team_id,
    )

# ---------------------------------------------------------------------------
# Friend-match lobby: real two-player matchmaking by shareable code.
#
# Replaces the old client-only "Friends Room" simulation (a code that only
# ever resolved inside one browser's in-memory `state.rooms` array) with an
# actual server-side handshake:
#   1. Host calls POST /friend-match/create with their squad -> gets a code.
#   2. Guest calls POST /friend-match/join with the code + their squad.
#   3. Both open /ws/lobby/{pending_id} and send {"action":"ready"}.
#   4. Only once BOTH sides have marked ready does the server provision the
#      squads, create the `matches` row, and broadcast a `kickoff` message
#      with the match_id + a per-side secret token — kickoff never happens
#      on `/quick-match` returning alone, unlike the old solo flow.
#
# Cross-worker note: the canonical lobby record (squads, tokens, ready
# flags, status) is stored via api/state_backend's kv_get/kv_set — Redis-
# backed when REDIS_URL is set, otherwise an in-process dict, so a single
# uvicorn worker needs nothing extra. `_LOBBY_LOCAL_WS` below is always
# purely local (a live WebSocket object can't be moved between processes
# anyway) — cross-worker delivery to a socket held by a *different* worker
# goes through `state_backend.publish`/`subscribe` on a per-lobby Redis
# channel instead (see `_send_lobby_event` / `_relay_lobby_events`), which
# is a no-op in single-worker/no-Redis mode since local delivery already
# covers that case.
# ---------------------------------------------------------------------------

_CODE_ALPHABET = "".join(c for c in string.ascii_uppercase + string.digits if c not in "0O1I")
_LOBBY_TTL_SECONDS = 6 * 3600


def _generate_lobby_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))


# Local mirror (canonical store when Redis is disabled) and the local-only
# websocket registry + per-lobby kickoff lock (guards against the same
# worker processing two near-simultaneous "ready" messages before the
# first has finished provisioning — a real race even single-process, since
# asyncio can interleave between the two awaiting coroutines).
_LOBBIES: dict[str, dict] = {}
_LOBBIES_BY_CODE: dict[str, str] = {}
_LOBBY_LOCAL_WS: dict[str, dict[str, WebSocket]] = {}
_LOBBY_KICKOFF_LOCKS: dict[str, asyncio.Lock] = {}

# match_id -> {"home": token, "away": token}, populated at kickoff. Only
# friend-matches born from this lobby flow get an entry here — the solo
# `/quick-match` endpoint (playing vs. a bot, or both sides on one browser)
# never does, so its tactics socket stays open/unauthenticated as before.
# Mirrors the same local-dict-with-Redis-mirror pattern as the lobby store
# (see `_get_side_tokens` / `_set_side_tokens` further down, next to
# `_PENDING_SUBS`).
_MATCH_SIDE_TOKENS: dict[str, dict[str, str]] = {}


class FriendMatchCreateRequest(BaseModel):
    manager_name: str
    home_name: str
    home_squad: list[SimplePlayer]
    home_bench: list[SimplePlayer] = []
    # Logged-in manager's persistent team_id (see QuickMatchRequest) — same
    # reuse-instead-of-reprovision behavior as /quick-match.
    home_team_id: str | None = None


class FriendMatchJoinRequest(BaseModel):
    code: str
    manager_name: str
    away_name: str
    away_squad: list[SimplePlayer]
    away_bench: list[SimplePlayer] = []
    away_team_id: str | None = None


async def _lobby_save(lobby: dict) -> None:
    _LOBBIES[lobby["pending_id"]] = lobby
    if state_backend.redis_enabled():
        await state_backend.kv_set(f"lobby:{lobby['pending_id']}", lobby, ttl_seconds=_LOBBY_TTL_SECONDS)


async def _lobby_load(pending_id: str) -> dict | None:
    if state_backend.redis_enabled():
        remote = await state_backend.kv_get(f"lobby:{pending_id}")
        if remote is not None:
            _LOBBIES[pending_id] = remote
            return remote
    return _LOBBIES.get(pending_id)


async def _lobby_code_register(code: str, pending_id: str) -> None:
    _LOBBIES_BY_CODE[code] = pending_id
    await state_backend.kv_set(f"lobby_code:{code}", pending_id, ttl_seconds=_LOBBY_TTL_SECONDS)


async def _lobby_code_lookup(code: str) -> str | None:
    if state_backend.redis_enabled():
        remote = await state_backend.kv_get(f"lobby_code:{code}")
        if remote:
            return remote
    return _LOBBIES_BY_CODE.get(code)


async def _send_lobby_event(pending_id: str, payload: dict, target_role: str | None = None) -> None:
    """Delivers a lobby event either to one specific role ("host"/"guest",
    e.g. the per-side kickoff message, which carries a different secret
    token for each) or to both (target_role=None, e.g. a state refresh).
    Cross-worker via Redis pub/sub when enabled; otherwise sent directly to
    whichever local sockets this worker is holding."""
    if state_backend.redis_enabled():
        await state_backend.publish(f"lobby:{pending_id}:events",
                                     {**payload, "target_role": target_role})
        return
    local = _LOBBY_LOCAL_WS.get(pending_id, {})
    targets = [local[target_role]] if target_role else list(local.values())
    for ws in targets:
        with contextlib.suppress(Exception):
            await ws.send_json(payload)


async def _relay_lobby_events(pending_id: str, websocket: WebSocket, role: str) -> None:
    """Background task per lobby-socket connection: forwards Redis-pub/sub
    lobby events addressed to this role (or to everyone) onto this specific
    websocket. No-ops for its whole lifetime when Redis isn't configured,
    since `_send_lobby_event` already delivered locally in that mode."""
    if not state_backend.redis_enabled():
        return
    async with state_backend.subscribe(f"lobby:{pending_id}:events") as events:
        async for payload in events:
            target_role = payload.get("target_role")
            if target_role is not None and target_role != role:
                continue
            with contextlib.suppress(Exception):
                await websocket.send_json({k: v for k, v in payload.items() if k != "target_role"})


@app.post("/friend-match/create")
async def friend_match_create(req: FriendMatchCreateRequest) -> dict:
    if len(req.home_squad) != 11:
        return JSONResponse(status_code=400, content={"error": "home_squad must have exactly 11 players."})

    pending_id = str(uuid.uuid4())
    code = _generate_lobby_code()
    while await _lobby_code_lookup(code) is not None:  # astronomically unlikely, but be sure
        code = _generate_lobby_code()

    host_token = secrets.token_urlsafe(18)
    lobby = {
        "code": code, "pending_id": pending_id, "status": "waiting_guest",
        "match_id": None, "created_at": time.time(),
        "host": {
            "token": host_token, "manager_name": req.manager_name, "team_name": req.home_name,
            "squad": [p.model_dump() for p in req.home_squad], "bench": [p.model_dump() for p in req.home_bench],
            "ready": False, "team_id": req.home_team_id,
        },
        "guest": None,
    }
    await _lobby_save(lobby)
    await _lobby_code_register(code, pending_id)

    return {"pending_id": pending_id, "code": code, "host_token": host_token}


@app.post("/friend-match/join")
async def friend_match_join(req: FriendMatchJoinRequest) -> dict:
    pending_id = await _lobby_code_lookup(req.code.strip().upper())
    lobby = await _lobby_load(pending_id) if pending_id else None
    if lobby is None:
        return JSONResponse(status_code=404, content={"error": "No room found for that code."})
    if lobby["status"] != "waiting_guest" or lobby["guest"] is not None:
        return JSONResponse(status_code=409, content={"error": "That room already has two players."})
    if len(req.away_squad) != 11:
        return JSONResponse(status_code=400, content={"error": "away_squad must have exactly 11 players."})

    guest_token = secrets.token_urlsafe(18)
    lobby["guest"] = {
        "token": guest_token, "manager_name": req.manager_name, "team_name": req.away_name,
        "squad": [p.model_dump() for p in req.away_squad], "bench": [p.model_dump() for p in req.away_bench],
        "ready": False, "team_id": req.away_team_id,
    }
    lobby["status"] = "waiting_ready"
    await _lobby_save(lobby)
    await _send_lobby_event(pending_id, _lobby_state_payload(lobby))

    return {
        "pending_id": lobby["pending_id"], "guest_token": guest_token,
        "host_manager_name": lobby["host"]["manager_name"], "home_name": lobby["host"]["team_name"],
    }


def _lobby_state_payload(lobby: dict) -> dict:
    host, guest = lobby["host"], lobby["guest"]
    return {
        "type": "state",
        "code": lobby["code"],
        "status": lobby["status"],
        "host_manager_name": host["manager_name"],
        "host_team_name": host["team_name"],
        "host_ready": host["ready"],
        "guest_joined": guest is not None,
        "guest_manager_name": guest["manager_name"] if guest else None,
        "guest_team_name": guest["team_name"] if guest else None,
        "guest_ready": guest["ready"] if guest else False,
    }


async def _broadcast_lobby_state(lobby: dict) -> None:
    await _send_lobby_event(lobby["pending_id"], _lobby_state_payload(lobby))


async def _try_kickoff(lobby: dict) -> bool:
    """Both-players-ready gate: the match is only provisioned and started
    once BOTH host.ready and guest.ready are true. Returns True if kickoff
    happened. Guarded by a local asyncio.Lock (same-process race) plus a
    Redis distributed lock (cross-worker race) so two near-simultaneous
    'ready' messages can never provision the match twice."""
    pending_id = lobby["pending_id"]
    lock = _LOBBY_KICKOFF_LOCKS.setdefault(pending_id, asyncio.Lock())
    async with lock:
        lobby = await _lobby_load(pending_id) or lobby
        guest = lobby.get("guest")
        if guest is None or not lobby["host"]["ready"] or not guest["ready"]:
            return False
        if lobby["status"] == "started":
            return False

        owner = await state_backend.try_acquire_owner(f"lobby:{pending_id}:kickoff", ttl_seconds=30)
        if owner is None:  # another worker already won the race
            return False

        client = get_client()
        host, guest = lobby["host"], lobby["guest"]
        result = await _provision_and_create_match(
            client, host["team_name"], [SimplePlayer(**p) for p in host["squad"]],
            [SimplePlayer(**p) for p in host["bench"]],
            guest["team_name"], [SimplePlayer(**p) for p in guest["squad"]],
            [SimplePlayer(**p) for p in guest["bench"]],
            host.get("team_id"), guest.get("team_id"),
        )
        lobby["status"] = "started"
        lobby["match_id"] = result["match_id"]
        await _lobby_save(lobby)
        await _set_side_tokens(result["match_id"], {"home": host["token"], "away": guest["token"]})

        for side_name, side_token in (("home", host["token"]), ("away", guest["token"])):
            await _send_lobby_event(pending_id, {
                "type": "kickoff", "match_id": result["match_id"],
                "home_team_id": result["home_team_id"], "away_team_id": result["away_team_id"],
                "home_player_map": result["home_player_map"], "away_player_map": result["away_player_map"],
                "your_side": side_name, "your_token": side_token,
            }, target_role="host" if side_name == "home" else "guest")
        return True


@app.websocket("/ws/lobby/{pending_id}")
async def lobby_socket(websocket: WebSocket, pending_id: str, role: str | None = None, token: str | None = None):
    """Real-time companion to /friend-match/create + /friend-match/join.
    `role` is "host" or "guest"; `token` must match the token that endpoint
    returned. Clients send {"action": "ready"} once their manager is happy
    to kick off; the match only actually starts once both sides have."""
    await websocket.accept()
    lobby = await _lobby_load(pending_id)
    if lobby is None or role not in ("host", "guest"):
        await websocket.send_json({"type": "error", "error": "Room not found."})
        await websocket.close(code=4404)
        return

    side_obj = lobby.get(role)
    if side_obj is None or token != side_obj.get("token"):
        await websocket.send_json({"type": "error", "error": "Invalid room token."})
        await websocket.close(code=4403)
        return

    _LOBBY_LOCAL_WS.setdefault(pending_id, {})[role] = websocket
    await websocket.send_json(_lobby_state_payload(lobby))

    relay_task = asyncio.create_task(_relay_lobby_events(pending_id, websocket, role))
    try:
        while True:
            msg = await websocket.receive_json()
            lobby = await _lobby_load(pending_id) or lobby
            side_obj = lobby.get(role)
            if side_obj is None:
                continue
            if msg.get("action") == "ready":
                side_obj["ready"] = True
                await _lobby_save(lobby)
                await _broadcast_lobby_state(lobby)
                await _try_kickoff(lobby)
            elif msg.get("action") == "unready":
                side_obj["ready"] = False
                await _lobby_save(lobby)
                await _broadcast_lobby_state(lobby)
    except WebSocketDisconnect:
        local = _LOBBY_LOCAL_WS.get(pending_id)
        if local is not None and local.get(role) is websocket:
            local.pop(role, None)
        lobby = await _lobby_load(pending_id) or lobby
        if lobby.get("status") != "started":
            side_obj = lobby.get(role)
            if side_obj is not None:
                side_obj["ready"] = False
                await _lobby_save(lobby)
                await _broadcast_lobby_state(lobby)
    finally:
        relay_task.cancel()
        with contextlib.suppress(Exception):
            await relay_task


SECONDS_PER_GAME_MINUTE = 1.33
HALF_TIME_BREAK_SECONDS = 20.0  # real seconds paused server-side at half-time
DISCONNECT_GRACE_SECONDS = 45.0  # real seconds a friend-match side gets to reconnect before forfeit
SIM_LOCK_TTL_SECONDS = 900  # generous vs. the ~2-minute compressed match, so it never expires mid-sim

# One live-substitution queue per in-flight match, shared between the two
# websocket handlers below (`stream_match` reads it, `receive_tactics`
# writes to it). Local-dict-with-Redis-mirror, same pattern as the lobby
# store above: `_pending_subs_push`/`_pending_subs_drain` route to Redis
# lists when REDIS_URL is set (so the tactics socket and the simulation
# loop can be on different workers), otherwise operate on this dict
# directly, exactly as the old single-worker version did.
_PENDING_SUBS: dict[str, list[dict]] = {}

# match_id -> {"home": token, "away": token} local mirror (see _MATCH_SIDE_TOKENS above).
async def _set_side_tokens(match_id: str, tokens: dict[str, str]) -> None:
    _MATCH_SIDE_TOKENS[match_id] = tokens
    await state_backend.kv_set(f"match:{match_id}:tokens", tokens, ttl_seconds=SIM_LOCK_TTL_SECONDS)


async def _get_side_tokens(match_id: str) -> dict[str, str] | None:
    if match_id in _MATCH_SIDE_TOKENS:
        return _MATCH_SIDE_TOKENS[match_id]
    remote = await state_backend.kv_get(f"match:{match_id}:tokens")
    if remote is not None:
        _MATCH_SIDE_TOKENS[match_id] = remote
    return remote


async def _clear_side_tokens(match_id: str) -> None:
    _MATCH_SIDE_TOKENS.pop(match_id, None)
    await state_backend.kv_delete(f"match:{match_id}:tokens")


async def _pending_subs_push(match_id: str, msg: dict) -> None:
    if state_backend.redis_enabled():
        await state_backend.queue_push(f"match:{match_id}:subs", msg)
    else:
        _PENDING_SUBS.setdefault(match_id, []).append(msg)


async def _pending_subs_drain(match_id: str) -> list[dict]:
    if state_backend.redis_enabled():
        return await state_backend.queue_drain(f"match:{match_id}:subs")
    items = _PENDING_SUBS.get(match_id, [])
    _PENDING_SUBS[match_id] = []
    return items


async def _mark_match_active(match_id: str) -> None:
    _ACTIVE_MATCHES.add(match_id)
    await state_backend.kv_set(f"match:{match_id}:active", True, ttl_seconds=SIM_LOCK_TTL_SECONDS)


async def _mark_match_inactive(match_id: str) -> None:
    _ACTIVE_MATCHES.discard(match_id)
    await state_backend.kv_delete(f"match:{match_id}:active")


async def _is_match_active(match_id: str) -> bool:
    if match_id in _ACTIVE_MATCHES:
        return True
    return await state_backend.kv_get(f"match:{match_id}:active") is not None


_ACTIVE_MATCHES: set[str] = set()


async def _side_presence_delta(match_id: str, side: str, delta: int) -> int:
    """Adjusts (and returns the new value of) the count of currently-open
    `/ws/match/{id}` connections authenticated as `side`. Redis INCRBY when
    available (so presence is correct even if the two players' sockets
    land on different workers); otherwise a local counter, which is exactly
    right in single-worker mode since that's the only worker there is."""
    key = f"match:{match_id}:conn:{side}"
    new_val = await state_backend.counter_incrby(key, delta, ttl_seconds=SIM_LOCK_TTL_SECONDS)
    if new_val is not None:
        return max(0, new_val)
    counts = _SIDE_CONN_COUNTS.setdefault(match_id, {"home": 0, "away": 0})
    counts[side] = max(0, counts.get(side, 0) + delta)
    return counts[side]


async def _side_presence_count(match_id: str, side: str) -> int:
    key = f"match:{match_id}:conn:{side}"
    val = await state_backend.counter_get(key)
    if val is not None:
        return max(0, val)
    return max(0, _SIDE_CONN_COUNTS.get(match_id, {}).get(side, 0))


_SIDE_CONN_COUNTS: dict[str, dict[str, int]] = {}


def _row_to_player(row) -> PlayerSnapshot:
    return PlayerSnapshot(
        id=row["id"], name=row["name"], position=row["position"],
        overall=row["overall"], consistency=row["consistency"],
        finishing=row["finishing"], vision=row["vision"],
        positioning=row["positioning"], tackling=row["tackling"],
        gk_reflexes=row["gk_reflexes"], pace=row["pace"],
        current_form=row["current_form"], fatigue=row["fatigue"],
    )


async def _load_squad(client, team_id: str) -> tuple[TeamSnapshot, list[PlayerSnapshot]]:
    """Returns (starting XI as a TeamSnapshot, bench players available to sub on)."""
    rs = await client.execute(
        """
        SELECT p.id, p.name, p.position, p.overall, p.consistency, p.finishing,
               p.vision, p.positioning, p.tackling, p.gk_reflexes, p.pace,
               p.current_form, p.fatigue, t.name AS team_name, pr.squad_role
        FROM players p
        JOIN player_rights pr ON pr.player_id = p.id
        JOIN teams t ON t.id = pr.owner_team_id
        WHERE pr.owner_team_id = ?
        ORDER BY
            CASE pr.squad_role WHEN 'starter' THEN 0 WHEN 'bench' THEN 1 ELSE 2 END,
            p.overall DESC
        """,
        [team_id],
    )
    rows = rs.rows
    has_explicit_roles = any(r["squad_role"] in ("starter", "bench") for r in rows)
    if has_explicit_roles:
        starters = [r for r in rows if r["squad_role"] == "starter"]
        bench = [r for r in rows if r["squad_role"] == "bench"]
    else:
        # No quick-match roles set (e.g. a regular league fixture drawing
        # from a full transfer-market roster) -> fall back to picking the
        # best 11 by overall, same as before.
        starters, bench = rows[:11], rows[11:18]
    lineup = [_row_to_player(row) for row in starters]
    bench_players = [_row_to_player(row) for row in bench]
    team_name = rows[0]["team_name"] if rows else team_id
    return TeamSnapshot(id=team_id, name=team_name, lineup=lineup), bench_players


async def _persist_match_start(client, match_id: str, seed: int) -> None:
    await client.execute(
        "UPDATE matches SET status = 'live', rng_seed = ? WHERE id = ?",
        [seed, match_id],
    )


async def _persist_event(client, match_id: str, event: dict) -> None:
    await client.execute(
        """
        INSERT INTO match_events (match_id, minute, event_type, description, team_id, player_id, momentum_home, momentum_away)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            match_id, event["minute"], event["type"], event["description"],
            event.get("team_id"), event.get("player_id"),
            event.get("momentum"), -(event.get("momentum") or 0),
        ],
    )


async def _persist_match_end(client, match_id: str, event: dict) -> None:
    await client.execute(
        "UPDATE matches SET status = 'completed', home_score = ?, away_score = ?, "
        "home_xg = ?, away_xg = ? WHERE id = ?",
        [
            event["home_score"], event["away_score"],
            event.get("stats", {}).get("home_xg", 0),
            event.get("stats", {}).get("away_xg", 0),
            match_id,
        ],
    )


async def _persist_match_forfeit(client, match_id: str, forfeited_side: str,
                                  home_score: int, away_score: int) -> None:
    """Ends a match early because one side never reconnected within the
    grace period (see `_watchdog_tick`). Resolves to 'completed' — same
    status a normal full_time gets — so nothing downstream needs a new
    status value; `forfeited_side` is what distinguishes a walkover from a
    match actually played to the final whistle. The score is left as
    whatever it actually was at the moment of forfeit rather than an
    arbitrary walkover scoreline."""
    await client.execute(
        "UPDATE matches SET status = 'completed', home_score = ?, away_score = ?, "
        "forfeited_side = ? WHERE id = ?",
        [home_score, away_score, forfeited_side, match_id],
    )


async def _apply_pending_subs(match_id: str, home: TeamSnapshot, away: TeamSnapshot,
                               home_bench: list[PlayerSnapshot], away_bench: list[PlayerSnapshot]) -> list[dict]:
    """Drains any substitution/mentality messages queued by receive_tactics,
    mutates the live TeamSnapshot objects in place, and returns the
    match_event-shaped dicts to emit/persist for whichever ones landed."""
    pending = await _pending_subs_drain(match_id)

    emitted = []
    for msg in pending:
        action = msg.get("action")

        if action == "substitution":
            side = msg.get("side")  # "home" | "away"
            team, bench = (home, home_bench) if side == "home" else (away, away_bench)
            out_id = msg["payload"]["player_out_id"]
            in_id = msg["payload"]["player_in_id"]
            sub_in = next((p for p in bench if p.id == in_id), None)
            if sub_in is None:
                continue
            sub_out = next((p for p in team.lineup if p.id == out_id), None)
            ok = apply_substitution(team, out_id, sub_in)
            if ok:
                bench.remove(sub_in)
                if sub_out is not None:
                    bench.append(sub_out)
                emitted.append({
                    "minute": msg.get("minute", 0), "type": "substitution",
                    "description": f"Substitution for {team.name} — {sub_in.name} replaces "
                                    f"{sub_out.name if sub_out else out_id}.",
                    "team_id": team.id, "player_id": sub_in.id,
                    "home_score": None, "away_score": None, "momentum": None,
                })

        elif action == "mentality":
            side = msg.get("side")
            team = home if side == "home" else away
            team.mentality = max(-1.0, min(1.0, float(msg["payload"].get("value", 0.0))))

    return emitted


@dataclass
class _MatchRunner:
    """One shared simulation per match_id, fanned out to every connected
    viewer *on this worker*. Two real players both opening `/ws/match/{id}`
    for the SAME friend-match need to watch (and tactically affect) the
    identical running simulation — exactly one worker ever runs the actual
    `simulate_match(...)` loop for a given match_id (see `try_acquire_owner`
    in `stream_match` below); every other worker's viewers relay events in
    over Redis pub/sub instead of starting their own competing loop.

    `log`/`subscribers` only fan out to sockets held by THIS worker — a
    viewer on a different worker replays from the Redis-backed log
    (`state_backend.log_read_all`) and receives live events via the
    `match:{id}:stream` channel instead. In single-worker/no-Redis mode
    this is exactly the previous behavior (there's only one worker, so
    "this worker" is every viewer)."""
    task: asyncio.Task | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    log: list[str] = field(default_factory=list)  # serialized events emitted so far, for same-worker late joiners
    finished: bool = False
    # Disconnect/pause/forfeit watchdog state (only meaningful for
    # friend-matches, i.e. matches with side tokens) — see `_watchdog_tick`.
    paused: bool = False
    disconnect_since: dict[str, float | None] = field(default_factory=lambda: {"home": None, "away": None})
    last_home_score: int = 0
    last_away_score: int = 0


_MATCH_RUNNERS: dict[str, _MatchRunner] = {}


async def _broadcast(runner: _MatchRunner, match_id: str, payload: dict) -> None:
    text = json.dumps(payload)
    runner.log.append(text)
    for q in list(runner.subscribers):
        await q.put(text)
    # Cross-worker fan-out: no-ops when Redis isn't configured, since local
    # subscribers above already got it.
    await state_backend.log_append(f"match:{match_id}:log", text)
    await state_backend.publish(f"match:{match_id}:stream", payload)


async def _relay_remote_events(match_id: str, q: asyncio.Queue) -> None:
    """Runs on a worker that does NOT own this match's simulation: forwards
    events published by the owning worker into this viewer's local queue,
    exactly as if `_broadcast` had pushed to it directly. No-ops (never
    yields anything) when Redis isn't configured — irrelevant in that mode
    since there's only ever one worker, which is always the owner."""
    async with state_backend.subscribe(f"match:{match_id}:stream") as events:
        async for payload in events:
            if payload.get("type") == "_end_":
                await q.put(None)
                return
            await q.put(json.dumps(payload))


async def _watchdog_tick(match_id: str, runner: _MatchRunner, client, last_minute: int) -> bool:
    """Called once per simulation tick (and once per second during the
    half-time break) for matches that have side tokens (i.e. real
    friend-matches — solo/bot matches have no `_get_side_tokens` entry and
    never call this at all). Tracks each side's live `/ws/match/{id}`
    connection count:
      * a side dropping to 0 connections starts a grace-period timer and
        broadcasts `match_paused` (once);
      * reconnecting before the grace period elapses broadcasts
        `match_resumed`;
      * failing to reconnect in time ends the match as a forfeit in favor
        of whichever side is still connected, broadcasts `match_forfeited`,
        and persists the result.
    While paused, this function blocks (polling once a second) so the
    match doesn't keep advancing real time for a side that's still there
    while its opponent is gone. Returns True if the match was just
    forfeited (caller should stop the simulation loop)."""
    tokens = await _get_side_tokens(match_id)
    if tokens is None:
        return False  # not a tracked friend-match — no watchdog behavior

    while True:
        counts = {s: await _side_presence_count(match_id, s) for s in ("home", "away")}
        now = time.time()
        newly_gone = []
        for s in ("home", "away"):
            if counts[s] == 0 and runner.disconnect_since[s] is None:
                runner.disconnect_since[s] = now
                newly_gone.append(s)
            elif counts[s] > 0 and runner.disconnect_since[s] is not None:
                runner.disconnect_since[s] = None

        any_gone = any(runner.disconnect_since[s] is not None for s in ("home", "away"))

        if newly_gone and not runner.paused:
            runner.paused = True
            for s in newly_gone:
                await _broadcast(runner, match_id, {
                    "type": "match_paused", "match_id": match_id, "minute": last_minute,
                    "side": s, "grace_seconds": DISCONNECT_GRACE_SECONDS,
                    "description": f"{'Home' if s == 'home' else 'Away'} manager's connection dropped — "
                                    f"match paused, {int(DISCONNECT_GRACE_SECONDS)}s to reconnect before forfeit.",
                    "home_score": runner.last_home_score, "away_score": runner.last_away_score, "momentum": None,
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                })
        elif runner.paused and not any_gone:
            runner.paused = False
            await _broadcast(runner, match_id, {
                "type": "match_resumed", "match_id": match_id, "minute": last_minute,
                "description": "Both managers connected — match resumes.",
                "home_score": runner.last_home_score, "away_score": runner.last_away_score, "momentum": None,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            })

        if not runner.paused:
            return False

        for s in ("home", "away"):
            since = runner.disconnect_since[s]
            if since is not None and (now - since) >= DISCONNECT_GRACE_SECONDS:
                winner = "away" if s == "home" else "home"
                await _persist_match_forfeit(client, match_id, s, runner.last_home_score, runner.last_away_score)
                await _broadcast(runner, match_id, {
                    "type": "match_forfeited", "match_id": match_id, "minute": last_minute,
                    "forfeited_side": s, "winning_side": winner,
                    "description": f"{'Home' if s == 'home' else 'Away'} manager never reconnected — "
                                    f"match forfeited, {winner} wins.",
                    "home_score": runner.last_home_score, "away_score": runner.last_away_score, "momentum": None,
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                })
                return True

        await asyncio.sleep(1.0)


async def _run_match_simulation(match_id: str, owner_token: str) -> None:
    runner = _MATCH_RUNNERS[match_id]
    client = get_client()
    ml_bridge = MLBridge.load()
    try:
        row_rs = await client.execute(
            "SELECT home_team_id, away_team_id, rng_seed FROM matches WHERE id = ?",
            [match_id],
        )
        if not row_rs.rows:
            await _broadcast(runner, match_id, {"type": "error", "description": "Match not found", "match_id": match_id})
            return

        row = row_rs.rows[0]
        seed = row["rng_seed"] or random.randint(1, 2**31 - 1)

        home, home_bench = await _load_squad(client, row["home_team_id"])
        away, away_bench = await _load_squad(client, row["away_team_id"])

        await _persist_match_start(client, match_id, seed)
        await _mark_match_active(match_id)

        last_minute = 0
        # `home`/`away` are the SAME objects the tactics socket mutates via
        # apply_substitution — simulate_match reads their .lineup fresh
        # every minute, so a sub queued mid-stream takes effect immediately.
        for event in simulate_match(home, away, seed=seed, duration_minutes=90, ml_bridge=ml_bridge):
            if event.get("home_score") is not None:
                runner.last_home_score = event["home_score"]
            if event.get("away_score") is not None:
                runner.last_away_score = event["away_score"]

            # Blocks here for as long as a connected friend-match side is
            # gone, and returns True (stopping the whole simulation) if the
            # grace period expired and the match was forfeited. No-op for
            # solo/bot matches (no side tokens to watch).
            if await _watchdog_tick(match_id, runner, client, last_minute):
                return

            elapsed_minutes = max(0, event["minute"] - last_minute)
            delay = elapsed_minutes * SECONDS_PER_GAME_MINUTE
            if delay > 0:
                # NOTE: no cap on the TOTAL gap here. Commentary/ML-decision
                # events are sparse (only "notable" minutes get one — see
                # engine/match_engine.py), so gaps of 15-30 game-minutes
                # between events are normal. The whole point of
                # SECONDS_PER_GAME_MINUTE is that 90 game-minutes' worth of
                # these delays sums to ~2 real minutes; artificially capping
                # any single gap (this used to cap at 6s) throws that budget
                # away and makes the match finish in under a minute instead.
                #
                # BUT we don't just sleep(delay) in one block any more — that
                # made the on-screen clock sit frozen for the whole gap and
                # then jump straight to the next event's minute the instant
                # the ML/event generator produced something, i.e. the timer
                # only ever moved on an ML decision instead of running
                # continuously. Instead, sleep in ~1-real-second steps and
                # broadcast a lightweight `clock_tick` after each one, with
                # the minute interpolated from real elapsed time. The
                # generator above is completely unaffected — it has already
                # decided everything up to `event`; this loop is purely
                # pacing the *visible* clock smoothly while we wait to reveal
                # it, so ML decisions keep happening at their own pace
                # "behind the scenes" and the clock no longer stalls between
                # them.
                start_minute = last_minute
                remaining = delay
                while remaining > 0:
                    step = min(1.0, remaining)
                    await asyncio.sleep(step)
                    remaining -= step
                    if await _watchdog_tick(match_id, runner, client, last_minute):
                        return
                    elapsed_real = delay - remaining
                    interp_minute = start_minute + elapsed_real / SECONDS_PER_GAME_MINUTE
                    interp_minute = min(interp_minute, event["minute"])
                    await _broadcast(runner, match_id, {
                        "type": "clock_tick", "match_id": match_id,
                        "minute": round(interp_minute, 2),
                        "home_score": runner.last_home_score,
                        "away_score": runner.last_away_score,
                        "momentum": None,
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                    })
            last_minute = event["minute"]

            payload = {**event, "match_id": match_id, "sent_at": datetime.now(timezone.utc).isoformat()}
            await _broadcast(runner, match_id, payload)

            if event["type"] not in ("kickoff", "half_time", "half_time_break"):
                await _persist_event(client, match_id, event)

            if event["type"] == "half_time_break":
                # The ACTUAL real-time pause lives here, not inside the pure
                # generator. Poll every second so a substitution made partway
                # through the break still gets applied before kickoff of the
                # second half, and gets its own event emitted immediately.
                remaining = event.get("break_seconds", HALF_TIME_BREAK_SECONDS)
                while remaining > 0:
                    await asyncio.sleep(1.0)
                    remaining -= 1.0
                    if await _watchdog_tick(match_id, runner, client, last_minute):
                        return
                    for sub_event in await _apply_pending_subs(match_id, home, away, home_bench, away_bench):
                        sub_event["minute"] = last_minute
                        sub_payload = {**sub_event, "match_id": match_id,
                                        "sent_at": datetime.now(timezone.utc).isoformat()}
                        await _broadcast(runner, match_id, sub_payload)
                        await _persist_event(client, match_id, sub_event)

            # Outside the break too: subs can land on any live minute, not
            # just at half-time (e.g. a 60th-minute tactical change).
            for sub_event in await _apply_pending_subs(match_id, home, away, home_bench, away_bench):
                sub_event["minute"] = last_minute
                sub_payload = {**sub_event, "match_id": match_id,
                                "sent_at": datetime.now(timezone.utc).isoformat()}
                await _broadcast(runner, match_id, sub_payload)
                await _persist_event(client, match_id, sub_event)

            if event["type"] == "full_time":
                await _persist_match_end(client, match_id, event)

    finally:
        runner.finished = True
        for q in list(runner.subscribers):
            await q.put(None)  # sentinel: tells each same-worker handler to close its socket
        await state_backend.publish(f"match:{match_id}:stream", {"type": "_end_", "match_id": match_id})
        await state_backend.kv_delete(f"match:{match_id}:subs")
        _PENDING_SUBS.pop(match_id, None)
        await _clear_side_tokens(match_id)
        await _mark_match_inactive(match_id)
        await state_backend.release_owner(f"match:{match_id}:sim_lock", owner_token)


async def _authenticate_match_side(websocket: WebSocket, match_id: str,
                                    side: str | None, token: str | None) -> str | None:
    """Optional side auth for `/ws/match/{id}`, layered on top of the
    existing unauthenticated-spectator behavior: a plain viewer that
    doesn't pass `side`/`token` connects exactly as before (no presence
    tracking, no effect on pause/forfeit). A connection that DOES pass
    `side`/`token` for a match with side tokens (i.e. a real player's own
    socket in a friend-match) is verified against `_get_side_tokens` — a
    wrong/stale token is refused outright — and, once authenticated, that
    connection's presence is what the disconnect/pause/forfeit watchdog in
    `_watchdog_tick` tracks for that side. Returns the authenticated side,
    None (untracked spectator), or the sentinel "INVALID" (caller must
    close and return)."""
    if side is None:
        return None
    tokens = await _get_side_tokens(match_id)
    if tokens is None:
        return None  # solo/bot match — no tokens to check against
    if side not in ("home", "away") or token != tokens.get(side):
        await websocket.send_json({
            "type": "error", "match_id": match_id,
            "error": "Invalid side/token for this match — reconnect with the token from kickoff.",
        })
        await websocket.close(code=4403)
        return "INVALID"
    return side


@app.websocket("/ws/match/{match_id}")
async def stream_match(websocket: WebSocket, match_id: str, side: str | None = None, token: str | None = None):
    await websocket.accept()

    authed_side = await _authenticate_match_side(websocket, match_id, side, token)
    if authed_side == "INVALID":
        return

    runner = _MATCH_RUNNERS.get(match_id)
    if runner is None or runner.finished:
        runner = _MatchRunner()
        _MATCH_RUNNERS[match_id] = runner

    q: asyncio.Queue = asyncio.Queue()
    runner.subscribers.append(q)
    relay_task: asyncio.Task | None = None

    if authed_side is not None:
        await _side_presence_delta(match_id, authed_side, +1)

    try:
        # Late joiner (e.g. the second player's socket connects a beat after
        # the first): replay whatever's already been emitted so both players
        # end up looking at the same match state, not out of sync. Prefer
        # the local log (covers no-Redis mode and the owning worker), else
        # fall back to the Redis-backed log for a viewer on another worker.
        backlog = runner.log if runner.log else await state_backend.log_read_all(f"match:{match_id}:log")
        for text in backlog:
            await websocket.send_text(text)

        owner_token = await state_backend.try_acquire_owner(f"match:{match_id}:sim_lock", SIM_LOCK_TTL_SECONDS)
        if owner_token is not None:
            if runner.task is None:
                runner.task = asyncio.create_task(_run_match_simulation(match_id, owner_token))
        elif runner.task is None:
            # Some other worker already owns the simulation for this match —
            # relay its events in over Redis instead of starting a second,
            # competing `simulate_match` loop (which is exactly the bug this
            # replaces: two independent simulations racing each other).
            relay_task = asyncio.create_task(_relay_remote_events(match_id, q))

        while True:
            text = await q.get()
            if text is None:  # simulation finished
                break
            await websocket.send_text(text)

        await websocket.close()

    except WebSocketDisconnect:
        # This viewer dropped — the shared simulation (and the other
        # player's connection, if any) keeps running, UNLESS this was a
        # friend-match player's own authenticated socket, in which case
        # the presence-count decrement below feeds the pause/forfeit
        # watchdog running inside `_run_match_simulation`.
        pass
    finally:
        try:
            runner.subscribers.remove(q)
        except ValueError:
            pass
        if relay_task is not None:
            relay_task.cancel()
            with contextlib.suppress(Exception):
                await relay_task
        if authed_side is not None:
            await _side_presence_delta(match_id, authed_side, -1)


@app.websocket("/ws/match/{match_id}/tactics")
async def receive_tactics(websocket: WebSocket, match_id: str, side: str | None = None, token: str | None = None):
    """
    Companion channel for live tactical intervention (substitutions,
    mentality changes) during the match. Messages are pushed onto the
    `_PENDING_SUBS`/Redis-backed queue for `match_id` (see
    `_pending_subs_push`), which `stream_match`'s owning worker drains and
    applies to the live TeamSnapshot every tick (and continuously during
    the half-time break) — see `_apply_pending_subs`.

    Expected message shape:
        {"action": "substitution", "side": "home"|"away",
         "payload": {"player_out_id": "...", "player_in_id": "..."}}
        {"action": "mentality", "side": "home"|"away",
         "payload": {"value": -1.0..1.0}}

    Side ownership: if `match_id` came out of the friend-match lobby flow
    (two real players), `_get_side_tokens(match_id)` holds a secret token
    per side. The connecting client must pass `?side=home|away&token=...`
    matching the token it was handed at kickoff — a connection with a wrong
    or missing token is refused outright, and once connected, every message
    on this socket is forced onto the side it authenticated as regardless
    of what "side" the message body itself claims. This stops one player's
    browser from ever substituting on the other player's team, whether by
    bug or by a tampered client. Matches that never went through the lobby
    (solo quick-match vs. a bot) have no side tokens, so they keep the old
    open behavior — both sides on one browser, no auth.
    """
    await websocket.accept()
    required = await _get_side_tokens(match_id)
    authed_side: str | None = None
    if required is not None:
        if side not in ("home", "away") or token != required.get(side):
            await websocket.send_json({
                "ack": False, "match_id": match_id,
                "error": "Invalid or missing side/token for this match — reconnect with the token from kickoff.",
            })
            await websocket.close(code=4403)
            return
        authed_side = side

    try:
        while True:
            msg = await websocket.receive_json()
            if not await _is_match_active(match_id):
                await websocket.send_json({
                    "ack": False, "match_id": match_id,
                    "error": "No live simulation for this match_id yet — connect /ws/match/{id} first.",
                })
                continue

            if authed_side is not None:
                claimed_side = msg.get("side")
                if claimed_side != authed_side:
                    await websocket.send_json({
                        "ack": False, "match_id": match_id,
                        "error": f"Rejected — this connection is authenticated as '{authed_side}', "
                                 f"not '{claimed_side}'. You can only act on your own side.",
                    })
                    continue
                # Redundant given the check above, but make it explicit that
                # the side actually applied is the authenticated one, never
                # a client-supplied value.
                msg["side"] = authed_side

            await _pending_subs_push(match_id, msg)
            await websocket.send_json({"ack": True, "match_id": match_id, "queued": msg})
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Precomputed-match replay with a retroactive substitution.
#
# For matches simulated ahead of time in a batch job (no one watching live),
# a substitution requested after the fact still needs to change the result,
# not just relabel a scorer. The trick: re-run the SAME deterministic seed,
# replaying byte-for-byte up to the substitution minute, then mutate the
# lineup and let the remaining minutes diverge naturally.
# ---------------------------------------------------------------------------

def replay_with_substitution(
    home: TeamSnapshot,
    away: TeamSnapshot,
    seed: int,
    sub_minute: int,
    side: str,
    player_out_id: str,
    player_in: PlayerSnapshot,
    duration_minutes: int = 90,
) -> list[dict]:
    """
    Returns the full new event list for the match, identical to the
    original precomputed run up through `sub_minute`, diverging after.

    NOTE: this does not require a websocket connection — it's the function
    a "resimulate tonight's matchday" background worker (or an admin/debug
    endpoint) would call directly.
    """
    events: list[dict] = []
    applied = False
    team = home if side == "home" else away

    for event in simulate_match(home, away, seed=seed, duration_minutes=duration_minutes):
        events.append(event)
        if not applied and event["minute"] >= sub_minute:
            apply_substitution(team, player_out_id, player_in)
            applied = True
    return events


# ---------------------------------------------------------------------------
# Live trading between two connected managers.
#
# The pieces to SETTLE a trade already existed (db/turso_client.execute_trade
# is a full ACID multi-table swap, engine/market_ml.score_trade is the
# fairness/anti-cheat check) but nothing in this file ever actually created
# an offer or exposed any of it over HTTP/WebSocket — index.html's Trade Hub
# was purely a client-side mock negotiating against a fake seeded inbox, not
# two real managers. This section is the missing wiring:
#   * POST /trade/propose, /trade/{id}/counter, /trade/{id}/accept,
#     /trade/{id}/decline — the negotiation state machine.
#   * GET  /trade/box/{team_id} — a plain poll fallback (works even if the
#     WebSocket below is unavailable/blocked by a restrictive network).
#   * WS   /ws/trades/{team_id} — a live push channel so BOTH sides see a
#     propose/counter/accept land in real time instead of waiting on a
#     manual refresh, mirroring the friend-match lobby's pattern of one
#     shared, subscribable channel per room (here: per team).
# ---------------------------------------------------------------------------

from engine.market_ml import PlayerValuationInput, TradeLeg, predict_value, score_trade
from db.turso_client import (
    create_trade_offer, respond_to_trade_offer, get_trade_offer_detail,
    list_team_trade_offers, TradeRejected,
)

_TRADE_SUBSCRIBERS: dict[str, list[asyncio.Queue]] = {}  # team_id -> live listeners


async def _broadcast_trade_event(team_ids: list[str], payload: dict) -> None:
    """Pushes to every connected /ws/trades/{team_id} listener for each
    team involved — both the proposer and the receiver get it instantly,
    whichever one didn't trigger the action."""
    for team_id in set(team_ids):
        for q in list(_TRADE_SUBSCRIBERS.get(team_id, [])):
            await q.put(payload)


class TradeOfferLeg(BaseModel):
    player_id: str
    age: int
    overall: int
    potential: int
    contract_years_left: int
    current_form: float = 0.0


class TradeClause(BaseModel):
    player_id: str
    clause_type: str  # 'buyback' | 'sell_on'
    buyback_fee: int | None = None
    buyback_expires_season: int | None = None
    sell_on_percentage: float | None = None


class TradeProposeRequest(BaseModel):
    from_team_id: str
    to_team_id: str
    players_offered: list[TradeOfferLeg] = []      # proposer -> receiver
    players_requested: list[TradeOfferLeg] = []    # receiver -> proposer
    cash_from_proposer: int = 0
    cash_from_receiver: int = 0
    clauses: list[TradeClause] = []


def _score_offer(req: "TradeProposeRequest"):
    """Values every player leg with the same market model the frontend's
    fairness meter mirrors client-side, then scores the whole package —
    this is the actual anti-cheat check, not just a UI preview."""
    from_side = [
        TradeLeg(player_value=predict_value(PlayerValuationInput(
            age=p.age, overall=p.overall, potential=p.potential,
            contract_years_left=p.contract_years_left, current_form=p.current_form,
        )))
        for p in req.players_offered
    ] + ([TradeLeg(player_value=0, cash=req.cash_from_proposer)] if req.cash_from_proposer else [])
    to_side = [
        TradeLeg(player_value=predict_value(PlayerValuationInput(
            age=p.age, overall=p.overall, potential=p.potential,
            contract_years_left=p.contract_years_left, current_form=p.current_form,
        )))
        for p in req.players_requested
    ] + ([TradeLeg(player_value=0, cash=req.cash_from_receiver)] if req.cash_from_receiver else [])
    return score_trade(from_side, to_side)


@app.post("/trade/propose")
async def propose_trade(req: TradeProposeRequest) -> dict:
    fairness = _score_offer(req)
    if fairness.flag == "blocked":
        return JSONResponse(status_code=422, content={
            "error": "Blocked by the valuation model — this trade is too lopsided.",
            "fairness_score": fairness.fairness_score, "flag": fairness.flag,
        })

    offer_id = await create_trade_offer(
        from_team_id=req.from_team_id, to_team_id=req.to_team_id,
        players_to_receiving_team=[p.player_id for p in req.players_offered],
        players_to_proposing_team=[p.player_id for p in req.players_requested],
        from_team_cash=req.cash_from_proposer, to_team_cash=req.cash_from_receiver,
        ml_fairness_score=fairness.fairness_score, ml_flag=fairness.flag,
        clauses=[c.model_dump() for c in req.clauses],
    )
    detail = await get_trade_offer_detail(offer_id)
    await _broadcast_trade_event([req.from_team_id, req.to_team_id],
                                  {"type": "trade_proposed", "offer": detail})
    return detail


@app.post("/trade/{offer_id}/counter")
async def counter_trade(offer_id: str, req: TradeProposeRequest) -> dict:
    """Same shape as /trade/propose — a counter-offer is a brand-new offer
    with the sides typically flipped, linked back via parent_offer_id."""
    fairness = _score_offer(req)
    if fairness.flag == "blocked":
        return JSONResponse(status_code=422, content={
            "error": "Blocked by the valuation model — this counter is too lopsided.",
            "fairness_score": fairness.fairness_score, "flag": fairness.flag,
        })

    new_offer_id = await create_trade_offer(
        from_team_id=req.from_team_id, to_team_id=req.to_team_id,
        players_to_receiving_team=[p.player_id for p in req.players_offered],
        players_to_proposing_team=[p.player_id for p in req.players_requested],
        from_team_cash=req.cash_from_proposer, to_team_cash=req.cash_from_receiver,
        ml_fairness_score=fairness.fairness_score, ml_flag=fairness.flag,
        clauses=[c.model_dump() for c in req.clauses],
        parent_offer_id=offer_id,
    )
    detail = await get_trade_offer_detail(new_offer_id)
    await _broadcast_trade_event([req.from_team_id, req.to_team_id],
                                  {"type": "trade_countered", "original_offer_id": offer_id, "offer": detail})
    return detail


@app.post("/trade/{offer_id}/accept")
async def accept_trade(offer_id: str) -> dict:
    detail = await get_trade_offer_detail(offer_id)
    if detail is None:
        return JSONResponse(status_code=404, content={"error": "Offer not found."})
    try:
        settlement = await respond_to_trade_offer(offer_id, "accept")
    except TradeRejected as e:
        return JSONResponse(status_code=409, content={"error": str(e)})
    await _broadcast_trade_event([detail["from_team_id"], detail["to_team_id"]],
                                  {"type": "trade_accepted", "offer_id": offer_id, "settlement": settlement})
    return settlement


@app.post("/trade/{offer_id}/decline")
async def decline_trade(offer_id: str) -> dict:
    detail = await get_trade_offer_detail(offer_id)
    if detail is None:
        return JSONResponse(status_code=404, content={"error": "Offer not found."})
    result = await respond_to_trade_offer(offer_id, "decline")
    await _broadcast_trade_event([detail["from_team_id"], detail["to_team_id"]],
                                  {"type": "trade_declined", "offer_id": offer_id})
    return result


@app.get("/team/{team_id}/roster")
async def team_roster(team_id: str) -> dict:
    """Read-only squad listing for building a trade offer. The friend-match
    tactical map deliberately only shows the opponent's placeholder dots
    (see README's "Known limitations"), but a live trade needs to know
    what they actually have — this exposes just enough (name/position/
    overall/valuation inputs) to build an offer against, not full match
    internals like consistency/fatigue/current stat rolls."""
    client = get_client()
    rs = await client.execute(
        "SELECT p.id, p.name, p.position, p.overall, p.age, p.potential, "
        "p.contract_years_left, p.current_form, pr.squad_role "
        "FROM player_rights pr JOIN players p ON p.id = pr.player_id "
        "WHERE pr.owner_team_id = ? ORDER BY p.overall DESC",
        [team_id],
    )
    return {"team_id": team_id, "players": [dict(r) for r in rs.rows]}


@app.get("/trade/box/{team_id}")
async def trade_box(team_id: str) -> dict:
    """Poll-based fallback — open offers (inbox + outbox combined, the
    frontend splits by from_team_id/to_team_id) for a team. Works even for
    a client that never opens the WebSocket below."""
    offers = await list_team_trade_offers(team_id)
    return {"team_id": team_id, "offers": offers}


@app.websocket("/ws/trades/{team_id}")
async def trade_socket(websocket: WebSocket, team_id: str):
    """Live push companion to the REST routes above: whenever a propose/
    counter/accept/decline touches this team on either side, it's pushed
    here immediately, so two managers who are both online see the
    negotiation update in real time instead of polling /trade/box."""
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue()
    _TRADE_SUBSCRIBERS.setdefault(team_id, []).append(q)
    try:
        # Send current open offers immediately on connect, same idea as
        # stream_match()'s event-replay-for-late-joiners.
        await websocket.send_json({"type": "snapshot", "offers": await list_team_trade_offers(team_id)})
        while True:
            payload = await q.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        _TRADE_SUBSCRIBERS.get(team_id, []).remove(q) if q in _TRADE_SUBSCRIBERS.get(team_id, []) else None
