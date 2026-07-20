"""
WebSocket match streamer.

Compresses a 90-minute match into ~2 real minutes: 1 in-game minute ==
1.33 real seconds (90 * 1.33 ≈ 120s). Only *emitted* events carry a real
delay proportional to how many in-game minutes elapsed since the last
event, so quiet stretches fly by and the 2-minute thrill still lands on
the goals/chances that matter.

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

from db.turso_client import get_client
from db.auth import authenticate_user, create_user, ensure_auth_schema
from engine.match_engine import PlayerSnapshot, TeamSnapshot, apply_substitution, simulate_match
from engine.ml_bridge import MLBridge

app = FastAPI(title="Football Sim Match Streamer")

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
async def health():
    """Used by the host's health check (Fly.io/Render/Railway all probe this).
    Also useful to hit manually first if you suspect a Render cold start —
    the very first request after idle can take 30-60s, and THAT delay is
    what usually shows up in the browser as a CORS/network failure on the
    real request that follows it."""
    return {"status": "ok"}


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """
    Belt-and-suspenders: CORSMiddleware normally attaches headers even to
    error responses, but a truly unhandled exception (e.g. TURSO_DATABASE_URL
    missing so get_client() throws a bare KeyError) can otherwise surface to
    the browser as a connection failure with NO headers at all — which shows
    up in devtools as a CORS error even though CORS was never the problem.
    This guarantees every response, success or failure, is well-formed.
    """
    return JSONResponse(
        status_code=500,
        content={"error": str(exc) or exc.__class__.__name__},
    )


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
    ('starter'/'bench'). Returns {frontend_player_id: backend_player_id}."""
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


async def _provision_and_create_match(
    client,
    home_name: str, home_squad: list[SimplePlayer], home_bench: list[SimplePlayer],
    away_name: str, away_squad: list[SimplePlayer], away_bench: list[SimplePlayer],
) -> dict:
    """Shared by the solo `/quick-match` endpoint (vs. a bot/your own two
    sides) and the two-real-player friend-match flow below: creates the two
    ephemeral team rows, provisions both squads, and inserts the `matches`
    row. Returns the same shape either caller can hand straight back to the
    frontend to open `/ws/match/{match_id}` against."""
    guest_user_id = "guest-demo-user"
    await client.execute(
        "INSERT INTO users (id, username, email) VALUES (?, ?, ?) ON CONFLICT(id) DO NOTHING",
        [guest_user_id, "guest", "guest@touchline.local"],
    )

    home_team_id, away_team_id = str(uuid.uuid4()), str(uuid.uuid4())
    await client.execute(
        "INSERT INTO teams (id, manager_user_id, name) VALUES (?, ?, ?)",
        [home_team_id, guest_user_id, home_name],
    )
    await client.execute(
        "INSERT INTO teams (id, manager_user_id, name) VALUES (?, ?, ?)",
        [away_team_id, guest_user_id, away_name],
    )

    home_map = await _provision_squad(client, home_squad, home_team_id, "starter")
    home_map.update(await _provision_squad(client, home_bench, home_team_id, "bench"))
    away_map = await _provision_squad(client, away_squad, away_team_id, "starter")
    away_map.update(await _provision_squad(client, away_bench, away_team_id, "bench"))

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
# Like `_PENDING_SUBS` below, lobby state lives in an in-process dict; a
# real multi-worker deployment would back this with Redis/a per-match actor
# instead so a lobby survives a request landing on a different worker.
# ---------------------------------------------------------------------------

_CODE_ALPHABET = "".join(c for c in string.ascii_uppercase + string.digits if c not in "0O1I")


def _generate_lobby_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))


@dataclass
class _LobbySide:
    token: str
    manager_name: str
    team_name: str
    squad: list[SimplePlayer]
    bench: list[SimplePlayer]
    ready: bool = False
    ws: WebSocket | None = None


@dataclass
class _Lobby:
    code: str
    pending_id: str
    host: _LobbySide
    guest: _LobbySide | None = None
    status: str = "waiting_guest"  # waiting_guest -> waiting_ready -> started
    match_id: str | None = None
    created_at: float = field(default_factory=time.time)


_LOBBIES: dict[str, _Lobby] = {}          # keyed by pending_id
_LOBBIES_BY_CODE: dict[str, str] = {}     # code -> pending_id

# match_id -> {"home": token, "away": token}, populated at kickoff. Only
# friend-matches born from this lobby flow get an entry here — the solo
# `/quick-match` endpoint (playing vs. a bot, or both sides on one browser)
# never does, so its tactics socket stays open/unauthenticated as before.
_MATCH_SIDE_TOKENS: dict[str, dict[str, str]] = {}


class FriendMatchCreateRequest(BaseModel):
    manager_name: str
    home_name: str
    home_squad: list[SimplePlayer]
    home_bench: list[SimplePlayer] = []


class FriendMatchJoinRequest(BaseModel):
    code: str
    manager_name: str
    away_name: str
    away_squad: list[SimplePlayer]
    away_bench: list[SimplePlayer] = []


@app.post("/friend-match/create")
async def friend_match_create(req: FriendMatchCreateRequest) -> dict:
    if len(req.home_squad) != 11:
        return JSONResponse(status_code=400, content={"error": "home_squad must have exactly 11 players."})

    pending_id = str(uuid.uuid4())
    code = _generate_lobby_code()
    while code in _LOBBIES_BY_CODE:  # astronomically unlikely, but be sure
        code = _generate_lobby_code()

    host_token = secrets.token_urlsafe(18)
    lobby = _Lobby(
        code=code, pending_id=pending_id,
        host=_LobbySide(token=host_token, manager_name=req.manager_name, team_name=req.home_name,
                         squad=req.home_squad, bench=req.home_bench),
    )
    _LOBBIES[pending_id] = lobby
    _LOBBIES_BY_CODE[code] = pending_id

    return {"pending_id": pending_id, "code": code, "host_token": host_token}


@app.post("/friend-match/join")
async def friend_match_join(req: FriendMatchJoinRequest) -> dict:
    pending_id = _LOBBIES_BY_CODE.get(req.code.strip().upper())
    lobby = _LOBBIES.get(pending_id) if pending_id else None
    if lobby is None:
        return JSONResponse(status_code=404, content={"error": "No room found for that code."})
    if lobby.status != "waiting_guest" or lobby.guest is not None:
        return JSONResponse(status_code=409, content={"error": "That room already has two players."})
    if len(req.away_squad) != 11:
        return JSONResponse(status_code=400, content={"error": "away_squad must have exactly 11 players."})

    guest_token = secrets.token_urlsafe(18)
    lobby.guest = _LobbySide(token=guest_token, manager_name=req.manager_name, team_name=req.away_name,
                              squad=req.away_squad, bench=req.away_bench)
    lobby.status = "waiting_ready"

    if lobby.host.ws is not None:
        await lobby.host.ws.send_json(_lobby_state_payload(lobby))

    return {
        "pending_id": lobby.pending_id, "guest_token": guest_token,
        "host_manager_name": lobby.host.manager_name, "home_name": lobby.host.team_name,
    }


def _lobby_state_payload(lobby: _Lobby) -> dict:
    return {
        "type": "state",
        "code": lobby.code,
        "status": lobby.status,
        "host_manager_name": lobby.host.manager_name,
        "host_team_name": lobby.host.team_name,
        "host_ready": lobby.host.ready,
        "guest_joined": lobby.guest is not None,
        "guest_manager_name": lobby.guest.manager_name if lobby.guest else None,
        "guest_team_name": lobby.guest.team_name if lobby.guest else None,
        "guest_ready": lobby.guest.ready if lobby.guest else False,
    }


async def _broadcast_lobby_state(lobby: _Lobby) -> None:
    payload = _lobby_state_payload(lobby)
    for side in (lobby.host, lobby.guest):
        if side is not None and side.ws is not None:
            try:
                await side.ws.send_json(payload)
            except Exception:
                pass


async def _try_kickoff(lobby: _Lobby) -> bool:
    """Both-players-ready gate: the match is only provisioned and started
    once BOTH `host.ready` and `guest.ready` are true. Returns True if
    kickoff happened."""
    if lobby.guest is None or not lobby.host.ready or not lobby.guest.ready:
        return False
    if lobby.status == "started":
        return False

    client = get_client()
    result = await _provision_and_create_match(
        client, lobby.host.team_name, lobby.host.squad, lobby.host.bench,
        lobby.guest.team_name, lobby.guest.squad, lobby.guest.bench,
    )
    lobby.status = "started"
    lobby.match_id = result["match_id"]
    _MATCH_SIDE_TOKENS[result["match_id"]] = {"home": lobby.host.token, "away": lobby.guest.token}

    for side_obj, side_name, side_token in ((lobby.host, "home", lobby.host.token),
                                             (lobby.guest, "away", lobby.guest.token)):
        if side_obj.ws is None:
            continue
        try:
            await side_obj.ws.send_json({
                "type": "kickoff", "match_id": result["match_id"],
                "home_team_id": result["home_team_id"], "away_team_id": result["away_team_id"],
                "home_player_map": result["home_player_map"], "away_player_map": result["away_player_map"],
                "your_side": side_name, "your_token": side_token,
            })
        except Exception:
            pass
    return True


@app.websocket("/ws/lobby/{pending_id}")
async def lobby_socket(websocket: WebSocket, pending_id: str, role: str | None = None, token: str | None = None):
    """Real-time companion to /friend-match/create + /friend-match/join.
    `role` is "host" or "guest"; `token` must match the token that endpoint
    returned. Clients send {"action": "ready"} once their manager is happy
    to kick off; the match only actually starts once both sides have."""
    await websocket.accept()
    lobby = _LOBBIES.get(pending_id)
    if lobby is None or role not in ("host", "guest"):
        await websocket.send_json({"type": "error", "error": "Room not found."})
        await websocket.close(code=4404)
        return

    side_obj = lobby.host if role == "host" else lobby.guest
    if side_obj is None or token != side_obj.token:
        await websocket.send_json({"type": "error", "error": "Invalid room token."})
        await websocket.close(code=4403)
        return

    side_obj.ws = websocket
    await websocket.send_json(_lobby_state_payload(lobby))

    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("action") == "ready":
                side_obj.ready = True
                await _broadcast_lobby_state(lobby)
                await _try_kickoff(lobby)
            elif msg.get("action") == "unready":
                side_obj.ready = False
                await _broadcast_lobby_state(lobby)
    except WebSocketDisconnect:
        side_obj.ws = None
        if lobby.status != "started":
            side_obj.ready = False
            await _broadcast_lobby_state(lobby)


SECONDS_PER_GAME_MINUTE = 1.33
HALF_TIME_BREAK_SECONDS = 20.0  # real seconds paused server-side at half-time

# One live-substitution queue per in-flight match, shared between the two
# websocket handlers below (`stream_match` reads it, `receive_tactics`
# writes to it). A production deployment would back this with Redis/a
# per-match actor instead of an in-process dict.
_PENDING_SUBS: dict[str, "asyncio.Queue[dict]"] = {}


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


def _apply_pending_subs(match_id: str, home: TeamSnapshot, away: TeamSnapshot,
                         home_bench: list[PlayerSnapshot], away_bench: list[PlayerSnapshot]) -> list[dict]:
    """Drains any substitution/mentality messages queued by receive_tactics,
    mutates the live TeamSnapshot objects in place, and returns the
    match_event-shaped dicts to emit/persist for whichever ones landed."""
    queue = _PENDING_SUBS.get(match_id)
    if queue is None:
        return []

    emitted = []
    while not queue.empty():
        msg = queue.get_nowait()
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
    viewer. Two real players both opening `/ws/match/{id}` for the SAME
    friend-match need to watch (and tactically affect) the identical
    running simulation — the old version of this handler instead started
    an independent `simulate_match(...)` loop per websocket connection AND
    unconditionally overwrote `_PENDING_SUBS[match_id]` with a fresh queue
    on every new connection, so a second viewer connecting would silently
    steal the substitution channel out from under the first. Now exactly
    one background task runs the simulation; connecting just subscribes a
    queue to it."""
    task: asyncio.Task | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    log: list[str] = field(default_factory=list)  # serialized events emitted so far, for late joiners
    finished: bool = False


_MATCH_RUNNERS: dict[str, _MatchRunner] = {}


async def _broadcast(runner: _MatchRunner, payload: dict) -> None:
    text = json.dumps(payload)
    runner.log.append(text)
    for q in list(runner.subscribers):
        await q.put(text)


async def _run_match_simulation(match_id: str) -> None:
    runner = _MATCH_RUNNERS[match_id]
    client = get_client()
    ml_bridge = MLBridge.load()
    try:
        row_rs = await client.execute(
            "SELECT home_team_id, away_team_id, rng_seed FROM matches WHERE id = ?",
            [match_id],
        )
        if not row_rs.rows:
            await _broadcast(runner, {"type": "error", "description": "Match not found", "match_id": match_id})
            return

        row = row_rs.rows[0]
        seed = row["rng_seed"] or random.randint(1, 2**31 - 1)

        home, home_bench = await _load_squad(client, row["home_team_id"])
        away, away_bench = await _load_squad(client, row["away_team_id"])

        await _persist_match_start(client, match_id, seed)
        _PENDING_SUBS[match_id] = asyncio.Queue()

        last_minute = 0
        # `home`/`away` are the SAME objects the tactics socket mutates via
        # apply_substitution — simulate_match reads their .lineup fresh
        # every minute, so a sub queued mid-stream takes effect immediately.
        for event in simulate_match(home, away, seed=seed, duration_minutes=90, ml_bridge=ml_bridge):
            elapsed_minutes = max(0, event["minute"] - last_minute)
            delay = elapsed_minutes * SECONDS_PER_GAME_MINUTE
            if delay > 0:
                await asyncio.sleep(min(delay, 6.0))  # cap any single gap
            last_minute = event["minute"]

            payload = {**event, "match_id": match_id, "sent_at": datetime.now(timezone.utc).isoformat()}
            await _broadcast(runner, payload)

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
                    for sub_event in _apply_pending_subs(match_id, home, away, home_bench, away_bench):
                        sub_event["minute"] = last_minute
                        sub_payload = {**sub_event, "match_id": match_id,
                                        "sent_at": datetime.now(timezone.utc).isoformat()}
                        await _broadcast(runner, sub_payload)
                        await _persist_event(client, match_id, sub_event)

            # Outside the break too: subs can land on any live minute, not
            # just at half-time (e.g. a 60th-minute tactical change).
            for sub_event in _apply_pending_subs(match_id, home, away, home_bench, away_bench):
                sub_event["minute"] = last_minute
                sub_payload = {**sub_event, "match_id": match_id,
                                "sent_at": datetime.now(timezone.utc).isoformat()}
                await _broadcast(runner, sub_payload)
                await _persist_event(client, match_id, sub_event)

            if event["type"] == "full_time":
                await _persist_match_end(client, match_id, event)

        _PENDING_SUBS.pop(match_id, None)
        _MATCH_SIDE_TOKENS.pop(match_id, None)
    finally:
        runner.finished = True
        for q in list(runner.subscribers):
            await q.put(None)  # sentinel: tells each handler to close its socket


@app.websocket("/ws/match/{match_id}")
async def stream_match(websocket: WebSocket, match_id: str):
    await websocket.accept()

    runner = _MATCH_RUNNERS.get(match_id)
    if runner is None or runner.finished:
        runner = _MatchRunner()
        _MATCH_RUNNERS[match_id] = runner

    q: asyncio.Queue = asyncio.Queue()
    runner.subscribers.append(q)

    try:
        # Late joiner (e.g. the second player's socket connects a beat after
        # the first): replay whatever's already been emitted so both players
        # end up looking at the same match state, not out of sync.
        for text in runner.log:
            await websocket.send_text(text)

        if runner.task is None:
            runner.task = asyncio.create_task(_run_match_simulation(match_id))

        while True:
            text = await q.get()
            if text is None:  # simulation finished
                break
            await websocket.send_text(text)

        await websocket.close()

    except WebSocketDisconnect:
        # This viewer dropped — the shared simulation (and the other
        # player's connection, if any) keeps running unaffected.
        pass
    finally:
        try:
            runner.subscribers.remove(q)
        except ValueError:
            pass


@app.websocket("/ws/match/{match_id}/tactics")
async def receive_tactics(websocket: WebSocket, match_id: str, side: str | None = None, token: str | None = None):
    """
    Companion channel for live tactical intervention (substitutions,
    mentality changes) during the match. Messages are pushed onto
    `_PENDING_SUBS[match_id]`, which `stream_match`'s loop drains and
    applies to the live TeamSnapshot every tick (and continuously during
    the half-time break) — see `_apply_pending_subs`.

    Expected message shape:
        {"action": "substitution", "side": "home"|"away",
         "payload": {"player_out_id": "...", "player_in_id": "..."}}
        {"action": "mentality", "side": "home"|"away",
         "payload": {"value": -1.0..1.0}}

    Side ownership: if `match_id` came out of the friend-match lobby flow
    (two real players), `_MATCH_SIDE_TOKENS[match_id]` holds a secret token
    per side. The connecting client must pass `?side=home|away&token=...`
    matching the token it was handed at kickoff — a connection with a wrong
    or missing token is refused outright, and once connected, every message
    on this socket is forced onto the side it authenticated as regardless
    of what "side" the message body itself claims. This stops one player's
    browser from ever substituting on the other player's team, whether by
    bug or by a tampered client. Matches that never went through the lobby
    (solo quick-match vs. a bot) have no entry in `_MATCH_SIDE_TOKENS`, so
    they keep the old open behavior — both sides on one browser, no auth.
    """
    await websocket.accept()
    required = _MATCH_SIDE_TOKENS.get(match_id)
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
            queue = _PENDING_SUBS.get(match_id)
            if queue is None:
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

            await queue.put(msg)
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
