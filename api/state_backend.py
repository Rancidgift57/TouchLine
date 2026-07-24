"""
Shared cross-worker state backend for api/match_stream.py.

The original implementation kept lobby state, per-match side tokens, the
live-substitution queue, and the running match's event log/subscriber list
in plain in-process dicts. That's fine for `uvicorn --workers 1`, but it
silently breaks the moment you run more than one worker (or more than one
region/instance): a lobby-join request, a tactics/substitution message, or
a viewer's websocket can land on a worker that never saw the rest of that
match's state, and it just looks like the feature doesn't work.

This module gives every piece of that state a Redis-backed implementation
(async, via `redis.asyncio`) that any worker can read/write/subscribe to,
and falls back to the old in-process behavior when `REDIS_URL` isn't set —
so local dev / single-worker deploys need nothing extra, and a real
multi-worker deployment just needs to set the env var.

Everything here is deliberately narrow: it's not a general pub/sub
framework, just the handful of primitives match_stream.py actually needs:
  * JSON blob storage with TTL (lobby state, side tokens)
  * an append-only log + fan-out channel (match event stream, for replay-
    to-late-joiners plus live push)
  * a FIFO queue (pending tactics/substitution messages)
  * a distributed lock (so exactly one worker runs a given match's
    simulation loop, no matter which worker's websocket kicked it off)
  * a plain broadcast channel (lobby state pushes, disconnect/reconnect
    control signals for the pause/forfeit watchdog)
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from typing import Any, AsyncIterator

_redis = None
_redis_checked = False


def redis_enabled() -> bool:
    return bool(os.environ.get("REDIS_URL"))


def get_redis():
    """Lazily create a single shared redis.asyncio client for the process.
    Returns None (and every helper below falls back to in-process state)
    if REDIS_URL isn't configured or the `redis` package isn't installed."""
    global _redis, _redis_checked
    if _redis_checked:
        return _redis
    _redis_checked = True
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis.asyncio as redis_asyncio
    except ImportError:
        print("REDIS_URL is set but the `redis` package isn't installed "
              "(pip install redis>=5.0) — falling back to single-worker "
              "in-process state.")
        return None
    _redis = redis_asyncio.from_url(url, decode_responses=True)
    return _redis


# ---------------------------------------------------------------------------
# JSON blob storage (lobby snapshots, side tokens) — get/set/delete with TTL.
# ---------------------------------------------------------------------------

async def kv_set(key: str, value: Any, ttl_seconds: int | None = None) -> None:
    r = get_redis()
    if r is None:
        return
    payload = json.dumps(value)
    if ttl_seconds:
        await r.set(key, payload, ex=ttl_seconds)
    else:
        await r.set(key, payload)


async def kv_get(key: str) -> Any | None:
    r = get_redis()
    if r is None:
        return None
    raw = await r.get(key)
    return json.loads(raw) if raw is not None else None


async def kv_delete(*keys: str) -> None:
    r = get_redis()
    if r is None:
        return
    if keys:
        await r.delete(*keys)


# ---------------------------------------------------------------------------
# Pub/sub broadcast — used for lobby state pushes and match control signals
# (side connected/disconnected) so the worker that owns the relevant loop
# hears about an event that happened on a websocket held by a *different*
# worker.
# ---------------------------------------------------------------------------

async def publish(channel: str, payload: dict) -> None:
    r = get_redis()
    if r is None:
        return
    await r.publish(channel, json.dumps(payload))


@contextlib.asynccontextmanager
async def subscribe(channel: str) -> AsyncIterator[Any]:
    """Async context manager yielding an async generator of decoded
    payloads published to `channel`. No-ops (yields an empty generator) if
    Redis isn't configured — callers that only need pub/sub for the
    multi-worker case should treat "nothing arrives" as fine, since in
    single-worker/no-Redis mode the equivalent signal is delivered
    in-process instead."""
    r = get_redis()
    if r is None:
        async def _empty():
            return
            yield  # pragma: no cover - makes this an async generator
        yield _empty()
        return

    pubsub = r.pubsub()
    await pubsub.subscribe(channel)

    async def _messages():
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            try:
                yield json.loads(msg["data"])
            except (TypeError, ValueError):
                continue

    try:
        yield _messages()
    finally:
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(channel)
        with contextlib.suppress(Exception):
            await pubsub.aclose()


# ---------------------------------------------------------------------------
# Append-only event log — RPUSH for durability/replay, paired with a
# `publish()` on the same logical channel for live fan-out. match_stream.py
# uses this so a late-joining viewer (or a viewer whose websocket lands on
# a different worker than the one running the simulation) can catch up via
# the log, then keep receiving live events via the channel.
# ---------------------------------------------------------------------------

async def log_append(key: str, entry: str, ttl_seconds: int = 3600) -> None:
    r = get_redis()
    if r is None:
        return
    await r.rpush(key, entry)
    await r.expire(key, ttl_seconds)


async def log_read_all(key: str) -> list[str]:
    r = get_redis()
    if r is None:
        return []
    return await r.lrange(key, 0, -1)


# ---------------------------------------------------------------------------
# FIFO queue — backs `_PENDING_SUBS`. RPUSH/LPOP so `receive_tactics`
# (whichever worker holds that websocket) and `_run_match_simulation`
# (whichever worker owns the sim loop) can be on different workers.
# ---------------------------------------------------------------------------

async def queue_push(key: str, item: dict) -> None:
    r = get_redis()
    if r is None:
        return
    await r.rpush(key, json.dumps(item))


async def queue_drain(key: str) -> list[dict]:
    """Pops every currently-queued item off (non-blocking) and returns them
    in FIFO order."""
    r = get_redis()
    if r is None:
        return []
    out: list[dict] = []
    while True:
        raw = await r.lpop(key)
        if raw is None:
            break
        with contextlib.suppress(json.JSONDecodeError):
            out.append(json.loads(raw))
    return out


# ---------------------------------------------------------------------------
# Sorted set — backs the matchmaking queue (`mm:waiting` in match_stream.py).
# Members are ticket ids, score is each ticket's `joined_at` timestamp, so
# `sorted_set_all` naturally returns waiting tickets oldest-first (the order
# the matchmaking loop wants: someone who's been waiting longest gets first
# crack at a newly-widened tolerance band). `sorted_set_remove` is the
# pairing claim primitive — ZREM on multiple members is a single atomic
# Redis command, so calling it with both {my_ticket_id, candidate_id} and
# checking the return count == 2 is how two near-simultaneous workers
# racing to pair the same two tickets resolve without double-booking a
# match: exactly one caller sees count == 2, everyone else sees a smaller
# count and backs off. No-op / returns empty when Redis isn't configured,
# since match_stream.py's local fallback keeps its own in-process dict.
# ---------------------------------------------------------------------------

async def sorted_set_add(key: str, member: str, score: float) -> None:
    r = get_redis()
    if r is None:
        return
    await r.zadd(key, {member: score})


async def sorted_set_all(key: str) -> list[tuple[str, float]]:
    r = get_redis()
    if r is None:
        return []
    pairs = await r.zrange(key, 0, -1, withscores=True)
    return [(member, score) for member, score in pairs]


async def sorted_set_remove(key: str, *members: str) -> int:
    """Atomic multi-member ZREM. Returns how many of `members` were
    actually present and removed — callers use this as an all-or-nothing
    claim: a caller only 'wins' a pairing if this equals len(members)."""
    r = get_redis()
    if r is None:
        return 0
    if not members:
        return 0
    return await r.zrem(key, *members)


# open for this side" presence tracking (see _side_presence_delta in
# match_stream.py). Returns None when Redis isn't configured so callers
# fall back to a local in-process counter, which is authoritative in that
# mode anyway (only one worker exists).
# ---------------------------------------------------------------------------

async def counter_incrby(key: str, amount: int = 1, ttl_seconds: int | None = None) -> int | None:
    r = get_redis()
    if r is None:
        return None
    val = await r.incrby(key, amount)
    if val < 0:
        # Defensive: never let a stale/duplicate decrement push a presence
        # counter negative and get stuck reading "still connected".
        await r.set(key, 0)
        val = 0
    if ttl_seconds:
        await r.expire(key, ttl_seconds)
    return val


async def counter_get(key: str) -> int | None:
    r = get_redis()
    if r is None:
        return None
    raw = await r.get(key)
    return int(raw) if raw is not None else 0


# ---------------------------------------------------------------------------
# Distributed lock — one worker "wins" ownership of a given match's
# simulation loop. SET NX EX is atomic in Redis, so exactly one concurrent
# caller across every worker gets True.
# ---------------------------------------------------------------------------

async def try_acquire_owner(key: str, ttl_seconds: int, owner_id: str | None = None) -> str | None:
    """Returns an opaque owner token if this call won the lock, else None.
    Falls back to always-True/local-uuid when Redis isn't configured,
    since in-process there's only ever one candidate owner anyway (the
    single worker that's running)."""
    r = get_redis()
    token = owner_id or str(uuid.uuid4())
    if r is None:
        return token
    won = await r.set(key, token, nx=True, ex=ttl_seconds)
    return token if won else None


async def release_owner(key: str, owner_token: str) -> None:
    r = get_redis()
    if r is None:
        return
    # Only release if we're still the recorded owner (avoid clobbering a
    # lock some other worker legitimately re-acquired after our TTL lapsed).
    current = await r.get(key)
    if current == owner_token:
        await r.delete(key)
