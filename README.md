<div align="center">

# ⚽ TOUCHLINE

### *A football club, run from your browser tab.*

**Build a squad. Work the transfer market. Watch every kick.**

`FastAPI` · `WebSockets` · `PyTorch` · `Turso/libSQL` · `Vanilla JS` — no build step, no bloat

[![Status: Live](https://img.shields.io/badge/status-live-2ea043)](#running-it-locally)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)](#running-it-locally)
[![Two-Player: Real](https://img.shields.io/badge/two--player-real--time-orange)](#two-player-quick-match)
[![ML: Trained](https://img.shields.io/badge/ML-2%20trained%20nets-9146FF)](#the-brain-behind-the-ball)

</div>

---

Most "football manager in a browser" side projects fake it: a random-number
generator dressed up in a scoreboard, a commentary feed that's really just
`Math.random()` wearing a jersey. **Touchline doesn't.** Every match is a
minute-by-minute simulation — a Gaussian duel here, a chaos roll there, two
trained neural networks quietly voting on what a footballer would actually
do with the ball — streamed to your browser over a real WebSocket, live,
whether you're playing a bot or a friend three time zones away.

This README is the whole map: what it does, how it's built, why certain
decisions were made the hard way instead of the easy way, and how to run
the whole thing yourself.

<img width="1917" height="868" alt="image" src="https://github.com/user-attachments/assets/30821235-6ea7-47b5-9dd8-b4ce336453b2" />
---

## Contents

- [The pitch](#the-pitch)
- [File tree](#file-tree)
- [Core gameplay](#core-gameplay)
- [The brain behind the ball](#the-brain-behind-the-ball)
- [ML model performance](#ml-model-performance)
- [Two-player Quick Match](#two-player-quick-match)
- [Quick Match Online (skill-based matchmaking)](#quick-match-online-skill-based-matchmaking)
- [Live Trade (real two-player trading)](#live-trade-real-two-player-trading)
- [The database layer](#the-database-layer)
- [Running it locally](#running-it-locally)
- [Deployment](#deployment)
- [Troubleshooting: CORS / "Failed to fetch"](#troubleshooting-cors--failed-to-fetch)
- [Known limitations](#known-limitations)
- [Roadmap ideas](#roadmap-ideas)
- [Contact](#contact)

---

## The pitch

You're a manager. You've got eleven starters, a bench, a balance sheet, and
90 minutes of football that hasn't been written yet — because it gets
*simulated*, live, the moment you hit kickoff. Trade players with real
managers under an ML fairness check that can flag a lopsided deal before it
ever touches the database. Enter a 48-hour tournament, or skip the queue
entirely and get skill-matched into a live 1v1 in seconds. Make a
substitution at minute 61 and watch it change minute 62 — not cosmetically,
*structurally*, because the engine re-reads your lineup fresh every single
tick.

It's built to feel less like a toy and more like infrastructure: real
auth, a real ACID-safe trade ledger, real websocket-authenticated sides so
one player's browser can never touch the other's team, and a match engine
that degrades gracefully instead of crashing when the ML weights aren't
installed.

---

## File tree

```
Touchline/
├── frontend/
│   └── index.html                   # Full SPA (auth, squad, market, tournaments, Match Centre UI)
├── manifest.json / sw.js / icon.svg
├── requirements.txt                  # fastapi, uvicorn, libsql, torch, scikit-learn, ...
│
├── engine/                           # Pure-Python match logic (no I/O)
│   ├── match_engine.py               #   Gaussian rolls + chaos-die + momentum + ML blending
│   ├── ml_bridge.py                  #   Loads/wraps the two trained nets for the engine
│   └── market_ml.py                  #   Trade-offer fairness / anti-cheat scoring
│
├── api/
│   ├── match_stream.py               # FastAPI app: auth, quick-match provisioning, the
│   │                                  # two-player friend-match lobby, and the WebSocket
│   │                                  # match streamer (see below)
│   └── state_backend.py              # Optional Redis-backed shared state for multi-worker deploys
│
├── db/
│   ├── schema.sql                    # Turso/libSQL schema (players, teams, matches, trades, ...)
│   ├── auth.py                       # PBKDF2 password hashing + user/team CRUD
│   └── turso_client.py               # Async DB client wrapper + ACID trade settlement
│
├── ml/                                # Model training + trained weights
│   ├── xg_model_architecture.py      #   Trains the xG net on StatsBomb shot data
│   ├── decision_model_architecture.py#   Trains the Pass/Dribble/Shot decision net
│   ├── model_usage_demo.py           #   Standalone CLI demo (no DB/web needed)
│   ├── xg_model.pth / xg_scaler.pkl
│   └── decision_model.pth / decision_scaler.pkl
│
├── scripts/
│   └── load_kaggle_players.py        # ETL: Kaggle football-players-data CSV -> players table
│
├── test_model_scenarios.py           # Sanity-check harness for the ML-blended engine
└── data/                              # (empty) drop the downloaded Kaggle CSV here
```

`statsbomb_shots_data.csv` and `statsbomb_decision_data.csv` (the cached
training sets) aren't included, to keep this light — `ml/xg_model_architecture.py`
/ `ml/decision_model_architecture.py` will re-download them via `statsbombpy`
on first run and cache them locally.

---

## Core gameplay

- 🧑‍💼 **Squad building** — pick your starting XI and bench from a synthetic
  league of clubs, or a roster of real players seeded via
  `scripts/load_kaggle_players.py`.
- 💰 **Transfer market** — propose and counter trade offers, with
  `engine/market_ml.py` scoring each one's fairness and flagging lopsided
  deals for review before they can settle.
- 🏆 **Tournaments** — a 48-hour, 4-team group-stage-into-final format; join
  a public one, spin up your own, or open a **Friends Room** (a shareable
  code) to fill the bracket with people you actually know.
- 📺 **Match Centre** — the live matchday screen: scoreboard, a momentum
  meter that tugs back and forth, a tactical map with real player and ball
  movement, a scrolling commentary feed, and a half-time break where
  substitutions genuinely change what happens next.

---

## The brain behind the ball

`engine/match_engine.py` runs the simulation minute by minute: a Gaussian
performance roll per duel, a chaos d100 for the moments football is famous
for, and a decaying momentum tug-of-war between the two sides. Blended into
all of that, via `engine/ml_bridge.py`, are two neural nets trained on real
StatsBomb event data:

1. **Decision net** — given a created chance, decides whether it actually
   turns into a shot attempt or gets recycled into another pass/dribble.
   This sits *on top of*, not instead of, the existing Gaussian creation
   duel.
2. **xG net** — feeds both the reported xG number and the finishing duel's
   win probability, blended alongside player attributes (finishing,
   consistency), momentum, and the chaos die.

If `torch` or the `.pth` weight files aren't available, the engine falls
back to pure rule-based/Gaussian mode automatically. No crash, no missing
match — just a slightly less opinionated one.

**Tactics feed the ML layer, not just the Gaussian one.** The trained nets
only ever see shot geometry (position, angle, pressure) — on their own they
have no way to notice a manager pushed the mentality dial mid-match.
`ml_bridge.decide()` takes the attacking team's live mentality and blends a
tilt into the trained decision probabilities (attacking shifts mass toward
Shot, defensive shifts it toward Pass), and `match_engine.py` applies
mentality as a genuine two-sided trade-off: it raises your own
chance-creation rate when you push forward, but also raises the
*opponent's* chance rate against you (a thinner defensive line), and
stiffens or softens the individual creation duel via the defending side's
effective positioning stat. A mentality of 0 reproduces the model's
untouched output.

Substitutions are **not cosmetic**. `apply_substitution()` mutates a
`TeamSnapshot.lineup` in place, and `simulate_match` re-reads
`team.outfield()`/`goalkeeper()` fresh every minute — so any caller holding
that same object sees the change on the very next duel. For matches
simulated ahead of time in a batch job (no one watching live),
`api/match_stream.replay_with_substitution()` re-runs the same RNG seed and
applies the sub at the requested minute: byte-for-byte identical up to that
point, genuinely different after.

**Real-time pacing.** A live match compresses 90 simulated minutes into
roughly two real minutes of watch time (`SECONDS_PER_GAME_MINUTE` in
`api/match_stream.py`), plus a real ~20-second half-time break where
substitutions actually land before kickoff of the second half. Commentary
only fires on "notable" minutes — quiet stretches between them are normal,
by design — so the frontend runs its own clock ticker between events
rather than leaving the scoreboard frozen until the next headline lands.
The offline/no-backend demo mode in `index.html` (`simTick()`) is tuned to
the same ~2-minute rhythm, so solo play and live play feel identical.

---

## ML model performance

Both nets are trained with 5-fold stratified cross-validation on StatsBomb
open event data.

### xG model (`ml/xg_model_architecture.py`)

Shot-quality classifier — 5,381 shots, 18 engineered features (shot
location, angle, body part, assist type, pressure, etc.), scored by
ROC-AUC.

| Fold | Val Loss | Val ROC-AUC |
|------|---------:|------------:|
| 1    | 0.9163   | 0.8255      |
| 2    | 0.9801   | 0.7952      |
| 3    | 0.9532   | 0.7992      |
| 4    | 0.9552   | 0.7889      |
| 5    | 0.9462   | 0.8008      |

**Final average ROC-AUC: 0.8019 ± 0.0125**

### Decision model (`ml/decision_model_architecture.py`)

Pass / Dribble / Shot classifier — 71,802 in-possession actions, 17
features, scored by accuracy.

| Fold | Val Accuracy |
|------|-------------:|
| 1    | 91.69%       |
| 2    | 90.59%       |
| 3    | 89.79%       |
| 4    | 88.38%       |
| 5    | 90.91%       |

**Final average accuracy: 90.27% ± 1.12%**

Final-fold classification breakdown (14,360 held-out actions) — read this
one carefully, since the dataset skews heavily toward passes:

| Class        | Precision | Recall | F1-score | Support |
|--------------|----------:|-------:|---------:|--------:|
| Pass (0)     | 0.99      | 0.91   | 0.95     | 13,703  |
| Dribble (1)  | 0.24      | 0.68   | 0.36     | 359     |
| Shot (2)     | 0.39      | 0.93   | 0.55     | 298     |
| **Accuracy** |           |        | **0.91** | 14,360  |
| Macro avg    | 0.54      | 0.84   | 0.62     | 14,360  |
| Weighted avg | 0.96      | 0.91   | 0.93     | 14,360  |

**Reading this honestly:** overall accuracy (90–91%) looks strong, but it's
inflated by the Pass class dominating the support (13,703 of 14,360
actions). Recall on Shot and Dribble is actually high (0.93 / 0.68) — the
model rarely *misses* a real shot or dribble — but precision is low
(0.39 / 0.24), meaning it also flags a fair number of passes as
shots/dribbles that weren't. In practice this biases `match_engine.py`
toward *more* shot attempts being created from borderline chances than a
human scout would call — a reasonable trade-off for an engine optimizing
for watchable matches, but worth knowing if you're tuning
`ml_bridge.py`'s decision threshold: pulling it toward Pass would trade
away some of that shot recall for cleaner precision.

---

## Two-player Quick Match

Real matchmaking between two people, not the offline bot sim. Requires the
live backend (`?api=` param or `LIVE_API_BASE` set — see
[Deployment](#deployment)).

**Flow:**

1. **Host** clicks *"🤝 Play vs. a Friend"* → *Create room* — their current
   starting XI + bench is sent to `POST /friend-match/create`, which
   returns a 6-character room code and a secret host token.
2. **Guest** clicks the same button → *Join room*, enters the code —
   `POST /friend-match/join` validates the code, attaches their squad to
   the same lobby, and returns a secret guest token.
3. Both browsers hold open `/ws/lobby/{pending_id}` and can toggle
   **Ready**. The match is **only** provisioned and kicked off once *both*
   sides have marked ready — not the instant either request lands.
4. On kickoff the server provisions both squads into real `teams`/`players`
   rows, creates the `matches` row, and pushes each browser a `match_id` +
   its own per-side secret token over the lobby socket. Both browsers then
   open the same `/ws/match/{match_id}` and `/ws/match/{match_id}/tactics`
   sockets used by solo mode.

**Side ownership is enforced server-side**, not just assumed client-side:
the tactics socket requires `?side=home|away&token=...`, checked against a
per-match token pair generated at kickoff. A connection with the wrong or
missing token is refused outright, and once connected, every message's
declared `side` is forced to match the side that socket actually
authenticated as — a substitution claiming the other side is rejected, so
one player's browser can never sub on the other player's team, whether by
bug or a tampered client.

**Shared live simulation, not two independent copies:** `/ws/match/{id}`
runs a single background simulation per `match_id`, fanned out to every
connected viewer (with event replay for whichever side connects a beat
later), instead of each connecting socket accidentally starting its own
`simulate_match` loop and racing over the same substitution queue.

**Known limitation:** each browser only has full player data (names,
attributes) for its own squad — the opponent's tactical-map dots render as
generic placeholders. Goals and commentary still show real names correctly,
since the backend renders that description text server-side and sends it
to both sides as plain text.

---

## League — persistent manager careers

Everywhere else in the app, a match is a one-off: pick an opponent, play,
done. The **League** tab (`frontend/index.html`) is the opposite —  a
career that keeps going: real seasons, a real table, and promotion/
relegation, picking back up exactly where you left off even after you
close the tab.

**Structure:** two divisions of 6 clubs each, seeded from the `division:
0|1` field already on every entry in `CLUBS`. Each season is a single
round-robin per division — 5 matchdays, 3 fixtures each, generated by the
standard "circle method" (`makeRoundRobinFixtures`) so every pair of clubs
meets exactly once, home/away alternating by round.

**Playing a season:**

1. The **League** tab always shows your current matchday: your fixture,
   opponent, and two ways to resolve it — **"⚽ Play this match"** (drops
   you into the live Match Centre, same engine as Quick Match, tagged with
   a `SEASON — MATCHDAY N` banner) or **"⏭ Sim this match & advance"**
   (resolves it instantly via the same statistical model already used to
   auto-resolve tournament group games you're not part of).
2. Either way, the moment your fixture is settled, every *other*
   still-unplayed fixture in that matchday — across **both** divisions —
   auto-resolves the same way. Divisions always stay in lockstep by
   matchday; there's never a table with a half-played round sitting in it.
3. The standings table (`standingsTable()`) sorts by points, then goal
   difference, then goals scored — the standard football tiebreak order,
   same logic the tournament groups already used, generalized to also
   track goals for/against for display. Toggle "My Division" / "Other
   Division" to check how the rest of the pyramid is doing.
4. **End of season:** once every matchday is played, the top 2 of Division
   1 swap places with the bottom 2 of Division 2 (`endOfSeason()`), a
   summary modal announces the champion plus who went up/down, and a fresh
   season starts immediately — new fixtures, table back to zero, same two
   divisions' worth of clubs just reshuffled by result.

**Persistence:** `state.career` (season number, both divisions' fixtures
and current matchday) is saved to `localStorage` under
`touchline_career_v1` on every change, and reloaded on boot
(`ensureCareer()`) — so unlike `LEAGUE` (squads — regenerated fresh every
page load from a deterministic seed) or tournament progress (memory-only,
resets on refresh), your season genuinely carries over between visits.
There's no server-side career storage in this pass — it's a browser-local
save, same tier of persistence as `LIVE_API_BASE`'s saved backend URL.

---

## Quick Match Online (skill-based matchmaking)

A third way into a real two-player match, alongside the offline solo bot
sim and the code-based Friend Match lobby above: click
**"🌐 Quick Match Online"** and the server pairs you with whichever other
*currently waiting* manager has the closest starting-XI level — no code to
share, no picking who you play.

**Flow:**

1. Your browser computes your level as the average `overall` of your
   starting XI, then calls `POST /matchmaking/join` with your squad — this
   returns a `ticket_id` + secret token and adds you to an in-memory queue
   (`api/match_stream.py`'s `_MATCHMAKING_QUEUE`).
2. It opens `/ws/matchmaking/{ticket_id}`, which repeatedly attempts to
   pair you with the closest-level *other* waiting manager, and pushes a
   `searching` status update (elapsed wait, current tolerance, queue size)
   every ~1.5s while none qualifies yet.
3. **Tolerance widens over time**, not just at the moment you join: a
   candidate pairing is accepted once the level gap is within the *wider*
   of either side's own current tolerance band (`_MM_BASE_TOLERANCE` at
   0s, growing by `_MM_GROWTH_PER_SEC` every second waited, capped at
   `_MM_MAX_TOLERANCE`). That means someone who's been queued a long time
   can still pull in an opponent who only just joined, even before that
   fresh arrival's own tolerance has had time to widen.
4. Once a pair is found, the server reuses the same
   `_provision_and_create_match` the Friend Match lobby uses — real
   `teams`/`players` rows, a real `matches` row, and per-side tokens
   registered in `_MATCH_SIDE_TOKENS` — and pushes a `kickoff` message
   (match_id, team ids, player maps, your side, your token, and the
   opponent's name/level) to both sockets. The frontend hands this straight
   to the same `beginFriendMatch()` used by the lobby flow, so the live
   match itself — side-token enforcement, shared simulation,
   substitutions, Live Trade — behaves identically no matter which route
   got you into it.
5. **Cancelling:** closing the modal (or the "Cancel search" button) calls
   `POST /matchmaking/leave`, which removes your ticket from the queue — as
   long as you haven't already been paired (a pairing that lands in the
   same instant as a cancel just proceeds; there's no way to un-pair once
   both sides have been provisioned).

**Known limitation:** like the Friend Match lobby, the queue lives in an
in-process dict, so this only matches people whose requests land on the
same backend worker — a multi-worker/multi-region deployment needs this
backed by something shared (Redis, a dedicated matchmaking service)
instead.

---

## Live Trade (real two-player trading)

`engine/market_ml.py`'s fairness scoring and `db/turso_client.py`'s ACID
`execute_trade()` existed from the start, but for a while nothing exposed
them end-to-end — the Transfer Market tab only ever negotiated against a
synthetic, client-seeded inbox, so two real managers could never actually
trade. That's now wired into the two-player **friend match** flow (trading
needs a real opposing `team_id`, which only exists once both sides are in
a live match together — see [Known limitations](#known-limitations)):

- `POST /trade/propose`, `/trade/{id}/counter`, `/trade/{id}/accept`,
  `/trade/{id}/decline` — the negotiation state machine, each one
  re-running `score_trade()` server-side so the fairness/anti-cheat check
  can't be bypassed by a tampered client.
- `GET /trade/box/{team_id}` — poll fallback.
- `WS /ws/trades/{team_id}` — live push so both managers see a propose,
  counter, or accept land instantly, mirroring the friend-match lobby's
  one-shared-channel pattern.
- `GET /team/{team_id}/roster` — lets your browser see what the opponent's
  *real* squad actually has (name/position/overall/valuation inputs only,
  not full match internals), since the tactical map itself only shows
  their side as placeholder dots.

Once a friend match is live, a **🔁 Live Trade** button appears next to the
mentality/substitution controls in the Match Centre.

Every trade settles atomically through `execute_trade()`: cash moves
between team balances, `player_rights.owner_team_id` flips for every player
in the deal, buyback/sell-on clauses get written, and the offer is marked
resolved — all inside a single transaction, so a failure partway through
(say, insufficient funds discovered mid-settlement) rolls back the *entire*
trade. No manager ever ends up with a player gone and no cash to show for
it.

---

## The database layer

Touchline runs on [Turso](https://turso.tech) (managed libSQL — SQLite,
distributed). The client wrapper in `db/turso_client.py` uses the
officially supported **`libsql`** package, talking HTTP rather than raw
websockets — a deliberate choice, since Turso retired websocket-based
Python driver support for AWS-hosted databases. Because `libsql`'s
connection API is synchronous (sqlite3-style), the wrapper drives it
through `asyncio.to_thread` behind a small async-compatible adapter, so the
rest of the codebase gets to keep writing `await client.execute(...)` and
`await client.batch([...])` without ever touching a thread itself.

Schema migrations (`db/schema.sql`) are additive and idempotent —
`ensure_auth_schema()` / `ensure_match_schema()` run on every boot and
no-op safely if a column or index already exists, so redeploying never
requires a manual migration step.

---

## Running it locally

```bash
pip install -r requirements.txt --break-system-packages

# (optional) train the models — or skip this and the engine runs rule-based only
cd ml && python xg_model_architecture.py && python decision_model_architecture.py && cd ..

# (optional) seed real players from the Kaggle dataset
python scripts/load_kaggle_players.py --csv data/players.csv --dry-run

# apply schema + start the API
python -c "import asyncio; from db.turso_client import run_migrations; asyncio.run(run_migrations('db/schema.sql'))"
uvicorn api.match_stream:app --reload

# open frontend/index.html — the SPA also runs a fully self-contained
# offline demo sim if you're not wiring up the WebSocket backend
```

To point a locally-opened `index.html` at a running backend without editing
code: `frontend/index.html?api=http://localhost:8000` (saved to
`localStorage`, so you only need the param once).

---

## Deployment

Frontend on a static host, backend on Fly.io/Render/Railway, database on
Turso. Short version:

```bash
# Backend env vars (Render dashboard / flyctl secrets / railway variables):
TURSO_DATABASE_URL=libsql://your-db.turso.io
TURSO_AUTH_TOKEN=ey...
FRONTEND_ORIGIN=https://your-frontend-url        # exact match, no trailing slash
```

Open the deployed frontend once with `?api=https://your-backend-url` — it's
remembered in `localStorage` after that.

**Not recommended as a backend host at all:** Cloudflare Workers,
Vercel/Netlify Functions, AWS Lambda — these are request/response
serverless models that can't hold a WebSocket open for the ~2 real minutes
a compressed match runs.

---

## Troubleshooting: CORS / "Failed to fetch"

If the browser reports something like:

```
Access to fetch at '.../quick-match' from origin 'https://your-frontend'
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header
is present on the requested resource.
```

This is very often **not actually a CORS bug** — `api/match_stream.py`'s
comments call this out directly. Work through these in order:

1. **Is the backend actually up?** Hit `/health` directly in a new tab.
   Free-tier hosts (Render especially) spin down on idle; the first
   request after that can take 30–60s, which the browser reports as a
   generic network/CORS failure rather than "still loading." `GET
   /health?db=1` additionally round-trips a `SELECT 1` to Turso and reports
   a specific diagnosis (bad token, unreachable database, missing env var)
   instead of a stack trace.

2. **Does `FRONTEND_ORIGIN` exactly match your frontend's real origin?** Hit
   the backend's root `/` — it echoes back `cors_origins` from what it
   actually parsed at boot. Compare that character-for-character against
   the `Origin` header on the failing request in devtools. Common
   mismatches: a trailing `/`, `http` vs `https`, or the env var only
   picked up after a redeploy. Multiple allowed origins are
   comma-separated: `FRONTEND_ORIGIN=https://a.example.com,https://b.example.com`.

3. **Stale preflight cache.** The browser caches a successful `OPTIONS`
   preflight for up to `max_age` (10 minutes by default). Test in an
   incognito window to rule this out immediately.

4. **Bypass the browser and check the raw response:**
   ```bash
   curl -i -X OPTIONS https://your-backend/quick-match \
     -H "Origin: https://your-frontend" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type"
   ```
   If `Access-Control-Allow-Origin` is present here but the browser still
   fails, it's a browser-side cache or a mismatched origin. If it's
   genuinely absent, it's a server-side config issue (step 2) or the
   host's proxy is mishandling `OPTIONS` specifically.

5. **If the route itself is erroring** (preflight succeeds but the real
   request doesn't), test it directly:
   ```bash
   curl -i -X POST https://your-backend/auth/signup \
     -H "Origin: https://your-frontend" -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"testpass","manager_name":"Test","club_name":"Test FC"}'
   ```
   A `500` with a JSON `{"error": ...}` body will name the actual failure
   directly — almost always a missing/misconfigured `TURSO_DATABASE_URL` /
   `TURSO_AUTH_TOKEN`.

---

## Known limitations

- The two-player friend-match lobby and `_MATCH_SIDE_TOKENS`/`_PENDING_SUBS`
  live in an in-process dict (see the comments in `api/match_stream.py`) —
  fine for a single backend worker, but a multi-worker/multi-region
  deployment needs this backed by Redis or a per-match actor instead, or a
  lobby/tactics message can land on a worker that never sees the rest of
  that match's state.
- Every "Kick off (live)" click (solo or friend-match) provisions
  brand-new `teams`/`players` rows rather than reusing existing ones —
  fine for casual/demo use, but a real deployment would want persistent
  squads tied to logged-in managers instead of re-provisioning from
  scratch each match.
- A disconnected friend-match player's socket dropping doesn't currently
  pause or forfeit the match instantly — there's a grace period before a
  forfeit is called, during which the simulation keeps running server-side
  for whoever's still connected.

---

## Roadmap ideas

Things that would be natural next steps for anyone picking this up:

- Persistent manager careers — seasons, promotion/relegation, a real
  league table instead of one-off matches.
- Player development over time (aging curves, training gains) rather than
  a static `overall`.
- A proper matchmaking backend (Redis-backed queue) so Quick Match Online
  works correctly across multiple workers/regions.
- Expanding the ML layer: a set-piece-specific model, or an expected-
  threat (xT) possession-value model layered under the existing xG/decision
  pair.

---

## Contact

- **Email:** nnair7598@gmail.com
- **LinkedIn:** [linkedin.com/in/nikhil-nair-809248286](https://www.linkedin.com/in/nikhil-nair-809248286)

<div align="center">

*Built by someone who'd rather simulate the beautiful game properly than fake it.*

**Thank you for reading this far — now go kick off.** ⚽

</div>
