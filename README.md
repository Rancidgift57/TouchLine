# TouchLine

A browser-based football management sim: squad building, a transfer market
with an ML fairness/anti-cheat check, 48-hour tournaments, and a live Match
Centre whose simulation blends a rule-based/Gaussian engine with two trained
PyTorch nets (shot quality, and pass/dribble/shot decision-making) — playable
solo against a bot, or as a **real two-player match** against a friend.

---

## Contents

- [File tree](#file-tree)
- [Core gameplay](#core-gameplay)
- [Match Centre & the ML models](#match-centre--the-ml-models)
- [ML model performance](#ml-model-performance)
- [Two-player Quick Match](#two-player-quick-match)
- [Running it locally](#running-it-locally)
- [Deployment](#deployment)
- [Troubleshooting: CORS / "Failed to fetch"](#troubleshooting-cors--failed-to-fetch)
- [Known limitations](#known-limitations)

---

## File tree

```
TouchLine/
├── index.html                      # Full SPA (auth, squad, market, tournaments, Match Centre UI)
├── manifest.json / sw.js / icon.svg
├── requirements.txt                 # fastapi, uvicorn, libsql-client, torch, scikit-learn, ...
│
├── engine/                          # Pure-Python match logic (no I/O)
│   ├── match_engine.py              #   Gaussian rolls + chaos-die + momentum + ML blending
│   ├── ml_bridge.py                 #   Loads/wraps the two trained nets for the engine
│   └── market_ml.py                 #   Trade-offer fairness / anti-cheat scoring
│
├── api/
│   └── match_stream.py              # FastAPI app: auth, quick-match provisioning, the
│                                     # two-player friend-match lobby, and the WebSocket
│                                     # match streamer (see below)
│
├── db/
│   ├── schema.sql                   # Turso/libSQL schema (players, teams, matches, trades, ...)
│   └── turso_client.py              # DB client + ACID trade settlement
│
├── ml/                              # Model training + trained weights
│   ├── xg_model_architecture.py     # Trains the xG net on StatsBomb shot data
│   ├── decision_model_architecture.py # Trains the Pass/Dribble/Shot decision net
│   ├── model_usage_demo.py          # Standalone CLI demo (no DB/web needed)
│   ├── xg_model.pth / xg_scaler.pkl
│   └── decision_model.pth / decision_scaler.pkl
│
├── scripts/
│   └── load_kaggle_players.py       # ETL: Kaggle football-players-data CSV -> players table
│
└── data/                            # (empty) drop the downloaded Kaggle CSV here
```

`statsbomb_shots_data.csv` and `statsbomb_decision_data.csv` (the cached
training sets) are left out of this delivery to keep it light —
`ml/xg_model_architecture.py` / `ml/decision_model_architecture.py` will
re-download them via `statsbombpy` on first run and cache them locally.

---

## Core gameplay

- **Squad building** — pick your starting XI and bench from a synthetic
  league of clubs (or real players seeded via `scripts/load_kaggle_players.py`).
- **Transfer market** — propose/counter trade offers, with `engine/market_ml.py`
  scoring each offer's fairness and flagging lopsided ones for review.
- **Tournaments** — 48-hour, 4-team group → final format; join public ones,
  create your own, or open a **Friends Room** (client-side lobby with a
  shareable code) to fill a bracket with people you know.
- **Match Centre** — the live matchday screen: scoreboard, momentum meter,
  a tactical map with real player + ball movement, a commentary feed, and
  a half-time break where you can make substitutions that actually change
  the outcome (not cosmetic — see below).

---

## Match Centre & the ML models

`engine/match_engine.py` runs the simulation minute-by-minute: a Gaussian
performance roll per duel, a chaos d100 for upsets, and a decaying momentum
tug-of-war between the two sides. Two trained PyTorch nets are blended into
that, via `engine/ml_bridge.py`:

1. **Decision net** — given a created chance, decides whether it actually
   becomes a shot attempt (vs. getting recycled into another pass/dribble).
   This sits on top of, not instead of, the existing Gaussian creation duel.
2. **xG net** — contributes to both the reported xG number and the finishing
   duel's win probability, blended alongside player attributes
   (finishing/consistency), momentum, and the chaos die.

If `torch` or the `.pth` weight files aren't available, the engine degrades
gracefully to pure rule-based/Gaussian mode automatically — no crash.

**Tactics feed the ML layer, not just the Gaussian one.** The trained nets
only ever see shot geometry (position/angle/pressure) — on their own they
have no way to notice a manager pushed the mentality dial mid-match.
`ml_bridge.decide()` now takes the attacking team's live `mentality` and
blends a tilt into the trained decision probabilities (attacking shifts
mass toward Shot, defensive shifts it toward Pass), and `match_engine.py`
applies mentality as a genuine two-sided trade-off rather than a one-sided
bonus: it raises your own chance-creation rate when you're on the front
foot, but also raises the *opponent's* chance rate against you (thinner
defensive line), and stiffens/softens the individual creation duel via the
defending side's effective positioning stat. A mentality of 0 reproduces
the model's untouched output.

Substitutions are **not cosmetic**: `apply_substitution()` mutates a
`TeamSnapshot.lineup` in place, and `simulate_match` re-reads
`team.outfield()`/`goalkeeper()` fresh every minute, so any caller holding
that same object sees the change on the very next duel. For matches
simulated ahead of time in a batch job (no one watching live),
`api/match_stream.replay_with_substitution()` re-runs the same RNG seed and
applies the sub at the requested minute — byte-for-byte identical up to that
point, genuinely different after.

---

## Offline demo pacing

Playing solo without a live backend runs `index.html`'s self-contained
`simTick()` loop instead of the WebSocket streamer. Its per-minute delay is
the `SIM_TICK_MS` constant (1100ms), tuned so 90 ticks plus the 20s
half-time pause land on ~2 real minutes — matching what the live backend
(`SECONDS_PER_GAME_MINUTE` in `api/match_stream.py`) already delivers for a
real match. (This used to be hardcoded at 260ms, worth calling out since it
made the whole 90-minute match blow by in under 30 real seconds.)

---

## ML model performance

Both nets are trained with 5-fold stratified cross-validation on StatsBomb
open event data.

### xG model (`ml/xg_model_architecture.py`)

Shot-quality classifier — 5,381 shots, 18 engineered features (shot location,
angle, body part, assist type, pressure, etc.), scored by ROC-AUC.

| Fold | Val Loss | Val ROC-AUC |
|------|---------:|------------:|
| 1    | 0.9163   | 0.8255      |
| 2    | 0.9801   | 0.7952      |
| 3    | 0.9532   | 0.7992      |
| 4    | 0.9552   | 0.7889      |
| 5    | 0.9462   | 0.8008      |

**Final average ROC-AUC: 0.8019 ± 0.0125**

### Decision model (`ml/decision_model_architecture.py`)

Pass / Dribble / Shot classifier — 71,802 in-possession actions, 17 features,
scored by accuracy.

| Fold | Val Accuracy |
|------|-------------:|
| 1    | 91.69%       |
| 2    | 90.59%       |
| 3    | 89.79%       |
| 4    | 88.38%       |
| 5    | 90.91%       |

**Final average accuracy: 90.27% ± 1.12%**

Final-fold classification breakdown (14,360 held-out actions) — this is the
one to read carefully, since the dataset is heavily imbalanced toward passes:

| Class        | Precision | Recall | F1-score | Support |
|--------------|----------:|-------:|---------:|--------:|
| Pass (0)     | 0.99      | 0.91   | 0.95     | 13,703  |
| Dribble (1)  | 0.24      | 0.68   | 0.36     | 359     |
| Shot (2)     | 0.39      | 0.93   | 0.55     | 298     |
| **Accuracy** |           |        | **0.91** | 14,360  |
| Macro avg    | 0.54      | 0.84   | 0.62     | 14,360  |
| Weighted avg | 0.96      | 0.91   | 0.93     | 14,360  |

**Reading this honestly:** overall accuracy (90–91%) looks strong, but it's
inflated by the Pass class dominating the support (13,703 of 14,360 actions).
Recall on Shot and Dribble is actually high (0.93 / 0.68) — the model rarely
*misses* a real shot or dribble — but precision is low (0.39 / 0.24), meaning
it also flags a lot of passes as shots/dribbles that weren't. In practice
this biases `match_engine.py` toward *more* shot attempts being created from
borderline chances than a human would call, which is a reasonable trade-off
for an engine that's aiming for watchable matches, but worth knowing if
you're tuning `ml_bridge.py`'s decision threshold — pulling it toward Pass
would trade away some of that shot recall for cleaner precision.

---

## Live Trade (real two-player trading)

`engine/market_ml.py`'s fairness scoring and `db/turso_client.py`'s ACID
`execute_trade()` existed from the start, but nothing used to expose them —
the Transfer Market tab only ever negotiated against a synthetic, client-
seeded inbox, so two real managers could never actually trade. That's now
wired end-to-end for the two-player **friend match** flow (trading needs a
real opposing `team_id`, which only exists once both sides are in a live
match together — see "Known limitations" below):

- `POST /trade/propose`, `/trade/{id}/counter`, `/trade/{id}/accept`,
  `/trade/{id}/decline` — the negotiation state machine, each one
  re-running `score_trade()` server-side so the fairness/anti-cheat check
  can't be bypassed by a tampered client.
- `GET /trade/box/{team_id}` — poll fallback.
- `WS /ws/trades/{team_id}` — live push so both managers see a propose/
  counter/accept land instantly, mirroring the friend-match lobby's
  one-shared-channel pattern.
- `GET /team/{team_id}/roster` — lets your browser see what the opponent's
  *real* squad actually has (name/position/overall/valuation inputs only,
  not full match internals), since the tactical map itself only shows
  their side as placeholder dots.

In the Match Centre, once a friend match is live, a **🔁 Live Trade**
button appears next to the mentality/substitution controls.

---

## Two-player Quick Match

Real matchmaking between two people, not the offline bot sim. Requires the
live backend (`?api=` param or `LIVE_API_BASE` set — see
[Deployment](#deployment)).

**Flow:**

1. **Host** clicks *"🤝 Play vs. a Friend"* → *Create room* — their current
   starting XI + bench is sent to `POST /friend-match/create`, which returns
   a 6-character room code and a secret host token.
2. **Guest** clicks the same button → *Join room*, enters the code —
   `POST /friend-match/join` validates the code, attaches their squad to the
   same lobby, and returns a secret guest token.
3. Both browsers hold open `/ws/lobby/{pending_id}` and can toggle **Ready**.
   The match is **only** provisioned and kicked off once *both* sides have
   marked ready — not the instant either request lands.
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
simulate_match loop and racing over the same substitution queue.

**Known limitation:** each browser only has full player data (names,
attributes) for its own squad — the opponent's tactical-map dots render as
generic placeholders. Goals/commentary still show real names correctly,
since the backend renders that description text server-side and sends it
to both sides as plain text.

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

# open index.html (the SPA also runs a fully self-contained offline demo
# sim if you're not wiring up the WebSocket backend)
```

To point a locally-opened `index.html` at a running backend without editing
code: `index.html?api=http://localhost:8000` (saved to `localStorage`, so
you only need the param once).

---

## Deployment

See `DEPLOY.md` for the full walkthrough (frontend on a static host, backend
on Fly.io/Render/Railway, database on Turso). Short version:

```bash
# Backend env vars (Render dashboard / flyctl secrets / railway variables):
TURSO_DATABASE_URL=libsql://your-db.turso.io
TURSO_AUTH_TOKEN=ey...
FRONTEND_ORIGIN=https://your-frontend-url        # exact match, no trailing slash
```

Open the deployed frontend once with `?api=https://your-backend-url` — it's
remembered in `localStorage` after that.

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
   Free-tier hosts (Render especially) spin down on idle; the first request
   after that can take 30–60s, which the browser reports as a generic
   network/CORS failure rather than "still loading." Also confirm it isn't
   crash-looping — check host logs for a startup `KeyError` from
   `get_client()` if `TURSO_DATABASE_URL` isn't set.

2. **Does `FRONTEND_ORIGIN` exactly match your frontend's real origin?**
   Hit the backend's root `/` — it echoes back `cors_origins` from what it
   actually parsed at boot. Compare that character-for-character against
   the `Origin` header on the failing request in devtools (not the address
   bar — they can differ, e.g. behind a proxy or CDN). Common mismatches:
   a trailing `/`, `http` vs `https`, or the env var only picked up after a
   redeploy (it's read once at process start into `_frontend_origins`).
   Multiple allowed origins are comma-separated:
   `FRONTEND_ORIGIN=https://a.example.com,https://b.example.com`.

3. **Stale preflight cache.** The browser caches a successful `OPTIONS`
   preflight for up to `max_age` (10 minutes by default). If you fixed the
   env var *after* a failing preflight got cached, you'll keep seeing the
   old error until it expires — test in an incognito window to rule this
   out immediately.

4. **Bypass the browser and check the raw response:**
   ```bash
   curl -i -X OPTIONS https://your-backend/quick-match \
     -H "Origin: https://your-frontend" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type"
   ```
   If `Access-Control-Allow-Origin` is present here but the browser still
   fails, the problem is a browser-side cache or the page is actually
   loading from a different origin than expected (common with
   preview-deployment URLs that change per-push). If it's genuinely absent
   here too, it's a server-side config issue (step 2) or the host's proxy
   is mishandling `OPTIONS` specifically — try the same request against
   `/health` (a `GET`, no preflight) to isolate whether it's `OPTIONS`-
   specific.

5. **If the route itself is erroring** (preflight succeeds but the real
   `POST`/`GET` doesn't), test it directly:
   ```bash
   curl -i -X POST https://your-backend/auth/signup \
     -H "Origin: https://your-frontend" -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"testpass","manager_name":"Test","club_name":"Test FC"}'
   ```
   A `500` with a JSON `{"error": ...}` body (from the app's catch-all
   exception handler) will name the actual failure directly — almost always
   a missing/misconfigured `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN`.

**Not recommended as a backend host at all:** Cloudflare Workers, Vercel/
Netlify Functions, AWS Lambda — these are request/response serverless
models that can't hold a WebSocket open for the ~2 real minutes a
compressed match runs, and are a common source of exactly this class of
error when someone accidentally deploys `api/match_stream.py` to one. If
your frontend's origin keeps changing between attempts (e.g. a Workers
preview URL), that's the underlying cause, not `FRONTEND_ORIGIN` itself —
pin down one canonical frontend URL first.

---

## Known limitations

- The two-player friend-match lobby and `_MATCH_SIDE_TOKENS`/`_PENDING_SUBS`
  live in an in-process dict (see the comments in `api/match_stream.py`) —
  fine for a single backend worker, but a multi-worker/multi-region
  deployment needs this backed by Redis or a per-match actor instead, or a
  lobby/tactics message can land on a worker that never sees the rest of
  that match's state.
- Every "Kick off (live)" click (solo or friend-match) provisions brand-new
  `teams`/`players` rows rather than reusing existing ones — fine for
  casual/demo use, but a real deployment would want persistent squads tied
  to logged-in managers instead of re-provisioning from scratch each match.
- A disconnected friend-match player's socket dropping doesn't currently
  pause or forfeit the match — the simulation keeps running server-side for
  whoever's still connected.
