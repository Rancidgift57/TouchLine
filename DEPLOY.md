# Deploying TouchLine

Two independent pieces, deployed separately:

| Piece         | What it is                              | Where it goes                          |
|---------------|------------------------------------------|-----------------------------------------|
| **Frontend**  | `index.html` (+ `manifest.json`, `sw.js`, `icon.svg`) — a self-contained PWA | Any static host: Cloudflare Pages, Netlify, Vercel, GitHub Pages |
| **Backend**   | `api/match_stream.py` (FastAPI + WebSockets) + `engine/` + `ml/` | Fly.io (recommended), Render, or Railway — needs a host that keeps long-lived WebSocket connections open, so **not** a serverless/Lambda-style platform |
| **Database**  | `db/schema.sql`                          | Turso (managed libSQL) |

Ship the frontend today (step 1) — it works standalone, no backend required. Steps 2–4 get the real backend live; step 5 is the follow-up work to point the Match Centre at it instead of its offline demo sim.

---

## 1. Frontend — deploy now

It's already a working PWA with zero build step. Easiest path, **Cloudflare Pages**:

```bash
npm install -g wrangler
cd TouchLine
wrangler pages deploy . --project-name=touchline
```

That uploads `index.html`, `manifest.json`, `sw.js`, `icon.svg` (it'll also upload the Python/`ml/` folders harmlessly — to keep the deploy clean, put just the 4 frontend files in their own folder first):

```bash
mkdir -p web-deploy
cp index.html manifest.json sw.js icon.svg web-deploy/
wrangler pages deploy web-deploy --project-name=touchline
```

Alternatives, same idea (drag-and-drop the `web-deploy` folder in the dashboard):
- **Netlify**: [app.netlify.com/drop](https://app.netlify.com/drop)
- **Vercel**: `vercel web-deploy --prod`
- **GitHub Pages**: push `web-deploy/` contents to a `gh-pages` branch, enable Pages in repo settings

You'll get a URL like `https://touchline.pages.dev` — that's your `FRONTEND_ORIGIN` for CORS in step 4.

---

## 2. Database — Turso

```bash
curl -sSfL https://get.tur.so/install.sh | bash     # install the CLI
turso auth login
turso db create touchline
turso db show touchline --url                        # -> TURSO_DATABASE_URL
turso db tokens create touchline                      # -> TURSO_AUTH_TOKEN
```

Apply the schema:

```bash
export TURSO_DATABASE_URL="libsql://touchline-yourname.turso.io"
export TURSO_AUTH_TOKEN="ey..."
pip install libsql-client --break-system-packages
python -c "import asyncio; from db.turso_client import run_migrations; asyncio.run(run_migrations('db/schema.sql'))"
```

(Optional) seed real players from the Kaggle dataset — see `scripts/load_kaggle_players.py`'s docstring for the download step, then:

```bash
python scripts/load_kaggle_players.py --csv data/players.csv
```

---

## 3. Backend — Fly.io (recommended)

Fly.io keeps machines warm for WebSockets and has a Dockerfile + `fly.toml` already set up in this repo.

```bash
curl -L https://fly.io/install.sh | sh
flyctl auth login
cd TouchLine
flyctl launch --no-deploy      # picks up fly.toml, asks to confirm/rename the app
flyctl secrets set TURSO_DATABASE_URL="libsql://touchline-yourname.turso.io"
flyctl secrets set TURSO_AUTH_TOKEN="ey..."
flyctl secrets set FRONTEND_ORIGIN="https://touchline.pages.dev"
flyctl deploy
```

Your API is now live at `wss://touchline-api.fly.dev/ws/match/{id}` (and `https://touchline-api.fly.dev/health` for a sanity check).

**Alternatives:**
- **Render**: New → Web Service → point at this repo → it detects the `Dockerfile` automatically. Add the same 3 env vars in the dashboard. Render's free tier spins down on idle, which drops any match in progress — fine for testing, not for real usage.
- **Railway**: `railway init && railway up`, then set the env vars in the dashboard. Similar WebSocket support to Fly.

**Not recommended for this backend:** Vercel/Netlify Functions, AWS Lambda, Cloudflare Workers (default) — these are request/response serverless models that don't hold a WebSocket open for the ~2 real minutes a compressed match runs.

---

## 4. Sanity check

```bash
curl https://touchline-api.fly.dev/health
# {"status":"ok"}
```

For the WebSocket itself, `wscat` is the quickest check:

```bash
npm install -g wscat
wscat -c wss://touchline-api.fly.dev/ws/match/some-existing-match-id
```

(You'll need a row in `matches` with that id first — there's no "create random match" endpoint yet; that's part of the wiring work below.)

---

## 5. Connecting the frontend to the live backend

This is now wired up — no extra code needed once both pieces are deployed:

1. Deploy the backend (step 3) and note its URL, e.g. `https://touchline-api.fly.dev`.
2. Open the deployed frontend with an `?api=` param pointing at it, once:
   `https://touchline.pages.dev/?api=https://touchline-api.fly.dev`
   (this is saved to `localStorage`, so you only need the param the first time — after that just visit the plain URL).
3. Open the Match Centre and set up a match as normal. The "Kick off simulation" button becomes **"Kick off (live)"** and a `● LIVE` badge appears whenever `LIVE_API_BASE` is set. Clicking it:
   - `POST`s your starting XI + bench to `/quick-match`, which provisions real `teams`/`players`/`player_rights` rows and returns a `match_id`
   - opens `wss://.../ws/match/{match_id}` (events) and `wss://.../ws/match/{match_id}/tactics` (substitutions)
   - renders the exact same commentary feed / tactical map / player+ball movement / half-time overlay as the offline demo, just driven by the server's ML-blended events instead of local JS
4. Substitutions made through the existing "Make a substitution" drawer are sent over the tactics socket (mapped to the backend's player ids via the id map `/quick-match` returns) instead of just mutating local state — they change the live server-side simulation, per the point in the README about `apply_substitution`.

**Known limitations of this first wiring pass** (fine for testing, worth tightening before real users):
- The AI-controlled away side has no bench provisioned yet (`away_bench: []` in the request), so away-side substitutions aren't available in live mode.
- If `/quick-match` or the WebSocket fails to connect (backend asleep, wrong URL, CORS misconfigured), it falls back to the offline demo automatically and toasts an error — check `FRONTEND_ORIGIN` on the backend and the `?api=` value first.
- Every "Kick off (live)" click provisions brand-new `teams`/`players` rows rather than reusing existing ones — fine for casual/demo use, but a real deployment would want persistent squads tied to logged-in managers instead of re-provisioning from scratch each match.
