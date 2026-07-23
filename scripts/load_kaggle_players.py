"""
load_kaggle_players.py — imports real player data into the `players` table
from the Kaggle dataset:

    https://www.kaggle.com/datasets/maso0dahmed/football-players-data

This is a FIFA/EA-Sports-style ratings export (overall, potential, pace,
shooting, passing, dribbling, defending, physical, position, age, etc.) —
exactly the shape our schema.sql `players` table wants, so this script is
mostly a column-mapping + derived-stat exercise rather than a from-scratch
feature build.

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
1. Download the CSV (requires a free Kaggle account + API token):

     pip install kaggle --break-system-packages
     # place kaggle.json (from kaggle.com/settings -> API -> Create New Token)
     # at ~/.kaggle/kaggle.json
     kaggle datasets download -d maso0dahmed/football-players-data -p data/ --unzip

   This drops one or more CSVs into ./data/. Point --csv at whichever file
   contains the player rows (Kaggle dataset pages sometimes ship more than
   one file — check the "Data Explorer" tab on the dataset page for the
   exact filename before running this, since the loader below only knows
   the *columns*, not the *filename*).

2. Run:

     python scripts/load_kaggle_players.py --csv data/players.csv --limit 3000

   `--dry-run` prints the first few mapped rows without touching the DB —
   use it first to sanity-check the column mapping against whatever the
   actual CSV header looks like (Kaggle uploaders vary column names across
   dataset versions; adjust COLUMN_ALIASES below if `--dry-run` shows
   everything falling back to defaults).

--------------------------------------------------------------------------
COLUMN MAPPING
--------------------------------------------------------------------------
FIFA-style ratings export columns are not perfectly standardized across
Kaggle re-uploads of this dataset. COLUMN_ALIASES below lists every column
name variant we've seen for each schema field; the first alias found in the
CSV header wins. If your copy has a column not listed here, add it to the
relevant list.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.turso_client import get_client  # noqa: E402

# ---------------------------------------------------------------------------
# Column mapping: schema_field -> list of possible source column names
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, list[str]] = {
    "name":        ["name", "long_name", "short_name", "player_name"],
    "age":         ["age"],
    "position":    ["position", "player_positions", "best_position", "club_position"],
    "overall":     ["overall", "overall_rating", "rating"],
    "potential":   ["potential", "potential_rating"],
    "pace":        ["pace", "pac", "sprint_speed", "acceleration"],
    "finishing":   ["finishing", "shooting", "sho"],
    "vision":      ["vision", "passing", "pas"],
    "positioning": ["positioning", "att_position", "attacking_position"],
    "tackling":    ["standing_tackle", "defending", "def", "sliding_tackle"],
    "gk_reflexes": ["gk_reflexes", "goalkeeping_reflexes", "gk_diving"],
    "stamina":     ["stamina", "physic", "physical"],
    "wage":        ["wage_eur", "wage", "value_eur"],
}

# FIFA-style detailed position strings ("CAM", "RM", "CDM", ...) collapsed
# down to the 10 buckets match_engine.py / schema.sql actually use.
POSITION_MAP = {
    "GK": "GK",
    "CB": "CB", "RCB": "CB", "LCB": "CB", "SW": "CB",
    "LB": "LB", "LWB": "LB",
    "RB": "RB", "RWB": "RB",
    "CDM": "DM", "RDM": "DM", "LDM": "DM",
    "CM": "CM", "RCM": "CM", "LCM": "CM",
    "CAM": "AM", "RAM": "AM", "LAM": "AM", "AM": "AM",
    "LM": "LW", "LW": "LW", "LF": "LW",
    "RM": "RW", "RW": "RW", "RF": "RW",
    "ST": "ST", "CF": "ST", "LS": "ST", "RS": "ST",
}


def _first_present(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _map_position(raw: str) -> str:
    if not isinstance(raw, str):
        return "CM"
    first = raw.split(",")[0].strip().upper()
    return POSITION_MAP.get(first, "CM")


def _consistency_from_traits(row: pd.Series) -> int:
    """
    The Kaggle export has no direct 'consistency' rating (that's specific
    to our engine's Gaussian-stdev system), so we derive a plausible one:
    higher overall + higher "international_reputation"/"skill_moves" (if
    present) -> more consistent (proven quality); very young high-potential
    players ("wonderkids") get a wider variance since they're volatile.
    Falls back to a mid-range 55-70 band driven only by overall if none of
    the optional source columns exist.
    """
    overall = float(row.get("overall", 65))
    potential = float(row.get("potential", overall))
    reputation = float(row.get("international_reputation", 2)) if "international_reputation" in row else 2.0

    base = 40 + overall * 0.35 + reputation * 4
    volatility_penalty = max(0.0, (potential - overall) * 0.6)  # big potential gap = boom/bust
    return int(np.clip(base - volatility_penalty, 10, 99))


def load_and_map(csv_path: str, limit: int | None) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, low_memory=False)
    if limit:
        raw = raw.head(limit)

    out = pd.DataFrame()
    out["id"] = [str(uuid.uuid4()) for _ in range(len(raw))]

    name_col = _first_present(raw, COLUMN_ALIASES["name"])
    out["name"] = raw[name_col] if name_col else [f"Player {i}" for i in range(len(raw))]

    age_col = _first_present(raw, COLUMN_ALIASES["age"])
    out["age"] = raw[age_col].fillna(24).astype(int).clip(15, 45) if age_col else 24

    pos_col = _first_present(raw, COLUMN_ALIASES["position"])
    out["position"] = raw[pos_col].apply(_map_position) if pos_col else "CM"

    for field in ("overall", "potential", "pace", "finishing", "vision",
                  "positioning", "tackling", "gk_reflexes", "stamina"):
        col = _first_present(raw, COLUMN_ALIASES[field])
        if col:
            out[field] = pd.to_numeric(raw[col], errors="coerce").fillna(50).clip(1, 100).astype(int)
        else:
            out[field] = 50  # neutral default if this dataset export lacks the column

    # potential can't be below overall in our schema's implied semantics
    out["potential"] = np.maximum(out["potential"], out["overall"])

    wage_col = _first_present(raw, COLUMN_ALIASES["wage"])
    out["wage"] = (pd.to_numeric(raw[wage_col], errors="coerce").fillna(10000) if wage_col else 10000).astype(int)

    out["consistency"] = [
        _consistency_from_traits(pd.Series({**row.to_dict(), "overall": out["overall"][i], "potential": out["potential"][i]}))
        for i, row in raw.iterrows()
    ]

    out["current_form"] = 0.0
    out["fatigue"] = 0.0
    out["injury_risk"] = 0.05
    out["contract_years_left"] = 3
    out["is_youth_product"] = False
    out["generated_season"] = None

    return out


async def insert_players(df: pd.DataFrame, owner_team_id: str | None) -> None:
    client = get_client()
    statements = []
    for _, row in df.iterrows():
        statements.append((
            """
            INSERT INTO players (id, name, age, position, overall, potential, consistency,
                                  pace, finishing, vision, positioning, tackling, gk_reflexes,
                                  stamina, current_form, fatigue, injury_risk,
                                  contract_years_left, wage, is_youth_product, generated_season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["id"], row["name"], int(row["age"]), row["position"],
                int(row["overall"]), int(row["potential"]), int(row["consistency"]),
                int(row["pace"]), int(row["finishing"]), int(row["vision"]),
                int(row["positioning"]), int(row["tackling"]), int(row["gk_reflexes"]),
                int(row["stamina"]), float(row["current_form"]), float(row["fatigue"]),
                float(row["injury_risk"]), int(row["contract_years_left"]),
                int(row["wage"]), bool(row["is_youth_product"]), row["generated_season"],
            ],
        ))
        if owner_team_id:
            statements.append((
                "INSERT INTO player_rights (player_id, owner_team_id) VALUES (?, ?)",
                [row["id"], owner_team_id],
            ))

    from db.turso_client import Statement
    # Batched in chunks (a single multi-thousand-statement batch can hit
    # payload limits on some Turso plans).
    CHUNK = 200
    for i in range(0, len(statements), CHUNK):
        chunk = statements[i:i + CHUNK]
        await client.batch([Statement(sql, args) for sql, args in chunk])
        print(f"  inserted {min(i + CHUNK, len(statements))}/{len(statements)} statements...")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to the downloaded Kaggle CSV")
    ap.add_argument("--limit", type=int, default=None, help="Only import the first N rows")
    ap.add_argument("--owner-team-id", default=None,
                     help="If set, also inserts a player_rights row assigning every imported "
                          "player to this team (useful for seeding a single club's roster; "
                          "omit to import free agents / a full player pool for the transfer market)")
    ap.add_argument("--dry-run", action="store_true", help="Print mapped rows, don't touch the DB")
    args = ap.parse_args()

    print(f"Reading {args.csv} ...")
    df = load_and_map(args.csv, args.limit)
    print(f"Mapped {len(df)} players.")
    print(df[["name", "age", "position", "overall", "potential", "consistency", "finishing", "vision"]].head(10))

    if args.dry_run:
        print("\n--dry-run set: not writing to the database.")
        return

    print("\nInserting into Turso...")
    asyncio.run(insert_players(df, args.owner_team_id))
    print("Done.")


if __name__ == "__main__":
    main()
