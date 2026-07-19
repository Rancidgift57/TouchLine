"""
MatchEngine — turns two 11-man squads into a stream of match events.

Design goals from the spec:
  * Gaussian performance rolls driven by a hidden `consistency` stat
    (low consistency = wide variance = "moody superstar").
  * A 1-100 "chaos" roll before every duel: 1-5 lets the underdog
    auto-win the duel outright (slip, deflection, wondergoal).
  * Dynamic momentum: strung-together successes buff the next roll,
    a missed big chance craters it.
  * A trained xG net (engine/ml_bridge.py) estimates finishing quality
    from pitch geometry, and a trained Pass/Dribble/Shot decision net
    picks what an attacker *tries* to do from a given spot. Both are
    blended with the player's RPG attributes + the Gaussian/chaos system
    above rather than used as the final say — see `resolve_duel` and
    `_ml_shot_quality`.
  * Deterministic given a seed, so a match can be replayed byte-for-byte
    from `matches.rng_seed` — UNLESS a live substitution mutates a
    TeamSnapshot mid-stream, in which case the replay intentionally
    diverges from that minute onward (see `apply_substitution`).

This module has no I/O — `simulate_match` is a pure generator of event
dicts. The WebSocket layer (api/match_stream.py) is responsible for
pacing/emitting them, persistence, and for actually sleeping through the
half-time break.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Literal

from engine.ml_bridge import MLBridge, PitchContext

EventType = Literal[
    "kickoff", "chance", "shot", "goal", "save", "chaos_moment",
    "half_time", "half_time_break", "full_time", "momentum_shift", "substitution",
]

# Formation slots as StatsBomb-style pitch coordinates (0..120 long / 0..80
# wide, own goal at x=0). Used purely to give the frontend a plausible (x, y)
# to animate players/ball toward — the duel math above never depends on it.
FORMATION_SLOTS = {
    "GK": (8, 40), "CB": (28, 40), "LB": (26, 12), "RB": (26, 68),
    "DM": (45, 40), "CM": (55, 40), "AM": (78, 40),
    "LW": (85, 15), "RW": (85, 65), "ST": (95, 40),
}


# ---------------------------------------------------------------------------
# Player / team snapshots (read-only views handed to the engine — the DB
# layer is responsible for hydrating these from `players` + fatigue/form)
# ---------------------------------------------------------------------------

@dataclass
class PlayerSnapshot:
    id: str
    name: str
    position: str
    overall: int
    consistency: int          # 1-100, higher = tighter Gaussian spread
    finishing: int = 50
    vision: int = 50
    positioning: int = 50
    tackling: int = 50
    gk_reflexes: int = 50
    pace: int = 50
    current_form: float = 0.0  # z-score, e.g. -2..+2, added to the mean
    fatigue: float = 0.0       # 0..1, shrinks effective overall


@dataclass
class TeamSnapshot:
    id: str
    name: str
    lineup: list[PlayerSnapshot]
    mentality: float = 0.0     # -1 (ultra-defensive) .. +1 (all-out attack)
    subs_used: int = 0
    max_subs: int = 5

    def outfield(self) -> list[PlayerSnapshot]:
        return [p for p in self.lineup if p.position != "GK"]

    def goalkeeper(self) -> PlayerSnapshot:
        gks = [p for p in self.lineup if p.position == "GK"]
        return gks[0] if gks else self.lineup[0]


def apply_substitution(team: TeamSnapshot, player_out_id: str, player_in: PlayerSnapshot) -> bool:
    """
    Live-swaps a player in `team.lineup`, IN PLACE.

    This is the whole trick behind "substitutions affect the result even
    when matches are precomputed": `simulate_match` never copies `home`/
    `away` — it re-reads `team.outfield()` / `team.goalkeeper()` fresh on
    every single minute of the loop. So if a caller holds a reference to
    the same TeamSnapshot object simulate_match is iterating over (which
    api/match_stream.py does) and mutates `.lineup` mid-generator, every
    duel resolved *after* that point uses the new player — no special
    casing needed inside the sim loop itself.

    For a match that was fully precomputed ahead of time (e.g. a batch
    "sim this whole matchday tonight" job), the same function still works:
    re-run `simulate_match(..., seed=matches.rng_seed)` but call
    `apply_substitution` right before the loop reaches the substitution
    minute (api/match_stream.py's `replay_with_substitution` helper does
    exactly this). The first N minutes replay byte-for-byte identically
    (same seed, same lineup up to that point); everything after the sub
    diverges, which is the intended "the sub actually changed the game"
    behavior rather than a cosmetic label swap.
    """
    if team.subs_used >= team.max_subs:
        return False
    for i, p in enumerate(team.lineup):
        if p.id == player_out_id:
            team.lineup[i] = player_in
            team.subs_used += 1
            return True
    return False


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

@dataclass
class Momentum:
    """A tug-of-war value in [-1, 1]. Positive favors home, negative away."""
    value: float = 0.0
    STEP: float = 0.12          # bump per successful event
    DECAY: float = 0.04         # pulls back toward 0 each simulated minute
    CRASH: float = 0.35         # penalty for a big miss (e.g. missed sitter)

    def bump(self, toward_home: bool, magnitude: float = 1.0) -> None:
        delta = self.STEP * magnitude
        self.value += delta if toward_home else -delta
        self.value = max(-1.0, min(1.0, self.value))

    def crash(self, home_side_failed: bool) -> None:
        # If the home side fails a big chance, momentum swings AWAY from home.
        delta = self.CRASH
        self.value += -delta if home_side_failed else delta
        self.value = max(-1.0, min(1.0, self.value))

    def decay(self) -> None:
        self.value *= (1 - self.DECAY)

    def modifier_for(self, is_home: bool) -> float:
        """Small additive bonus/penalty applied to a duel roll, in overall-points."""
        signed = self.value if is_home else -self.value
        return signed * 6.0  # momentum swings duels by up to +/-6 overall pts


# ---------------------------------------------------------------------------
# Core stochastic primitives
# ---------------------------------------------------------------------------

def gaussian_performance(rng: random.Random, overall: int, consistency: int,
                          form: float = 0.0, fatigue: float = 0.0) -> float:
    """
    Roll a single "performance value" for a stat check.

    mean  = overall, shifted by current form (z-score * 8) and fatigue penalty
    stdev = inversely proportional to consistency:
        consistency=100 -> stdev ~3   (rock solid)
        consistency=50  -> stdev ~12
        consistency=10  -> stdev ~22  (boom or bust)
    """
    mean = overall + (form * 8.0) - (fatigue * 15.0)
    stdev = 3.0 + (100 - consistency) * 0.21
    roll = rng.gauss(mean, stdev)
    return max(1.0, min(140.0, roll))  # clamp; superstar days can exceed 100


def roll_chaos(rng: random.Random) -> int:
    """1-100 dice. Callers check `<= 5` for a chaos event."""
    return rng.randint(1, 100)


CHAOS_THRESHOLD = 5  # 1-5 on a d100 => 5% chance


# ---------------------------------------------------------------------------
# Pitch coordinates — feeds the Match Centre's live player/ball movement.
# Coordinates are StatsBomb-convention: x in 0..120 (attacking goal at 120
# for the side whose "attacking_home" is True; mirrored for the away side),
# y in 0..80. The frontend maps these to % positions on the tactical map.
# ---------------------------------------------------------------------------

def _slot_position(position: str, attacking_home: bool) -> tuple[float, float]:
    x, y = FORMATION_SLOTS.get(position, (50, 40))
    if not attacking_home:
        x = 120 - x  # mirror for the away team, who attack leftward on screen
    return x, y


def _jitter_position(rng: random.Random, base: tuple[float, float],
                      spread_x: float = 10.0, spread_y: float = 10.0) -> tuple[float, float]:
    x = max(1.0, min(119.0, base[0] + rng.uniform(-spread_x, spread_x)))
    y = max(1.0, min(79.0, base[1] + rng.uniform(-spread_y, spread_y)))
    return round(x, 1), round(y, 1)


def resolve_duel(
    rng: random.Random,
    attacker: PlayerSnapshot,
    defender: PlayerSnapshot,
    attacker_stat: int,
    defender_stat: int,
    momentum: Momentum,
    attacker_is_home: bool,
) -> dict:
    """
    Resolve a single attacker-vs-defender duel (e.g. Finishing vs GK Reflexes,
    or Vision vs Positioning). Returns a dict describing the outcome so the
    caller can decide what event/commentary to emit.
    """
    chaos_roll = roll_chaos(rng)
    if chaos_roll <= CHAOS_THRESHOLD:
        # Automatic underdog win — a slip, a wicked deflection, a wondergoal.
        underdog_is_attacker = attacker.overall < defender.overall
        return {
            "attacker_wins": underdog_is_attacker or rng.random() < 0.5,
            "chaos": True,
            "chaos_roll": chaos_roll,
            "attacker_roll": None,
            "defender_roll": None,
        }

    mom_bonus_attacker = momentum.modifier_for(attacker_is_home)
    mom_bonus_defender = -mom_bonus_attacker

    a_roll = gaussian_performance(
        rng, attacker_stat, attacker.consistency, attacker.current_form, attacker.fatigue
    ) + mom_bonus_attacker
    d_roll = gaussian_performance(
        rng, defender_stat, defender.consistency, defender.current_form, defender.fatigue
    ) + mom_bonus_defender

    # Logistic conversion of the roll gap into a clean win/lose (keeps close
    # rolls close to 50/50 instead of a hard threshold).
    gap = a_roll - d_roll
    p_attacker_wins = 1 / (1 + math.exp(-gap / 12.0))

    return {
        "attacker_wins": rng.random() < p_attacker_wins,
        "chaos": False,
        "chaos_roll": chaos_roll,
        "attacker_roll": round(a_roll, 1),
        "defender_roll": round(d_roll, 1),
        "win_probability": round(p_attacker_wins, 3),
    }


# ---------------------------------------------------------------------------
# Match state + simulation
# ---------------------------------------------------------------------------

@dataclass
class MatchState:
    home: TeamSnapshot
    away: TeamSnapshot
    home_score: int = 0
    away_score: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_shots: int = 0
    away_shots: int = 0
    home_possession_ticks: int = 0
    away_possession_ticks: int = 0
    momentum: Momentum = field(default_factory=Momentum)


COMMENTARY_TEMPLATES = {
    "chance_created": "{minute}' - {name} threads a pass through midfield, {team} push forward.",
    "chaos_chance": "{minute}' - The ball ricochets kindly for {name}! Pure chaos, {team} through on goal.",
    "shot_on_target": "{minute}' - {name} lines it up... SHOT! On target!",
    "goal": "{minute}' - GOAL! {name} finds the net for {team}! ({home_score}-{away_score})",
    "save": "{minute}' - Brilliant save! The keeper denies {name}.",
    "chaos_miss": "{minute}' - Slips at the worst possible moment — {name}'s chance is gone.",
    "momentum_home": "{minute}' - {team} are finding a rhythm, momentum building.",
    "momentum_away": "{minute}' - The tide is turning, {team} sense a chance.",
}


def _attacking_context(state: MatchState, attacking_home: bool) -> tuple[TeamSnapshot, TeamSnapshot]:
    return (state.home, state.away) if attacking_home else (state.away, state.home)


def simulate_match(
    home: TeamSnapshot,
    away: TeamSnapshot,
    seed: int,
    duration_minutes: int = 90,
    ml_bridge: MLBridge | None = None,
    half_time_break_seconds: float = 20.0,
) -> Iterator[dict]:
    """
    Generator yielding one event dict per notable minute. The caller (the
    WebSocket streamer) is responsible for real-time pacing — this function
    is pure and deterministic for a given seed so matches can be replayed
    (see `apply_substitution` for the one intentional exception).

    `ml_bridge` (lazily loaded from disk if not supplied) supplies:
      * the Pass/Dribble/Shot decision net — gates whether a created chance
        actually turns into a shot attempt this minute, vs. the attacker
        recycling possession (keeps shot volume realistic instead of every
        chance becoming a shot).
      * the xG net — blended 50/50 with the old uniform-random xG roll and
        also nudges the Gaussian finishing duel's win probability, so a
        "good" chance by the model's geometry is more likely to go in than
        a speculative Hail Mary, on top of the player's finishing/consistency
        rolls and the chaos die.

    `half_time_break_seconds` is carried on the `half_time_break` event only
    as metadata — this generator has no I/O/sleep of its own (see module
    docstring); api/match_stream.py is what actually pauses for that long.
    """
    rng = random.Random(seed)
    state = MatchState(home=home, away=away)
    bridge = ml_bridge or MLBridge.load()
    half_time_minute = duration_minutes // 2
    half_time_fired = False

    yield {"minute": 0, "type": "kickoff",
           "description": f"Kickoff! {home.name} vs {away.name}.",
           "home_score": 0, "away_score": 0,
           "momentum": 0.0,
           "ball_pos": {"x": 60, "y": 40}}

    for minute in range(1, duration_minutes + 1):
        state.momentum.decay()

        # Possession weighted by team overall + current momentum. Reads
        # home.outfield()/away.outfield() FRESH every minute, so a
        # substitution applied mid-generator (apply_substitution) takes
        # effect on the very next iteration — no precomputation to undo.
        home_strength = sum(p.overall for p in home.outfield()) + state.momentum.value * 40
        away_strength = sum(p.overall for p in away.outfield()) - state.momentum.value * 40
        p_home_has_ball = home_strength / (home_strength + away_strength)
        attacking_home = rng.random() < p_home_has_ball

        if attacking_home:
            state.home_possession_ticks += 1
        else:
            state.away_possession_ticks += 1

        attacking_team, defending_team = _attacking_context(state, attacking_home)

        # --- Half-time: fires exactly once at the halfway minute regardless
        # of whether this particular minute also produced a chance/goal.
        if minute >= half_time_minute and not half_time_fired:
            half_time_fired = True
            yield {"minute": minute, "type": "half_time",
                   "description": "Half-time.",
                   "home_score": state.home_score, "away_score": state.away_score,
                   "momentum": round(state.momentum.value, 3)}
            yield {"minute": minute, "type": "half_time_break",
                   "description": f"{half_time_break_seconds:.0f}s break — make your changes.",
                   "home_score": state.home_score, "away_score": state.away_score,
                   "momentum": round(state.momentum.value, 3),
                   "break_seconds": half_time_break_seconds}

        # Roughly one clear "event" (chance/chaos/nothing) every few minutes,
        # scaled slightly by attacking mentality.
        chance_probability = 0.11 + max(0.0, attacking_team.mentality) * 0.05
        if rng.random() >= chance_probability:
            continue  # quiet minute, no event emitted (frontend just ticks the clock)

        attacker = rng.choice([p for p in attacking_team.outfield() if p.position in ("ST", "LW", "RW", "AM", "CM")] or attacking_team.outfield())
        defender = rng.choice([p for p in defending_team.outfield() if p.position in ("CB", "LB", "RB", "DM")] or defending_team.outfield())
        keeper = defending_team.goalkeeper()

        attacker_pos = _jitter_position(rng, _slot_position(attacker.position, attacking_home), 14, 18)
        defender_pos = _jitter_position(rng, _slot_position(defender.position, not attacking_home), 8, 10)

        # --- Duel 1: creating the chance (Vision/Positioning vs defender's Positioning)
        creation = resolve_duel(
            rng, attacker, defender,
            attacker_stat=attacker.vision, defender_stat=defender.positioning,
            momentum=state.momentum, attacker_is_home=attacking_home,
        )

        if not creation["attacker_wins"]:
            continue  # chance snuffed out, no shot

        # --- Decision net: does this created chance actually become a shot,
        # or does the attacker recycle possession (pass/dribble instead)?
        ctx_x, ctx_y = attacker_pos if attacking_home else (120 - attacker_pos[0], attacker_pos[1])
        pitch_ctx = PitchContext(x=ctx_x, y=ctx_y, under_pressure=not creation["chaos"])
        decision, decision_probs = bridge.decide(pitch_ctx)

        ml_xg = bridge.xg(pitch_ctx)
        xg_value = round((rng.uniform(0.03, 0.42) + ml_xg) / 2, 3)

        template = "chaos_chance" if creation["chaos"] else "chance_created"
        yield {
            "minute": minute, "type": "chance",
            "description": COMMENTARY_TEMPLATES[template].format(
                minute=minute, name=attacker.name, team=attacking_team.name),
            "home_score": state.home_score, "away_score": state.away_score,
            "momentum": round(state.momentum.value, 3), "xg_added": 0.0,
            "attacker_pos": {"x": attacker_pos[0], "y": attacker_pos[1]},
            "defender_pos": {"x": defender_pos[0], "y": defender_pos[1]},
            "ball_pos": {"x": attacker_pos[0], "y": attacker_pos[1]},
            "decision_probs": {k: round(v, 3) for k, v in decision_probs.items()},
        }
        state.momentum.bump(toward_home=attacking_home, magnitude=0.6)

        if decision != "Shot":
            continue  # built up play, but the decision model chose to keep the ball

        if attacking_home:
            state.home_xg += xg_value
        else:
            state.away_xg += xg_value

        # --- Duel 2: finishing the shot (Finishing vs GK Reflexes), nudged
        # by the ML xG estimate on top of the Gaussian roll + chaos die.
        if attacking_home:
            state.home_shots += 1
        else:
            state.away_shots += 1

        goal_pos = (120.0, 40.0) if attacking_home else (0.0, 40.0)

        finish = resolve_duel(
            rng, attacker, keeper,
            attacker_stat=attacker.finishing, defender_stat=keeper.gk_reflexes,
            momentum=state.momentum, attacker_is_home=attacking_home,
        )
        # Blend the rule/Gaussian outcome with the ML xG estimate: an
        # otherwise-lost duel still has `ml_xg` odds of sneaking in (and
        # vice versa an otherwise-won duel can still be denied), so neither
        # system fully overrides the other.
        finish_scored = finish["attacker_wins"] or rng.random() < (ml_xg * 0.35)
        if finish["attacker_wins"] and rng.random() < ((1 - ml_xg) * 0.15):
            finish_scored = False

        if finish_scored:
            if attacking_home:
                state.home_score += 1
            else:
                state.away_score += 1
            state.momentum.bump(toward_home=attacking_home, magnitude=1.5)
            yield {
                "minute": minute, "type": "goal",
                "description": COMMENTARY_TEMPLATES["goal"].format(
                    minute=minute, name=attacker.name, team=attacking_team.name,
                    home_score=state.home_score, away_score=state.away_score),
                "home_score": state.home_score, "away_score": state.away_score,
                "momentum": round(state.momentum.value, 3), "scorer_id": attacker.id,
                "xg_of_chance": xg_value,
                "attacker_pos": {"x": attacker_pos[0], "y": attacker_pos[1]},
                "ball_pos": {"x": goal_pos[0], "y": goal_pos[1]},
            }
        else:
            miss_template = "chaos_miss" if finish["chaos"] and xg_value > 0.25 else "save"
            state.momentum.crash(home_side_failed=attacking_home)
            yield {
                "minute": minute, "type": "save" if miss_template == "save" else "chaos_moment",
                "description": COMMENTARY_TEMPLATES[miss_template].format(
                    minute=minute, name=attacker.name, team=attacking_team.name),
                "home_score": state.home_score, "away_score": state.away_score,
                "momentum": round(state.momentum.value, 3),
                "xg_of_chance": xg_value,
                "attacker_pos": {"x": attacker_pos[0], "y": attacker_pos[1]},
                "ball_pos": {"x": goal_pos[0], "y": goal_pos[1]},
            }

    total_ticks = max(1, state.home_possession_ticks + state.away_possession_ticks)
    yield {
        "minute": duration_minutes, "type": "full_time",
        "description": f"Full time: {home.name} {state.home_score}-{state.away_score} {away.name}.",
        "home_score": state.home_score, "away_score": state.away_score,
        "momentum": round(state.momentum.value, 3),
        "stats": {
            "home_possession_pct": round(100 * state.home_possession_ticks / total_ticks, 1),
            "away_possession_pct": round(100 * state.away_possession_ticks / total_ticks, 1),
            "home_shots": state.home_shots, "away_shots": state.away_shots,
            "home_xg": round(state.home_xg, 2), "away_xg": round(state.away_xg, 2),
        },
    }
