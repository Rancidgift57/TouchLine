"""
Scenario-based test suite for the two trained nets (xG + decision), run
through the REAL inference path in engine/ml_bridge.py — not a
re-implementation of the models, so this exercises exactly what
engine/match_engine.py calls during a live match.

Run from anywhere:
    python ml/test_model_scenarios.py

If xg_model.pth/xg_scaler.pkl or decision_model.pth/decision_scaler.pkl
haven't been trained yet, MLBridge.load() falls back to rule-based mode
automatically (per its own design) — this script detects that and clearly
labels every result as FALLBACK MODE rather than silently testing the
wrong thing.

What's covered
---------------
1. "Good" scenarios       — situations with an obvious right answer
                             (point-blank tap-in, deep build-up play).
                             These are graded PASS/FAIL against a loose
                             absolute range.
2. "Tough" scenarios       — long range, tight angle, heavy pressure,
                             genuinely ambiguous shot-vs-pass calls, and
                             out-of-bounds/edge coordinates. Graded
                             PASS/WARN only — there's no ground truth for
                             "should a 35-yard header under pressure be
                             3% or 8% xG", so these are diagnostic, not
                             hard assertions.
3. Sanity relationships    — things ANY competently trained model should
                             respect regardless of exact calibration:
                             xG should fall as distance grows, a central
                             shot should out-score a tight-angle one, deep
                             build-up should favor Pass. These ARE hard
                             checks (the script exits non-zero if they
                             fail) because they're about the model's
                             internal consistency, not a specific number.
4. Fuzz / robustness       — hundreds of random (including out-of-range)
                             pitch coordinates, checking the model never
                             crashes and never returns an invalid
                             probability.
5. Fallback-mode check     — deliberately flips MLBridge into its
                             `.available = False` degraded mode and
                             re-runs a few scenarios, so the "no weights
                             trained yet" code path is exercised even
                             when you DO have weights trained locally.
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass, field

# Let this run from any cwd (ml/, repo root, wherever) by putting the repo
# root — the parent of this file's directory — on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.ml_bridge import MLBridge, PitchContext  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def tag(status: str) -> str:
    return {"PASS": "\033[92m✅ PASS\033[0m", "WARN": "\033[93m⚠️  WARN\033[0m",
            "FAIL": "\033[91m❌ FAIL\033[0m"}[status]


# ---------------------------------------------------------------------------
# 1 & 2. Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class XGScenario:
    name: str
    ctx: PitchContext
    expected_range: tuple[float, float]  # loose acceptable band
    hard: bool                            # True = FAIL outside range, False = WARN only
    note: str = ""


@dataclass
class DecisionScenario:
    name: str
    ctx: PitchContext
    expected_favored: str | None  # 'Pass' | 'Dribble' | 'Shot' | None (ambiguous, informational)
    hard: bool
    note: str = ""


GOOD_XG_SCENARIOS = [
    XGScenario(
        "Tap-in — 2 yards out, dead center, right foot",
        PitchContext(x=118, y=40, under_pressure=False, pattern="Regular Play", body_part="Right Foot"),
        expected_range=(0.30, 0.99), hard=True,
        note="Should clearly be one of the highest-xG situations in the suite.",
    ),
    XGScenario(
        "Penalty-spot equivalent — 12 yards, central, no pressure",
        PitchContext(x=108, y=40, under_pressure=False, pattern="Regular Play", body_part="Right Foot"),
        expected_range=(0.15, 0.95), hard=True,
        note="Textbook high-quality chance.",
    ),
    XGScenario(
        "Header from a corner, 6 yards, central",
        PitchContext(x=114, y=40, under_pressure=False, pattern="From Corner", body_part="Head"),
        expected_range=(0.10, 0.90), hard=False,
        note="Good chance, but headers score lower than footed shots historically.",
    ),
]

TOUGH_XG_SCENARIOS = [
    XGScenario(
        "Long range — 35 yards, central, right foot",
        PitchContext(x=85, y=40, under_pressure=False, pattern="Regular Play", body_part="Right Foot"),
        expected_range=(0.0, 0.20), hard=True,
        note="Should be low — this is the single most reliable xG fact there is.",
    ),
    XGScenario(
        "Near-post, tight angle, 8 yards from byline",
        PitchContext(x=118, y=5, under_pressure=False, pattern="Regular Play", body_part="Right Foot"),
        expected_range=(0.0, 0.35), hard=False,
        note="Close but almost no angle — genuinely tricky for a model trained mostly on central shots.",
    ),
    XGScenario(
        "35-yard header under heavy pressure, weak side",
        PitchContext(x=82, y=15, under_pressure=True, pattern="From Counter", body_part="Head"),
        expected_range=(0.0, 0.10), hard=False,
        note="About as low-quality as a 'shot' gets — worth eyeballing it isn't wildly overconfident.",
    ),
    XGScenario(
        "On the goal line, dead center (x=120 exactly)",
        PitchContext(x=120, y=40, under_pressure=False, pattern="Regular Play", body_part="Right Foot"),
        expected_range=(0.0, 1.0), hard=False,
        note="Boundary coordinate (distance_to_goal == 0) — just checking it doesn't NaN/crash.",
    ),
    XGScenario(
        "Out-of-bounds coordinate (behind the goal, x=126)",
        PitchContext(x=126, y=40, under_pressure=False, pattern="Regular Play", body_part="Right Foot"),
        expected_range=(0.0, 1.0), hard=False,
        note="Never happens in real data — checking the model degrades sanely instead of erroring.",
    ),
]

GOOD_DECISION_SCENARIOS = [
    DecisionScenario(
        "Deep in own half, totally unmarked",
        PitchContext(x=25, y=40, under_pressure=False, pattern="Regular Play"),
        expected_favored="Pass", hard=True,
        note="No footballer dribbles or shoots from inside their own half unpressured.",
    ),
    DecisionScenario(
        "Own-box clearance under heavy pressure",
        PitchContext(x=12, y=40, under_pressure=True, pattern="Regular Play"),
        expected_favored="Pass", hard=True,
        note="Forced clearance/pass situation — should not favor Shot or Dribble.",
    ),
    DecisionScenario(
        "Six yards out, dead center, no pressure",
        PitchContext(x=115, y=40, under_pressure=False, pattern="Regular Play"),
        expected_favored="Shot", hard=False,
        note="Textbook 'just shoot it' — graded soft since the training data still has plenty of "
             "central-box passes (lay-offs, cutbacks) to compete with.",
    ),
]

TOUGH_DECISION_SCENARIOS = [
    DecisionScenario(
        "Edge of the box, under pressure, tight angle",
        PitchContext(x=100, y=15, under_pressure=True, pattern="Regular Play"),
        expected_favored=None, hard=False,
        note="Genuinely ambiguous — a real player could shoot, cut back, or lay it off. Informational only.",
    ),
    DecisionScenario(
        "Counter-attack breakaway, open space, final third",
        PitchContext(x=95, y=40, under_pressure=False, pattern="From Counter"),
        expected_favored=None, hard=False,
        note="Could easily be Dribble (drive at goal) or Shot depending on the exact yard — no hard call.",
    ),
    DecisionScenario(
        "Throw-in deep in the attacking corner",
        PitchContext(x=118, y=2, under_pressure=True, pattern="From Throw In"),
        expected_favored=None, hard=False,
        note="Unusual combination (attacking corner + throw-in pattern + pressure) — stress-testing the "
             "one-hot pattern encoding on a rare combo, not asserting an answer.",
    ),
]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_xg_scenarios(bridge: MLBridge, scenarios: list[XGScenario], label: str) -> list[str]:
    print(f"\n--- xG model: {label} ---")
    results = []
    for s in scenarios:
        p = bridge.xg(s.ctx)
        lo, hi = s.expected_range
        in_range = lo <= p <= hi
        status = PASS if in_range else (FAIL if s.hard else WARN)
        results.append(status)
        print(f"  {tag(status)}  {s.name}")
        print(f"           xG = {p:.3f}  (expected [{lo:.2f}, {hi:.2f}])  {s.note}")
    return results


def run_decision_scenarios(bridge: MLBridge, scenarios: list[DecisionScenario], label: str) -> list[str]:
    print(f"\n--- Decision model: {label} ---")
    results = []
    for s in scenarios:
        action, probs = bridge.decide(s.ctx)
        favored = max(probs, key=probs.get)
        if s.expected_favored is None:
            status = PASS  # informational — always "passes", we're just reading the numbers
        else:
            ok = favored == s.expected_favored
            status = PASS if ok else (FAIL if s.hard else WARN)
        results.append(status)
        probs_str = ", ".join(f"{k}={v:.2f}" for k, v in probs.items())
        print(f"  {tag(status)}  {s.name}")
        print(f"           sampled={action}  favored={favored}  [{probs_str}]  {s.note}")
    return results


def run_sanity_relationships(bridge: MLBridge) -> list[str]:
    """Checks that don't depend on exact calibration — just basic football
    logic any competently trained model should respect."""
    print("\n--- Sanity relationships (hard checks) ---")
    results = []

    def check(name: str, condition: bool, detail: str):
        status = PASS if condition else FAIL
        results.append(status)
        print(f"  {tag(status)}  {name}")
        print(f"           {detail}")

    # xG should fall as distance from goal grows, all else equal.
    near = bridge.xg(PitchContext(x=115, y=40, body_part="Right Foot"))
    mid = bridge.xg(PitchContext(x=95, y=40, body_part="Right Foot"))
    far = bridge.xg(PitchContext(x=75, y=40, body_part="Right Foot"))
    check("xG decreases with distance (5yd > 25yd > 45yd, central)",
          near >= mid >= far,
          f"xG(5yd)={near:.3f}  xG(25yd)={mid:.3f}  xG(45yd)={far:.3f}")

    # A central shot should out-score an equal-distance, near-byline shot.
    central = bridge.xg(PitchContext(x=110, y=40, body_part="Right Foot"))
    wide = bridge.xg(PitchContext(x=110, y=2, body_part="Right Foot"))
    check("Central beats tight-angle at the same distance",
          central >= wide,
          f"xG(central)={central:.3f}  xG(near-byline)={wide:.3f}")

    # Deep, unpressured build-up should favor Pass over Shot.
    _, deep_probs = bridge.decide(PitchContext(x=20, y=40, under_pressure=False))
    check("Own-half build-up favors Pass over Shot",
          deep_probs["Pass"] > deep_probs["Shot"],
          f"Pass={deep_probs['Pass']:.3f}  Shot={deep_probs['Shot']:.3f}")

    # Probabilities are a real distribution.
    _, any_probs = bridge.decide(PitchContext(x=90, y=40))
    total = sum(any_probs.values())
    check("Decision probabilities sum to ~1.0", abs(total - 1.0) < 1e-3, f"sum={total:.5f}")

    return results


def run_fuzz_test(bridge: MLBridge, n: int = 300) -> tuple[int, int]:
    """Hundreds of random pitch situations, including out-of-range and
    negative coordinates the training data would never contain, checking
    the model never crashes and never returns an invalid probability."""
    print(f"\n--- Fuzz / robustness test ({n} random contexts, including out-of-bounds) ---")
    rng = random.Random(42)
    patterns = ["Regular Play", "From Corner", "From Counter", "From Free Kick",
                "From Goal Kick", "From Keeper", "From Kick Off", "From Throw In", "Other"]
    body_parts = ["Right Foot", "Left Foot", "Head", "Other"]

    crashes = 0
    invalid = 0
    for _ in range(n):
        ctx = PitchContext(
            x=rng.uniform(-20, 140),   # deliberately overshoots the real 0..120 pitch
            y=rng.uniform(-20, 100),   # deliberately overshoots the real 0..80 pitch
            under_pressure=rng.random() < 0.5,
            pattern=rng.choice(patterns),
            body_part=rng.choice(body_parts),
        )
        try:
            xg = bridge.xg(ctx)
            _, probs = bridge.decide(ctx)
        except Exception as e:  # noqa: BLE001
            crashes += 1
            print(f"  {tag(FAIL)}  Crashed on x={ctx.x:.1f}, y={ctx.y:.1f}: {e!r}")
            continue
        bad = not (0.0 <= xg <= 1.0) or any(not (0.0 <= p <= 1.0) for p in probs.values())
        if bad:
            invalid += 1
            print(f"  {tag(FAIL)}  Invalid output on x={ctx.x:.1f}, y={ctx.y:.1f}: xg={xg}, probs={probs}")

    status = PASS if crashes == 0 and invalid == 0 else FAIL
    print(f"  {tag(status)}  {n - crashes - invalid}/{n} random contexts produced valid, non-crashing output")
    return crashes, invalid


def run_fallback_mode_check() -> list[str]:
    """Forces MLBridge into its degraded (.available=False) mode and
    re-runs a couple of scenarios — this is the exact path a fresh clone
    of the repo runs in before anyone has trained the models, and it's
    easy for that code path to silently rot since it's not exercised once
    real weights exist locally."""
    print("\n--- Fallback mode (no trained weights) — testing the degraded path directly ---")
    fallback = MLBridge()  # no models/scalers passed in -> .available == False
    results = []

    status = PASS if not fallback.available else FAIL
    results.append(status)
    print(f"  {tag(status)}  MLBridge() with no weights reports .available == False")

    try:
        xg = fallback.xg(PitchContext(x=118, y=40))
        _, probs = fallback.decide(PitchContext(x=20, y=40, under_pressure=False))
        ok = (0.0 <= xg <= 1.0) and abs(sum(probs.values()) - 1.0) < 1e-6
        status = PASS if ok else FAIL
        print(f"  {tag(status)}  Fallback xg()={xg:.3f}, decide() probs sum to "
              f"{sum(probs.values()):.5f} — both valid without crashing")
    except Exception as e:  # noqa: BLE001
        status = FAIL
        print(f"  {tag(status)}  Fallback mode crashed: {e!r}")
    results.append(status)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("TouchLine ML model test suite — xG net + decision net")
    print("=" * 70)

    bridge = MLBridge.load()
    if bridge.available:
        print("Loaded trained weights — testing the REAL models.")
    else:
        print("No trained weights found (ml/*.pth missing) — MLBridge has degraded to "
              "FALLBACK MODE automatically. Every result below tests that fallback logic, "
              "not the trained nets. Run ml/xg_model_architecture.py and "
              "ml/decision_model_architecture.py first to test the real models.")

    all_results: list[str] = []
    all_results += run_xg_scenarios(bridge, GOOD_XG_SCENARIOS, "good/confident cases")
    all_results += run_xg_scenarios(bridge, TOUGH_XG_SCENARIOS, "tough/edge cases")
    all_results += run_decision_scenarios(bridge, GOOD_DECISION_SCENARIOS, "good/confident cases")
    all_results += run_decision_scenarios(bridge, TOUGH_DECISION_SCENARIOS, "tough/ambiguous cases")
    all_results += run_sanity_relationships(bridge)
    crashes, invalid = run_fuzz_test(bridge)
    all_results += [FAIL] * (crashes + invalid) or [PASS]

    if bridge.available:
        all_results += run_fallback_mode_check()
    else:
        # Already IN fallback mode above — running it again is redundant,
        # but confirm the explicit MLBridge() constructor path still works.
        all_results += run_fallback_mode_check()

    n_pass = all_results.count(PASS)
    n_warn = all_results.count(WARN)
    n_fail = all_results.count(FAIL)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} passed, {n_warn} warned (informational), {n_fail} failed"
          f"  (of {len(all_results)} checks)")
    print("=" * 70)

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
