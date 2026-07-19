"""
ml_bridge.py — loads the two trained PyTorch nets (xG, Pass/Dribble/Shot
decision) and exposes a small, engine-friendly API:

    bridge = MLBridge.load()                 # once, at process start
    action, probs = bridge.decide(ctx)        # -> "Pass" | "Dribble" | "Shot"
    xg = bridge.xg(ctx)                       # -> float 0..1

`ctx` is a PitchContext: the (x, y) StatsBomb-style coordinate (120x40 half
pitch, origin at defended goal... same convention the training scripts use:
0-120 long axis, 0-80 wide axis, attacking goal at x=120, y=40 center),
whether the ball-carrier is under pressure, and the play pattern.

Design notes
------------
* This module has ZERO hard dependency on the rest of match_engine — it's a
  pure "given a pitch situation, what does the model say" function, exactly
  mirroring Model Usage.py, just refactored into an importable class instead
  of a script with globals.
* If the .pth/.pkl files are missing (e.g. a dev checked out the repo but
  hasn't trained/downloaded weights yet), MLBridge.load() returns a bridge
  in `.available == False` state and match_engine falls back to the pure
  Gaussian/rule-based duel system. This keeps the sim runnable out of the box.
* Outputs are NOT used verbatim as the final outcome. match_engine blends
  them with player attributes (finishing, consistency), momentum, and the
  chaos roll — the ML models describe "what a neutral, attribute-less
  professional footballer would do/score from this spot", the rest of the
  engine layers the RPG stats + randomness on top.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # torch not installed yet -> engine degrades to rule-based mode
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

ML_DIR = os.path.dirname(os.path.abspath(__file__)).replace("engine", "ml")

DECISION_FEATURES = [
    "x", "y", "distance_to_goal", "angle_to_goal", "under_pressure",
    "zone_def", "zone_mid", "zone_att", "pattern_From Corner",
    "pattern_From Counter", "pattern_From Free Kick", "pattern_From Goal Kick",
    "pattern_From Keeper", "pattern_From Kick Off", "pattern_From Throw In",
    "pattern_Other", "pattern_Regular Play",
]

XG_FEATURES = [
    "x", "y", "distance_to_goal", "angle_to_goal", "under_pressure",
    "bp_Head", "bp_Left Foot", "bp_Other", "bp_Right Foot",
    "pattern_From Corner", "pattern_From Counter", "pattern_From Free Kick",
    "pattern_From Goal Kick", "pattern_From Keeper", "pattern_From Kick Off",
    "pattern_From Throw In", "pattern_Other", "pattern_Regular Play",
]

PlayPattern = Literal[
    "Regular Play", "From Corner", "From Counter", "From Free Kick",
    "From Goal Kick", "From Keeper", "From Kick Off", "From Throw In", "Other",
]
BodyPart = Literal["Right Foot", "Left Foot", "Head", "Other"]


@dataclass
class PitchContext:
    x: float                       # 0..120, attacking goal at x=120
    y: float                       # 0..80, center line y=40
    under_pressure: bool = False
    pattern: PlayPattern = "Regular Play"
    body_part: BodyPart = "Right Foot"

    @property
    def distance_to_goal(self) -> float:
        return float(np.sqrt((120 - self.x) ** 2 + (40 - self.y) ** 2))

    @property
    def angle_to_goal(self) -> float:
        return float(np.abs(np.arctan2(40 - self.y, 120 - self.x)))


if _TORCH_AVAILABLE:
    class _TwoLayerNet(nn.Module):
        """Shared architecture for both nets — only `output` width differs."""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, 32)
            self.bn1 = nn.BatchNorm1d(32)
            self.dropout1 = nn.Dropout(0.3)
            self.layer2 = nn.Linear(32, 16)
            self.bn2 = nn.BatchNorm1d(16)
            self.dropout2 = nn.Dropout(0.3)
            self.output = nn.Linear(16, output_dim)

        def forward(self, x):
            x = self.dropout1(torch.relu(self.bn1(self.layer1(x))))
            x = self.dropout2(torch.relu(self.bn2(self.layer2(x))))
            return self.output(x)
else:
    _TwoLayerNet = None  # type: ignore


class MLBridge:
    def __init__(self, decision_model=None, decision_scaler=None,
                 xg_model=None, xg_scaler=None):
        self.decision_model = decision_model
        self.decision_scaler = decision_scaler
        self.xg_model = xg_model
        self.xg_scaler = xg_scaler
        self.available = all(
            m is not None for m in (decision_model, decision_scaler, xg_model, xg_scaler)
        )

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, ml_dir: str = ML_DIR) -> "MLBridge":
        if not _TORCH_AVAILABLE:
            # `pip install -r requirements.txt` includes torch, but if this
            # runs before that, don't crash the whole app — just fall back.
            return cls()
        try:
            import joblib

            decision_model = _TwoLayerNet(len(DECISION_FEATURES), 3)
            decision_model.load_state_dict(
                torch.load(os.path.join(ml_dir, "decision_model.pth"), map_location="cpu")
            )
            decision_model.eval()
            decision_scaler = joblib.load(os.path.join(ml_dir, "decision_scaler.pkl"))

            xg_model = _TwoLayerNet(len(XG_FEATURES), 1)
            xg_model.load_state_dict(
                torch.load(os.path.join(ml_dir, "xg_model.pth"), map_location="cpu")
            )
            xg_model.eval()
            xg_scaler = joblib.load(os.path.join(ml_dir, "xg_scaler.pkl"))

            return cls(decision_model, decision_scaler, xg_model, xg_scaler)
        except (FileNotFoundError, OSError):
            # Weights not trained/downloaded yet -> engine runs in
            # pure-rule-based mode until `python ml/*_architecture.py` is run.
            return cls()

    # -- feature vector builders ------------------------------------------
    def _decision_vector(self, ctx: PitchContext) -> np.ndarray:
        row = {c: 0 for c in DECISION_FEATURES}
        row.update({
            "x": ctx.x, "y": ctx.y,
            "distance_to_goal": ctx.distance_to_goal,
            "angle_to_goal": ctx.angle_to_goal,
            "under_pressure": int(ctx.under_pressure),
            "zone_def": int(ctx.x <= 40),
            "zone_mid": int(40 < ctx.x <= 80),
            "zone_att": int(ctx.x > 80),
        })
        key = f"pattern_{ctx.pattern}"
        if key in row:
            row[key] = 1
        return np.array([[row[c] for c in DECISION_FEATURES]])

    def _xg_vector(self, ctx: PitchContext) -> np.ndarray:
        row = {c: 0 for c in XG_FEATURES}
        row.update({
            "x": ctx.x, "y": ctx.y,
            "distance_to_goal": ctx.distance_to_goal,
            "angle_to_goal": ctx.angle_to_goal,
            "under_pressure": int(ctx.under_pressure),
        })
        bp_key = f"bp_{ctx.body_part}"
        if bp_key in row:
            row[bp_key] = 1
        pat_key = f"pattern_{ctx.pattern}"
        if pat_key in row:
            row[pat_key] = 1
        return np.array([[row[c] for c in XG_FEATURES]])

    # -- public API ---------------------------------------------------------
    def decide(self, ctx: PitchContext) -> tuple[str, dict[str, float]]:
        """Returns (chosen_action, {'Pass':p,'Dribble':p,'Shot':p})."""
        if not self.available:
            # Deterministic-ish fallback so callers never branch on
            # `.available` themselves: favors passing, more shots near goal.
            near_goal = max(0.0, 1 - ctx.distance_to_goal / 40)
            probs = np.array([0.65 - 0.3 * near_goal, 0.2, 0.15 + 0.3 * near_goal])
            probs = probs / probs.sum()
        else:
            vec = self.decision_scaler.transform(self._decision_vector(ctx))
            tensor = torch.tensor(vec, dtype=torch.float32)
            with torch.no_grad():
                logits = self.decision_model(tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]

        actions = ["Pass", "Dribble", "Shot"]
        chosen = np.random.choice(actions, p=probs)
        return chosen, dict(zip(actions, (float(p) for p in probs)))

    def xg(self, ctx: PitchContext) -> float:
        """Returns model-estimated goal probability for a shot from ctx."""
        if not self.available:
            # Simple geometric fallback: closer + more central -> higher.
            d = ctx.distance_to_goal
            a = ctx.angle_to_goal
            return float(np.clip(0.5 * np.exp(-d / 22) * np.cos(a) ** 1.5, 0.01, 0.9))

        vec = self.xg_scaler.transform(self._xg_vector(ctx))
        tensor = torch.tensor(vec, dtype=torch.float32)
        with torch.no_grad():
            logits = self.xg_model(tensor)
            return float(torch.sigmoid(logits).item())
