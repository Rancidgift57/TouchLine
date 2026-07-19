"""
Market valuation model + trade anti-cheat.

Predicts a player's "True Market Value" from Age, Overall, Potential,
Contract Length, Form, and a Market Volatility index, then scores any
proposed trade for fairness. Trades far outside a fair band are flagged
'review' or 'blocked' (protects against farm-account dumping: trading a
star for 0 cash to funnel currency to an alt account).

Swap in a trained XGBoost/PyTorch model at inference time by replacing
`_predict_raw`; the public API (`predict_value`, `score_trade`) stays
the same so the rest of the app doesn't need to change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerValuationInput:
    age: int
    overall: int
    potential: int
    contract_years_left: int
    current_form: float      # z-score, -2..+2
    position_scarcity: float = 1.0  # >1 for scarce positions (e.g. elite CBs)


# Coefficients below approximate a fitted regression; in production these
# come from `python -m tuning.train_valuation_model` writing a joblib/ONNX
# artifact that this module loads at startup.
_BASE_VALUE_PER_OVERALL_POINT = 45_000
_AGE_CURVE_PEAK = 27


def _age_multiplier(age: int) -> float:
    """Peaks at 27, decays faster after 30, penalizes very young/unproven too."""
    distance = abs(age - _AGE_CURVE_PEAK)
    return max(0.35, 1.15 - 0.045 * distance)


def _predict_raw(inp: PlayerValuationInput) -> float:
    base = (inp.overall ** 2.1) * _BASE_VALUE_PER_OVERALL_POINT / 1000
    base *= _age_multiplier(inp.age)
    base *= 1 + max(0.0, (inp.potential - inp.overall)) * 0.015   # upside premium
    base *= 1 + max(-0.3, min(0.3, inp.current_form * 0.08))       # hot/cold form
    base *= 1 + min(0.5, inp.contract_years_left * 0.06)            # longer deal = more value
    base *= inp.position_scarcity
    return round(base)


def predict_value(inp: PlayerValuationInput) -> int:
    return int(_predict_raw(inp))


@dataclass
class TradeLeg:
    player_value: int
    cash: int = 0


@dataclass
class TradeFairnessResult:
    fairness_score: float   # 0 (lopsided) .. 1 (fair)
    flag: str                # 'ok' | 'review' | 'blocked'
    proposer_total: int
    receiver_total: int


def score_trade(from_side: list[TradeLeg], to_side: list[TradeLeg]) -> TradeFairnessResult:
    """
    from_side  = assets the PROPOSING team is giving up
    to_side    = assets the RECEIVING team is giving up
    A fair trade has both totals reasonably close. A near-total mismatch
    (e.g. a star player for 0 cash and a bench scrub) scores near 0 and
    gets blocked -- this is the farm-account / manipulation guard.
    """
    proposer_total = sum(leg.player_value + leg.cash for leg in from_side)
    receiver_total = sum(leg.player_value + leg.cash for leg in to_side)

    bigger = max(proposer_total, receiver_total, 1)
    smaller = min(proposer_total, receiver_total)
    fairness = smaller / bigger  # 1.0 = perfectly balanced

    if fairness >= 0.65:
        flag = "ok"
    elif fairness >= 0.35:
        flag = "review"
    else:
        flag = "blocked"

    return TradeFairnessResult(
        fairness_score=round(fairness, 3),
        flag=flag,
        proposer_total=proposer_total,
        receiver_total=receiver_total,
    )
