"""Regression guard for the project's central lesson: reward economy.

Per-step penalties must stay small enough that standing/idling near a
blocker for a whole episode never costs more than a single crash --
otherwise PPO learns to crash on purpose to escape the penalty (the
"suicide" dynamic documented in HANDOFF.md). Concretely, for every bounded
per-step penalty term P with weight w:

    w * horizon < |crash_penalty|

This test walks the *actual* reward configuration and horizon deployed in
``main.py`` and checks that rule -- not a hand-picked example -- so it
fails the moment someone changes a weight enough to reintroduce the bug.
"""

import importlib
import sys
from pathlib import Path

import pytest

from reward import (
    AvoidCollisionRewardStrategy,
    CompositeRewardStrategy,
    MinSpeedPenaltyStrategy,
    MotionGatedRewardStrategy,
    ProximityPenaltyStrategy,
)


def _flatten(strategies):
    """Yield every strategy, unwrapping MotionGatedRewardStrategy."""
    for strategy in strategies:
        yield strategy
        if isinstance(strategy, MotionGatedRewardStrategy):
            yield from _flatten([strategy.strategy])


@pytest.fixture
def deployed_main():
    """Import main.py as a module (safe: no env/model is built at import
    time -- TrainingOrchestrator is only constructed under the
    ``if __name__ == "__main__":`` guard)."""
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    added = src_path not in sys.path
    if added:
        sys.path.insert(0, src_path)
    try:
        if "main" in sys.modules:
            module = importlib.reload(sys.modules["main"])
        else:
            module = importlib.import_module("main")
        yield module
    finally:
        if added:
            sys.path.remove(src_path)


def test_bounded_penalties_stay_under_crash_penalty(deployed_main):
    config = deployed_main.config
    assert isinstance(config.reward_strategy, CompositeRewardStrategy)

    horizon = config.map_config["horizon"]
    strategies = list(_flatten(config.reward_strategy.strategies))

    crash_terms = [s for s in strategies
                   if isinstance(s, AvoidCollisionRewardStrategy)]
    assert crash_terms, "expected an AvoidCollisionRewardStrategy in main.py"
    crash_penalty = abs(crash_terms[0].crash_penalty)

    bounded_penalty_types = (MinSpeedPenaltyStrategy, ProximityPenaltyStrategy)
    penalty_terms = [s for s in strategies
                     if isinstance(s, bounded_penalty_types)]

    for term in penalty_terms:
        worst_case = term.weight * horizon
        assert worst_case < crash_penalty, (
            f"{type(term).__name__}(weight={term.weight}) * "
            f"horizon={horizon} = {worst_case} >= crash penalty "
            f"{crash_penalty} -- this can make crashing on purpose "
            f"cheaper than idling (the 'suicide' dynamic)."
        )
