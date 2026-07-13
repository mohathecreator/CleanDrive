"""Reward strategies for the driving agent (Strategy pattern).

Each strategy maps a transition ``(observation, action, info)`` to a scalar
reward. :class:`CompositeRewardStrategy` sums several of them, so the overall
reward function is assembled declaratively in ``main.py``.

The ``info`` dict carries the values each strategy needs. Besides the
MetaDrive-native keys (``velocity``, ``crash``, ``out_of_road``) it is
augmented by :class:`environment.EnvironmentWrapper` with ``lateral_offset``,
``lateral_velocity``, ``heading_error`` and ``front_distance``.

Rewards are *not* normalised downstream (``VecNormalize(norm_reward=False)``),
so the raw per-step magnitudes documented below are exactly what PPO sees.
"""

import math
from abc import ABC, abstractmethod


class RewardStrategy(ABC):
    """Abstract base for a single reward term.

    Subclasses implement :meth:`compute`. Stateful strategies (those that
    remember something between steps) additionally override :meth:`reset`,
    which the environment calls at the start of every episode.
    """

    @abstractmethod
    def compute(self, observation, action, info) -> float:
        """Return this term's scalar reward for one transition."""

    def reset(self):
        """No-op default. Stateful strategies override this."""
        pass


class SpeedRewardStrategy(RewardStrategy):
    """Reward proportional to forward speed.

    Linear in ``info['velocity']`` up to ``max_speed``; per-step range
    ``[0, weight]``. Encourages the agent to keep moving.

    Raises ValueError if ``max_speed`` is not positive (it is a
    denominator; zero would divide by zero at every step).
    """

    def __init__(self, weight: float = 1.0, max_speed: float = 40.0):
        if max_speed <= 0:
            raise ValueError(f"max_speed must be positive. "
                             f"Current value: {max_speed}")
        self.weight = weight
        self.max_speed = max_speed

    def compute(self, observation, action, info) -> float:
        return self.weight * (info["velocity"] / self.max_speed)


class ProgressRewardStrategy(RewardStrategy):
    """Reward forward progress along the route (the primary objective).

    Rewards the per-step change in ``info['route_completion']`` (fraction
    of the reference trajectory covered, 0..1). Summed over a full route
    this yields ``weight`` regardless of speed; with discounting, finishing
    sooner is worth more, so it rewards *reaching the goal quickly*. Unlike
    raw speed it cannot be farmed by standing still or circling -- neither
    advances completion -- and backward motion yields a negative delta, so
    oscillating in place does not pay.

    Stateful: remembers the previous completion, so it overrides
    :meth:`reset`. The first step of an episode returns 0 (no baseline
    yet), which also absorbs any non-zero spawn completion.
    """

    def __init__(self, weight: float = 200.0):
        self.weight = weight
        self._prev = None

    def compute(self, observation, action, info) -> float:
        completion = info["route_completion"]
        if self._prev is None:
            self._prev = completion
            return 0.0
        delta = completion - self._prev
        self._prev = completion
        return self.weight * delta

    def reset(self):
        self._prev = None


class LaneCenteringRewardStrategy(RewardStrategy):
    """Reward for staying near the lane centre.

    Uses ``info['lateral_offset']`` (metres from the centreline). Full
    reward at the centre, decaying linearly to 0 at ``half_lane_width``;
    per-step range ``[0, weight]``. The default ``half_lane_width=0.6``
    matches the empirical ``out_of_road`` threshold.

    Raises ValueError if ``half_lane_width`` is not positive (it is a
    denominator; zero would divide by zero at every step).
    """

    def __init__(self, weight: float = 2.0, half_lane_width: float = 0.6):
        if half_lane_width <= 0:
            raise ValueError(f"half_lane_width must be positive. "
                             f"Current value: {half_lane_width}")
        self.weight = weight
        self.half_lane_width = half_lane_width

    def compute(self, observation, action, info) -> float:
        center = max(
            0.0, 1.0 - abs(info["lateral_offset"]) / self.half_lane_width
        )
        return self.weight * center


class HeadingAlignmentRewardStrategy(RewardStrategy):
    """Reward for pointing along the lane tangent.

    Uses ``info['heading_error']`` (radians, wrapped to [-pi, pi]). Full
    reward when aligned, decaying to 0 at ``max_error`` (~29 deg at the
    default); per-step range ``[0, weight]``.

    Closes the zig-zag loophole that lane-centering alone leaves open:
    centering can be satisfied on average while the car oscillates about
    the centre, whereas a heading term cannot.

    Raises ValueError if ``max_error`` is not positive (it is a
    denominator; zero would divide by zero at every step).
    """

    def __init__(self, weight: float = 2.0, max_error: float = 0.5):
        if max_error <= 0:
            raise ValueError(f"max_error must be positive. "
                             f"Current value: {max_error}")
        self.weight = weight
        self.max_error = max_error

    def compute(self, observation, action, info) -> float:
        align = max(0.0, 1.0 - abs(info["heading_error"]) / self.max_error)
        return self.weight * align


class AvoidCollisionRewardStrategy(RewardStrategy):
    """Terminal penalty for crashing or leaving the road.

    Returns ``crash_penalty`` when ``info['crash']`` or
    ``info['out_of_road']`` is set, else 0. The large magnitude dominates
    every shaping term so these events are strongly avoided.
    """

    def __init__(self, crash_penalty: float = -10.0):
        self.crash_penalty = crash_penalty

    def compute(self, observation, action, info) -> float:
        if info["crash"] or info["out_of_road"]:
            return self.crash_penalty
        return 0.0


class RouteCompletionBonusStrategy(RewardStrategy):
    """One-off bonus for actually reaching the destination.

    Returns ``bonus`` on the step where ``info['arrive_dest']`` is set,
    else 0. Offsets the incentive to avoid the goal (arriving ends the
    reward stream): completing must out-earn dragging the episode out.

    Accesses ``info['arrive_dest']`` directly, like every sibling
    strategy accesses its own info key -- MetaDrive always populates it
    on every step, so a missing key means a real wiring bug that should
    surface immediately rather than being silently masked to 0.
    """

    def __init__(self, bonus: float = 100.0):
        self.bonus = bonus

    def compute(self, observation, action, info) -> float:
        if info["arrive_dest"]:
            return self.bonus
        return 0.0


class SteeringSmoothRewardStrategy(RewardStrategy):
    """Penalty for jerky steering (anti-oscillation).

    Penalises both the steering magnitude and its change between steps,
    quadratically::

        -(weight_magnitude * steer**2 + weight_delta * (steer - prev)**2)

    where ``steer = action[0]``. The quadratic form leaves gentle
    cornering almost free (~-0.03) while making rapid full-lock swings
    expensive (~-1.5) -- something the earlier linear ``|delta|`` form
    could not do. Stateful: remembers the previous steering command, so it
    overrides :meth:`reset`.
    """

    def __init__(self, weight_magnitude: float = 0.3,
                 weight_delta: float = 0.5):
        self.weight_magnitude = weight_magnitude
        self.weight_delta = weight_delta
        self._prev_steering = 0.0

    def compute(self, observation, action, info) -> float:
        steer = action[0]
        magnitude = steer ** 2
        delta = (steer - self._prev_steering) ** 2
        self._prev_steering = steer
        return -(self.weight_magnitude * magnitude
                 + self.weight_delta * delta)

    def reset(self):
        """Clear the remembered steering at the start of an episode."""
        self._prev_steering = 0.0


class BrakingRewardStrategy(RewardStrategy):
    """Reward braking when an obstacle is close ahead.

    Active only when ``info['front_distance'] < safe_distance``. Scales
    with proximity and braking effort (throttle ``action[1] < 0``). Not
    part of the current reward set, but kept for reuse.

    Raises ValueError if ``safe_distance`` is not positive (it is a
    denominator; zero would divide by zero at every step).
    """

    def __init__(self, weight: float = 2.0, safe_distance: float = 8.0):
        if safe_distance <= 0:
            raise ValueError(f"safe_distance must be positive. "
                             f"Current value: {safe_distance}")
        self.weight = weight
        self.safe_distance = safe_distance

    def compute(self, observation, action, info) -> float:
        front_dist = info["front_distance"]
        if front_dist >= self.safe_distance:
            return 0.0
        proximity = 1.0 - (front_dist / self.safe_distance)
        if action[1] < 0:
            return self.weight * proximity * abs(action[1])
        return 0.0


class MinSpeedPenaltyStrategy(RewardStrategy):
    """Penalty for creeping below a minimum speed.

    Zero at or above ``min_speed``, ramping linearly to ``-weight`` at a
    standstill; per-step range ``[-weight, 0]``. Complements
    :class:`SpeedRewardStrategy` to discourage the agent from stalling.

    Raises ValueError if ``min_speed`` is not positive (it is a
    denominator; zero would divide by zero at every step).
    """

    def __init__(self, weight: float = 2.0, min_speed: float = 10.0):
        if min_speed <= 0:
            raise ValueError(f"min_speed must be positive. "
                             f"Current value: {min_speed}")
        self.weight = weight
        self.min_speed = min_speed

    def compute(self, observation, action, info) -> float:
        velocity = info["velocity"]
        if velocity >= self.min_speed:
            return 0.0
        return -self.weight * (1 - velocity / self.min_speed)


class ProximityPenaltyStrategy(RewardStrategy):
    """Continuous penalty for a close obstacle ahead (anti-freeze shaping).

    Fills the silence in front of the binary crash cliff with a smooth
    gradient. As ``info['front_distance']`` -- the normalised lidar
    fraction in [0, 1], where 1.0 is clear road and 0.0 is contact --
    drops below ``safe_distance``, the penalty ramps linearly from 0 to
    ``-weight``. That gives PPO a graded "ease off" signal *before* the
    terminal collision, so it can learn to slow and trail rather than
    binary-freeze next to every car (which the flat 0-until--30 shape
    rewards). ``safe_distance=0.3`` corresponds to ~15 m at a 50 m lidar.

    Bounded on purpose: ``weight * horizon`` must stay under the crash
    penalty, else stopping close would eventually cost more than crashing
    and recreate the documented suicide dynamic. At ``weight=0.02`` and
    horizon 1000 the worst-case budget is 20 < 30. This is a *shaping*
    term (smoothing the cliff into a ramp), not a strong deterrent -- the
    -30 terminal stays the real deterrent.

    Raises ValueError if ``safe_distance`` is not positive (it is a
    denominator; zero would divide by zero at every step).
    """

    def __init__(self, weight: float = 0.02, safe_distance: float = 0.3):
        if safe_distance <= 0:
            raise ValueError(f"safe_distance must be positive. "
                             f"Current value: {safe_distance}")
        self.weight = weight
        self.safe_distance = safe_distance

    def compute(self, observation, action, info) -> float:
        front = info["front_distance"]
        if front >= self.safe_distance:
            return 0.0
        proximity = 1.0 - front / self.safe_distance
        return -self.weight * proximity


class MotionGatedRewardStrategy(RewardStrategy):
    """Scale another strategy's reward by how fast the car is moving.

    Decorator: wraps any :class:`RewardStrategy` and multiplies its output
    by ``min(1, velocity / ref_speed)``. Applied to posture terms (lane
    centering, heading), it stops them being farmed while stationary -- a
    standing car earns zero posture reward, so "sit still, stay centred" is
    no longer a net-positive plateau. Full reward returns once the car
    reaches ``ref_speed``.

    Forwards :meth:`reset` to the wrapped strategy so stateful inners keep
    working.

    Raises ValueError if ``ref_speed`` is not positive (it is a
    denominator; zero would divide by zero at every step).
    """

    def __init__(self, strategy: RewardStrategy, ref_speed: float = 10.0):
        if ref_speed <= 0:
            raise ValueError(f"ref_speed must be positive. "
                             f"Current value: {ref_speed}")
        self.strategy = strategy
        self.ref_speed = ref_speed

    def compute(self, observation, action, info) -> float:
        factor = min(1.0, max(0.0, info["velocity"] / self.ref_speed))
        return factor * self.strategy.compute(observation, action, info)

    def reset(self):
        self.strategy.reset()


class CompositeRewardStrategy(RewardStrategy):
    """Sum of several reward strategies (Composite pattern).

    Lets the full reward function be declared as a list in ``main.py``.
    :meth:`reset` propagates to every child so stateful terms are cleared
    together at the start of each episode.

    :meth:`compute` rejects a non-finite (NaN/inf) result from any child,
    naming the offending strategy. Without this, a NaN or inf -- e.g.
    from a malformed observation slipping through -- would silently reach
    PPO as the training signal, far harder to trace back to its source
    than a clear exception raised at the point it was produced.
    """

    def __init__(self, strategies: list[RewardStrategy]):
        self.strategies = strategies

    def compute(self, observation, action, info) -> float:
        total = 0.0
        for strategy in self.strategies:
            value = strategy.compute(observation, action, info)
            if not math.isfinite(value):
                raise ValueError(
                    f"{type(strategy).__name__}.compute() returned a "
                    f"non-finite reward ({value!r}) -- check for NaN/inf "
                    f"in the info dict this step."
                )
            total += value
        return total

    def reset(self):
        for strategy in self.strategies:
            strategy.reset()
