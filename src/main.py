"""Entry point: assembles the config and launches a training run.

Defines the map, the composite reward and the training hyper-parameters,
then hands everything to :class:`orchestrator.TrainingOrchestrator`. Run
with ``python src/main.py``.

Fresh curriculum, stage 4 -- the goal (``map=3``). Combines every learnt
building block on long, procedurally generated multi-block routes: straight
driving, both curve directions, and light traffic. Resumes the working
traffic base (``ml_traffic2``: det. 7/9 at density 0.05) and only changes
the map from the ``"SC"`` fixed geometry to the full ``map=3``. Traffic stays
at the mastered density 0.05 (overtaking dense traffic is out of scope).
lr=1e-4 stays gentle -- this is a combine-and-generalise step, not a new
skill; use_sde=False. Reward unchanged (mild wide lane-keeping included).
"""

from pathlib import Path
from config import TrainingConfig
from reward import (ProgressRewardStrategy, HeadingAlignmentRewardStrategy,
                    LaneCenteringRewardStrategy, MotionGatedRewardStrategy,
                    SteeringSmoothRewardStrategy, ProximityPenaltyStrategy,
                    AvoidCollisionRewardStrategy, RouteCompletionBonusStrategy,
                    CompositeRewardStrategy)
from orchestrator import TrainingOrchestrator


map_config = dict(use_render=False, manual_control=False,
                  num_scenarios=20, map=3, horizon=1000,
                  traffic_density=0.05,
                  vehicle_config=dict(lidar=dict(num_lasers=120, distance=50)))

# map=3 polish: still multi-lane (lane-keeping stays relative to the current
# lane -- any lane is fine, it just must not oscillate once on one). Nudged
# up a little to hold the lane better and damp the residual oscillation the
# agent shows near traffic: LaneCentering 1.0 -> 1.5, SteeringSmooth
# 0.3/0.5 -> 0.4/0.8. Paired with the narrower front lidar cone (environment
# _front_distance, +/-5) so ProximityPenalty stops reacting to adjacent-lane
# cars the agent could just pass.
reward_strategy = CompositeRewardStrategy(strategies=[
    ProgressRewardStrategy(weight=200.0),
    MotionGatedRewardStrategy(HeadingAlignmentRewardStrategy(weight=3.0)),
    MotionGatedRewardStrategy(
        LaneCenteringRewardStrategy(weight=1.5, half_lane_width=1.5)),
    SteeringSmoothRewardStrategy(weight_magnitude=0.4, weight_delta=0.8),
    ProximityPenaltyStrategy(weight=0.02, safe_distance=0.3),
    AvoidCollisionRewardStrategy(crash_penalty=-30.0),
    RouteCompletionBonusStrategy(bonus=100.0),
])

# Resume the 2M map=3 model and keep training: map=3 has complex, varied
# scenarios (curves + junctions + traffic), so it needs more steps to learn.
# 4M this run, lr=1e-4 gentle so the tuned reward only refines behaviour.
ckpt_path = Path("./checkpoint")
config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy,
                        learning_rate=1e-4,
                        total_timesteps=4_000_000,
                        use_sde=False,
                        checkpoint_path=ckpt_path)

if __name__ == "__main__":
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run(resume_from=config.checkpoint_path / "ppo_model")
