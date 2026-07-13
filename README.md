# CleanDrive

A PPO reinforcement-learning agent that learns to drive in the
[MetaDrive](https://github.com/metadriverse/metadrive) simulator — lane
keeping, cornering, light traffic, and reaching a destination — trained
entirely from a shaped reward signal, no hand-written driving rules.

Built with [stable-baselines3](https://stable-baselines3.readthedocs.io/)
on top of a small, SOLID-structured Python codebase (`src/`).

---

## Requirements

- **Python 3.11** (the project venv was built against 3.11.9; pinned in
  `.python-version`)
- Windows, macOS, or Linux (MetaDrive runs on all three; commands below use
  Windows/PowerShell paths where they differ from bash)

## Setup

### Option A: with `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package/version manager.
Its advantage here: it does **not** require Python 3.11 to already be
installed on your machine — it downloads the exact version pinned in
`.python-version` automatically. This avoids any "wrong Python version"
mismatch when reproducing the project on a different machine.

Install `uv` once (skip if you already have it):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repository root:

```bash
uv venv .venv_rl --python 3.11
uv pip install -r requirements.txt --python .venv_rl
```

### Option B: plain `venv` (if Python 3.11 is already installed)

```bash
# from the repository root
python -m venv .venv_rl

# Windows
.venv_rl\Scripts\activate

# macOS / Linux
source .venv_rl/bin/activate

pip install -r requirements.txt
```

---

Either way, this installs MetaDrive, stable-baselines3, PyTorch, and
everything else needed for both training and evaluation.

To run any script without activating the venv, call the venv's Python
directly, e.g. `.venv_rl/Scripts/python.exe src/main.py` (Windows) or
`.venv_rl/bin/python src/main.py` (macOS/Linux).

---

## Project structure

```
src/
  config.py        TrainingConfig — frozen dataclass of all run settings
  reward.py        Reward building blocks (Strategy + Composite + Decorator)
  environment.py    Gymnasium wrapper around MetaDrive (observations + reward)
  agent.py          Thin wrapper around the SB3 PPO model
  orchestrator.py   Wires everything together, drives one training run
  main.py           Entry point: defines map / reward / hyperparameters, trains
  evaluate.py       Entry point: loads a checkpoint, renders it driving
```

Each reward term (progress, lane centering, heading alignment, collision
avoidance, ...) is its own small class in `reward.py` implementing
`RewardStrategy`. `CompositeRewardStrategy` sums a list of them into the
reward actually optimised — the whole reward function is assembled
declaratively in `main.py`, without touching the environment or agent code.

---

## How training works

`src/main.py` is the single entry point for training. It:

1. Builds a MetaDrive `map_config` (which map/scenario set, traffic
   density, lidar setup, ...).
2. Assembles the reward as a list of `RewardStrategy` objects passed to
   `CompositeRewardStrategy`.
3. Builds a `TrainingConfig` (learning rate, total timesteps, checkpoint
   directory, ...).
4. Hands it to `TrainingOrchestrator`, which spins up 24 parallel
   simulated environments (`SubprocVecEnv`) with running observation
   normalisation (`VecNormalize`), and trains a PPO agent
   (`stable_baselines3.PPO`, `MlpPolicy [256, 256]`).

Run it with:

```bash
.venv_rl/Scripts/python.exe src/main.py
```

(macOS/Linux: `.venv_rl/bin/python src/main.py`)

- Progress prints to the console (episode outcomes: `arrive_dest`,
  `out_of_road`, `crash vehicle`, `max step`).
- A checkpoint (model + `VecNormalize` stats) is saved every ~50k steps
  into the run's `checkpoint_path`, so you can evaluate a run without
  waiting for it to finish.
- On completion, the final model (`ppo_model.zip`) and normaliser
  (`vec_normalize.pkl`) are saved to `checkpoint_path`.
- `orchestrator.run(resume_from=...)` resumes training from an existing
  checkpoint (used to continue an earlier run rather than starting from
  scratch) — see `main.py` for the current setting.

### Training was done as a curriculum

Rather than training directly on the hardest scenario, the agent was
built up in stages, each resuming from the *clean* result of the previous
one — straight driving → cornering (both directions) → light traffic →
combined long routes (`map=3`). Each stage changed exactly one variable at
a time (map, one reward term, or one hyperparameter), so every result can
be attributed to a single cause. `HANDOFF.md` in the repository root has
the full history of what was tried, what worked, and why.

### Configuring a run

Everything you'd want to change lives at the top of `main.py`:

```python
map_config = dict(use_render=False, manual_control=False,
                  num_scenarios=20, map=3, horizon=1000,
                  traffic_density=0.05,
                  vehicle_config=dict(lidar=dict(num_lasers=120, distance=50)))
```

- `map`: a MetaDrive map string (e.g. `"S"` straight, `"C"` curve, `"SC"`
  straight-into-curve) or an integer seed for procedurally generated
  multi-block maps (`map=3` combines straights, curves, and junctions on
  long routes).
- `traffic_density`: 0.0–1.0, fraction of lanes populated with other
  vehicles.
- `num_scenarios`: how many procedurally varied scenarios to sample from.

```python
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
```

Add, remove, or reweight terms freely — every class is documented in
`reward.py` with its exact per-step reward range. `MotionGatedRewardStrategy`
wraps another strategy and scales it by current speed, so posture rewards
(lane centering, heading) can't be farmed by sitting still.

```python
config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy,
                        learning_rate=1e-4,
                        total_timesteps=4_000_000,
                        use_sde=False,
                        checkpoint_path=Path("./checkpoint"))
```

- `learning_rate`: use a higher rate (e.g. `3e-4`) when training a genuinely
  new skill from scratch, and a lower rate (e.g. `1e-4`) when resuming a
  model to gently adapt it without destabilising what it already learnt.
- `use_sde`: gSDE (state-dependent exploration). Keep this `False` — an
  earlier run found it produces a bang-bang steering limit cycle (steering
  saturating at full lock) instead of the smooth control it's meant to
  encourage.
- `checkpoint_path` / `log_path`: created automatically (including any
  missing parent directories) if they don't already exist — no need to
  `mkdir` first. When resuming, the directory should already contain
  `ppo_model.zip` and `vec_normalize.pkl` from the checkpoint you're
  resuming from (otherwise there is nothing to resume, and loading it
  fails with its own clear error).

A 2M-step run takes roughly 45 minutes on 24 CPU cores.

---

## Evaluating a trained model

`src/evaluate.py` loads a checkpoint and renders it driving in a MetaDrive
window (`use_render=True`), running the deterministic policy
(`model.predict(..., deterministic=True)`) for 2000 steps.

```bash
.venv_rl/Scripts/python.exe src/evaluate.py
```

Before running it, make sure the `map_config` and `checkpoint_path` at the
top of `evaluate.py` match the run you want to inspect (same map settings
the model was trained/evaluated on — the observation normaliser
`vec_normalize.pkl` is specific to a checkpoint directory, not to a map).

```python
map_config = dict(use_render=True, manual_control=False,
                  num_scenarios=20, map="SC", horizon=1000,
                  traffic_density=0.05,
                  vehicle_config=dict(lidar=dict(num_lasers=120, distance=50)))
...
config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy,
                        checkpoint_path=Path("./checkpoint"))
```

Episode outcomes (`arrive_dest` / `out_of_road` / `crash vehicle` /
`max step`) are printed by MetaDrive itself as each episode ends.

## Testing

Automated tests live in `tests/` (pytest). Run the full suite from the
repository root:

```bash
.venv_rl/Scripts/python.exe -m pytest
```

(macOS/Linux: `.venv_rl/bin/python -m pytest`)

`pytest.ini` at the repository root points pytest at `tests/` and puts
`src/` on the import path, so no packaging/installation step is needed.

What's covered:

- **`test_reward.py`** — every `RewardStrategy` in `reward.py` (progress,
  lane centering, heading alignment, collision, steering smoothness,
  proximity, motion-gating, composition, ...), including edge cases like
  clipping at the boundary and `reset()` clearing stateful terms.
- **`test_reward_economy.py`** — a regression guard for the project's
  central lesson (see `HANDOFF.md`): per-step penalties must stay small
  enough that idling near a blocker for a whole episode never costs more
  than a single crash, or PPO learns to crash on purpose to escape the
  penalty. This test reads the *actual* reward configuration deployed in
  `main.py` and checks that rule automatically, so it fails the moment a
  future weight change would reintroduce that bug.
- **`test_config.py`** — `TrainingConfig.validate()`'s range checks and
  fail-fast behaviour, plus that the config is genuinely immutable.
- **`test_environment.py`** — the pure observation-assembly logic
  (channel order, π-normalisation) always; a handful of tests that build
  a real, minimal MetaDrive environment are marked `integration` (see
  below).
- **`test_agent.py`** — `AgentManager`'s create/train/save/load lifecycle,
  using a tiny synthetic Gymnasium environment instead of MetaDrive so
  these stay fast.
- **`test_orchestrator.py`** — that an invalid config (e.g. a bad
  learning rate) fails *before* any environment is built.

**Scope note:** tests that would require spinning up the real 24-process
training workload (`TrainingOrchestrator`'s happy path) are intentionally
not part of the automated suite — that's the training run itself, not a
unit test. A handful of `environment.py` tests do build one real,
minimal MetaDrive environment and are marked `@pytest.mark.integration`;
they're a few seconds slower and need MetaDrive's assets. Run just the
fast subset with:

```bash
.venv_rl/Scripts/python.exe -m pytest -m "not integration"
```
