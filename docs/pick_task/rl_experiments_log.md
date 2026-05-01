# RL experiments — pick-cube OSC

Research-perspective journal for the cube-pick task. **What** changed,
**why**, and **what we got**. One entry per training run + one nested
block per evaluation. Append; don't rewrite.

The drawer task's logs (`docs/experiments_log.md`,
`docs/rl_experiments_log.md`, `docs/hyperparam_experiments.md`) are the
prior-art reference — we lean heavily on lessons learned there:

- Physics DR on most axes is absorbed by OSC and doesn't help (drawer
  Phase 2 result).
- Init-pose / spawn-pose DR with curriculum is the only DR that
  consistently moved the operating envelope (drawer baseline_dr).
- The plateau-at-0.77 ceiling on the drawer was structural (controller
  resolution + reward shape), not a DR coverage issue.

## Task

| Field | Value |
|---|---|
| Task ID (train) | `Mjlab-Pick-Cube-Osc-Kinova` |
| Task ID (eval)  | (none yet — register a `-Eval` variant when needed) |
| Robot | Kinova Gen3 7-DOF + Robotiq 2F-85 gripper |
| Scene | Tabletop cube; goal = aerial 3D position |
| Episode length | 10.0 s @ 10 Hz outer policy → 100 control steps |
| Success criterion | `cube_to_goal_error < 0.02 m` (per-step dwell — see Note) |
| Engine | mujoco_warp (training) |

**Note on success metric:** the file currently registers
`cube_to_goal_error` and `ee_to_cube_error` only as `MetricsTermCfg`,
not as an explicit `success_rate`. Add a thresholded indicator
metric before the first OOD sweep so we have a comparable headline
number across runs.

## Observation / action / reward (training reference)

### Obs space — 33D
Mirrors the drawer task structure (state-based, no vision).

| # | Name | Dim | Notes |
|---|---|---|---|
| 1 | `joint_vel`     | 7 | arm dq, noise U(−1.5, 1.5) rad/s |
| 2 | `ee_pose`       | 6 | EE pos (3) + axis-angle (3), noise ±0.01 |
| 3 | `gripper_state` | 1 | `right_driver_joint / 0.8`, noise ±0.01 |
| 4 | `ee_to_cube`    | 3 | cube_world − ee_world, noise ±0.01 |
| 5 | `cube_pos`      | 3 | cube_world (env-local), noise ±0.01 |
| 6 | `cube_to_goal`  | 3 | goal − cube, noise ±0.01 |
| 7 | `goal_pos`      | 3 | goal_world (env-local), noise ±0.01 |
| 8 | `last_action`   | 7 | previous policy action |

Critic uses the same terms with `enable_corruption=False`.

### Action space — 7D

| # | Component | Range | Scale → world |
|---|---|---|---|
| 1–3 | OSC Δposition | [−1, 1]³ | × `delta_pos_scale=0.01` m |
| 4–6 | OSC Δorientation (axis-angle) | [−1, 1]³ | × `delta_ori_scale=0.02` rad |
| 7   | Gripper command | [−1, 1] | linearly mapped to `fingers_actuator` ctrl ∈ [0, 255] |

OSC controller gains: kp_pos=kp_ori=50.0, kd_pos=kd_ori=10.0,
posture_weight=0.0, max_torque=[39,39,39,39,9,9,9] Nm.

### Reward terms

| Term | Func | Weight | Notes |
|---|---|---|---|
| `reach_cube`        | `ee_to_cube_reward` (Gauss, std=0.15)       | +1.0 | EE→cube distance |
| `lift_to_goal`      | `cube_at_goal_reward` (Gauss, std=0.10)     | +1.0 | cube→goal distance |
| `goal_precise`      | `cube_at_goal_reward` (Gauss, std=0.05)     | +2.0 | tight placement bonus |
| `action_rate_l2`    | `mdp.action_rate_l2`                        | −0.01 (curriculum: −0.01 → −0.10 over 0–7200 env-steps) | smoothness |
| `joint_pos_limits`  | `mdp.joint_pos_limits`                      | −10.0 | hard penalty near joint limits |
| `joint_vel_hinge`   | hinge above max_vel=0.5 rad/s               | −0.01 (curriculum: −0.01 → −0.10) | low-velocity bias |
| ~~`ee_ground_force`~~ | exponential force penalty on EE-floor contact | disabled (commented out) | re-enable if EE slamming becomes an issue |

No `grasp` / `gripper_state` reward — the cube can't be dragged to an
aerial goal, so the policy must close the gripper to succeed (in
contrast to the drawer task, where the policy could hook+drag with
open fingers — that pathology shouldn't occur here).

### Termination terms

| Term | TimeOut? | Notes |
|---|---|---|
| `time_out`        | yes  | episode_length_s = 10.0 s |
| `nan_detection`   | no   | should stay at 0 |
| `cube_out_of_bounds` | no | cube outside workspace box (centered at home, half-extents (0.35, 0.25, 0.40) m). Widened from (0.20, 0.20, 0.30) to envelope the curriculum-end cube spawn ranges. |

`ee_ground_collision` termination is commented out (sensor still
present). Same rationale as drawer: ground collision shouldn't
auto-terminate during exploration.

### Reset / init-pose ranges (curriculum start)

These are the **iter ≤ 500** values; ramps below describe how they
widen.

| Knob | Start (iter ≤ 500) |
|---|---|
| `reset_robot_joints.joint_delta_deg` | 5° |
| `reset_base.pose_range` | `{}` (no base offset) |
| `reset_cube_position` | x ∈ ±0.10, y ∈ −0.5 ± 0.10, z ∈ [0.02, 0.03] |
| `pick_goal.x/y/z` | x ∈ ±0.10, y ∈ −0.5 ± 0.10, z ∈ [0.10, 0.30] |
| Fingertip friction (slide) | log-U(0.3, 1.5) μ |

### Domain randomization (startup events, per-env at scene reset)

| Knob | Distribution | Source target |
|---|---|---|
| Fingertip slide friction | U(0.3, 1.5) | `(left|right)_pad[12]` geom friction axis 0 |
| Fingertip spin friction | log-U(1e-4, 2e-2) | same geoms, axis 1 |
| Fingertip roll friction | log-U(1e-5, 5e-3) | same geoms, axis 2 |
| Cube slide friction | U(0.3, 1.5) | `cube_geom` friction axis 0 |
| Cube mass (~50 g nominal) | log-U scale [0.5, 2.0] (α = ±0.347) via `pseudo_inertia` | `cube` body |

Skipped (lessons from drawer):
- Arm-link mass DR — drawer P0 resweep showed bare baseline already
  robust to ±50% arm-mass perturbations; OSC absorbs it.
- Drawer-style external impulses on the cube — the cube isn't
  anchored, so an impulse just teleports it.
- `delta_pos_scale` (action-scale) DR — drawer showed action_scale is
  brittle and DR doesn't fix it.

### Curricula (linear ramp on PPO iter)

All four init-distribution curricula linearly interpolate between
`_RAMP_START_ITER=500` and `_RAMP_END_ITER=3000` (PPO iters, computed
from `common_step_counter // num_steps_per_env`). Held at start
values for iter ≤ 500 and at end values for iter ≥ 3000.

| Knob | Start (≤ iter 500) | End (≥ iter 3000) |
|---|---|---|
| Cube spawn x half-extent | ±0.10 m | **±0.30 m** |
| Cube spawn y half-extent (centered −0.5) | ±0.10 m | **±0.20 m** |
| Cube spawn z range | [0.02, 0.03] m (fixed) | [0.02, 0.03] m |
| Goal x half-extent | ±0.10 m | **±0.30 m** |
| Goal y half-extent (centered −0.5) | ±0.10 m | **±0.20 m** |
| Goal z range | [0.10, 0.30] m | **[0.025, 0.40] m** |
| Joint init delta | 5° | **20°** |
| Base xy half-extent | 0 | **±0.05 m** |
| Base yaw half-extent | 0 | **±10°** |

Plus existing reward-weight curricula on `action_rate_l2` and
`joint_vel_hinge` (−0.01 → −0.10 over env-steps 0 → 7200), inherited
from the original task config.

Schedule rationale (matches `baseline_dr` on the drawer task):
- 500 iters frozen at narrow init lets the policy learn the basic
  reach-grasp-lift before the distribution widens. Avoids cold-start
  failure.
- Linear ramp 500 → 3000 matches the time the drawer phases spent
  reaching SR ≈ 0.7 plateau on the narrower distribution.
- 2000 iters frozen at full width gives the policy ≥2× plateau time
  at peak before evaluation.

### PPO / agent

| Field | Value |
|---|---|
| Algorithm | PPO (rsl_rl `OnPolicyRunner`) |
| Actor / critic | MLP [512, 256, 128] elu, obs-normalization on |
| Steps per env per rollout | 24 |
| Learning epochs / minibatches | 5 / 4 |
| LR | 1e-3 (adaptive schedule, desired_kl=0.01) |
| γ / λ_GAE | 0.99 / 0.95 |
| entropy_coef | 0.005 |
| Clip param | 0.2 |
| `num_envs` | 1024 (default — set at launch) |
| Steps/iter | 24 × 1024 = 24 576 |

Same as the drawer baseline; inherited via `kinova_ppo_runner_cfg`.

## Eval methodology

Lane A (in-process OOD sweep) and Lane B (headless async sim driver)
should mirror the drawer setup, but **neither is wired up yet for
this task.** TODO before the first evaluation:

1. Register a `Mjlab-Pick-Cube-Osc-Kinova-Eval` task variant that
   freezes the curriculum at the iter ≥ 3000 end-state, disables obs
   corruption, and uses finite episodes.
2. Define an explicit `success_rate` metric — per-step indicator
   `cube_to_goal_error < 0.02 m`, averaged across the episode (dwell
   metric, same as drawer). Optionally also a terminal-step success.
3. Define the OOD sweep axes for pick. Likely starting set:
   - cube spawn x/y extent (extrapolate beyond curriculum end)
   - goal x/y/z extent (same)
   - cube mass scale (extrapolate beyond DR range)
   - cube friction (extrapolate beyond DR range)
   - fingertip friction (extrapolate beyond DR range)
   - init joint delta (extrapolate beyond curriculum end)
   - base xy / yaw (extrapolate beyond curriculum end)
   - action_scale (delta_pos_scale)
   - obs noise (object position σ)
4. A clean Lane B headless async-sim driver (mirrors
   `nn_policy/sim_open_drawer_osc_async.py` if it exists; new file
   `nn_policy/sim_pick_cube_osc_async.py`).

## Run history

Newest at bottom. Each entry: hypothesis, knobs changed, training
trajectory at key checkpoints, eval result, decision.

### baseline_dr (planned — not yet trained)

- **Hypothesis:** With cube friction + cube mass DR + a 4-curriculum
  init-distribution widening (cube spawn, goal, joint init, base
  pose), the policy should learn a general cube-picker that is
  robust to:
  - cube placement anywhere in a 60 cm × 40 cm tabletop region
  - goals anywhere in a 60 cm × 40 cm × 37.5 cm aerial volume
    (including near-ground placements at z = 2.5 cm)
  - ±35% cube mass variation
  - cube material variation (slide friction U(0.3, 1.5))
  - ±20° joint init noise
  - ±5 cm base xy and ±10° base yaw mounting variation
- **Change vs. previous baseline:** there is no previous trained
  pick-cube checkpoint to compare to — this is the first run with
  DR + curricula. If a quick "Phase 0" baseline (no DR, no
  curriculum) is wanted as a comparison reference, train one
  separately and slot it in above this entry.
- **Run name:** `pick_baseline_dr` (suggested)
- **Task ID:** `Mjlab-Pick-Cube-Osc-Kinova`
- **Launcher:** TBD — model after
  `slurm/open_drawer_osc_baseline_dr.sh`
- **Decision criteria:** plateau dwell-success ≥ 0.6 by iter ~4000;
  OOD sweep robustness on extrapolation axes (TBD once eval harness
  is wired up). Specifically watch:
  - cube spawn / goal extent passing at the curriculum end-state
    (interpolation regime)
  - generalization beyond curriculum end (extrapolation regime)
  - sim2real-relevant axes (cube mass, cube friction, fingertip
    friction)
- **Status:** not yet launched.

## Cross-task lessons being applied (from drawer)

- **Physics DR is mostly a no-op on OSC-controller policies.** Skip
  arm-link mass, gravity, torque-channel noise — confirmed
  irrelevant on the drawer task. Keep DR only at the contact
  interface (cube friction, cube mass, fingertip friction) where it
  *might* affect grasp quality.
- **Init-distribution curriculum is the high-leverage knob.** Drawer
  baseline_dr's 3-curriculum init widening (drawer cube extent,
  joint delta, base pose) was the only intervention that
  meaningfully widened the operating envelope. We expect the
  4-curriculum widening here to do the same.
- **Don't stack experiments.** One change per run, plateau, eval.
  Mixing init-pose DR + impulse curriculum + reward shaping in one
  run makes it hard to attribute gains/regressions.
- **The dwell-SR plateau may be structural.** Drawer hit a 0.77
  ceiling regardless of DR — the bottleneck was OSC resolution +
  reward Gaussian std, not data. If pick plateaus low, look at
  `goal_precise.std` (0.05 m) and `delta_pos_scale` (0.01 m) before
  adding more DR.
- **Cancel doomed runs early.** Drawer cancel-signals apply: NaN
  rising, mean_reward flat or decreasing for ≥200 iters past
  warmup, success_rate not rising past iter 1000 in a config that
  previously trained, episode_length collapsed.

## Open questions

- Will the policy actually close the gripper to grasp the cube, or
  will it find some sim-only "scoop" trick? The drawer task showed
  surprising open-gripper-drag behavior under a similar reward
  setup; pick should be safer (no aerial goal is reachable by
  dragging) but worth verifying on the trained policy via video.
- Are 6D OSC actions (Δpos + Δori) overkill for a top-down cube
  grasp? The orientation channel adds 3 action dims that may just
  be exploration noise. Defer until we see the trained behavior;
  if the policy thrashes orientation, consider freezing it (3D
  Δpos + 1D gripper = 4D total).
- Does `delta_pos_scale=0.01` give the right max-velocity envelope
  for pick? Drawer policies were brittle to action_scale changes;
  reaching a moving aerial goal needs more EE travel per step than
  a drawer pull, so 0.01 may bottleneck the lift phase. Watch
  trajectories on a trained policy.
- Should the eval `success_rate` be dwell-based (drawer's choice)
  or terminal-based? For pick, the meaningful success is "cube at
  goal at end of episode," not "cube at goal for many timesteps in
  a row" — terminal may be more honest. Pick a definition before
  the first eval and freeze it across runs.

## Notes on metric definitions (training vs. eval)

When the OOD sweep harness is wired up, document here:
- **Training `success_rate` (dwell):** per-env, per-step
  indicator `cube_to_goal_error < 0.02 m`, averaged across the
  episode and rolling window.
- **Eval `success_rate` (terminal):** at the step just before
  each env terminates, check `cube_to_goal_error < 0.02 m`;
  binary per episode, averaged across episodes.

These are two different estimators of the same underlying
competence; the drawer task's `Notes on metric definitions`
section explains the discrepancy in detail.
