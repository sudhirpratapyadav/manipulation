# Open-drawer task — robustness improvement plan

This is the *plan*. The journal is `experiments_log.md`; operational
recipes are in `cluster_workflow.md`; the autonomy contract is in
`AGENT.md`. Read `AGENT.md` first.

## Context

The OSC-based open-drawer task (`Mjlab-Open-Drawer-Osc-Kinova`) trains to
~90–100 % dwell-success at training nominal and transfers to the real
Kinova Gen3 + Robotiq 2F-85 — but transfer is **not robust**. The goal
here is a wider operating envelope (sim and real), surfaced through an
evaluation regime that *finds the breaking point* of each policy along
each axis instead of just confirming the training distribution.

State-based policy. No vision is used or needed.

## Constraints

- Cluster: SLURM, partition `1gpu`, 8-GPU holder pattern (see
  `cluster_workflow.md`). Use `uv run` always.
- W&B: project `mjlab-kinova-tasks-osc`, entity
  `sudhirpratapyadav-indian-institute-of-technology-jodhpur`. One W&B
  run per phase, tagged `phase=N`.
- No vision. Don't add camera or light DR.
- Mjlab APIs available: per-env DR via `mdp.dr.*`, post-scale action
  clipping via `ActionTermCfg.clip` (v1.3.0+), termination/reward
  curricula via `mdp.reward_curriculum` / `mdp.termination_curriculum`,
  `MetricsTermCfg` (per-step, episode-averaged).

## Success metrics

- **Primary — `success_rate` (dwell):** fraction of episode steps where
  `object_to_goal_error < 0.02 m`. Defined in
  `tasks/open_drawer_osc.py:success_rate`. `MetricsTermCfg` averages
  per-step values across the episode → "dwell" success. Stricter than
  terminal success but fixed across phases; comparable phase-to-phase.
- **Secondary:** `object_to_goal_error` (mean), `joint_vel` L2,
  action L2 (used in execution-speed phases).
- **Robustness score:** mean over the 10 sweep axes of the normalized
  envelope width where `success_rate ≥ 0.80`. Computed by
  `eval_sweep.py`. Range [0, 1]. This is the headline number that should
  improve across phases.

**Per-phase decision rule:** keep the change iff `success_rate` doesn't
drop *and* at least one of {robustness score, speed metric} improves.

## Phased training plan

Each phase = one change, one training run, one in-dist eval, one OOD
sweep along the new axis. Read results, decide, then move on.

### Phase 0 — Baseline

No model changes. Establishes reference numbers under the new
metric/eval setup.

Exit: baseline `success_rate` and per-axis breaking points written to
`experiments_log.md`.

### Phase 1 — Initial-pose randomization

The current config is too narrow:
- `joint_delta_deg=5.0`
- `pose_range={}` (base never moves)

Changes:
- `joint_delta_deg` 5° → 15° (consider per-joint deltas if uniform 15
  hurts the distal joints).
- `reset_root_state_uniform.pose_range`: ±2 cm in x/y, ±2° yaw.

If the reward curve degrades, halve before declaring failure.

### Phase 2 — Targeted DR

| Axis | Why for drawer-opening | mjlab API |
|---|---|---|
| Drawer slide frictionloss | Governs pull force; biggest sim2real gap | `dr.dof_frictionloss` |
| Drawer slide damping | Couples with friction; controls stiction | `dr.dof_damping` |
| Drawer base mass+inertia | Cabinet weight unknown in real | `dr.pseudo_inertia(alpha_range=...)` |
| Arm link mass+inertia | Robustness to model error | `dr.pseudo_inertia` on robot links |

Skip: gripper friction (already done), camera/lighting (no vision), terrain.

Use `dr.pseudo_inertia` over `dr.body_mass` — the latter only changes
mass and leaves inertia stale.

Start ranges narrow. Widen in Phase 4 if everything else holds.

### Phase 3 — Slow execution

In order, one at a time. Stop at the first that gives enough slowdown
without dropping success_rate.

- **3a** `ActionTermCfg.clip` on the OSC delta — softer commands without
  cutting the learning signal.
- **3b** Quadratic vel penalty: replace `joint_velocity_hinge_penalty`
  with `-w * sum((qvel/qvel_ref)**2)` so there's gradient inside the
  hinge dead-zone.
- **3c** Curriculum on `max_vel`: ramp the *target speed* down (1.5 →
  0.3 over training), not just the weight.
- **3d** Add the previous *processed* action (post-scale) to actor obs
  alongside `last_action` to stabilize commands.
- **3e** Per-env action-scale DR. Last resort. Custom OSC wrapper.
  Mjlab's `mdp.dr.*` doesn't expose `delta_pos_scale` because it's a
  config field, not a model field.

### Phase 4 — Combined deploy run

All keepers from Phases 1–3, possibly with wider DR. Final candidate
for sim2real.

## Evaluation strategy

Two modes; both reuse the trained checkpoint.

### In-distribution eval

Same task config as training but with `Mjlab-Open-Drawer-Osc-Kinova-Eval`
(corruption off, no curriculum, finite episodes). Confirms no
train→eval regression. Reports: `success_rate`, mean error, mean
episode length.

### OOD sweep — find the breaking point

`eval_sweep.py` runs the per-axis sweep against a checkpoint. For each
axis, it loops over a grid of values (others held nominal), runs N
episodes per setting, classifies as `pass` (≥ 80 %), `degraded`
(50–80 %), or `fail` (< 50 %), and writes:

- `docs/results/<run_name>/sweep_summary.csv` — one row per (axis, value).
- `docs/results/<run_name>/breaking_points.md` — per-axis envelope
  summary plus the headline robustness score.

Sweep axes (per plan §7.2; nominal values are the training defaults):

| Axis | Range | Steps |
|---|---|---|
| Drawer slide friction | nominal × {0.25, 0.5, 1, 2, 4} | 5 |
| Drawer slide damping | nominal × {0.25, 0.5, 1, 2, 4} | 5 |
| Drawer base mass scale | × {0.5, 1, 2, 5} via `pseudo_inertia` | 4 |
| Drawer goal depth | {-0.10, -0.15, -0.20, -0.25, -0.28} m | 5 |
| Drawer init slide | {0, -0.05, -0.10} m | 3 |
| Robot base x offset | {0, 2, 5, 10} cm | 4 |
| Initial joint delta | {5, 10, 20, 30, 45} ° | 5 |
| Action scale (`delta_pos_scale`) | {0.005, 0.01, 0.02, 0.05} m | 4 |
| Arm link mass perturbation | ±{0, 10, 25, 50} % via `pseudo_inertia` | 4 |
| Fingertip slide friction | {0.1, 0.3, 0.6, 1.0, 1.5} | 5 |

Total ~44 settings × N episodes (default `num_envs=64`,
`episodes_per_setting=64`). Single GPU, minutes.

For settings tagged `fail`, dump 3 example trajectories
(joint_pos, ee_pose, action, drawer_slide, terminal reason) under
`docs/results/<run_name>/failures/` for offline inspection.

### What "robust" actually means here

Maximize the *width* of each axis where `success_rate ≥ 0.80`, not the
mean. After each phase the headline is "how much wider did the
operating envelope get along axis X?", not "did mean error go down?".

## Operations

- All training runs go through `slurm/open_drawer_osc_phase<N>.sh`. Per
  phase, copy the previous phase's script, change `experiment-name` to
  `open_drawer_osc_phaseN[_variant]`, set the `phase=N` tag, edit the
  one config knob the phase introduces, then submit (or `srun
  --jobid=<holder>` inside the holder allocation).
- All eval runs go through `slurm/eval_open_drawer_osc.sh` or call
  `eval_sweep.py` directly inside the holder allocation.
- Append a per-phase results entry to `experiments_log.md`.
