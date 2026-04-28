# Open-Drawer Task: Robustness & Sim2Real Improvement Plan

## 1. Context

The OSC-based open-drawer task (`Mjlab-Open-Drawer-Osc-Kinova`) already trains
to ~90–100% success on the current setup, and the trained policy *does* transfer
to the real Kinova Gen3 with Robotiq 2F-85 — but transfer is **not robust**.
This plan defines a phased, low-risk path to a more robust policy and, just as
importantly, an **evaluation regime that surfaces failure modes** rather than
just confirming the training distribution.

The deliverable for each training phase is a numerical comparison against a
fixed Phase-0 baseline; the deliverable for each eval cycle is a sweep that
*finds the breaking point* of the policy along a chosen axis.

State-based policy. **No vision is or will be required** for this task.

## 2. Goals

1. **Train a more robust policy** for opening the drawer to a sampled goal
   slide depth.
2. **Slow execution speed** of the resulting policy without losing learning
   throughput.
3. **Build a diverse evaluation harness** that:
   - confirms in-distribution performance (sanity check),
   - sweeps each DR / action-scale / payload axis to find the policy's
     breaking points,
   - auto-summarizes failure modes per category.
4. Keep the iteration loop tight: **one change per phase**, retrain, evaluate,
   decide whether to keep it before the next phase.

## 3. Constraints / Environment

- Cluster: SLURM, single-GPU jobs in `slurm/` (e.g. `slurm/open_drawer_osc.sh`)
  using `uv run train ...`.
- W&B logging: project `mjlab-kinova-tasks-osc`, entity
  `sudhirpratapyadav-indian-institute-of-technology-jodhpur`.
- Mjlab APIs available: per-env DR via `mdp.dr.*`, per-term metrics via
  `MetricsTermCfg(reduce="last", per_substep=...)`, post-scale action clipping
  via `ActionTermCfg.clip` (v1.3.0+), termination/reward curricula via
  `mdp.reward_curriculum` / `mdp.termination_curriculum`.

## 4. Decisions & non-decisions (recorded so we don't relitigate)

- **Per-env link length / mesh swap:** out of scope. Mjlab uses a single shared
  `MjModel`; geometry is structural, not per-env. Per-env DR is supported for
  *parameters* (mass, inertia, friction, damping, gains, limits).
- **Per-env action scale DR:** not natively supported by `mdp.dr.*`
  (`delta_pos_scale` is a config field, not a model field). If we end up
  needing it (Phase 3e), we'll do it inside the OSC action term with a per-env
  scale buffer resampled on reset.
- **Vision:** not used. Don't add camera or light DR.
- **Eval scope:** in-distribution sanity + **out-of-distribution sweeps**
  along single axes are first-class deliverables, not afterthoughts.

## 5. Success metrics

A single numerical comparison across phases requires a fixed metric. Use:

- **Primary:** `success_rate` = fraction of episodes where final
  `object_to_goal_error < 0.02 m` (2 cm). Logged via
  `MetricsTermCfg(reduce="last")` so we get the *terminal* value, not the
  episode average.
- **Secondary:** mean episode `object_to_goal_error`, mean `joint_vel` L2,
  mean action L2 (for execution-speed phases).
- **Per-phase decision rule:** keep the change iff success rate doesn't drop
  *and* one of {speed metric, robustness sweep score} improves.

## 6. Phased training plan

Each phase = **one change**, one training run, one in-distribution eval, one
OOD sweep along the new axis. Read results, then decide.

### Phase 0 — Baseline (no model changes)

- Add a `success_rate` metric (`MetricsTermCfg(reduce="last")`).
- Train current task end-to-end; archive checkpoint as `baseline.pt`.
- Run the in-distribution eval and one full sweep set (Section 7) so we have
  comparison numbers from day one.

**Exit criterion:** baseline numbers are checked in / written to W&B and
referenced in this doc's "Results" section.

### Phase 1 — Initial-pose randomization

What's missing right now:

- `joint_delta_deg=5.0` is tiny.
- `pose_range={}` in `reset_root_state_uniform` — base never moves.
- Drawer base position *is* randomized — keep it.

Changes:

- Increase `joint_delta_deg` from 5.0 → 15.0 (per-joint deltas if needed —
  bigger for `joint_1`/`joint_4`, smaller distal).
- Add small base XY offset via `reset_root_state_uniform` `pose_range`
  (e.g. ±2 cm in x and y, ±2° yaw).

**Risk:** too aggressive and learning slows. If reward curve degrades, halve
the deltas before deciding it failed.

### Phase 2 — Targeted DR (most-relevant only)

| Axis | Why for drawer-opening | API |
|---|---|---|
| Drawer slide friction loss | Governs pull force needed; biggest sim2real gap | `dr.dof_frictionloss` on drawer slide |
| Drawer slide damping | Couples with friction; controls stiction | `dr.dof_damping` on drawer slide |
| Drawer base mass | Cabinet weight unknown in real | `dr.body_mass` on drawer base |
| Robot link mass perturbation | Robustness to payload / model error | `dr.body_mass` on arm links, ±10 % |

Skip: gripper friction (already done), camera/lighting (no vision), terrain.

Start ranges narrow. Widen in Phase 4 if everything else holds.

### Phase 3 — Slow execution

In order, **one at a time**. Stop at the first one that gives enough slowdown
without dropping success.

- **3a `ActionTermCfg.clip`** (v1.3.0). Clip processed OSC delta after
  scaling. Forces softer commands without cutting the learning signal.
- **3b Quadratic vel penalty.** Replace `joint_velocity_hinge_penalty` with
  `-w * sum((qvel/qvel_ref)**2)` — gives gradient toward "even slower" inside
  the hinge dead-zone.
- **3c Curriculum on `max_vel`.** Today only the *weight* ramps. Make the
  *target speed* shrink: `max_vel` 1.5 → 0.3 over training. Learn fast first,
  slow down later.
- **3d Previous-action obs.** Add the previous *processed* (post-scale)
  action to observations alongside `last_action`. Helps stabilize commands.
- **3e Per-env action scale DR.** Last resort. Custom OSC wrapper; only if
  3a–d are insufficient.

### Phase 4 — Combined / "deploy" run

Once each phase has individually moved the metric in the right direction, do
one combined run with all keepers, possibly widening DR ranges. This is the
candidate for sim2real deployment.

## 7. Evaluation strategy

Two distinct evaluation modes. Both reuse the trained checkpoint.

### 7.1 In-distribution eval ("training sanity check")

Same task config as training, just with `play=True` and corruption disabled.
Goal: confirm we didn't regress between train and eval (matches what you
already do today).

Reports: success rate, mean error, mean episode length.

### 7.2 OOD sweep eval ("find the breaking point")

Build a separate task variant — call it `Mjlab-Open-Drawer-Osc-Kinova-Eval`
— that exposes each DR/action axis as a CLI override and runs a fixed N
episodes per setting. The script:

1. Loops over a grid of values along **one axis at a time** (held-out
   distribution per axis).
2. Records success rate + mean error per setting.
3. Auto-tags each setting as `pass` / `degraded` / `fail` against
   configurable thresholds (e.g. ≥80% / 50–80% / <50% success).
4. Writes a per-axis breaking-point report (last value at which the policy
   still passes) to W&B and to `docs/results/`.

Initial sweep axes (each run independently while others held at nominal):

| Axis | Range | Steps |
|---|---|---|
| Drawer slide friction loss | nominal × {0.25, 0.5, 1, 2, 4} | 5 |
| Drawer slide damping | nominal × {0.25, 0.5, 1, 2, 4} | 5 |
| Drawer base mass | nominal × {0.5, 1, 2, 5} | 4 |
| Drawer goal depth | {-0.10, -0.15, -0.20, -0.25, -0.28} m | 5 |
| Drawer init slide | {0, -0.05, -0.10} m (start partially open) | 3 |
| Robot base XY offset | {0, 2, 5, 10} cm in random dir | 4 |
| Initial joint delta | {5, 10, 20, 30, 45} ° | 5 |
| Action scale (`delta_pos_scale`) | {0.005, 0.01, 0.02, 0.05} m | 4 |
| Arm link mass perturbation | ±{0, 10, 25, 50} % | 4 |
| Fingertip friction (slide axis) | {0.1, 0.3, 0.6, 1.0, 1.5} | 5 |

Total: ~44 settings × N episodes (e.g. N=64 envs × 1 episode) — single GPU,
minutes.

Output:
- `docs/results/<run_name>/sweep_summary.csv` — one row per (axis, value).
- `docs/results/<run_name>/breaking_points.md` — human summary: for each axis,
  the last value where success ≥ 80 %.
- Per-axis line plot pushed to W&B.

### 7.3 Failure-case capture

For settings tagged `fail`, dump 3 example trajectories (joint_pos, ee_pose,
action, drawer_slide, terminal reason) to `docs/results/<run_name>/failures/`
for offline inspection. Helps build intuition about *why* the policy breaks
(e.g. "loses grip when drawer mass > X", "overshoots when delta_pos_scale > Y").

### 7.4 What "robust" actually means here

We're not optimizing average performance — we're maximizing the **width of
each axis where success ≥ 80 %**. After each phase the headline number is
"how much wider did the operating envelope get along axis X?", not "did mean
error go down?".

## 8. Operations

### Training

- All runs go through `slurm/open_drawer_osc.sh`. Per phase, copy the script
  to `slurm/open_drawer_osc_phaseN.sh`, change `--agent.experiment-name` to
  `open_drawer_osc_phaseN`, and `sbatch` it.
- W&B project: `mjlab-kinova-tasks-osc`. Each phase is a separate run, tagged
  `phase=N` for grouping.

### Eval

- New script `slurm/eval_open_drawer_osc.sh` that runs the OOD sweep against
  a checkpoint.
- Eval task ID: `Mjlab-Open-Drawer-Osc-Kinova-Eval` (created in Phase 0,
  alongside `success_rate` metric).

### Records (this doc)

Append a "Results" entry per phase below. Each entry: phase ID, W&B run URL,
key numbers (success in-dist, breaking points along each axis), keep/drop
decision.

## 9. Results log

### Phase 0 — Baseline
- Run URL:
- In-dist success rate:
- Sweep breaking points:
- Notes:

### Phase 1 — Init-pose randomization
- Run URL:
- In-dist success rate:
- Sweep breaking points:
- Decision:

<!-- Add Phase 2, 3, 4 entries as we go. -->
