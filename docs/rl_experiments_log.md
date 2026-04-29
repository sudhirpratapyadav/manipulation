# RL experiments — open-drawer OSC

Research-perspective journal. **What** changed, **why**, and **what we
got** — separate from infra/cluster details (those live in
`experiments_log.md` and `cluster_workflow.md`). One entry per training
run + one nested block per evaluation. Append; don't rewrite.

## Task

| Field | Value |
|---|---|
| Task ID (train) | `Mjlab-Open-Drawer-Osc-Kinova` (+ `-PhaseN` variants) |
| Task ID (eval)  | `Mjlab-Open-Drawer-Osc-Kinova-Eval` |
| Robot | Kinova Gen3 7-DOF + Robotiq 2F-85 gripper |
| Scene | Cabinet drawer; goal = pull handle to target slide |
| Episode length | 10.0 s @ 10 Hz outer policy → 100 control steps |
| Success criterion (dwell metric) | `object_to_goal_error < 0.02 m` for the step (fraction averaged across episode) |
| Engine | mujoco_warp (training/Lane A); mujoco native (Lane B async sim driver) |

## Observation / action / reward (training reference)

### Obs space — 33 D
Concatenation order (matches the policy checkpoint and
`sim2real_open_drawer_osc.py` / `sim_open_drawer_osc_async.py`):

| # | Name | Dim | Notes |
|---|---|---|---|
| 1 | `joint_vel`        | 7 | arm dq, noise U(−1.5, 1.5) rad/s |
| 2 | `ee_pose`          | 6 | EE pos (3) + axis-angle (3), noise ±0.01 |
| 3 | `gripper_state`    | 1 | `right_driver_joint / 0.8`, noise ±0.01 |
| 4 | `ee_to_object`     | 3 | handle_world − ee_world, noise ±0.01 |
| 5 | `object_pos`       | 3 | handle_world (env-local), noise ±0.01 |
| 6 | `object_to_goal`   | 3 | goal − handle, noise ±0.01 |
| 7 | `goal_pos`         | 3 | goal_world (env-local), noise ±0.01 |
| 8 | `last_action`      | 7 | previous policy action |

Critic uses the same terms with `enable_corruption=False` (no obs noise
on the value function).

### Action space — 7 D
| # | Component | Range | Scale → world |
|---|---|---|---|
| 1–3 | OSC Δposition         | [−1, 1]³ | × `delta_pos_scale=0.01` m |
| 4–6 | OSC Δorientation (axis-angle) | [−1, 1]³ | × `delta_ori_scale=0.02` rad |
| 7   | Gripper command        | [−1, 1]  | linearly mapped to `fingers_actuator` ctrl ∈ [0, 255] |

OSC controller gains: kp_pos=kp_ori=50.0, kd_pos=kd_ori=10.0,
posture_weight=0.0, max_torque=[39,39,39,39,9,9,9] Nm.

### Reward terms

| Term | Func | Weight | Notes |
|---|---|---|---|
| `reach_object`     | `ee_to_object_reward` (Gauss, std=0.15) | +1.0 | EE→handle distance |
| `move_to_goal`     | `object_at_goal_reward` (Gauss, std=0.10) | +1.0 | handle→goal distance |
| `goal_precise`     | `object_at_goal_reward` (Gauss, std=0.05) | +2.0 | tight placement bonus |
| `action_rate_l2`   | `mdp.action_rate_l2` | −0.01 (curriculum: −0.01 → −0.10 over 0–7200 iters) | smoothness |
| `joint_pos_limits` | `mdp.joint_pos_limits` | −10.0 | hard penalty near joint limits |
| `joint_vel_hinge`  | hinge above max_vel=0.5 rad/s | −0.01 (curriculum: −0.01 → −0.10) | low-velocity bias |

### Termination terms

| Term | TimeOut? | Notes |
|---|---|---|
| `time_out` | yes | episode_length_s = 10.0 s |
| `nan_detection` | no | should stay at 0 |
| `ee_ground_collision` | no | bracelet_link vs floor |

### Reset / init-pose ranges (Phase 0 nominal)

| Knob | Phase 0 value |
|---|---|
| `reset_robot_joints.joint_delta_deg` | 5° |
| `reset_base.pose_range` | `{}` (no base offset) |
| `reset_drawer.x_range` | (0.7, 0.9) m |
| `reset_drawer.y_range` | (−0.05, 0.15) m |
| `reset_drawer.init_slide_range` | (−0.02, 0.0) m |
| `drawer_goal.slide_range` | (−0.25, −0.15) m |
| Fingertip friction (slide) | log-U(0.3, 1.5) μ |

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
| `num_envs` | 1024 |
| Steps/iter | 24 × 1024 = 24 576 |

## Eval methodology

### Lane A — in-process OOD sweep
- Module: `kinova_tasks.eval_sweep` (driver) +
  `kinova_tasks.eval_sweep_smoketest` (per-axis override-plumbing test) +
  `kinova_tasks.eval_sweep_handcheck` (harness vs. hand rollout).
- Per axis, hold all other DR/init knobs at their **training nominal**
  and sweep the target value over a small grid. For each (axis, value):
  64 envs × 64 episodes, vec-env step until each env terminates,
  record `object_to_goal_error` at the terminal step → success_rate &
  mean error.
- Output: `docs/results/<run_name>/sweep_summary.csv` and
  `breaking_points.md`.
- Pass / degraded / fail thresholds: 0.80 / 0.50 / below.
- Robustness score = mean normalized envelope width across axes.

### Lane B — headless async sim driver
- Script: `nn_policy/sim_open_drawer_osc_async.py` (self-contained;
  does not import the hardware path).
- Engine: raw `mujoco` native (not `mujoco_warp`).
- Loop shape: 50 Hz outer policy thread, 500 Hz inner OSC + `mj_step`
  thread. Mirrors how the real Kinova would be driven.
- Per-episode `Setting` mutates `mj_model.body_mass`,
  `mj_model.dof_frictionloss`, `mj_model.dof_damping` for the drawer
  base body / slide DOF before the episode starts.
- Used as a sanity check: a policy that's robust in Lane A but flat-zero
  in Lane B is overfitting to mjlab specifics (warp, batched sync
  stepping, etc.).

## Sweep axes (Lane A)

| Axis | Values | Nominal | Apply target |
|---|---|---|---|
| `drawer_slide_friction`     | 0.0025, 0.005, 0.01, 0.02, 0.04 N | 0.01 | `dof_frictionloss` on `drawer_slide` |
| `drawer_slide_damping`      | 0.25, 0.5, 1.0, 2.0, 4.0 N·s/m | 1.0 | `dof_damping` on `drawer_slide` |
| `drawer_base_mass_scale`    | 0.5, 1.0, 2.0, 5.0 × | 1.0 | `body_mass` on `drawer/drawer_base` |
| `goal_depth`                | −0.10, −0.15, −0.20, −0.25, −0.28 m | −0.20 | `drawer_goal.slide_lo/hi` |
| `init_slide`                | 0.0, −0.05, −0.10 m | 0.0 | `reset_drawer.init_slide_range` |
| `robot_base_x_offset`       | 0, 2, 5, 10 cm | 0 | `reset_base.pose_range.x` |
| `init_joint_delta_deg`      | 5, 10, 20, 30, 45 ° | 5 | `reset_robot_joints.joint_delta_deg` |
| `action_scale`              | 0.005, 0.01, 0.02, 0.05 m | 0.01 | `osc_pose.delta_pos_scale` |
| `arm_link_mass_pct`         | 0, 10, 25, 50 % | 0 | per-link `body_mass` perturbation |
| `fingertip_friction_slide`  | 0.1, 0.3, 0.6, 1.0, 1.5 μ | 0.6 | `geom_friction` on pad geoms |

10 axes × ~5 values = ~46 settings/checkpoint.

---

## Run history

Newest at bottom. Each entry: hypothesis, knobs changed, training
trajectory at key checkpoints, eval result, decision.

### Phase 0 — baseline (`open_drawer_osc_phase0`)

- **Hypothesis:** Establish a clean reference under the new metric
  (dwell `success_rate` at 0.02 m) and the new sweep harness.
- **Change vs. previous:** none — reference run.
- **Run name:** `baseline`
- **W&B run:** [52n3yt9o](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/52n3yt9o) (project `mjlab-kinova-tasks-osc`)
- **Checkpoint (canonical):** `model_3800.pt`
- **Iterations:** 3979 of 5000 (early-cancelled at plateau)

Training trajectory:

| Iter | success_rate | object_to_goal_error (m) | mean_reward | mean_ep_length |
|---:|---:|---:|---:|---:|
| 1000 | 0.242 | — | 19.06 | — |
| 2000 | 0.725 | — | 28.48 | — |
| 3000 | ~0.760 | — | ~30.0 | — |
| 3979 (final) | **0.7723** | **0.0395** | **30.32** | 100.00 |

- **In-dist eval:** TBD (Lane A nominal-value setting will give this)
- **OOD sweep (Lane A):** running (job-step 18278.1).
  Output: `docs/results/open_drawer_osc_phase0/sweep_summary.csv`
- **Lane B:** smoke test pending (driver self-contained, ready to run).
- **Decision:** baseline reference. Keep.

---

### Phase 1 — wider initial-pose DR (`open_drawer_osc_phase1`)

- **Hypothesis:** The cabinet may not always be flush-closed and the
  technician places the robot with small position/yaw error. Wider
  init-pose distribution should be absorbed at no in-distribution cost
  and may improve robustness on the
  `init_joint_delta_deg` and `robot_base_x_offset` sweep axes.
- **Change vs. Phase 0:**
  - `reset_robot_joints.joint_delta_deg`: 5° → **15°**
  - `reset_base.pose_range`: `{}` → `{x: ±2 cm, y: ±2 cm, yaw: ±2°}`
  - All other knobs unchanged.
- **Run name:** `init_pose_dr`
- **W&B run:** [4gn2sbub](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/4gn2sbub)
- **Checkpoint (canonical):** `model_3800.pt`
- **Iterations:** 3888 of 5000 (early-cancelled at plateau)

Training trajectory:

| Iter | success_rate | object_to_goal_error (m) | mean_reward | mean_ep_length |
|---:|---:|---:|---:|---:|
| 1500 | 0.703 | — | 28.56 | — |
| 2000 | 0.751 | — | 29.67 | — |
| 3000 | 0.770 | — | 30.11 | — |
| 3888 (final) | **0.7734** | **0.0392** | **30.67** | 100.00 |

- **Read on hypothesis:** ✅ in-distribution SR matches Phase 0 within
  ~0.001 — wider init-pose was absorbed without measurable cost.
  Whether it actually buys robustness is for the OOD sweep to show.
- **OOD sweep (Lane A):** running (job-step 18278.2).
  Output: `docs/results/open_drawer_osc_phase1/sweep_summary.csv`
- **Decision (so far):** keep, pending OOD sweep.

---

### Phase 2 — drawer + arm DR (`open_drawer_osc_phase2`)

- **Hypothesis:** Real drawer slide has friction + damping that vary
  cabinet-to-cabinet and over wear; drawer mass and arm-link mass have
  manufacturing tolerance. DR over these should buy us breaking-point
  margin on the `drawer_slide_friction`, `drawer_slide_damping`,
  `drawer_base_mass_scale`, and `arm_link_mass_pct` axes.
- **Change vs. Phase 1:** add four DR knobs (Phase 1 init-pose deltas
  inherited):
  - drawer slide frictionloss → U(0.005, 0.02) (≈ ±2× nominal 0.01)
  - drawer slide damping → U(0.5, 2.0)
  - drawer base mass → log-U alpha ±0.347 (≈ ±34.7%)
  - arm link mass → log-U alpha ±0.0477 (≈ ±4.77%)
- **Run name:** `drawer_dr`
- **W&B runs:**
  - first attempt (NaN'd, killed): [99v32mk7](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/99v32mk7)
  - second attempt (also NaN'd; cancelled): [ukd2fffm](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/ukd2fffm)
  - **third attempt (clean, current):** [mso8ooz7](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/mso8ooz7)
- **Iterations target:** 4000 (cut from 5000 plan default since
  Phase 0/1 plateaued by iter ~3000; 30% headroom for slower DR
  convergence).
- **Bug found and fixed:** the first launch NaN'd at iter 0 because
  `dr_arm_link_mass` used regex `r".*_link"` for `pseudo_inertia`,
  which matched the Kinova's massless `end_effector_link` (mass=0 →
  NaN) and the gripper `*_spring_link` finger bodies (~0.02 kg, not
  arm dynamics). Tightened to
  `r"(base|shoulder|half_arm_[12]|forearm|spherical_wrist_[12]|bracelet)_link"`.
  Verified env reset + 5 steps NaN-free post-fix.

Training trajectory: TBD (running).

| Iter | success_rate | object_to_goal_error (m) | mean_reward | mean_ep_length |
|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD |

- **OOD sweep (Lane A):** pending — schedule after training.
- **Decision:** TBD.

---

### Phase 3 — slow-execution / safety-aware (planned)

- **Hypothesis:** A real Kinova at high joint velocities is unsafe.
  Reduce action space, tighten the velocity hinge, and/or curriculum
  on `max_vel`. Plan §3 lists 3a–3e in increasing intervention order;
  stop at the first that holds in-dist SR while reducing observed
  joint velocity.
- **3a — quadratic vel penalty:** replace hinge with quadratic
  `−w·||dq||²` after iter 2400.
- **3b — tighter hinge:** drop `max_vel` 0.5 → 0.3 rad/s.
- **3c — vel curriculum:** schedule `max_vel` 0.6 → 0.4 → 0.3 over
  3 stages.
- **3d — OSC delta clip:** clip `delta_pos_scale` action component to
  ±0.5 (effective Δpos ≤ 5 mm).
- **3e — processed-action obs:** add post-clip processed action to obs.
- **Status:** pending; Phase 2 outcome will inform which 3x to try.

---

### Phase 4 — combined deploy run (planned)

- **Hypothesis:** Stack the surviving knobs from Phase 1+2+3 into a
  single training run that's the candidate for sim2real deploy.
- **Status:** pending.

---

## Cross-phase decisions (research)

- **2026-04-29** — Cancelled Phase 0 + Phase 1 at iter ~3900 (plan
  said 5000) because both plateaued at SR ≈ 0.77 by iter 3000.
  Last 1000 iters delivered ≤ 0.02 SR — not worth 2 GPU-h with
  Phase 2 blocked behind Phase 1. `model_3800.pt` adopted as
  canonical for both phases.
- **2026-04-29** — Picked Phase 2 deltas to be **only** drawer slide
  friction/damping + drawer/arm masses, not all of `mdp.dr.*`.
  Reason: keep the experiment isolated so a robustness gain (or loss)
  on the drawer-mass / friction sweep axes is causally attributable to
  these knobs. A "kitchen sink" run conflates effects.
- **2026-04-29** — Did not raise `num_envs` for Phase 2 even though
  the wider DR distribution would benefit from more samples per
  rollout. Rationale: keep all phase-to-phase comparisons at the same
  `num_envs=1024` so the per-iteration delta is meaningful. Will
  revisit if Phase 2 SR fails to recover Phase 0 levels by iter ~3000.

## Notes on metric definitions (training vs. eval)

The training metric `Episode_Metrics/success_rate` is the **dwell**
metric: per-env, per-step, count the fraction of steps where
`object_to_goal_error < 0.02 m`, then average across envs and steps in
the rolling window. So a policy that reaches the goal late and dwells
briefly scores low even if it succeeded.

The OOD-sweep `success_rate` (Lane A, `eval_sweep._run_setting`) is the
**terminal-step** metric: at the step just before each env terminates
(time-out at 100 steps), check `object_to_goal_error < 0.02 m`; success
is binary per episode, then averaged across episodes.

These are not the same number. Phase 0 trained to dwell-SR ≈ 0.77,
which corresponds to terminal-SR ≈ 1.0 on most settings (the policy
*does* reach the goal; it just doesn't dwell perfectly within the
0.02 m threshold for every step of every episode). When comparing
training-time SR to eval-sweep SR, treat them as two different
estimators of the same underlying competence; for cross-phase
comparison, prefer **mean terminal error** over success-binarized
numbers when both eval-SRs saturate near 1.0.

## Open questions

- Does Phase 1 actually move the needle on the `init_joint_delta_deg`
  and `robot_base_x_offset` sweep axes? In-distribution SR matched
  Phase 0; OOD sweep result needed to answer this.
- Phase 1 dwell SR plateaued at 0.77 (same as Phase 0). Is the
  remaining 0.23 floor structural (reach geometry, distal joint
  reach), or is it a converged-policy noise floor we can push by
  tuning `goal_precise.std` or `success_rate.threshold`?
- Lane B (raw mujoco, async) hasn't been smoke-tested on a
  high-success checkpoint yet. The `model_400.pt` smoke test produced
  SR=0 (expected — policy hadn't learned). Re-run on `model_3800.pt`
  is pending.
