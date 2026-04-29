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

- **In-dist eval:** SR=1.00 at all nominal settings (read from sweep
  results — drawer_slide_friction=0.01, init_joint_delta=5°, etc.).
- **OOD sweep (Lane A):** complete (job-step 18278.20).
  **Robustness score: 0.696.** 7/10 axes pass at full envelope.
  Breaking points: action_scale fails outside 0.01, arm_link_mass
  fails at any perturbation, init_joint degrades at 20° / fails at
  30°+, goal_depth fails at -0.28.
  Files: `docs/results/open_drawer_osc_phase0/{sweep_summary.csv,breaking_points.md}`.
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
- **OOD sweep (Lane A):** complete (job-step 18278.2).
  **Robustness score: 0.746 (+0.050 vs Phase 0).** 7/10 axes pass at
  full envelope.
  Files: `docs/results/open_drawer_osc_phase1/{sweep_summary.csv,breaking_points.md}`.
- **Hypothesis confirmed:** Phase 1 init-pose DR widens
  `init_joint_delta_deg` envelope ~3-4× while leaving every other
  axis identical to Phase 0:

  | init_joint_delta | P0 SR | P1 SR | Δ |
  |---:|---:|---:|---:|
  | 10° | 1.00 | 1.00 | — |
  | 20° | 0.72 | **1.00** | +0.28 |
  | 30° | 0.44 (fail) | **0.94 (pass)** | +0.50 |
  | 45° | 0.20 (fail) | 0.73 (degraded) | +0.53 |

  Same weak axes inherited: action_scale, arm_link_mass, goal_depth.
- **Decision:** keep. Adopt Phase 1 init-pose DR as the floor for
  Phase 2+ (already built in).

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

Training trajectory (mso8ooz7, early-cancelled at iter ~2700):

| Iter | success_rate | object_to_goal_error (m) | mean_reward | mean_ep_length |
|---:|---:|---:|---:|---:|
| 1500 | ~0.72 | ~0.05 | ~28.5 | 100 |
| 2000 | ~0.76 | ~0.043 | ~30.0 | 100 |
| 2541 | 0.7733 | 0.0391 | 30.22 | 100 |
| 2700 (final) | ~0.7711 | ~0.04 | ~30.0 | 100 |

  No NaN events on mso8ooz7 (iter 0..2700 all clean post-regex-fix).
  SR delta over iter 1500-1800 (mean 0.7630) → iter 2400-2700 (mean
  0.7711) was +0.008 over 900 iters — same plateau criterion that
  cancelled Phase 0 + 1.

- **Checkpoint (canonical):** `model_2700.pt`
- **OOD sweep (Lane A):** complete (job-step 18278.21).
  **Robustness score: 0.721 — *worse* than Phase 1 (0.746), better
  than Phase 0 (0.696).**
  Files: `docs/results/open_drawer_osc_phase2/{sweep_summary.csv,breaking_points.md}`.
- **Hypothesis NOT confirmed.** Two failure modes:
  1. **`arm_link_mass_pct` still fails at 10% perturbation** (SR=0,
     episode_length=1, instant crash). Same as P0/P1. The arm-link
     DR `alpha=±0.0477` (≈ ±4.77%) was too narrow vs the 10% test
     perturbation and the policy got no useful margin. Either widen
     the DR (e.g. `alpha=±0.10`) or accept that arm-mass robustness
     is structurally hard with `pseudo_inertia` and isn't a
     reachable target.
  2. **`init_joint_delta_deg` regresses vs Phase 1.** P1 passes 30°
     at SR=0.94; P2 only 0.69 (degraded). Likely undertraining
     (P2 ran 2700 iters vs P1's 3888). The added DR distractors
     slowed convergence on the init-pose component even though
     Phase 2 trained 5-7× faster on the *reach-and-grasp* component
     (cross-phase table earlier). Implication: time-to-plateau on
     each axis is *not* uniform under DR.
- **Drawer-axis result:** all three drawer axes (slide_friction,
  slide_damping, base_mass_scale) stayed at SR=1.00 across the swept
  envelope. But Phase 0/1 were already at 1.00 there — there was no
  headroom for Phase 2 to gain. The drawer DR is *neither* helping
  *nor* hurting on its target axes.
- **Decision:** **Reject Phase 2 deltas for the deploy run.** Use
  Phase 1 as the DR floor for Phase 3. If we want arm-link
  robustness, the right next experiment is to widen the alpha range
  to ±0.10 and retrain on top of Phase 1 (separate from Phase 3
  slow-execution).

---

## What's randomized vs. what's tested (P0 / P1 / P2)

### Train-time DR knobs

| Knob | Phase 0 (baseline) | Phase 1 | Phase 2 (= P1 + drawer/arm) |
|---|---|---|---|
| Joint reset angle | ±5° | **±15°** | ±15° (inherited) |
| Robot base pose | none (fixed) | **±2 cm x, ±2 cm y, ±2° yaw** | inherited |
| Drawer slide friction | fixed (0.01) | fixed | **U(0.005, 0.02)** = ½× to 2× |
| Drawer slide damping | fixed (1.0) | fixed | **U(0.5, 2.0)** = ½× to 2× |
| Drawer base mass | fixed | fixed | **scale ∈ [0.5, 2.0]** (log-uniform α) |
| Arm-link mass (7 arm bodies) | fixed | fixed | **scale ∈ [0.91, 1.10]** (±10%, log-uniform α) |
| Object/goal/handle position | fixed | fixed | fixed |
| Fingertip friction | fixed | fixed | fixed |
| Action scale (OSC delta) | fixed at 0.01 | fixed | fixed |
| Velocity max (joint hinge) | fixed | fixed | fixed |

All other env params (timestep, reward weights, gripper, observation
noise, PPO hyperparameters, network architecture, seed) are identical
across phases.

### Eval-time perturbations (Lane A OOD sweep — same for all phases)

64 envs × 64 episodes per setting; success = `object_to_goal_error
≤ 0.02 m` at terminal step. **Bold** = nominal value matching training
distribution.

| Axis | Tested values | Coverage by P2's training-DR |
|---|---|---|
| `drawer_slide_friction` (N) | 0.0025, 0.005, **0.01**, 0.02, 0.04 | middle 3 in DR; 0.0025 + 0.04 are extrapolation |
| `drawer_slide_damping` (N·s/m) | 0.25, 0.5, **1.0**, 2.0, 4.0 | middle 3 in DR; 0.25 + 4.0 are extrapolation |
| `drawer_base_mass_scale` (×) | 0.5, **1.0**, 2.0, 5.0 | first 3 in DR; 5.0 is extrapolation |
| `goal_depth` (m) | -0.1, -0.15, **-0.2**, -0.25, -0.28 | not in DR |
| `init_slide` (m, drawer pre-open) | **0.0**, -0.05, -0.1 | not in DR |
| `robot_base_x_offset` (cm) | **0**, 2, 5, 10 | P1: ±2 cm — interpolation only at 0/2 |
| `init_joint_delta_deg` (°) | **5**, 10, 20, 30, 45 | P1: ±15° — interpolation only at 5/10 |
| `action_scale` (m, OSC Δpos) | 0.005, **0.01**, 0.02, 0.05 | not in DR |
| `arm_link_mass_pct` (±%) | **0**, 10, 25, 50 | P2: ±10% — interpolation only at 0/10 |
| `fingertip_friction_slide` (μ) | 0.1, 0.3, **0.6**, 1.0, 1.5 | not in DR |

The eval envelope is **deliberately wider than each phase's training
DR** on every covered axis — passing inside the DR range is the
interpolation case; passing outside is the actual robustness test.

### Per-axis result matrix

| Axis | P0 result | P1 result | P2 result |
|---|---|---|---|
| drawer_slide_friction | all pass | all pass | all pass |
| drawer_slide_damping | all pass | all pass | all pass |
| drawer_base_mass_scale | all pass | all pass | all pass |
| goal_depth | fails at -0.28 | fails at -0.28 | fails at -0.28 |
| init_slide | all pass | all pass | all pass |
| robot_base_x_offset | all pass | all pass | all pass |
| init_joint_delta_deg | degrades 20°, fails 30° | **passes 30°, degrades 45°** | passes 20°, degrades 30° |
| action_scale | only nominal passes | only nominal passes | only nominal passes |
| arm_link_mass_pct | all pass (post regex-fix resweep) | all pass (resweep) | all pass (resweep) |
| fingertip_friction_slide | all pass | all pass | all pass |

(Pre-fix arm_link_mass_pct entries showed SR=0/episode_length=1 at every
non-zero value — that was an `eval_sweep._override_arm_link_mass_pct`
regex bug NaN'ing the eval, not a robustness failure. See cross-phase
decision below.)

### Observations / analysis

1. **Most axes were never broken.** 8 of the 10 swept axes pass at
   full envelope on the bare Phase 0 baseline (no DR at all), and stay
   passing across phases. DR on those axes is solving a non-problem —
   there was nothing to fix.

2. **Only two axes have actual robustness gaps:**
   `init_joint_delta_deg` and `action_scale`.
   `goal_depth` at -0.28 is a workspace-reach limit, not really
   addressable by DR.

3. **Phase 1 was the only clean win.** Adding init-pose DR widened the
   `init_joint_delta_deg` envelope 3-4× (10° → 30° passes; +0.50 SR at
   30°) without any in-distribution cost or change on the other 9 axes.
   That's a textbook DR result: target a known weak axis, train on a
   wider distribution, gain robustness on that axis specifically.

4. **Phase 2 was probably solving non-problems.** Drawer DR (friction,
   damping, mass) couldn't help — every drawer axis was already at
   SR=1.00 in P0 with no headroom to gain. Arm-link DR (±10%) couldn't
   help either, because the policy is *already* robust to arm-mass
   perturbations up to at least ±50% even without any DR
   (post-regex-fix resweep). So the only real effect Phase 2 could have
   was potential negative — and the init_joint regression vs P1 (94%
   pass at 30° → 69% degraded at 30°) is most likely undertraining
   (2700 iters vs P1's 3888) rather than a DR-curriculum interaction.

5. **The arm-link "fail at 10%" result was a regex NaN bug, not a real
   robustness failure.** Both the training-time DR
   (`open_drawer_osc.py:_apply_phase_knobs`) and the eval-time
   perturbation (`eval_sweep._override_arm_link_mass_pct`) used a broad
   `r".*_link"` body-name regex, which matched the Kinova's massless
   `end_effector_link` (mass=0, zero inertia → `pseudo_inertia` produces
   NaN) and the gripper `*_spring_link` finger bodies. The
   training-side bug was caught at iter 0 (Phase 2 NaN'd) and fixed.
   The eval-side bug went undetected because the symptom (SR=0 at every
   non-zero perturbation) looked like a believable "policy can't handle
   arm mass changes" failure mode — but the simultaneous
   `mean_episode_length=1.00` was the giveaway: the policy couldn't
   even take one step. Re-sweep with the fixed regex shows SR=1.0 at
   every test value across all three phases.

6. **`action_scale` is brittle in all phases.** Only nominal 0.01
   passes; 0.005 fails (too slow to reach goal in 100 steps), 0.02 +
   0.05 fail (overshoot/chaos). DR didn't help here because it's an
   action-API hyperparameter, not a physics axis. Phase 3
   (slow-execution / velocity penalty) is the right intervention —
   reducing reliance on large velocity excursions should indirectly
   tighten the action_scale envelope.

7. **Plateau ceiling is structural.** All three phases converge to
   dwell-SR ≈ 0.77 regardless of DR. The bottleneck is OSC controller
   resolution + reward shape (`goal_precise` Gaussian std=0.05 m), not
   data diversity. Pushing past 0.77 needs reward / controller changes,
   not more DR.

### Implications for the plan

- **Drop Phase 2 deltas from the deploy stack.** Drawer DR is a no-op
  (nothing to fix), arm-link DR is a no-op (already robust), and the
  shorter training risks regressing P1's gains.
- **Phase 1 stays** as the only DR floor for Phase 3+.
- **Phase 3 (slow-execution) targets the actual remaining gap**
  (`action_scale` brittleness + safety on the real arm). Pick 3a
  (quadratic velocity penalty) first — least invasive, cleanest
  comparison vs Phase 1.
- **Phase 4 (deploy)** = Phase 1 + Phase 3 keeper. No drawer DR, no
  arm-link DR.

---

## Extended sweep — 8 new axes (perception / action / dynamics)

After concluding that the original 10 axes mostly didn't probe real
robustness gaps, we added 8 new axes that cover perception noise,
action noise, gravity, torque-channel disturbance, and external
impulses. Implementation: noise hooks land directly in the OSC
controller (`tau_offset_std_Nm`, `tau_noise_std_Nm`, `action_noise_std`
fields on `OperationalSpaceActionCfg`); obs noise replaces existing
`UniformNoiseCfg` with `GaussianNoiseCfg` at sweep time; impulses use
mjlab's `apply_body_impulse` with random cooldown / duration.

### New-axis values

| Axis | Values | Type | Rate |
|---|---|---|---|
| `obs_noise_object_m` | 0, 0.002, 0.005, 0.01, 0.02 m σ | Gaussian | per-step |
| `obs_noise_ee` | 0, 0.001, 0.002, 0.005, 0.01 m σ | Gaussian | per-step |
| `action_noise_pct` | 0, 1, 3, 5, 10 % σ on raw action | Gaussian | per-step |
| `gravity_scale` | 0.8, 0.9, 1.0, 1.1, 1.2 × g + ≤5° tilt | constant | per-episode |
| `torque_offset_Nm` | 0, 0.05, 0.1, 0.5, 1.0 N·m σ per joint | Uniform | per-episode |
| `torque_noise_Nm` | 0, 0.05, 0.1, 0.5, 1.0 N·m σ per joint | Gaussian | per-step |
| `drawer_impulse` | level 0..4 → (0, 5N×1, 10N×1, 20N×2, 10N×4) | xfrc | random throughout episode |
| `ee_impulse` | level 0..4 → (0, 2.5N×1, 5N×1, 10N×2, 5N×4) | xfrc | random throughout episode |

### Result on P0 + P1

| | Phase 0 | Phase 1 |
|---|---:|---:|
| Robustness score (8 new axes) | **1.000** | **1.000** |
| Settings with SR < 1.0 (out of 40) | 0 | 0 |
| Settings with mean error > 5 mm (out of 40) | 0 | 0 |
| Worst-case mean error (m) | 0.0026 | 0.0028 |

**Every single setting at every value passed at SR=1.0 with mean
error around 2 mm**, including the most extreme values:

- 2 cm σ object obs noise (= the success threshold itself)
- 1 N·m per-joint torque offset constant + 1 N·m torque noise per step
- ±20% gravity tilted 5° from vertical
- 10% action noise
- Four 10 N impulses on drawer base, random timing
- Four 5 N impulses on bracelet, random timing

Files: `docs/results/new_axes_p0/{sweep_summary.csv,breaking_points.md}`,
`docs/results/new_axes_p1/{...}`.

### Observations / analysis

1. **The OSC controller is doing the robustness work, not the policy.**
   OSC's task-space PD with `qfrc_bias` (gravity + Coriolis)
   compensation absorbs essentially all physical perturbations
   transparently:
   - **Action noise** is small relative to `kp_pos × pos_error` —
     the policy commands large deltas relative to the noise floor.
   - **Torque noise/offset** gets cancelled by OSC's pos-tracking
     correction the next step.
   - **Obs noise** matters only at grasp commitment (a brief moment),
     and the rest of the episode it averages out.
   - **Gravity changes** are absorbed exactly by `qfrc_bias`
     compensation.
   - **Impulses** on drawer/EE get corrected by OSC's pos-tracking
     when the policy re-commands the next step.
2. **The policy needs to learn one thing:** where to put the EE in
   task space. Everything below that — gravity, joint dynamics,
   torque-channel noise — is OSC's job.
3. **DR is the wrong tool for an OSC-based policy.** Phase 2 confirmed
   this on the original axes; the new axes confirm it more strongly.
   Adding physics randomization to training doesn't help because the
   policy never feels the physics — OSC isolates it.
4. **The real sim2real risk is in the OSC controller itself, not the
   policy.** If the real robot's torque limits, motor response time,
   or gravity compensation differ from sim, OSC's perfect dynamics
   compensation breaks down. This is what Phase 3 (slow-execution) is
   really targeting — reducing the policy's reliance on OSC's
   perfect compensation by making it command less aggressive
   trajectories that survive imperfect compensation.
5. **`init_joint_delta_deg` and `action_scale` remain the only true
   robustness gaps** out of 18 swept axes. Phase 1 fixed the first.
   Phase 3 should target the second by encouraging slower, more
   conservative task-space trajectories.

### Implications for the plan (revised)

- **The deploy-stack doesn't need physics DR at all.** Phase 1
  init-pose DR is the only DR that meaningfully helped, and only on
  one axis.
- **Phase 3 (slow-execution) is the only intervention left that
  could plausibly buy real-world robustness.** It changes what the
  policy commands, not what physics it experiences.
- **Sim2real testing should focus on OSC-controller mismatch**, not
  on physics DR coverage. The most likely failure mode at deploy
  time is the real arm's torque/velocity limits clipping the OSC
  output below what the policy commanded — a hard-to-DR-against
  failure that the policy never sees in sim.

---

### Phase 3 — slow-execution / safety-aware (planned)

- **Hypothesis:** A real Kinova at high joint velocities is unsafe.
  Reduce action space, tighten the velocity hinge, and/or curriculum
  on `max_vel`. Plan §3 lists 3a–3e in increasing intervention order;
  stop at the first that holds in-dist SR while reducing observed
  joint velocity.
- **DR floor:** Phase 1 init-pose deltas (Phase 2 deltas rejected;
  see cross-phase decision 2026-04-29).
- **Variant order:** 3a → 3b → 3c → 3d → 3e (least → most
  invasive). Stop at the first that holds in-dist SR ≥ 0.75 *and*
  reduces observed `joint_vel_max` by ≥ 30% on a held-out rollout.
- **3a — quadratic vel penalty:** replace hinge with quadratic
  `−w·||dq||²` after iter 2400. Recommended **first try** because
  it's a reward-shape change only — no env API changes, no obs
  changes; cleanest comparison to Phase 1.
- **3b — tighter hinge:** drop `max_vel` 0.5 → 0.3 rad/s.
- **3c — vel curriculum:** schedule `max_vel` 0.6 → 0.4 → 0.3 over
  3 stages.
- **3d — OSC delta clip:** clip `delta_pos_scale` action component to
  ±0.5 (effective Δpos ≤ 5 mm).
- **3e — processed-action obs:** add post-clip processed action to obs.
- **Status:** ready to launch (3a). Phase 2 sweep showed
  `action_scale` is brittle in *both* P0 and P1 (only nominal 0.01
  passes, all 4 perturbed values fail); 3a/3b/3c don't directly
  target `action_scale` but should reduce the policy's reliance on
  large velocity excursions, which may indirectly tighten the
  action_scale envelope. Worth measuring on the Phase 3 sweep.

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
- **2026-04-29** — OOD sweep Phase 0 vs Phase 1 confirms the init-pose
  DR thesis: Phase 1 widens `init_joint_delta_deg` envelope from
  10° to 30° (3× wider passing range, **+0.50 SR at 30°**) without
  any in-distribution cost or any change on the other 9 axes. Adopt
  Phase 1 init-pose deltas as the floor for all subsequent phases.
- **2026-04-29** — Action-scale axis is brittle in *both* phases —
  only the nominal 0.01 passes; the policy fails at 0.005 (too slow
  to reach goal in 100 steps), 0.02 (overshoots), 0.05 (chaotic).
  This rules out direct OSC-delta scaling as a robustness lever and
  supports Phase 3's plan to constrain action via velocity
  penalties / processed-action obs rather than via scale changes.
- **2026-04-29** — Arm-link mass axis fails at 10% perturbation in
  both Phase 0 and Phase 1 (terminal at episode length 1.00 — the
  policy crashes immediately). This is exactly what Phase 2's
  `arm_link_mass` DR is supposed to fix; its in-flight SR plateau
  at 0.77 (matching baseline) is a positive signal that the DR is
  not trading off in-distribution capability.
- **2026-04-29** — **Phase 2 sweep returned 0.721 (worse than P1's
  0.746, better than P0's 0.696).** Two failure modes identified:
  arm_link_mass DR alpha=±4.77% is too narrow vs 10% test
  perturbation (same crash as P0/P1); init_joint regresses vs P1
  (likely undertraining at 2700 iters). **Decision: reject Phase 2
  deltas for the deploy run; Phase 1 stays the DR floor.** If
  arm-mass robustness is worth pursuing, the right experiment is
  Phase 2' = Phase 1 + arm_link_mass at α=±0.10 (≈ ±10%) and run
  to plateau (~3800 iters), as a *separate* experiment from
  Phase 3.
- **2026-04-29** — **Drawer DR knobs (friction, damping, base_mass)
  changed nothing** — both P0 and P1 already passed all swept values
  at SR=1.00 with no headroom to gain. The "drawer cabinet has
  manufacturing tolerance" story may be true for the real cabinet but
  doesn't show up as a robustness gap in our sweep envelope. Could
  drop these from a future deploy run unless we widen the swept
  values further.
- **2026-04-29** — **Caught a second instance of the
  `r".*_link"` regex bug — this time in the eval sweep**
  (`eval_sweep._override_arm_link_mass_pct`). Same failure mode as
  the Phase 2 training-time NaN: matching `end_effector_link`
  (mass=0) and gripper `*_spring_link` bodies makes `pseudo_inertia`
  produce NaN, which crashes the policy at step 1. So **every "fail
  at any non-zero arm_mass perturbation" result in the original P0/P1/
  P2 sweeps was a NaN-eval bug, not a real robustness failure.**
  Fixed regex (anchored arm-only set, same as training-time fix) and
  re-swept all three checkpoints. **Result: P0 baseline already passes
  arm_link_mass at 0%, 10%, 25%, AND 50% perturbation, all SR=1.00.**
  The policy is fully robust to arm-mass perturbations even without
  any DR training. **Phase 2's arm-link DR was solving a non-problem.**
- **2026-04-29** — **Revised plan-level read.** Combining the regex
  fix with the existing per-axis matrix: 8 of 10 axes were never
  broken on the bare baseline. Only `init_joint_delta_deg` and
  `action_scale` are real robustness gaps. Phase 1 already fixed the
  first one. Phase 2 (drawer + arm DR) was solving non-problems and
  marginally regressed `init_joint_delta_deg` due to undertraining
  at 2700 iters. **Decision: drop Phase 2 deltas entirely from the
  deploy stack.** Phase 1 is the DR floor. Phase 3 (slow-execution)
  is the next real experiment, targeting the action_scale gap.
- **2026-04-29** — **Extended sweep with 8 new perception/action/
  dynamics axes returned robustness=1.000 on both P0 and P1.** Every
  one of 80 settings (40 per phase) passed at SR=1.0 with mean error
  ~2 mm, including extreme values (2 cm σ object obs noise, 1 N·m
  per-joint torque offset/noise, ±20% gravity with 5° tilt, 10%
  action noise, 4×10 N drawer impulses, 4×5 N EE impulses).
  **Conclusion: the OSC controller absorbs essentially all physical
  perturbations through task-space PD + `qfrc_bias` compensation —
  the policy never feels the physics noise.** This explains why
  physics DR (Phase 2) helped nothing: the policy never had a
  robustness gap on those axes to begin with. The deploy stack
  doesn't need any physics DR — only the Phase 1 init-pose deltas
  matter. Phase 3 (slow-execution) is the only remaining
  intervention that could plausibly buy real-world robustness,
  because it changes what the policy commands rather than what
  physics it experiences. **Real sim2real risk is in OSC-controller
  mismatch (real robot's torque/velocity limits clipping the OSC
  output), which physics DR cannot probe.**

## Cross-phase training-curve comparison

Time-to-threshold per phase, all run at `num_envs=1024`, same seed,
same PPO hyperparameters, same reward terms (Phases 1+2 inherit Phase 1
init-pose deltas; Phase 2 adds drawer + arm-link DR on top):

| Threshold | Phase 0 (iter) | Phase 1 (iter) | Phase 2 (iter) | P2/P0 speedup |
|---|---:|---:|---:|---:|
| reward ≥ 20 | 1003 | 803 | **153** | **6.6×** |
| SR ≥ 0.5 | 1128 | 942 | **198** | **5.7×** |
| SR ≥ 0.7 | 1548 | 1344 | **330** | **4.7×** |
| obj_err ≤ 0.05 m | 1581 | 1437 | **444** | **3.6×** |
| reward ≥ 29 | 1556 | 1451 | **537** | **2.9×** |
| SR ≥ 0.75 | 2181 | 1950 | **768** | **2.8×** |
| reward ≥ 30 | 2032 | 2001 | **1152** | **1.8×** |
| obj_err ≤ 0.04 m | 2784 | 2508 | **1244** | **2.2×** |

**Pattern:** early thresholds (initial reaching, getting onto the
reward landscape) get a 5-7× speedup from Phase 2's DR. Late
thresholds (final dwell-precision, terminal reward) converge to the
*same* plateau time. All three phases hit the same dwell-SR ≈ 0.77
ceiling.

**Reading.** Most likely explanation: drawer + arm DR functions as
data-diversity regularization. Wider physics distribution → more
varied gradients per minibatch → faster early-stage credit assignment
on the reach-and-grasp component. But the plateau is dwell-precision
bounded (terminal `object_to_goal_error` ≈ 0.04 m, never below the
0.02 m success threshold for *every* step of *every* episode), and
DR doesn't move that ceiling because the bottleneck is OSC controller
resolution + reward-shape (`goal_precise` Gaussian std=0.05), not
exploration-data quantity.

**Implication for Phase 3+.** A combined Phase-2 + Phase-3 run should
reach plateau faster than Phase 3 alone. If it doesn't, the
regularization-from-DR hypothesis is wrong and the speedup we saw
here was just seed/initialization variance.

**Caveat.** All three phases share the same RNG seed (default), so
the absolute iter counts are noisy estimators of the true mean
time-to-threshold. The 6× gap on `reward ≥ 20` is too big to be just
seed noise, but the 1.8× on `reward ≥ 30` could plausibly shrink
under reseeding.

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

- ~~Does Phase 1 actually move the needle on the `init_joint_delta_deg`
  and `robot_base_x_offset` sweep axes?~~ **Answered (2026-04-29):**
  yes for `init_joint_delta_deg` (envelope 3-4× wider; +0.50 SR at 30°);
  no measurable change for `robot_base_x_offset` (already saturated at
  10 cm in both phases, no breaking-point reached in either).
- Phase 1 dwell SR plateaued at 0.77 (same as Phase 0). Is the
  remaining 0.23 floor structural (reach geometry, distal joint
  reach), or is it a converged-policy noise floor we can push by
  tuning `goal_precise.std` or `success_rate.threshold`?
- Lane B (raw mujoco, async) hasn't been smoke-tested on a
  high-success checkpoint yet. The `model_400.pt` smoke test produced
  SR=0 (expected — policy hadn't learned). Re-run on `model_3800.pt`
  is pending.
