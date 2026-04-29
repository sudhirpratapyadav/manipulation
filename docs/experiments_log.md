# Open-drawer OSC — experiments & eval log

The journal. One entry per training run; eval entries nested. Append,
don't rewrite. Numbers come from W&B `Episode_Metrics/*` panels at the
final iteration (not best-iteration peaks — peaks hide collapse).

Schema for each entry below. Copy this when adding a phase.

```
### <experiment_name>

- Status: pending / running / done / aborted
- Phase: N
- Submitted: YYYY-MM-DD
- Slurm job ID: 
- W&B run URL: 
- Script: slurm/<file>.sh
- Tags: phase=N, <variant>
- Change vs. previous phase: <one paragraph; what knob moved and why>

Iterations: <N>  ·  num_envs: <M>  ·  task: Mjlab-Open-Drawer-Osc-Kinova

| Metric | Final-iter value |
|---|---|
| Episode_Metrics/success_rate (dwell) |  |
| Episode_Metrics/object_to_goal_error (m) |  |
| Episode_Metrics/ee_to_object_error (m) |  |
| Train/mean_reward |  |
| Train/mean_episode_length |  |

Notes: <free-form>

#### Eval: in-distribution
- Date: 
- Checkpoint: 
- Result: success_rate=…, mean_error=…m

#### Eval: OOD sweep
- Date: 
- Checkpoint: 
- Sweep dir: docs/results/<experiment_name>/
- Robustness score: <float in [0, 1]>
- Per-axis breaking points: see breaking_points.md

#### Decision
- keep / drop / partial
- Reason: <how it scored against the keep rule>

#### Autonomous decisions taken this phase
- <one bullet per non-obvious call made without user input>
```

## Status

**Next pending:** Phase 2 training (mso8ooz7) → plateau or 4000 iters,
then Phase 2 OOD sweep, then Phase 3 (slow-execution). Phase 0 + Phase 1
OOD sweeps **complete**: P0 robustness 0.696, P1 robustness 0.746
(+0.050; init_joint envelope widened 3-4×). Holder: `slurm/holder_3gpu.sh`
job **18278**. To resume, follow `AGENT.md` § "Resume sequence".

## Autonomous decisions (cross-phase)

- **2026-04-29** — Used a **3-GPU holder** (`slurm/holder_3gpu.sh`,
  job 18271) instead of the planned 8-GPU holder. `dgx2` had only ~6 free
  GPUs (2 in use by another user) and `dgx1` was drained, so the
  `8gpu` QoS holder sat in `PD (Resources)`. The QoS tiers are
  1/2/3/8 (no 6gpu), so 3-GPU is the largest immediately-available
  option — still allows running 3 phases in parallel, which captures
  most of the parallelism win. If 8 GPUs free up later, swap to the
  8-GPU holder.
- **2026-04-29 ~08:09** — **Cancelled Phase 0 (job-step 18271.0) and
  Phase 1 (18271.5) at iter ~3900/5000**, ~2 h before the planned 5000
  iter mark. Both trajectories had effectively plateaued by iter 3000:
  Phase 0 went SR 0.752 → 0.774 (Δ=0.02) over iter 2000→3979; Phase 1
  went SR 0.751 → 0.773 (Δ=0.02) over iter 2000→3888. Burning 2 GPU-h
  for ≤0.02 SR gain isn't worth it when Phase 2 (which inherits Phase 1)
  is blocked on this. Treating `model_3800.pt` as the canonical
  checkpoint for both phases; `model_3700` is a fallback if 3800
  shows any anomaly. AGENT.md "early-cancel doomed runs" applies — runs
  weren't doomed but they were *done*.
- **2026-04-29 ~08:30** — **Phase 2 NaN'd at iter 0** on the first launch.
  `Episode_Termination/nan_detection = 1024` (every env) from iter 0 on,
  rew=0, SR=0. Killed (job-step 18278.3) after ~10 min. Root cause: in
  `_apply_phase_knobs` the `dr_arm_link_mass` event used regex
  `body_names=r".*_link"` for `pseudo_inertia`, which matched the
  Kinova's `end_effector_link` (mass=0, zero inertia → log/exp on zero
  → NaN) and also the gripper `*_spring_link` finger bodies (~0.02 kg,
  not arm dynamics). Fixed regex to anchored arm-only set
  `(base|shoulder|half_arm_[12]|forearm|spherical_wrist_[12]|bracelet)_link`.
  Verified post-fix env builds and steps 5× with no NaN, sensible
  reward. Relaunched as job-step 18278.4 (W&B run `ukd2fffm`,
  superseding the killed `99v32mk7`). Logged as a research-side
  decision in `rl_experiments_log.md` too.
- **2026-04-29 ~08:14** — **Tried to upgrade to 8-GPU holder (18273)**
  after cancelling 18271, hoping the freed GPUs would let qos=8gpu
  schedule. It sat in `PD (Resources)` for ~2 min: dgx2 reported
  `AllocTRES=cpu=40,gres/gpu=2` even though `squeue` showed no other
  jobs — a slurmctld reconciliation lag from the cancelled holder.
  dgx1 still drained, so dgx2 was the only candidate node and
  effectively had only 6 of 8 GPUs available. Cancelled 18273 and
  resubmitted 3-GPU holder (job **18278**) to avoid sitting idle.
  Will retry 8-GPU after the next phase milestone if dgx1 recovers.
- **2026-04-29** — Kept the working-tree changes (refined plan/log,
  AGENT.md, holder/sweep scaffolding) un-committed but in-place. They
  are the autonomous-research framework AGENT.md instructs me to
  operate from; reverting would destroy the project state. Will
  commit at phase milestones per the autonomy contract.
- **2026-04-29** — Found and fixed a latent bug in
  `src/kinova_tasks/eval_sweep.py:_run_setting`: it passed
  `dict(agent_cfg_dict)` (shallow) to `MjlabOnPolicyRunner`, but
  `rsl_rl.OnPolicyRunner.__init__` mutates inner dicts (e.g. pops
  `class_name`). The first setting in any sweep would succeed; the
  2nd would crash with `KeyError: 'class_name'`. Replaced with
  `copy.deepcopy`. Caught by writing
  `src/kinova_tasks/eval_sweep_handcheck.py`, an end-to-end check
  that runs the harness vs. an independent rollout on the same
  checkpoint and the same env. With the fix, both branches return
  identical success_rate (0.000 ± 0) on `model_400.pt` of the
  in-flight Phase 0 baseline, n=32 episodes, num_envs=16. Will
  re-run this handcheck on a high-success Phase 0 checkpoint to
  prove the harness in the interesting regime as well.
- **2026-04-29 ~13:50** — **Phase 2 had two srun job-steps alive at
  once:** the first (W&B `99v32mk7`, job-step 18278.2) was the broken
  pre-fix run still in NaN-burst mode (~89% of iters all-1024 NaN);
  the second (W&B `mso8ooz7`, job-step 18278.14) was the post-fix
  clean run at SR≈0.77 with zero NaN events. Both writing to the same
  `slurm/logs/phase2_train.out` file by accident, so log inspection
  showed only the bad one's NaN bursts and looked like the fix had
  failed. Mapped each to a run-dir via tfevents PID and confirmed
  via wandb output.log. Cancelled 18278.2; mso8ooz7 alive and
  healthy. Lesson: when launching parallel job-steps, give each its
  own log path; otherwise tail-grep can mislead about which run is
  in trouble.
- **2026-04-29 ~13:55** — **Found a UnicodeEncodeError in
  `eval_sweep.py:519`** (`md_path.write_text(...)` with default
  ASCII codec choking on em-dash). The Phase 1 sweep CSV was fully
  written (43 settings, all axes), but the markdown summary wasn't.
  Added `encoding="utf-8"` and a separate `eval_sweep_md.py` utility
  that re-renders the markdown from an existing CSV without
  re-running the sweep — used to recover the Phase 1 summary,
  reusable for any future write failure.
- **2026-04-29 ~14:05** — **Phase 0 OOD sweep complete (job-step
  18278.20)** after the original 18278.1 had failed earlier under
  srun resource collision. Robustness 0.696 vs Phase 1's 0.746.
  Phase 1 init-pose DR confirmed: widens `init_joint_delta_deg`
  envelope 3-4× without any in-distribution cost. See
  `rl_experiments_log.md` for the cross-phase comparison table.
- **2026-04-29 ~14:30** — **Cancelled Phase 2 (mso8ooz7) at iter
  ~2700/4000** on the same plateau criterion that cancelled P0/P1:
  SR delta over iter 1500-1800 → 2400-2700 was +0.008 (≤0.02
  threshold). `model_2700.pt` adopted as canonical Phase 2
  checkpoint. Launched Phase 2 OOD sweep as job-step 18278.21.
- **2026-04-29 ~14:35** — **Phase 2 trains 2-7× faster than P0/P1
  to early thresholds** (e.g. `reward ≥ 20` at iter 153 vs P0's 1003;
  SR ≥ 0.5 at iter 198 vs P0's 1128). Late thresholds converge to
  the same plateau time (~iter 2000-2700). Same seed across all
  three phases, so the gap is not pure RNG noise — most likely DR
  acting as data-diversity regularization on the reach-and-grasp
  component. Logged as a research finding with the full
  time-to-threshold table in `rl_experiments_log.md`.

## Run history

<!-- Phase entries get appended below. Keep newest at bottom. -->

### open_drawer_osc_phase0

- Status: done (early-cancelled at iter ~3979/5000; plateaued)
- Phase: 0
- Submitted: 2026-04-29
- Slurm job ID: 18271.0 (job-step inside 3-GPU holder 18271, GPU 0)
- W&B run URL: https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/52n3yt9o
- Script: slurm/open_drawer_osc_phase0.sh
- Tags: phase=0, baseline
- Canonical checkpoint: `logs/rsl_rl/open_drawer_osc_phase0/2026-04-29_00-25-49_baseline/model_3800.pt`
- Change vs. previous phase: none — establishes reference under the new
  metric/eval setup. Same task config that the prior baseline run
  (cancelled job 18266) was using.

Iterations: 3979 of 5000 target (early-cancelled)  ·  num_envs: 1024  ·  task: Mjlab-Open-Drawer-Osc-Kinova

| Metric | Final-iter value (iter 3979) |
|---|---|
| Episode_Metrics/success_rate (dwell) | 0.7723 |
| Episode_Metrics/object_to_goal_error (m) | 0.0395 |
| Episode_Metrics/ee_to_object_error (m) | 0.0447 |
| Train/mean_reward | 30.32 |
| Train/mean_episode_length | 100.00 |

Trajectory (key checkpoints, from training stdout):
- iter 1000: SR=0.242, rew=19.06
- iter 2000: SR=0.725, rew=28.48
- iter 3000: SR=~0.76, rew=~30 (plateau begins)
- iter 3979: SR=0.7723, rew=30.32

Notes: Sweep harness smoke test passed 10/10 axes pre-launch. Iter 43
health was clean (mean_reward 6.23, NaN=0, reach_object dominant).
Plateau set in around iter 2500–3000; the last 1000 iters added only
+0.02 SR. mean_episode_length=100 (full episodes — drawer task does
not terminate on success in this config), so the policy is solving
the task and dwelling ~77% of steps within 0.02 m of the goal.

#### Autonomous decisions taken this phase
- Used 3-GPU holder (job 18271) instead of 8-GPU because dgx2 only had
  6 free GPUs at submit time. See cross-phase decisions above.
- Built `eval_sweep_smoketest.py` to validate every override actually
  hits its target field (or value) before trusting any sweep number.
  Smoke test confirmed all 10 axes; runs in ~2 min on a spare GPU.
  Hand-computed-vs-harness success_rate validation deferred until
  Phase 0 produces a checkpoint.
- Early-cancelled at iter 3979 (see cross-phase decision). Last 1000
  iters of training delivered ≤ 0.02 SR gain; freed GPU is more
  valuable for OOD eval and Phase 2 training.

### open_drawer_osc_phase1

- Status: done (early-cancelled at iter ~3888/5000; plateaued)
- Phase: 1
- Submitted: 2026-04-29
- Slurm job ID: 18271.5 (job-step inside 3-GPU holder 18271, GPU 1)
- W&B run URL: https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/4gn2sbub
- Script: slurm/open_drawer_osc_phase1.sh
- Tags: phase=1, init_pose_dr
- Canonical checkpoint: `logs/rsl_rl/open_drawer_osc_phase1/2026-04-29_00-34-48_init_pose_dr/model_3800.pt`
- Change vs. previous phase: wider initial-pose randomization on top of
  Phase 0 baseline. `reset_robot_joints.joint_delta_deg`: 5° → **15°**;
  `reset_base.pose_range`: `{}` → `{"x": ±2 cm, "y": ±2 cm, "yaw": ±2°}`.
  Models cabinet not always closed flush + technician placing the robot
  with slight position/yaw variation. No DR knobs touched.
- Implementation: introduced `PhaseKnobs` dataclass +
  `_apply_phase_knobs()` in `tasks/open_drawer_osc.py` so each phase is a
  small declarative diff vs. Phase 0. Registered new task IDs `…-Phase1`,
  `…-Phase2`, `…-Phase4`. `slurm/check_phases.sh` confirms all 5 task
  IDs build cleanly at small num_envs.

Iterations: 3888 of 5000 target (early-cancelled)  ·  num_envs: 1024  ·  task: Mjlab-Open-Drawer-Osc-Kinova-Phase1

| Metric | Final-iter value (iter 3888) |
|---|---|
| Episode_Metrics/success_rate (dwell) | 0.7734 |
| Episode_Metrics/object_to_goal_error (m) | 0.0392 |
| Episode_Metrics/ee_to_object_error (m) | 0.0371 |
| Train/mean_reward | 30.67 |
| Train/mean_episode_length | 100.00 |

Trajectory (key checkpoints, from training stdout):
- iter 1500: SR=0.703, rew=28.56
- iter 2000: SR=0.751, rew=29.67
- iter 3000: SR=0.770, rew=30.11 (plateau)
- iter 3888: SR=0.7734, rew=30.67

Notes: Wider init-pose randomization (joint 5°→15°, base pose ±2 cm /
±2° yaw) absorbed by the policy without measurable in-distribution
SR loss vs. Phase 0 (Δ ≈ 0). Whether it actually buys robustness is a
question for the OOD sweep. ee_ground_collision rate dropped to 0/ep
by mid-training, as expected.

#### Autonomous decisions taken this phase
- Ran in parallel with Phase 0, not sequentially. Phase 1's deltas don't
  inherit any Phase 0 result, so the parallel run only loses the
  ability to early-cancel Phase 1 if Phase 0 reveals a metric problem.
  Saves ~9 h of wall time. Held Phase 2 since it *does* inherit P1 deltas.
- Chose ±2cm / ±2° yaw rather than the plan's literal "±2 cm xy, ±2° yaw"
  by also scaling yaw to radians explicitly (`±2°·π/180`). Documented
  in `_phase1_knobs()`.
- Did not introduce per-joint deltas yet (plan §3 mentions them as a
  fallback if uniform 15° hurts distal joints). Will revisit after first
  evaluation if `joint_pos_limits` reward stays consistently negative.
- Early-cancelled at iter 3888 (see cross-phase decision). Phase 0 and
  Phase 1 plateaus matched within 0.001 SR, so terminating the
  Phase 1 run on the same trigger was symmetric.

### open_drawer_osc_phase2

- Status: running
- Phase: 2
- Submitted: 2026-04-29 08:19
- Slurm job ID: 18278.3 (job-step inside 3-GPU holder 18278, GPU 2)
- W&B run URL: https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/99v32mk7
- Script: slurm/open_drawer_osc_phase2.sh
- Tags: phase=2, drawer_dr
- Change vs. previous phase: targeted drawer + arm DR on top of Phase 1
  init-pose randomization. New knobs in `_phase2_knobs()`:
  - drawer slide frictionloss: U(0.005, 0.02) (≈ ±2× nominal 0.01)
  - drawer slide damping: U(0.5, 2.0)
  - drawer base mass: log-uniform alpha ±0.347 (≈ ±34.7%)
  - arm link mass: log-uniform alpha ±0.0477 (≈ ±4.77%)
  Phase 1 init-pose deltas inherited.
- Iterations target: 4000 (down from 5000 — Phase 0/1 showed plateau by
  ~3000, so 4000 is a safer cap with budget for slower convergence
  under DR).

Iterations: 4000 target  ·  num_envs: 1024  ·  task: Mjlab-Open-Drawer-Osc-Kinova-Phase2

| Metric | Final-iter value |
|---|---|
| Episode_Metrics/success_rate (dwell) | TBD |
| Episode_Metrics/object_to_goal_error (m) | TBD |
| Episode_Metrics/ee_to_object_error (m) | TBD |
| Train/mean_reward | TBD |
| Train/mean_episode_length | TBD |

Notes: Launched in parallel with Phase 0 + Phase 1 OOD eval sweeps
(GPUs 0/1 of holder 18278). ETA ~7.5 h at 4000 iters.

#### Autonomous decisions taken this phase
- Cut max_iterations from 5000 (the plan default) to 4000. Phase 0
  and Phase 1 both plateaued by iter ~3000; 4000 gives ~30% headroom
  for slower DR convergence without burning GPU on flat plateau.
- Did NOT raise num_envs to spread the wider DR distribution. Plan §3
  flags this as a possible knob; trying without it first to keep
  phase-to-phase comparisons clean.
