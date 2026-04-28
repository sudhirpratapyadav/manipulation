# Open-Drawer OSC — Experiments & Eval Log

Tracking record for every training run and eval pass on
`Mjlab-Open-Drawer-Osc-Kinova`. Companion to
`docs/open_drawer_improvement_plan.md` — that doc is the *plan*; this one is
the *journal*. Append, don't rewrite.

## Conventions

- One section per training run. Use `### <experiment-name>` matching
  `--agent.experiment-name`.
- Eval passes go under their parent training run as `#### Eval: <date>`.
- Status legend: `running` / `done` / `aborted`.
- Always paste the W&B run URL once the run starts logging.
- Numbers come from W&B `Episode_Metrics/*` panels; record terminal-iteration
  values, not best-iteration peaks (peaks hide collapse).

## Metric notes

- `success_rate` is registered via `MetricsTermCfg`, which averages across the
  episode. Logged value = **fraction of steps within 2 cm of the goal handle
  position** (a "dwell" success), strictly stricter than terminal success.
  The plan calls for `reduce="last"` (terminal-only), but mjlab's metrics
  manager doesn't expose that yet. Numbers between phases are still
  comparable as long as we keep the metric definition fixed.
- `object_to_goal_error` and `ee_to_object_error` are episode-mean Euclidean
  distances in meters.

## Phase 0 — Baseline

### open_drawer_osc_phase0

- **Status:** queued (pending dgx1)
- **Submitted:** 2026-04-28
- **Slurm job ID:** 18265
- **W&B run URL:** _populated once job starts_
- **Script:** `slurm/open_drawer_osc_phase0.sh`
- **Tags:** `phase=0`, `baseline`
- **Config delta vs. existing `open_drawer_osc.sh`:** none (model unchanged);
  only env-cfg change is the new `success_rate` metric.

**Pre-submission environment fix (2026-04-28):** local `mjlab` checkout was
ahead of `kinova_constants.py`'s expected API — the unified `XmlActuatorCfg`
was split into `XmlMotorActuatorCfg` / `XmlPositionActuatorCfg` /
`XmlVelocityActuatorCfg` / `XmlMuscleActuatorCfg`. Migrated
`assets/kinova_gen3/kinova_constants.py`:
  - `KINOVA_ACTUATORS` → `XmlPositionActuatorCfg` (gen3_gripper.xml uses
    `<position>` actuators on arm joints; fingers_actuator is a
    `<general>` tendon actuator and is filtered out by joint-name regex,
    same as before).
  - `KINOVA_NO_GRIPPER_ACTUATORS` → `XmlMotorActuatorCfg`
    (gen3_no_gripper_torque.xml `<motor>` actuators).
  - `KINOVA_GRIPPER_TORQUE_ARM_ACTUATORS` → `XmlMotorActuatorCfg`
    (gen3_gripper_torque.xml arm `<motor>` actuators).

`ActuatorCfg` defaults (`armature=0.0`, `frictionloss=0.0`,
`transmission_type=JOINT`) match the prior unified config, so behavior
should be identical to the 2026-04-14 run except for any unrelated mjlab
changes that landed between then and now. Verified by building the env
with `num_envs=4` on CPU and stepping once: `Episode_Metrics/success_rate`
is logged alongside the existing distance metrics.

**Iterations:** 5 000  ·  **num_envs:** 1024  ·  **task:** `Mjlab-Open-Drawer-Osc-Kinova`

| Metric | Value @ final iter |
|---|---|
| `Episode_Metrics/success_rate` (dwell) |  |
| `Episode_Metrics/object_to_goal_error` (m) |  |
| `Episode_Metrics/ee_to_object_error` (m) |  |
| `Train/mean_reward` |  |
| `Train/mean_episode_length` |  |

**Notes:**
- 

#### Eval: in-distribution (training task, play=True)

- **Date:** 
- **Checkpoint:** 
- **Command:**
  ```
  sbatch --export=WANDB_RUN_PATH=<entity/project/run_id>,WANDB_CHECKPOINT_NAME=model_4900.pt \
      slurm/eval_open_drawer_osc.sh
  ```
- **Result:** 

#### Eval: OOD sweep (skeleton)

- **Date:** 
- **Status:** sweep harness not yet implemented (see plan §7.2). For Phase 0
  this slot just runs `Mjlab-Open-Drawer-Osc-Kinova-Eval` with nominal DR
  to confirm the variant boots and matches in-dist numbers within noise.
- **Result:** 

---

<!-- Future phase entries get appended below. Keep the schema. -->
