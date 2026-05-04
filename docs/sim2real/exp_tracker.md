# Sim2Real Experiment Tracker

Per-run log for the state→vision distillation pipeline. One row per
training run. Append, don't overwrite. See `plan.md` for the phase
plan and `state_to_vision_transfer.md` for context.

## Conventions

- **Run ID**: `<phase>_<NN>_<short_tag>`, e.g. `A_01_smoke`,
  `A_02_full`, `B_01_aux1.0`, `C_01_finetune`.
- **Status**: `running` / `done` / `failed` / `aborted`.
- **Success rate**: episodes with `cube_to_goal_error < 0.05` at
  episode end, over ≥256 eval envs. Note env config used.
- Always record the **teacher checkpoint** used and the **commit hash**
  the run was launched from.
- For failed runs, write what broke + the fix in the notes column.

---

## Reference: teacher policy (state-based)

| field | value |
|---|---|
| wandb run | `jn3l22j9` (run-20260430_220905) |
| Checkpoint path | `manipulation/wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt` |
| Iter trained | 5000 (final = `model_4999.pt`) |
| Env | `kinova_pick_cube_osc` |
| Obs dim (actor & critic) | 33 |
| Action dim | 7 (6D OSC + 1D gripper) |
| Actor architecture | MLP `33 → 512 → 256 → 128 → 7`, elu |
| Critic architecture | MLP `33 → 512 → 256 → 128 → 1`, elu |
| Distribution | Gaussian, scalar `std_param` (shape `(7,)`) |
| Has `obs_normalizer` | yes — running mean/var; persisted with the actor |
| Checkpoint top-level keys | `actor_state_dict`, `critic_state_dict`, `optimizer_state_dict`, `iter`, `infos` |
| RSL-RL distill loadable | yes — `Distillation.load()` auto-detects `actor_state_dict` and loads it as the teacher |
| Success rate (sim) | _to measure with play script_ |
| `cube_to_goal_error` mean | _to measure_ |
| Notes | Architecture is **(512, 256, 128)**, not the (256, 256, 128) shown in the default vision PPO cfg. The distill `teacher` cfg must use 512-256-128 to match. |

---

## Phase A — DAgger distillation

### A_01_smoke

- **Status:** done
- **Goal:** sanity check — pipeline runs end-to-end, losses computable.
- **Teacher ckpt:** `wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt`
- **Config:** 64 envs × 3 iter, then 256 envs × 30 iter. wandb disabled.
- **Result:** All checks pass. Student CNN (16/32 spatial-softmax) +
  MLP (256, 256, 128) elu, Gaussian scalar-std.  Teacher MLP
  (33→512→256→128→7) elu loaded from PPO ckpt with strict=True (no
  shape mismatch).  DAgger rollout + MSE backprop confirmed working.
  Loss trajectory at 256 envs over 30 iters: oscillates 3.6–5.3
  (expected — student exploring, teacher returns large deltas in
  off-distribution states; convergence requires many more iters).
  ~4s/iter at 256 envs on RTX A6000.
- **Issues found:** none. Cosmetic EGL teardown warning from mujoco
  is harmless (known mujoco issue).

### A_03_cams64 (cam upgrades)

- **Status:** running (launched 2026-05-03 ~22:30)
- **Goal:** Phase A re-run with improved cameras after qualitative
  inspection of A_02b showed grasping was unreliable.
- **Camera changes vs A_02b:**
  - Wrist: 32×32 → **64×64**, fovy 41.8° → **75°**, rotated **-10°
    about local X**, pos shifted to `(0, -0.06639, -0.098475)`
    (slightly down/back from default).
  - D455: 32×32 → **64×64**, world-x centered to **0** (was 0.25),
    z lowered to **0.3** (was 0.5), look-at **(0, -0.5, 0.05)**
    (was `0.2`).
- **Teacher ckpt:** unchanged (`model_4999.pt`)
- **Local log:** `manipulation/logs/distill_A_03_cams64.log`
- **wandb tags:** `phase_a`, `dagger_mse`, `teacher_jn3l22j9`,
  `cams64`, `wrist_-10x_fov75`, `d455_low_lookat0.05`

### A_02b_full (relaunch with d455 per-env fix)

- **Status:** running (launched 2026-05-03 16:18)
- **Goal:** full 2000-iter Phase A run with d455 attached per-env.
- **Teacher ckpt:** `wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt`
- **Bug fixed in this run:** d455 camera was previously created as a
  *worldbody* camera at absolute world coords `(0.25, -0.9, 0.5)`. With
  1024 parallel envs at different `env_origins`, only env 0 saw its
  workspace; envs 1–1023 saw an empty area. Fix: set
  `parent_body="robot/base_link"` on the `CameraSensorCfg` so the cam
  is cloned per-env and follows the robot base. pos/quat unchanged
  (base_link sits at env origin).
- **Verified by recorder:** d455 mp4 sizes jumped from ~7K (mostly
  static across envs 1-3) to ~25-27K (real per-env motion), on par
  with wrist cam sizes.
- **wandb tags:** `phase_a`, `dagger_mse`, `teacher_jn3l22j9`, `d455_fix`
- **Local log:** `manipulation/logs/distill_A_02b_full.log`

### A_02_full (CANCELLED)

- **Status:** cancelled at iter ~1100/2000 (~2h50m elapsed)
- **Reason:** d455 was a worldbody camera with absolute world coords —
  only env 0 framed its workspace; envs 1–1023 trained against
  meaningless d455 input. Killed and relaunched as A_02b_full with the
  fix above.
- **Old wandb run:** `275jdjkq`
  (https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc-vision/runs/275jdjkq)

### A_02_full (original, pre-cancel)

- **Status:** cancelled (see above)
- **Goal:** full 2000-iter Phase A run.
- **Teacher ckpt:** `wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt`
- **Config:** 1024 envs, 2000 iter, save_interval=100, video every
  200 iter, MSE loss, lr=1e-3, num_steps_per_env=24.
- **wandb run id:** `275jdjkq`
- **wandb URL:** https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc-vision/runs/275jdjkq
- **wandb project:** `mjlab-kinova-tasks-osc-vision`
- **wandb entity:** `sudhirpratapyadav-indian-institute-of-technology-jodhpur`
- **wandb tags:** `phase_a`, `dagger_mse`, `teacher_jn3l22j9`
- **GPU:** `cuda:0` (CUDA_VISIBLE_DEVICES=1, RTX A6000)
- **Iteration time:** ~4.65s/iter (1024 envs) → ETA 3h 14m
- **Local log:** `manipulation/logs/distill_A_02_full.log`
- **Local ckpt dir:** `manipulation/logs/rsl_rl/kinova_pick_cube_distill_osc/2026-05-03_13-25-13_A_02_full/`
- **Startup checks (all passed):**
  - `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=0` (rendering on dedicated GPU)
  - Teacher MLP `33→512→256→128→7` loaded with strict=True
  - Student MLP `(actor:24 + camera_features) → 256 → 256 → 128 → 7`
    with spatial-softmax CNN [16, 32]
  - Critic obs group = 33D (teacher's natural input)
  - Camera obs `(3, 32, 64)` = wrist `(3, 32, 32)` ⊕ d455 `(3, 32, 32)`
  - DAgger active: student rollouts, teacher relabels each obs
  - Iter 0–2 behavior loss: 3.21 → 3.61 → 4.06 (expected early
    DAgger trajectory; converges over later iters)

---

## Phase B — state-prediction aux loss

_Not started. Templates to be added when Phase A motivates this._

---

## Phase C — PPO fine-tune

_Not started._

---

## Phase D — sim-to-real

_Not started._

---

## Decisions log

Running notes on choices made and *why*, so we don't relitigate them.

- **Use RSL-RL `Distillation` instead of building from scratch.** Already
  installed in the venv, same algorithmic family as `mjlab.rl`, used by
  Isaac Lab as the official student-teacher workflow. Loss = MSE on
  action mean (RSL-RL default; KL not needed for deterministic deploy).
  DAgger-style: student rolls out, teacher relabels.
- **Teacher arch matches checkpoint exactly: (512, 256, 128) elu.**
  Confirmed from wandb run jn3l22j9 `config.yaml`. The default vision
  PPO cfg used (256, 256, 128); we override to match the trained
  teacher or `load_state_dict(strict=True)` would fail.
- **`obs_groups = {"student": ("actor", "camera"), "teacher": ("critic",)}`.**
  The `critic` group of `pick_cube_vision_osc` already carries the full
  33D privileged state (it was kept untouched when the actor's
  privileged terms were popped). So no env changes needed — just point
  the teacher at the existing group.
- **Custom `MjlabDistillationRunner` (subclasses RSL-RL's
  `DistillationRunner`).** Mirrors `MjlabOnPolicyRunner`'s mjlab-specific
  patches: env `common_step_counter` save/load, legacy ckpt key
  migration, `upload_model` flag.
- **Custom `train_distill.py` entrypoint.** The base mjlab `train.py`
  uses `--agent.resume` + `load_run` regex to find a checkpoint inside
  `log_root/experiment_name/`. The teacher is in a wandb run dir
  outside that tree, so a thin custom entrypoint that takes
  `--teacher-ckpt <path>` is cleaner than fighting the resume logic.
- **wandb project = `mjlab-kinova-tasks-osc-vision`** (state project
  was `mjlab-kinova-tasks-osc`; appended `-vision` per user instruction).

---

## Issues / TODO

- [x] Locate trained teacher checkpoint (path + commit). → wandb run
      `jn3l22j9`, `model_4999.pt`.
- [x] Confirm teacher actor architecture (hidden dims, activation).
      → (512, 256, 128), elu, scalar-std Gaussian.
- [ ] Confirm camera obs scale (`uint8 [0,255]` vs `float [0,1]`) in
      `manipulation_mdp.camera_rgb`.
- [ ] Decide: keep camera in critic for Phase C, or drop? (irrelevant
      for Phase A — distillation has no critic).
- [ ] Measure baseline teacher success rate on `pick_cube_osc` so we
      have a number to chase.
