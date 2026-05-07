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

### A_03_cams64_resume_20k (continuation of A_03_cams64)

- **Status:** done (resumed from `model_1999.pt`, ran iter 2000 → 10800)
- **Training-env metrics at iter 10800:**
  - `Loss/behavior` 0.54 → 0.17 (monotone)
  - `Episode_Reward/goal_precise` 0.71 (teacher 0.66)
  - `Episode_Metrics/object_to_goal_error` 0.156 m
  - `Episode_Metrics/ee_to_object_error` 0.055 m
  - `Episode_Termination/object_out_of_bounds` ~26%
- **Reading:** Pure DAgger plateaued reasonably. OOB ~26% is the
  dominant failure mode now, not "didn't reach the cube". User
  decided OOB is acceptable; visual encoder is the next move.
- **A_03 fresh-start reference curve** (used as ceiling for MCR runs):
  iter 10 → 4.76 / 30 → 3.89 / 100 → 3.58 / 200 → 2.86 / 500 → 1.71
  / 1000 → 0.68 / 1999 → 0.45 (read from
  `2026-05-03_22-28-12_A_03_cams64` events file).

---

## Phase A.MCR — frozen MCR ResNet-50 encoder swap

Plan: `plan_v2.md`. Encoder: `src/kinova_tasks/encoders/mcr_encoder.py`.
Goal: replace from-scratch spatial-softmax CNN with a frozen MCR
ResNet-50 (Jiang et al. 2024, [arXiv:2410.22325](https://arxiv.org/abs/2410.22325)).

Common setup unless noted:
- env: `Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova` (same vision env as A_03,
  64×64 wrist + 64×64 d455).
- encoder: `FrozenMCREncoder` — frozen ResNet-50, MCR weights,
  64→224 bilinear upsample, ImageNet normalize, per-camera LayerNorm
  (added during smoke testing — without it the head can't absorb the
  unbounded 4096-D ResNet activations).
- teacher: same `model_4999.pt` from wandb `jn3l22j9`.
- num_envs: 1024. num_steps_per_env: 24.
- throughput: ~680 fps (~36 s/iter) — **7.7× slower** than A_03's
  ~3000 fps (4.6 s/iter), driven by the ResNet-50 forward at 224×224.
  Combined with ~3-4× slower convergence rate, MCR is a **~12× total
  compute multiplier** vs the from-scratch CNN baseline.

### MCR_01_full

- **Status:** killed at iter 109
- **Encoder:** MCR ResNet-50, avg-pool, LayerNorm
- **lr:** 1e-3 (RSL-RL default for distillation)
- **Result:** rolling-min plateau ~5.5 from iter 20 onwards. lr too
  high — Adam updates large enough that stochastic student rollouts
  re-create the high-loss regime each iter.

### MCR_02_lr3e4

- **Status:** running (launched 2026-05-06 ~23:55, iter ~425 at
  this writeup, ETA ~46h to iter 5000)
- **wandb:** run name `MCR_02_lr3e4`, project `mjlab-kinova-tasks-osc-vision`
- **Encoder:** MCR ResNet-50, avg-pool (4096-D), LayerNorm
- **lr:** 3e-4 (lower than MCR_01)
- **Local log:** `logs/distill_MCR_02_lr3e4.log`
- **Curve so far** (rmin30 envelope):
  - iter 50: ~5.4    | A_03 ref: 3.97
  - iter 100: ~5.0   | A_03 ref: 3.58
  - iter 200: 4.51   | A_03 ref: 2.86
  - iter 300: 4.18   | A_03 ref: 2.40
  - iter 400: 3.19   | A_03 ref: ~2.0
  - **delta vs A_03: ~1.2-1.5 units behind, slowly closing.**
- **Reading:** Not plateaued — slow envelope decay. Oscillation
  amplitude ~1.5 units around a slowly-falling minimum. Will continue
  to iter 1000-2000 to see whether it can match A_03's iter-500 (1.71).

### MCR_03_ss

- **Status:** killed at iter 84
- **Variant:** Cut ResNet at `layer4` → spatial-softmax over
  `(2048, 7, 7)` map → `(B, 8192)` keypoint output. Hypothesis:
  avg-pool throws away spatial info that the policy needs.
- **lr:** 3e-4
- **Result:** rolling-min flat at 5.05–5.09 over iters 30-84 (delta
  -0.05 between iter-20-50 and last-30 windows — i.e. *no* improvement).
  At the same iter range MCR_02 had +0.86 improvement.
- **Reading:** Spatial-softmax over coarse layer4 features doesn't
  help.  Each of the 7×7 cells aggregates ~32×32 of input, so the
  resulting "keypoint" is too coarse to carry useful localization.
  The from-scratch CNN's spatial-softmax works because it operates on
  much-shallower feature maps with smaller receptive fields.

### MCR_04_widehead

- **Status:** killed at iter 39 (relaunched once — first try died on
  `uv run` failing to fetch `kortex_api`; future MCR runs use
  `.venv/bin/train-distill` directly)
- **Variant:** avg-pool encoder + wider MLP head `(1024, 512, 256, 128)`
  instead of `(512, 256, 128)`. Hypothesis: head too narrow to project
  4096-D feature into 7-D action.
- **lr:** 3e-4
- **Result:** apples-to-apples vs MCR_02 at iters 0-30: window-min
  diff ±0.1. **Wider head is identical to narrow head.** Head capacity
  is not the bottleneck.
- **Reading:** The slowness comes from the encoder–task mismatch, not
  from an under-parameterized head.

### R3M_05_droid

- **Status:** done (killed at iter 789; production winner)
- **wandb:** run name `R3M_05_droid` (logged under same MCR project +
  experiment by reusing `Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova` task
  with `MCR_WEIGHTS_PATH=assets/mcr/r3mdroid_resnet50.pth` env var)
- **Encoder:** ResNet-50 pretrained on DROID with R3M's
  time-contrastive + L1-sparsity objective, frozen. Same architecture
  / pipeline as MCR_02 but different pretrained weights.
- **lr:** 3e-4
- **Local log:** `logs/distill_R3M_05_droid.log`
- **Production ckpt:** `model_600.pt`
- **Full envelope (50-iter window-min):**
  - iter   0- 49: 3.15
  - iter 100-149: 3.73
  - iter 200-249: 2.22
  - iter 300-349: 1.78
  - iter 400-449: 1.65
  - iter 500-549: 1.51
  - **iter 550-599: 1.44 ← best**
  - iter 600-649: 1.52
  - iter 650-699: 1.69 (rising)
  - iter 700-749: 1.76 (continuing up)
- **Reading:** R3M-DROID dramatically outperforms MCR — at iter 600
  R3M_05 was at 1.44 while MCR_02 was at ~3 (killed).  R3M beat A_03's
  reference curve through iter ~500, then plateaued / diverged late
  while A_03 continued monotonically to 0.68@1000.  Sim2real-relevant
  conclusion: R3M-DROID frozen is the right encoder for this
  pipeline; from-scratch CNN still wins asymptotic loss.

### R3M_06_lr1e3

- **Status:** killed at iter 84
- **Variant:** R3M-DROID frozen + lr 1e-3 (tests if R3M tolerates a
  higher lr than MCR_01 did).
- **Result:** 0.4-0.7 units worse than R3M_05 across iters 25-74.
  Same divergence pattern as MCR_01 — bigger oscillations, higher
  envelope minimum.  lr=3e-4 is the right setting for this regime
  regardless of pretrained-weights choice.

### R3M_07_ll4

- **Status:** killed at iter 144 (relaunched once — first attempt OOM'd
  at 41 GB on 1024 envs because layer4 backprop activations don't fit;
  second attempt at 512 envs ran fine).
- **Variant:** R3M-DROID with `layer4` (last ResNet stage) unfrozen
  via the `unfreeze_layers=("layer4",)` cnn_cfg flag.  ~15M trainable
  params added.  Tests whether partial fine-tuning lets the encoder
  adapt to pick-cube geometry.
- **Result (env-step parity vs R3M_05):**
  - k=50-74 (R3M_07 iters / R3M_05 iters 100-149): R3M_07 +0.45 *worse*
  - k=75-99 (R3M_07 iters / R3M_05 iters 150-199): R3M_07 +1.06 *worse*
  - k=100-124 (R3M_07 iters / R3M_05 iters 200-249): R3M_07 +0.63 *worse*
- **Reading:** Layer4 fine-tuning **HURTS** at fixed env-step budget.
  Frozen R3M is the production setup.  The DROID-pretrained features
  are good as-is; allowing them to drift adds noise without helping
  pick-cube performance.
- **Caveat (drove the rerun decision):** killed at only 144 iters; the
  early phase is noisy and a longer run may behave differently.  Queued
  for full-3.5k rerun (R3M_07_ll4_v2).

### R3M_08_smallhead (running)

- **Status:** running (iter 359 at this writeup, target 5000)
- **Variant:** R3M-DROID frozen + hd=(256, 128) MLP head (smaller than
  R3M_05's (512, 256, 128)).  Tests whether the 4096-D pretrained
  feature is so well-structured that even a tiny head suffices.
- **Window-min comparison vs R3M_05 (head-to-head):**
  - iter   0- 24: 3.10 vs 3.15 (-0.05)
  - iter  25- 49: 5.24 vs 5.36 (-0.12)
  - iter  50- 74: 5.02 vs 4.67 (+0.35)
  - iter  75- 99: 4.40 vs 4.69 (-0.29)
  - iter 100-124: 4.24 vs 3.78 (+0.46)
- **Latest:** iter 348 rmin30 = 1.88 (vs R3M_05 at iter 308: 1.86).
  **Essentially tied with R3M_05.**
- **Reading:** Head capacity does not matter — even the small head
  matches the standard one.  R3M-DROID features are well-formatted
  enough that any reasonable readout works.

### R3M_09_lr1e4 (running)

- **Status:** running (just launched, iter 4 at this writeup)
- **Variant:** R3M-DROID frozen + lr=1e-4 (lower than R3M_05's 3e-4).
  Tests whether slower lr prevents the late-phase plateau / divergence
  R3M_05 showed past iter 600.

### Phase A.MCR — interim conclusions (after 9 runs)

1. **R3M-DROID >> MCR-DROID** for this task (1.44 vs 2.97 at iter 600).
   Contradicts the Vakil 2025 ranking; the pick-cube + DAgger + ResNet-50
   regime favors R3M's time-contrastive objective over MCR's
   action-prediction objective.
2. **Frozen R3M beats from-scratch CNN in mid-phase, loses in late.**
   - Iter 200-500: R3M_05 1.65 ≤ A_03 1.71
   - Iter 500-1000: A_03 0.68 << R3M_05 plateau ~1.44
   - Frozen pretrained features have a **convergence floor** the
     from-scratch CNN doesn't.
3. **Layer4 unfreezing HURTS** at fixed env-step budget. Production
   setup = fully frozen.
4. **Head capacity doesn't matter.** (256,128), (512,256,128), and
   (1024,512,256,128) all converge identically with R3M frozen.
5. **lr=3e-4 is right** for ResNet-50 distillation; lr=1e-3 destabilizes
   for both MCR and R3M.
6. **Spatial-softmax over coarse layer4 (7×7) features does NOT
   substitute for the from-scratch CNN's keypoint output.**
7. **Throughput: ~680 fps vs A_03's ~3000 fps** (7.7× slower wall
   clock per iter).  Combined with the asymptotic loss gap, MCR/R3M
   are ~8× more compute for a *worse* asymptote on sim alone.

### The kill-too-early correction

**Critical methodological correction:** several of the 9 runs above
were killed at iter 39-150 based on early-iter signal that turned out
NOT to predict end behavior.  R3M_05 looked tied with MCR at iter 39,
then broke away by iter 200 to become the production winner.  Without
that lucky long-watch, R3M would have been killed alongside MCR.

**New rule:** every run goes 3.5k iters minimum before any decision.
Reruns queued (in priority order, ~35h each on 2 GPUs):

- **MCR_02_v2** — MCR avg-pool full 3.5k (was killed at 528)
- **R3M_07_ll4_v2** — layer4 unfrozen full 3.5k at 512 envs (was killed at 144)
- **MCR_03_ss_v2** — spatial-softmax full 3.5k (was killed at 84)
- **MCR_04_widehead_v2** — wider head full 3.5k (was killed at 39)

Skipping reruns of MCR_01 and R3M_06 (lr=1e-3) — both showed
unambiguous early-iter divergence that's known not to recover.
Total queued time: ~3 days on 2 GPUs after R3M_08 + R3M_09 finish.

---

## Phase B — state-prediction aux loss

_Not started. Folded into `plan_v1.md` but plan_v2 deprioritized this
in favor of the encoder swap. Revisit once Phase A.MCR completes._

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
