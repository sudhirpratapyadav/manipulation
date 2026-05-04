# Sim2Real Plan: Pick-Cube OSC State → Vision

Concrete execution plan for transferring the trained state-based PPO
policy on `pick_cube_osc` to a vision-based deployable policy on
`pick_cube_vision_osc`. Background and literature: see
`state_to_vision_transfer.md`.

## Phases at a glance

| Phase | What | Algorithm | Output |
|---|---|---|---|
| A | DAgger distillation | Supervised (MSE on actions) | Vision student that imitates teacher in sim |
| B | + state-prediction aux loss | Supervised (MSE actions + MSE state) | Stronger CNN encoder |
| C | PPO fine-tune in vision env | RL, warm-started from B | Robust vision policy |
| D | Sim-to-real | Heavier visual DR, real-image fine-tune | Deployable policy |

We start with **Phase A** end-to-end, sanity-check, then move to B/C/D.
Each phase is a checkpoint we can stop at if results are good enough.

---

## Phase A — DAgger distillation

**Goal:** student vision policy reaches teacher-level success rate on
the *training* env (no extra DR yet).

### Tooling

Use RSL-RL's `Distillation` + `DistillationRunner` (already in venv).
Don't build from scratch.

- Loss: `mse` on action means.
- Style: on-policy student rollouts, teacher relabels each obs.
- Asymmetric obs: native via `obs_groups`.
- Teacher load: auto-detects `actor_state_dict` from the existing PPO
  checkpoint.

### Code to write

1. **`kinova_pick_cube_distill_cfg.py`** — distillation runner config.
   - `student` = vision model: same CNN+MLP architecture as
     `kinova_pick_cube_vision_osc_ppo_cfg.actor` (spatial-softmax CNN
     [16, 32], hidden dims [256, 256, 128], elu).
   - `teacher` = state model: MLP matching the trained `pick_cube_osc`
     actor — **hidden dims (512, 256, 128), elu**, scalar-std Gaussian,
     `obs_normalizer` enabled. Confirmed from
     `wandb/run-.../jn3l22j9/files/config.yaml`.
   - `obs_groups = {"student": ("actor", "camera"), "teacher": ("critic",)}`.
     The existing `critic` group already carries the full 33D
     privileged state.
   - `algorithm.class_name = "rsl_rl.algorithms:Distillation"`,
     `loss_type = "mse"`, `learning_rate = 1e-3`,
     `num_learning_epochs = 1`, `gradient_length = 15`.
   - `num_steps_per_env = 24`, `max_iterations = 2000` (≈2M env steps
     at 4096 envs).

2. **`train_distill.py`** — entrypoint.
   - Build `pick_cube_vision_osc` env (play=False).
   - Build `DistillationRunner(env, cfg)`.
   - `runner.load(<teacher_ckpt>.pt)` — auto-loads teacher only.
   - `runner.learn(num_iters)`.
   - Save student checkpoint.

3. **`play_distill.py`** — eval entrypoint.
   - Load student, run rollouts in vision env, log success rate +
     metrics. Reuse existing play script as template.

### Hyperparameters to start with

| param | value | rationale |
|---|---|---|
| `loss_type` | `mse` | RSL-RL default; KL not needed for deterministic deploy |
| `learning_rate` | `1e-3` | RSL-RL default for distillation |
| `num_learning_epochs` | 1 | DAgger is on-policy; 1 epoch per rollout batch |
| `gradient_length` | 15 | RSL-RL default; truncated BPTT for RNNs (not used here, but harmless) |
| `max_grad_norm` | 1.0 | mild safety |
| `num_steps_per_env` | 24 | matches existing PPO config |
| `num_envs` | 4096 | matches existing config |
| `max_iterations` | 2000 | ≈2M env steps; ANYmal-style distillations converge in 1–3M |

### Success criteria for Phase A

- Vision student success rate within **80–90%** of teacher's success
  rate on the same env.
- `cube_to_goal_error` curve flattens at a value close to teacher's.
- Action MSE loss plateaus below ~0.01 (sanity, not strict).
- No DR added yet beyond what the teacher trained with.

If we hit 90%+: Phase A done, decide whether B/C are needed.
If <60%: encoder is the bottleneck → go to Phase B.
If 60–80%: try Phase C (PPO finetune) before B.

### Risks / things to watch

- **Camera obs scale.** RGB in [0, 255] vs normalized — check what
  `manipulation_mdp.camera_rgb` returns; CNN expects /255.0 or
  per-channel normalization.
- **Teacher checkpoint architecture mismatch.** Trained PPO actor is
  `MLPModel` with specific hidden dims; the distillation `teacher` cfg
  must match exactly or `load_state_dict` fails.
- **Action space sanity.** Teacher and student both output 7D
  (6D OSC + 1D gripper). The distillation loss is plain MSE across all
  7 dims — gripper might want a heavier weight. Defer until we see
  results.
- **Gripper action sampling.** Teacher is a Gaussian policy; distilling
  the *mean* is what RSL-RL does. Stochastic teacher rollouts not used.
- **Memory.** Two 4096×32×32×3 cameras + storing
  `transition.observations` for full rollout. Watch GPU mem; drop
  num_envs if OOM.

### Deliverable

`student_phaseA.pt` checkpoint + eval numbers logged in
`exp_tracker.md`.

---

## Phase B — state-prediction auxiliary loss

**Trigger:** Phase A success rate < ~60%, or visibly poor cube
localization in rollouts.

**Goal:** force the CNN to produce features that linearly decode the
teacher's privileged state (`ee_to_cube`, `cube_pos`, `cube_to_goal`).

### Code to write

1. **Subclass `Distillation`** (~30 LOC) to add a regression head off
   the student CNN.
   - Head: small MLP (CNN_features → 9D = ee_to_cube + cube_pos +
     cube_to_goal).
   - Aux loss: `mse(head(cnn_feat), privileged_state_subset)` with
     coefficient `lambda_aux` (start 1.0, anneal to 0).
   - Add to total loss in `update()`.

2. **Plumb privileged state as a target** through `obs_groups`. Already
   accessible via `("critic",)` — extract the relevant 9 dims.

### Hyperparameters

- `lambda_aux`: 1.0 → 0 linearly over training, or constant 0.5.
- Same other hparams as Phase A.

### Success criteria

- Student outperforms Phase A by ≥10% success rate.
- Aux loss decreases (sanity that the CNN is actually learning the
  geometry).

---

## Phase C — PPO fine-tune in vision env

**Trigger:** Phase A or B doesn't close the gap enough, or we want
robustness to states the teacher never visited.

**Goal:** RL polish on top of distilled student. Same vision env,
real reward.

### Approach

- Start from `student_phaseA.pt` (or B).
- Run standard PPO via `kinova_pick_cube_vision_osc_ppo_cfg`,
  initialized from the distilled student weights.
- **Critic:** state-only (drop the CNN from critic — see open question
  in `state_to_vision_transfer.md`). Warm-start from the trained
  `pick_cube_osc` critic if architectures match.
- Fix `obs_groups` to drop `camera` from critic:
  `{"actor": ("actor", "camera"), "critic": ("critic",)}`.

### Optional: KL-to-teacher kickstarting

Not in RSL-RL out of the box. Skip for first attempt — most papers
just warm-start and let PPO take over. Add only if PPO destabilizes
the distilled policy.

### Success criteria

- Match or beat teacher success rate on the vision env.
- Stable training; no catastrophic forgetting of distilled behavior in
  early iters (watch first 100 iters carefully).

---

## Phase D — sim-to-real

**Trigger:** Phase C policy is good in sim.

### Steps

1. **Visual DR audit.** Add to vision env:
   - Lighting randomization (intensity, direction, color temp).
   - Camera pose jitter (±2cm position, ±5° rotation).
   - Background textures (groundplane variation).
   - Distractors (random cubes outside workspace).
   - Image-space augmentations applied at obs time: color jitter,
     gaussian blur, gaussian noise.

2. **Re-run Phase C** with the DR'd env, warm-start from clean Phase C
   checkpoint.

3. **Real-world eval.** Collect ~100 real rollouts on the Kinova,
   measure success rate.

4. **Optional real-image fine-tune.** If sim-trained policy gaps on
   real:
   - Collect real RGB while running policy.
   - DAgger-relabel with sim teacher? — only if we can register the
     real cube pose. Otherwise pure RL on real (slow) or
     image-translation (RCAN-style).

### Success criteria

- ≥70% real-world success rate. Below that, iterate on DR or collect
  real data.

---

## Open questions to resolve before Phase A starts

1. **Teacher checkpoint location** — what's the path? Need to confirm
   it loads via `Distillation.load()`.
2. **Teacher architecture** — confirm hidden dims / activation match
   `kinova_pick_cube_osc_ppo_cfg`'s actor exactly.
3. **Camera obs format** — `[0,255]` uint8 or `[0,1]` float? Affects
   CNN input normalization.
4. **Critic in vision env** — keep camera input or drop it? For Phase
   A this doesn't matter (no critic in distillation); revisit at C.
5. **Gripper action loss weight** — equal to OSC dims, or upweighted?
   Defer; check Phase A rollouts first.

---

## Execution order

1. Resolve open questions 1–3 above.
2. Write `kinova_pick_cube_distill_cfg.py`.
3. Write `train_distill.py`.
4. Smoke test: 50 iterations, confirm losses decrease, no crashes.
5. Full Phase A run: 2000 iterations.
6. Eval; record in `exp_tracker.md`.
7. Decide: ship Phase A, or proceed to B / C.
