# State → Vision Transfer for Pick-Cube OSC

Notes on transferring a trained state-based PPO policy
(`pick_cube_osc.py`) to a vision-based variant (`pick_cube_vision_osc.py`)
for sim-to-real deployment.

## Setup recap

- **Teacher (trained):** state-based PPO on `pick_cube_osc`. Actor obs (33D):
  `joint_vel (7) + ee_pose (6) + gripper_state (1) + ee_to_cube (3) +
  cube_pos (3) + cube_to_goal (3) + goal_pos (3) + last_action (7)`.
  Action: 6D OSC delta + 1D gripper.
- **Student (target):** `pick_cube_vision_osc`. Same env, same action space.
  Privileged terms (`ee_to_cube`, `cube_pos`, `cube_to_goal`) removed from
  actor; replaced with 32×32 RGB from wrist cam + static D455. `goal_pos`
  kept as proprioceptive goal conditioning. CNN: spatial-softmax,
  channels [16, 32], kernels [5, 3], stride [2, 2].
- **Existing infrastructure:** asymmetric actor-critic already wired
  (`obs_groups={"actor": ("actor","camera"), "critic": ("critic","camera")}`).
  DR on fingertip/cube friction, cube mass, cube color.

## Approaches in the literature

### 1. Pure teacher–student (privileged distillation)

Student sees only deployable obs; teacher acts on privileged state. Train
student to mimic teacher in sim, deploy student.

- **Pinto et al. "Asymmetric Actor Critic" (2017)** — privileged critic,
  image actor, joint RL. Cheap to bolt on.
- **Chen et al. "Learning by Cheating" (CoRL 2019)** — two-stage:
  (1) RL with state, (2) DAgger-style distill image policy from state
  policy. Closest match to this setup.
- **Kumar et al. "RMA: Rapid Motor Adaptation" (RSS 2021)** — teacher
  conditions on env params, student infers them from history.
- **Chen, Xu, Agrawal "A System for General In-Hand Object
  Re-Orientation" (CoRL 2022)** — canonical state→vision distillation
  recipe for dexterous manipulation.
- **Lee et al. ANYmal (Science Robotics 2020)** — same recipe in legged
  locomotion.

**DAgger over BC:** plain BC on teacher rollouts suffers covariate shift
once the student vision encoder adds noise. DAgger (roll out student,
label with teacher) fixes it — used by LbC and successors.

### 2. Distillation + RL fine-tune (usually wins)

Pure BC/DAgger leaves residual gap from camera noise, occlusions, DR.
Standard fix:

1. Distill state→vision teacher (DAgger).
2. PPO fine-tune in the vision env with KL-to-teacher regularization
   that decays to zero.

- **Schmitt et al. "Kickstarting" (DeepMind, 2018)** — KL-to-teacher
  auxiliary loss on top of RL.
- **DexPoint and similar** — distill then RL-finetune in vision env.

This reuses the existing checkpoint maximally; recommended starting
point.

### 3. Representation-focused (state-prediction aux loss)

Instead of (or in addition to) distilling actions, force the vision
encoder to produce a vector close to the privileged state.

- **Chen et al. "Visual Dexterity"** — state-prediction aux losses on
  CNN; frozen / fine-tuned MLP head reuses teacher weights.
- **DrQ-v2 (Hansen et al.)** — image augmentation stabilizes learning.
- **R3M / VC-1 / MVP** — pretrained visual encoders, frozen, small
  policy on top. Skip CNN training.

If the teacher MLP is small, freeze it and train a CNN whose output is
forced to ≈ the privileged state vector via regression loss. Cheap and
very effective when the privileged state is geometric (as here:
`ee_to_cube`, `cube_pos`, `cube_to_goal`).

### 4. Domain randomization + sim-to-real

Combine the above with:

- **OpenAI "Learning Dexterous In-Hand Manipulation" / ADR** —
  automatic domain randomization.
- **CycleGAN / RL-CycleGAN, RCAN (Rao et al. 2020)** — image-to-image
  translation sim↔real.
- Photorealistic sim (Isaac Lab ray tracing, Blender renders).

Already present: friction/mass/cube color. Still needed for vision:
lighting, camera pose jitter, background textures, distractors,
image-space augmentations (color jitter, blur, noise).

### 5. Other recent options

- **Diffusion Policy (Chi et al., RSS 2023)** — DAgger + BC + diffusion
  student. Overkill unless many demos available.
- **VLM/foundation-model bootstrapping (RT-2, OpenVLA)** — irrelevant
  for this sim2real loop; listed for completeness.

## Recommended recipe for this setup

Combines LbC + RL finetune + state-prediction aux loss.

### Stage A — DAgger distillation

- Roll out the student (CNN + MLP) in `pick_cube_vision_osc`.
- Label every step with the teacher's action computed on privileged
  state.
- MSE on the 6D OSC delta; BCE (or MSE) on the 1D gripper.
- ~1–2M env steps usually enough.

### Stage B — auxiliary state regression

- Add a head off the CNN that predicts `ee_to_cube`, `cube_pos`,
  `cube_to_goal`.
- Joint loss: action distillation + state regression.
- Forces the encoder to learn the geometry the teacher used.

### Stage C — PPO fine-tune

- Initialize from the distilled policy.
- Run PPO in the vision env with the same reward.
- Add `KL(student ‖ teacher_on_priv_state)` with a coefficient that
  decays from ~1.0 to 0.
- Keep current DR + add visual DR (lighting, camera jitter, textures,
  image augs).

### Stage D — sim-to-real

- Heavier visual DR.
- Optional: RCAN or fine-tuning on ~100 real rollouts.

## Tooling decision: use RSL-RL's `Distillation`

`mjlab.rl` is built on RSL-RL, and RSL-RL ships a working
state→vision distillation trainer that is also Isaac Lab's official
student-teacher workflow. Already installed at
`mjlab/.venv/.../rsl_rl/algorithms/distillation.py`.

Verified properties (from the source):

- **Loss:** MSE (default) or Huber via `loss_type`. Not KL — for
  continuous OSC actions MSE on the mean is the standard choice and is
  what Isaac Lab and ETH legged-gym use.
- **DAgger-style:** student rolls out, teacher relabels the same obs.
  Not vanilla BC from a frozen dataset.
  ```python
  self.transition.actions = self.student(obs, stochastic_output=True).detach()
  self.transition.privileged_actions = self.teacher(obs).detach()
  ```
- **Asymmetric obs:** native, via `obs_groups` with default sets
  `["student", "teacher"]`. Same mechanism the PPO config already uses.
- **Teacher loading:** `Distillation.load()` auto-detects a PPO
  checkpoint (`actor_state_dict`) and loads it as the teacher. The
  existing `pick_cube_osc` checkpoint plugs straight in.
- **Runner:** `DistillationRunner` is a 25-line subclass of
  `OnPolicyRunner` — same training loop, gated on teacher loaded.

### MSE vs KL — why MSE here

KL only matters when the teacher outputs a non-trivial distribution and
you want to preserve exploration noise. For deployment we want a
near-deterministic student; MSE on the mean is simpler, more stable,
and empirically close to KL for Gaussian policies. Stick with MSE.

### What still needs to be written

- A `kinova_pick_cube_distill_cfg()` returning a config with:
  - `student` = vision model (CNN + MLP, like current vision actor).
  - `teacher` = state model (matches the trained checkpoint).
  - `obs_groups = {"student": ("actor","camera"), "teacher": ("critic",)}`
    — the existing `critic` group already holds the full privileged
    state, so it's the teacher's natural input.
  - `algorithm.class_name = "rsl_rl.algorithms:Distillation"`,
    `loss_type = "mse"`.
- A `train_distill.py` entrypoint: build `DistillationRunner`, call
  `.load(teacher_ckpt)`, then `.learn(N)`.
- Optional Stage B (state-prediction aux loss): subclass `Distillation`
  to add a regression head + auxiliary loss. ~30 lines.

### What is *not* in RSL-RL

- **Stage C kickstarting (KL-to-teacher in PPO).** Not built in. The
  simpler equivalent — distill → PPO finetune from-scratch on the
  warm-started student — is what most papers actually do.
- **Visual domain randomization.** Orthogonal; handled at the env level.

### What to skip

- **`robomimic`**: offline BC only, no DAgger, no asymmetric obs.
- **Learning by Cheating original repo**: unmaintained, CARLA-specific.
- **Building from scratch**: no reason to — the RSL-RL implementation
  is the same one Isaac Lab and ETH locomotion work use, and matches
  this codebase's algorithmic conventions exactly.

## The full design space (alternatives to DAgger)

DAgger is one point in a wider space.  All these methods solve the
same problem — distribution shift between teacher's training states
and the states the student actually visits — but make different
tradeoffs.

### 1. Behavior Cloning (BC, offline)
- **How:** roll teacher once, save dataset, train student offline
  with MSE.
- **Pros:** dead simple, GPU-bound, parallel, teacher only runs once.
- **Cons:** compounding error.  Student's small mistakes drift it
  into states the dataset doesn't cover.  Errors compound
  *quadratically* with horizon.
- **When:** short horizons, expert dataset covers state space, or
  reactive single-step tasks.  Not pick-cube (100 steps).

### 2. DAgger (this project's Phase A)
- **How:** student rolls out, teacher relabels each visited state,
  retrain on aggregated dataset.
- **Pros:** solves covariate shift directly; bounded error
  *linear* in horizon (vs quadratic for BC).  RSL-RL ships it.
- **Cons:** teacher must be queryable online; pure imitation, can't
  exceed teacher; loss curves noisier (state distribution shifts).
- **When:** continuous control, fast teacher query, sim available.

### 3. BC + DAgger fine-tune
- **How:** Stage 1 BC on frozen dataset; Stage 2 DAgger from BC init.
- **Pros:** stage 1 cheap, parallel; stage 2 only fixes residual
  drift.
- **Cons:** two-stage; marginal gain when teacher is fast.
- **When:** teacher is *expensive* (MPC, big VLM).  Not here.

### 4. Distillation + RL fine-tune ("kickstarting", this project's Phase C)
- **How:** DAgger to bootstrap, then PPO with the real reward
  initialized from the student.  Optional KL-to-teacher anchor.
- **Pros:** student can *surpass* teacher; robust to states teacher
  never visited; closes residual sim2real gap.
- **Cons:** needs critic + reward (we have both); RL is sample-hungry;
  early-iter risk of forgetting distilled behavior.
- **When:** want best possible policy, not teacher copy.  Used by
  OpenAI hand, ANYmal extreme parkour, etc.

### 5. Asymmetric Actor-Critic from scratch (no teacher)
- **How:** PPO with state-only critic + vision-only actor, trained
  from scratch.
- **Pros:** single stage, no teacher.
- **Cons:** *slow* — vision encoder learns from noisy reward
  signal; ~10× wall time of distillation; throws away the existing
  trained state policy.
- **When:** no good teacher exists.  Not our case.

### 6. RMA (Rapid Motor Adaptation)
- **How:** Stage 1 teacher conditions on env params (mass,
  friction); Stage 2 student is teacher's MLP + small encoder that
  infers params from proprio history.  No vision.
- **Pros:** handles dynamics randomization beautifully; tiny student.
- **Cons:** doesn't add vision — orthogonal sub-problem.
- **When:** sim2real where the gap is dynamics, not perception.

### 7. Representation distillation / state-prediction aux loss (Phase B)
- **How:** force the vision encoder to predict privileged state from
  images alongside action MSE.
- **Pros:** encoder learns *what the teacher used* rather than
  whatever incidental features minimize action MSE; decouples
  perception from control.
- **Cons:** needs to know which state dims to predict (we do); extra
  hparam (aux loss weight).
- **When:** vision tasks where privileged state is geometric and
  factored.  Yes here.

### 8. Diffusion / flow-matching policies
- **How:** student is a denoising or flow model trained on teacher
  rollouts.
- **Pros:** captures multi-modal teacher behavior; SOTA for
  human-demo BC.
- **Cons:** 50–100× more parameters; iterative inference (multiple
  denoising steps per action) — bad at 10 Hz; overkill for
  deterministic OSC teacher.
- **When:** large-scale human BC datasets with multi-modal behavior.

### 9. Inverse RL / GAIL / AIRL
- **How:** learn a reward from teacher rollouts, RL the student
  against it.
- **Pros:** doesn't need queryable teacher.
- **Cons:** adversarial, unstable; for our case the real reward is
  known — IRL solves a problem we don't have.
- **When:** demos only, no env reward.

### 10. World-model / Dreamer-style distillation
- **How:** learn a world model from teacher rollouts, student plans
  in the model.
- **Pros:** sample-efficient at test time; imagined futures.
- **Cons:** whole new system to build (model + planner).
- **When:** offline RL, hard exploration.  Not our loop.

### Mental model: three axes

Most named methods are recombinations of three independent choices.

1. **Online student rollouts vs offline dataset** → DAgger vs BC.
2. **Imitate teacher vs maximize reward** → distillation vs RL.
3. **What does the encoder learn?** → action MSE only, +state aux,
   +contrastive, or +world-model.

DAgger + state-aux + PPO-finetune is the boring-but-reliable corner
of this space.  Hence the four-phase plan.

### What we'd realistically swap in

- Replace Phase A with **BC** (option 1) → faster but compounding
  error breaks 100-step episodes.  Not recommended.
- Replace Phase A+C with **AAC from scratch** (option 5) → throws
  away the trained state policy.  Why?
- Replace whole pipeline with **diffusion BC** (option 8) → over
  engineering for a deterministic teacher.
- Add **RMA** (option 6) later → orthogonal, useful if dynamics
  sim2real gap turns out to dominate the perception gap.

## Adding noise to the teacher (the "BC with coverage" idea)

A common intuition: if BC fails because the dataset is too narrow,
add noise to teacher rollouts to cover a tube around the optimal
path.  Two flavors:

- **Action noise:** sample `a = π_T(s) + ε` (Gaussian) or
  `a ~ π_T(·|s)` (use the teacher's full distribution, not the mean).
- **State noise:** perturb the initial state / inject disturbances
  during rollout.

### What it fixes — and what it doesn't

| | covered | not covered |
|---|---|---|
| **Local** drift (small student errors) | ✅ O(σ) tube around teacher path | |
| **Global** drift (qualitatively wrong actions) | | ❌ student lands far outside any reasonable σ |

Concretely: pick-cube is ~100 steps.  If student per-step error is
even 5% of action range, after 20 steps drift is way outside any
realistic noise tube.  Bigger σ produces unrealistic teacher
behavior — the dataset becomes garbage at the limit.

**Fundamental tradeoff in one sentence:**
> BC + noise covers an O(σ) tube around the teacher's path; DAgger
> covers wherever the student actually goes.

### When noise on the teacher is genuinely useful

1. **Warm-start before DAgger.** Pretrain student on noisy teacher
   rollouts (cheap, fully parallel, no env loop), then DAgger from
   that init.  Roughly halves DAgger time because the student isn't
   random at iter 0.
2. **Robustness to teacher idiosyncrasies.** Averages out the
   teacher's own uncertainty, useful when the teacher's std is
   non-trivial.
3. **Data augmentation for representation learning.** Adding
   *observation-space* noise (pixel jitter, proprio perturbation) is
   a different thing — *that* helps and is orthogonal.  This is part
   of Phase D's visual DR.

### What's already happening in our DAgger setup

Looking at `rsl_rl/algorithms/distillation.py:93–94`:

```python
self.transition.actions = self.student(obs, stochastic_output=True).detach()
self.transition.privileged_actions = self.teacher(obs).detach()
```

- **Student rollout is stochastic** (`stochastic_output=True`) — so
  exploration noise from the student already gives DAgger most of
  what noise-BC would.
- **Teacher action is the mean** (no `stochastic_output=True`).
  Could be changed to sample from the teacher's Gaussian for a richer
  target distribution; empirically a few-percent gain.

### Recommendation

Don't build a separate noise-BC pretraining stage.  DAgger already
covers what noise-BC would, with less engineering.  *If* Phase A
plateaus higher than expected, two cheap experiments worth one run
each:

1. Switch teacher to stochastic sampling in DAgger (one-line patch
   to a subclass of `Distillation`).
2. Add observation-space noise (image augmentations) — this is part
   of Phase D anyway and helps even more in Phase A.

## Quick baseline before the full recipe

The asymmetric AC plumbing is already in place. As a cheap first
checkpoint:

- Critic: privileged state (current `critic` group).
- Actor: camera + `goal_pos` only.
- Warm-start critic from the trained state-policy critic.
- PPO from scratch on the actor.

This isolates how much of the gap is "vision encoder learns slowly" vs
"policy needs to change." If it gets close to teacher performance, the
distillation stages are mostly about sample efficiency.
