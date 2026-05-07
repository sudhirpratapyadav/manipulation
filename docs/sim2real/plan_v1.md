# Sim2Real Vision — plan_v1 (concrete next steps)

This document is the **implementation-ready** continuation of the work tracked in `plan.md`, `state_to_vision_transfer.md`, `exp_tracker.md`, and `new_plan.md`. It is grounded in the actual code (`pick_cube_distill_osc.py`, `pick_cube_vision_osc.py`, `train_distill.py`, `distill_runner.py`) and the latest experiment (`A_03_cams64_resume_20k`, ckpts up to `model_10800.pt`).

It supersedes `plan.md`'s phase-A-onwards roadmap with what we've actually learned.

---

## 0. Where we are (snapshot, 2026-05-06)

- **Teacher** (state PPO, 33D obs): wandb `jn3l22j9 / model_4999.pt`. Loaded into `Distillation` via `actor_state_dict` auto-detect. Architecture (512,256,128) elu, scalar-std Gaussian, with `obs_normalizer`.
- **Student** (vision DAgger): `pick_cube_distill_osc.py`. Spatial-softmax CNN [16,32] + MLP (256,256,128) elu. Reads `actor` (24D proprio incl. `goal_pos`) + `camera` (wrist 64×64 + d455 64×64). Critic group carries 33D priv state (teacher input).
- **Latest run:** `A_03_cams64_resume_20k`, ran iter 2000 → ~10800 (~20 h wall-clock, 1024 envs, ~3000 fps, RTX A6000). Resumed from `A_03_cams64/model_1999.pt`.
- **Training-env metrics at end:**
  - `Loss/behavior` 0.54 → 0.17 (monotone)
  - `Episode_Reward/goal_precise` 0.49 → **0.71** (teacher 0.66 — comparable)
  - `Episode_Reward/reach_object` 0.76 → **0.85**
  - `Episode_Metrics/object_to_goal_error` → **0.156 m**
  - `Episode_Metrics/ee_to_object_error` → **0.055 m**
  - `Episode_Termination/object_out_of_bounds` ~26% (dominant failure mode)
- **What we have not yet measured:** held-out success rate at a fixed `cube_to_goal_error < 5 cm` threshold for student vs teacher in the same play harness. This is open TODO from `exp_tracker.md`.
- **DR currently active in vision env:** fingertip slide/spin/roll friction; object friction; object mass; object color (uniform RGB). No camera DR. No lighting DR. No image augs.

**Reading.** Phase A (DAgger only) plateaued reasonably close to teacher on the *training* env. The remaining ~25–30% failure rate is dominated by `object_out_of_bounds` — the cube gets knocked or pushed off the workspace, not "the policy never reaches it." Behavior loss is still slowly decreasing, but gains beyond iter ~9000 are small (Loss 0.19 → 0.17 over 1800 iters). **More pure DAgger iterations are not the next move.**

---

## 1. The actual question now

Phase A told us: a from-scratch CNN can imitate the teacher reasonably well *in the training env*. It did **not** tell us:

- **Q1.** How big is the sim2real visual gap when the only DR is object-color + dynamics? (No lighting, no camera pose, no background, no image augs.)
- **Q2.** Are the encoder features actually encoding cube geometry, or are they latching onto color-blob statistics that the DR happens to vary?
- **Q3.** Would a pretrained robotics-CNN backbone (MCR / R3M) close most of the residual gap for free?
- **Q4.** Does the dominant failure mode (`object_out_of_bounds`) come from the encoder (mis-localizing the cube) or from the policy head (overshooting on grasp)?

`new_plan.md §5.10` gives the literature-backed answer template; this plan picks the cheapest moves from there and orders them against our specific bottleneck (Q4).

---

## 2. Phase A1 — diagnose before you optimize (≤1 day)

Goal: turn Phase A from "training-env loss curves" into "do we have a deployable student or not?" Without this, every later move is guessing.

### A1.1 — Build a real eval harness (`eval_distill.py`)

Drop a thin script that:

1. Loads a checkpoint into the existing vision env at `play=True`, with `num_envs=256`.
2. Rolls out for one fixed-length episode each (env's `episode_length_s` or 200 steps).
3. Reports a flat dict:
   - `success_rate_5cm` = fraction of episodes ending with `cube_to_goal_error < 0.05`.
   - `success_rate_2cm` = same at 2 cm (hard threshold).
   - `mean_final_error`, `median_final_error`.
   - `oob_rate`, `timeout_rate`, `nan_rate`.
   - mean episode reward.
4. Writes the dict + per-episode trajectory CSV to `logs/eval/<run_name>/<ckpt>.json`.

Reuse: the existing `play` entrypoint already loads a checkpoint into the env via `--checkpoint-file`. Likely the simplest path is a `--no-viewer --eval-mode` flag on `play`, not a new entrypoint, but write what is shorter.

### A1.2 — Run the harness on three checkpoints

| Run | Checkpoint | Purpose |
|---|---|---|
| Teacher | `wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt` | Ceiling on the **state** env (`Mjlab-Pick-Cube-Osc-Kinova`). |
| Teacher in vision env | same ckpt, played in `Mjlab-Pick-Cube-Distill-Osc-Kinova` with the teacher acting through the **critic** obs group (33D priv state still available) | Tells us whether the env-config delta itself dropped success. |
| Student A_03 last | `logs/.../A_03_cams64_resume_20k/model_10800.pt` | Real student number. |
| Student A_03 mid | `model_5000.pt` | Detect over-/under-fit — pick the actual best ckpt to carry forward. |

Numbers we expect from `exp_tracker.md` plus the windowed averages:
- Teacher state env: success_rate_5cm probably ≥ 0.80 (the run summary's `goal_precise=0.66` is shaping reward, not success rate; needs measurement).
- Student A_03: unknown — best guess from `goal_precise=0.71` is something in the 0.50–0.75 band, dragged down by the 26% OOB.

### A1.3 — Decompose the failure mode

In the eval script, when an episode ends, log:
- final `ee_to_object_error` (did the student reach the cube?)
- final `object_z` and trajectory of `object_z` (did the cube get lifted at all?)
- whether the gripper closed at the moment of "best contact" (action[6] > 0.5 within 5 cm of cube)

Then bucket failures into:
- **R-fail**: `ee_to_object_error > 5 cm` always — never reached.
- **G-fail**: reached but gripper never closed in contact.
- **L-fail**: reached and grasped but cube never left the table (`max(object_z) < 5 cm`).
- **D-fail**: lifted but dropped before goal.
- **OOB**: cube ended out of bounds (and one of the above is the upstream cause).

This tells us whether the next investment is encoder (R-fail dominates → can't see the cube), gripper-policy (G/L-fail → action head issue), or environment (OOB before grasp — env is too unforgiving).

**Output of A1:** `logs/eval/A_03_cams64_resume_20k/diagnose.md` with the bucket histogram and the four success-rate numbers.

---

## 3. Phase A2 — eliminate the cheap fixes before adding aux losses

These are the "if this is the bottleneck, no aux loss is going to fix it" sanity moves. Order is by cost, not by expected impact.

### A2.1 — Stochastic teacher relabel (1-line patch)

`rsl_rl/algorithms/distillation.py:93–94` currently does:
```python
self.transition.actions = self.student(obs, stochastic_output=True).detach()
self.transition.privileged_actions = self.teacher(obs).detach()
```
Switch teacher to `stochastic_output=True` for a richer target distribution. `state_to_vision_transfer.md §"Adding noise to the teacher"` explicitly recommends this as a "few-percent gain" cheap experiment. Subclass `Distillation` in `distill_runner.py`, override the rollout step.

### A2.2 — Pick the actual best checkpoint, not the latest

A1.2 already gave us mid vs latest. If `model_5000.pt` beats `model_10800.pt` on success_rate_5cm, the run is saturating and we should *stop training pure DAgger here*. Otherwise pick `model_10800.pt` as the Phase A handoff.

### A2.3 — DAgger continuation only if Phase A1 says "very close to teacher" (<5 pp gap)

If A1.2 shows student is e.g. 70% success vs teacher 78%, **do not** push DAgger further — a 1-pp/1000-iter improvement rate isn't worth 13 h of compute. Move on to A3.

If the student is far from teacher (e.g. 50% vs 80%), continue DAgger to iter 20000 with a learning-rate decay (current `lr=1e-3` is constant per `pick_cube_distill_osc.py:48`). Cheap fix in `RslRlDistillationAlgorithmCfg`.

---

## 4. Phase B — auxiliary geometric loss (the highest-leverage move)

`new_plan.md §5.1` and `§5.10 step 1` agree: a small auxiliary head that L2-regresses **object 6D pose + ee position** from the encoder features is the single most consistent winner in the literature (DextrAH-G/RGB, Hora, HRP). Sim has the ground truth for free.

This is what we build *after* A1.

### B.1 — What the head predicts

The teacher reads 33D privileged obs:
```
joint_vel (7) + ee_pose (6) + gripper_state (1) + ee_to_object (3)
+ object_pos (3) + object_to_goal (3) + goal_pos (3) + last_action (7)
```
Of these, the **geometric subset** the encoder cannot directly know from proprio is:
- `object_pos` (3D): cube position in world.
- `ee_to_object` (3D): redundant with `object_pos` once `ee_pose` is known, but cheap to predict and acts as a separate signal.
- `object_to_goal` (3D): again redundant with `object_pos` + `goal_pos`, but separate signal.

Pick **`object_pos` (3D)** as the only target. It's the minimal sufficient statistic, and predicting redundant copies just diffuses the gradient.

### B.2 — Where the head attaches

The student's CNN currently emits a spatial-softmax feature vector (size depends on `output_channels[-1] * 2`; with `[16, 32]` and `spatial_softmax=True`, that's `32 * 2 = 64`-dim per camera, so 128-dim concat). The MLP then takes `(actor_24D ⊕ cnn_128D)` → (256, 256, 128) → 7D action.

Aux head:
```
cnn_features (128D)  ──►  MLP(64, 32)  ──►  3D object_pos prediction
```
Trained with L2 loss, weighted at `λ_aux = 0.1` (DextrAH default).

### B.3 — Implementation

- New file `src/kinova_tasks/distill_aux.py` subclassing `rsl_rl.algorithms.Distillation`. Override `update()` to add the aux loss term.
- New runner cfg `RslRlAuxDistillationRunnerCfg` (mirrors `RslRlDistillationRunnerCfg`) with `algorithm.class_name = "kinova_tasks.distill_aux:AuxDistillation"` and `lambda_aux: 0.1`.
- The aux target needs to flow through `obs_groups`. Easiest path: add a separate **"priv_geom"** observation group that exposes only `object_pos` (3D), readable by the algorithm but not by the student's policy MLP. Modify `obs_groups` to:
  ```
  "student": ("actor", "camera"),
  "teacher": ("critic",),
  "aux":     ("priv_geom",),
  ```
  and pull `obs["aux"]` inside `update()` as the regression target.
- New task id `Mjlab-Pick-Cube-Distill-Aux-Osc-Kinova` so we can A/B against vanilla A_03 cleanly.

### B.4 — Phase B run

Warm-start from the best Phase A checkpoint (whatever A1.2 picked). Train 5000 iters with aux on. Compare against pure-DAgger iter-15800 (= 10800+5000) baseline if we keep pure DAgger running in parallel as control.

**Success criterion:** ≥ +5pp success_rate_5cm vs the matched-iter pure-DAgger control on the same eval harness from A1.

### B.5 — Probe the encoder (free diagnostic)

`new_plan.md §5.10 step 8` is right that nobody has done this for a sim-trained manipulation CNN. We can:
- Snapshot 10k (encoder_features, object_pos) pairs from a Phase A and a Phase B rollout.
- Linear-probe `object_pos` prediction error.
- Linear-probe **`object_color`** prediction error.

If aux loss worked, the geometry probe error drops and the color probe error rises. This is direct evidence that decoupling happened. One-day diagnostic, written up in `exp_tracker.md`.

---

## 5. Phase B-alt — pretrained encoder swap (parallel branch)

This is the other "single change with the most consistent positive signal" from `new_plan.md §5.10 step 2`. Run **in parallel with Phase B** if compute allows (we have 2× A6000), not sequential.

### B-alt.1 — Pick MCR

`new_plan.md §5.3`: MCR (Jiang et al. 2024, [arXiv:2410.22325](https://arxiv.org/abs/2410.22325), [project page](https://robots-pretrain-robots.github.io/)) is **CNN backbone (ResNet-50)**, pretrained on the DROID dataset with action prediction + time-contrastive + dynamics-alignment losses. Vakil et al. ([arXiv:2501.16389](https://arxiv.org/abs/2501.16389)) ranks it #1 of 23 encoders on both Action Score and Domain Invariance Score for sim2real BC; **DINOv2-B is bottom**, so MCR is the right pick, not DINOv2.

### B-alt.2 — Implementation

- Replace `_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"` with a thin wrapper that:
  1. Loads frozen MCR ResNet-50 (download from the project page; pin the hash).
  2. Forwards each camera through MCR → 2048-dim feature per camera.
  3. Concatenates wrist+d455 → 4096-dim → small MLP head (e.g. 4096 → 256 → 128) before joining the proprio MLP.
- Frozen first; LoRA / full fine-tune is `B-alt.3` if frozen is close-but-not-good-enough.
- Camera resolution probably needs to go from 64×64 → 224×224 (MCR's native input). This will roughly **4× the per-step cost**; drop num_envs to 256 to keep wall-clock matched.

### B-alt.3 — Phase B-alt run

Warm-start the proprio MLP from Phase A (the CNN side gets thrown away). Distill for 5000 iters. Compare against Phase A and Phase B on the same eval harness.

**Decision rule.** After A1.2 + Phase B + Phase B-alt eval:
- If MCR-frozen ≥ aux-CNN by ≥ 5pp → carry MCR forward; Phase C uses MCR backbone.
- If aux-CNN ≥ MCR-frozen → keep our CNN, layer aux loss on, drop MCR.
- If they're within 5pp → keep CNN (cheaper at deploy, 64×64 vs 224×224).

---

## 6. Phase C — closing the residual gap (only if A1/B numbers warrant it)

Two options. Run only the one that the diagnose-bucket from A1.3 points to.

### C.1 (if R-fail dominates) — visual DR

The vision env currently has only `geom_rgba` color randomization. Add:
- **Lighting DR.** Random light direction + intensity per episode. Cheap.
- **Camera pose jitter.** ±2 cm position, ±5° rotation on the wrist cam parent transform; same on d455. Already supported by mjlab DR helpers — re-check `src/kinova_tasks/tasks/pick_cube_osc.py` events for the pattern.
- **Background texture.** Swap `_BROWN_TEXTURE` for a randomized texture pool at reset. Several Procedural / texture libraries are easy to wire into `MaterialCfg`.
- **Image augs at obs time.** Random shift (DrQ-v2 style — `new_plan.md §5.4`), gaussian noise, color jitter. Apply *frame-wise*, not episode-wise (per `new_plan.md §5.10 step 10`).

### C.2 (if G/D-fail dominates) — PPO finetune

Initialize the vision PPO from the best distilled student. Run `kinova_pick_cube_vision_osc_ppo_cfg` warm-started. The critic was trained on 33D priv obs already; reuse it. `plan.md §"Phase C"` has the recipe.

### C.3 (if OOB dominates and the cube *was* grasped) — env tightening

The 26% OOB rate suggests the workspace bounds are tight. Two cheap moves:
- Loosen `cube_out_of_bounds` by a few cm (eval will then re-bucket).
- Add a small in-bounds shaping reward.

Both are env changes, not policy changes; cheaper than another training run.

---

## 7. Phase D — deferred until C is decided

`plan.md §"Phase D"` already has the sim2real recipe. Don't touch until Phase B/C delivers a numbered student that's clearly worth deploying. Things to keep in mind from `new_plan.md §6`:

- Wrist cam is illumination-robust for free; don't drop it for d455-only.
- Mid-episode cube re-drop / camera occlusion in sim is the cheapest analog of "person grabs object." Add only if Phase C policy is actually being tested for disturbance recovery.
- Sim-and-Real co-training (+38% real per arXiv 2503.24361) is the smallest dose of real data with measurable bump; reserve for *after* the first real-robot eval shows where the gap is.

Photorealism (SplatSim) is parked in `sim2real_vision/PIPELINE.md`. The Kinova-port is non-trivial (Phase 1 of that doc); don't start that work until we know whether basic DR + frozen MCR closes the gap on its own.

---

## 8. Concrete first-day plan

In order; tick as we go.

1. **A1.1** — write `eval_distill.py` (or `--eval` mode on `play`). ~2 h.
2. **A1.2** — run on (a) teacher in state env, (b) teacher in vision env (priv obs), (c) student A_03 ckpts 5000, 8000, 10800. Same 256-env harness, fixed seed. ~30 min compute.
3. **A1.3** — fail-bucket histogram. ~1 h to write the analyzer; rolls together with A1.2.
4. Update `exp_tracker.md` with the four success-rate numbers per ckpt + bucket histogram + the *actual* teacher baseline number (closes the open TODO from `exp_tracker.md`).
5. **Decision point.** Read the bucket distribution; pick the Phase B and (independently) Phase B-alt branches per §4 / §5 above.

Only after step 5 do we touch the training code. The single biggest risk in this project is spending another 20 h of GPU time on a config we can't yet evaluate.

---

## 9. Open questions (to resolve as we go)

These were either inherited from `plan.md` or surfaced by reading the current code; they should be answered as part of A1 or B.

- **Camera obs scale.** `manipulation_mdp.camera_rgb` returns what — uint8 in [0,255], or float32 in [0,1]? CNN init assumes one of these; MCR backbone expects `(x − mean) / std` ImageNet normalization. Check before B-alt.
- **`enabled_geom_groups=(0, 3)`** — group 3 is what? The wrist + d455 cams both render groups 0 and 3 only; if the cube is in a different group, the cameras *can't see it*, which would explain a lot. **Verify by saving a single rendered frame from each cam and inspecting it.** Cheap, do this on day 1.
- **Goal pos in obs.** Currently the actor receives `goal_pos` as a 3D proprio. After the env refactor (cube → object), is this still wired in? Quick `print(obs.keys())` at env reset.
- **Privileged-state target for aux head.** Is `object_pos` the world-frame cube pose, or robot-base frame? The teacher's `obs_normalizer` expects the same frame the trained policy used; cross-check before training Phase B.
- **Per-camera vs concatenated CNN.** Currently both cameras concatenate into a single "camera" obs and share one CNN. A two-stream CNN (one per camera) is sometimes better — defer until A1 says cameras matter.

---

## 10. What this plan deliberately does NOT include

- **DINOv2.** Vakil et al. 2025 ranks it last for sim2real BC despite its strong geometry probes. Skip.
- **Diffusion policy head.** `new_plan.md §6` flags this as a real lever for *disturbance robustness*, but it's overkill for our deterministic OSC teacher. Re-evaluate only if Phase C plateaus.
- **Object-centric (POCR/SAM).** Adds a deploy-time segmentation step that could fail on real cameras; not worth it unless Phase B/B-alt + DR all underperform.
- **3D-aware encoders (F3RM, ManiGaussian).** Research-grade. The `sim2real_vision/PIPELINE.md` SplatSim port is the only 3D path we're tracking, and only as a Phase D appendix.
- **Stage B ("state-prediction aux loss" as a separate phase).** Folded into Phase B above. The original `plan.md` had this as Phase B → Phase C ordering; we're collapsing it because aux loss is the *first* move per `new_plan.md §5.10`, not a fallback after Phase A underperforms.
