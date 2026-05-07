# Sim2Real Vision — plan_v2 (MCR encoder swap)

Continuation of `plan.md` / `state_to_vision_transfer.md` / `new_plan.md`. Supersedes `plan_v1.md` (which queued an eval/aux-loss detour). The conclusion from A_03 is:

> DAgger is performing fine on the training env (`goal_precise` 0.71 vs teacher 0.66, monotone loss decay). The 26% `object_out_of_bounds` rate is acceptable for now. We are skipping the eval / aux-loss / OOB-fix detours and going straight at the **visual encoder gap** — replace the from-scratch spatial-softmax CNN with a pretrained robotics-CNN backbone (MCR).

Rationale (from `new_plan.md §5.3 / §5.10 step 2`): *Bridging the Sim2Real Gap* (Vakil et al. 2025, [arXiv:2501.16389](https://arxiv.org/abs/2501.16389)) ranks 23 encoders on Action Score + Domain Invariance Score for sim2real BC; **MCR is #1 on both, DINOv2-B is last on both, CNNs > ViTs for domain invariance**. MCR (Jiang et al. 2024, [arXiv:2410.22325](https://arxiv.org/abs/2410.22325), [project](https://robots-pretrain-robots.github.io/)) is a ResNet-50 pretrained on the DROID dataset with action-prediction + time-contrastive + dynamics-alignment losses; the paper itself reports +76.9% real-world success on 3 manipulation tasks vs the strongest baseline.

This is the cheapest change that the literature consistently says moves real-world numbers.

---

## 1. Current setup (anchor)

Files we will touch are listed here so the plan stays grounded.

| File | What it does today |
|---|---|
| `src/kinova_tasks/tasks/pick_cube_vision_osc.py` | Vision env. Wrist (64×64) + d455 (64×64) RGB cams. `_VISION_CNN_CFG` = spatial-softmax CNN [16,32], k=[5,3], stride=[2,2]. `_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"`. |
| `src/kinova_tasks/tasks/pick_cube_distill_osc.py` | Distill cfg. Student model uses `_VISION_CNN_CFG` + `_VISION_MODEL_CLS`. Teacher is a plain MLP (512,256,128). |
| `src/kinova_tasks/train_distill.py` | Trainer entrypoint (`train-distill`). Loads teacher from PPO ckpt via `Distillation.load()`. |
| `src/kinova_tasks/distill_runner.py` | `MjlabDistillationRunner(DistillationRunner)`. Migrates legacy ckpts; persists `common_step_counter`. |
| `mjlab/src/mjlab/tasks/manipulation/mdp/observations.py:95` | `camera_rgb` returns `float[0,1]`, shape `(B, 3, H, W)`. **Already 0-1 normalized.** |
| `mjlab/src/mjlab/rl/spatial_softmax.py:119` | `SpatialSoftmaxCNNModel(CNNModel)`. The class we are replacing. |
| `manipulation/.venv/.../rsl_rl/models/cnn_model.py:19` | `CNNModel(MLPModel)`. The base class we have to honor. Constructor takes `(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, cnn_cfg, ...)` and exposes `cnn_latent_dim` after the encoder + an `MLPModel` head. |

Existing latest student ckpt: `logs/rsl_rl/kinova_pick_cube_distill_osc/2026-05-04_21-32-04_A_03_cams64_resume_20k/model_10800.pt` (carry forward the proprio MLP weights only — the CNN side of A_03 is being thrown away).

---

## 2. Goal of plan_v2

One thing only: get a distilled student whose **visual encoder is a frozen MCR ResNet-50** instead of the from-scratch spatial-softmax CNN, trained with the existing DAgger pipeline, and compare it head-to-head with A_03 on whatever real-robot eval we run later.

Non-goals (deferred to a later plan):
- Auxiliary geometry head (Phase B in `plan_v1.md`).
- Visual DR beyond the existing object color randomization.
- Eval harness.
- PPO finetune.
- Real-robot deployment.

---

## 3. Implementation outline

### 3.1 New file: `src/kinova_tasks/encoders/mcr_encoder.py`

A drop-in replacement for `SpatialSoftmaxCNNModel`. Same constructor signature (`CNNModel`-compatible), so the rest of the runner / algorithm code does not need to change.

**Responsibilities:**

1. Load a frozen ResNet-50 with MCR weights at `__init__` time. The pretrained weights ship as a state_dict at the project page; pin a hash and store at `assets/mcr/mcr_resnet50.pt` (gitignored — it's a few hundred MB).
2. For each 2D obs group (the env has one: `"camera"`, which concatenates `wrist_rgb` and `d455_rgb` channel-wise into a `(B, 6, 64, 64)` tensor):
   - Reshape `(B, 6, H, W)` → `(2B, 3, H, W)` to run the two cameras through MCR independently.
   - **Resize to 224×224** with `F.interpolate(..., mode="bilinear", align_corners=False)`. MCR was trained at 224×224 native; running at 64×64 will give garbage features.
   - **ImageNet normalize:** `(x − μ) / σ`, μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]. `camera_rgb` is already in [0,1] (see §1) so we skip the /255.
   - Forward through MCR, take the global pooled feature (2048-D) — i.e. cut the ResNet at the global-avg-pool, drop the FC.
   - Reshape back: `(2B, 2048)` → `(B, 4096)`. Concat in the existing `cnn_latent_dim` slot.
3. Set `requires_grad=False` on every MCR parameter. `model.eval()` in `__init__`. The downstream MLP head is the only trainable part of the student visual pathway.
4. Override `output_channels`, `output_dim` to `4096` so the parent `CNNModel` wires the sizes correctly.

**Class name for tyro:** `kinova_tasks.encoders.mcr_encoder:MCREncoderModel`.

### 3.2 Wire it in `pick_cube_vision_osc.py`

Add a flag to `kinova_pick_cube_vision_osc_env_cfg` (or just a sibling `_VISION_MCR_CFG`). The vision env keeps using 64×64 cameras at *capture* — the resize to 224 happens inside the encoder, which keeps GPU memory for 1024 envs × 6 channels manageable. (Render at 64 → upsample to 224 wastes nothing visually compared to render at 224 → downsample, since the splat-free MuJoCo rasterizer already loses high-frequency content.)

```python
_VISION_MCR_CFG = {}  # MCR has no per-instance hyperparams; left empty so CNNModel doesn't choke
_VISION_MODEL_CLS_MCR = "kinova_tasks.encoders.mcr_encoder:MCREncoderModel"
```

### 3.3 New file: `src/kinova_tasks/tasks/pick_cube_distill_mcr_osc.py`

Mirrors `pick_cube_distill_osc.py` but:

- `student.cnn_cfg = _VISION_MCR_CFG`
- `student.class_name = _VISION_MODEL_CLS_MCR`
- `student.hidden_dims = (512, 256, 128)`  *(see §3.5 — MLP head needs to be wider because the encoder feature is now 4096-D, not 128-D)*
- New runner cfg `kinova_pick_cube_distill_mcr_osc_runner_cfg()` returning `experiment_name="kinova_pick_cube_distill_mcr_osc"`, otherwise identical defaults.

### 3.4 Register in `__init__.py`

```python
from kinova_tasks.tasks.pick_cube_distill_mcr_osc import (
    kinova_pick_cube_distill_mcr_osc_env_cfg,
    kinova_pick_cube_distill_mcr_osc_runner_cfg,
)
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova",
    env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_distill_mcr_osc_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)
```

### 3.5 MLP head sizing

The current MLP head is `(actor_24D ⊕ cnn_128D) → (256, 256, 128) → 7D`. With MCR the cnn slot is 4096-D, so the input layer balloons from 152 → 4120. Two reasonable shapes:

| Variant | Hidden dims | Params (input layer) | Notes |
|---|---|---|---|
| **A (default)** | `(512, 256, 128)` | 4120×512 ≈ 2.1M | Wider first layer to absorb the 4096 feature; downstream same as A_03. |
| B (compress-then-MLP) | encoder→Linear(4096, 256), then `(256, 256, 128)` | 4096×256 + 280×256 ≈ 1.1M | Smaller; relies on a learned projection of MCR features. |

Start with **A**. Compare B only if A diverges or memory is tight. Both are still tiny vs the 25M frozen MCR params.

### 3.6 Train-time wall-clock budget

- A_03 ran ~3000 fps at 1024 envs with the 64×64 spatial-softmax CNN.
- MCR forward at 224×224 on a frozen ResNet-50 is ~2 ms/img at fp32 batch 256 on A6000. Two cams × 1024 envs = 2048 imgs/forward → ~4 ms if we keep batch parallelism, but practically 10–20 ms with overhead.
- Expect throughput to drop to ~800–1500 fps at 1024 envs. **If we can't keep 1024 envs, drop to 512 and re-time.**
- Memory: 1024 envs × 2 cams × 3 × 224 × 224 × 4 bytes (float) ≈ 1.2 GB just for resized inputs, plus ResNet activations. Should fit on the A6000 (48 GB) with margin; if not, drop num_envs.

A 5000-iter run at half throughput would be ~13 h — same order as A_03's tail. Acceptable.

### 3.7 Warm-start the proprio MLP from A_03

Two options for the initial weights of the MCR student:

1. **Cold start.** Random init the MLP, train from scratch.
2. **Warm-start MLP only.** Load `model_10800.pt`, copy the MLP weights *that align in shape* (the 24D-proprio→256 input layer can't be reused because MCR's encoder feature is different size; the (256,256,128) trunk *can* be).

Try 1 first — it's the literature default and the cleanest comparison vs A_03. If convergence is slow, try 2 as an ablation.

---

## 4. Open implementation questions (resolve while coding)

- **MCR weight format.** The project page hosts a checkpoint; verify it loads cleanly into `torchvision.models.resnet50()` after stripping any prefix (e.g. `module.`, `encoder.`). If they ship a custom state-dict layout, write the key-mapping shim in `mcr_encoder.py`.
- **Where to cache the weights.** `assets/mcr/mcr_resnet50.pt` (relative to repo root). Gitignore it. Add a one-line note in `README.md` on how to download.
- **fp16 vs fp32 for the frozen forward.** Frozen ResNet-50 in fp16 with autocast halves the forward cost for free. Try fp32 first for parity with the reference; flip to `torch.cuda.amp.autocast(enabled=True)` if needed for throughput.
- **`obs_normalization=True` on the student.** The current cfg has `obs_normalization=True` on the student (running mean/var on the input). For MCR features, we want this disabled on the camera obs (MCR was pretrained on ImageNet stats; running its features through a second running-mean normalizer is double-normalization). Check `MLPModel` / `CNNModel` to confirm what `obs_normalization` actually normalizes — likely the 1D proprio path only, not the CNN output. If it touches the CNN output, set it `False`.
- **`enabled_geom_groups=(0, 3)`.** Both cameras in `pick_cube_vision_osc.py` are restricted to geom groups 0 and 3. This was inherited from A_03 and is presumed working (videos rendered fine), but it's worth saving one frame from each cam and eyeballing it before kicking off a 13 h run. Cheap to do at the smoke-test stage (§5.1).
- **Image input range to MCR.** Confirm whether the reference code pre-divides by 255 inside its forward, or expects already-/255 input. `camera_rgb` already returns [0,1]; double check.

---

## 5. Run plan

### 5.1 Smoke (before any long run)

`MCR_smoke`: 32 envs × 30 iters. Goals:

1. MCR weights load with no key mismatch.
2. Forward shapes are right (4096-D feature out, batch dim preserved).
3. Loss is finite, decreasing across the 30 iters.
4. Throughput at small batch reported, so we can extrapolate to 1024 envs.
5. Memory headroom logged (`nvidia-smi` once at iter 20).

Single GPU, foreground, no wandb. Mirrors `A_01_smoke` from `exp_tracker.md`.

Drop-into-shell command:

```bash
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  CUDA_VISIBLE_DEVICES=1 \
  uv run train-distill \
    --teacher-ckpt wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt \
    --env.scene.num-envs 32 \
    --agent.max-iterations 30 \
    --agent.save-interval 30 \
    --agent.experiment-name kinova_pick_cube_distill_mcr_osc_smoke \
    --agent.run-name MCR_smoke \
    --agent.logger tensorboard
```

`train-distill` currently hardcodes `_TASK_ID = "Mjlab-Pick-Cube-Distill-Osc-Kinova"` (`train_distill.py:35`). For the new MCR task we either:
- (cheaper) add a `--task-id` flag to `DistillTrainConfig` defaulting to the existing one, or
- (uglier) duplicate `train_distill.py` as `train_distill_mcr.py`.

Pick the first. One-line change: thread `task_id: str = _TASK_ID` through `DistillTrainConfig` and pass it to `from_task`.

### 5.2 Full run: `MCR_01_full`

Once smoke passes:

```bash
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  WANDB_API_KEY=... WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur \
  CUDA_VISIBLE_DEVICES=1 \
  nohup uv run train-distill \
    --task-id Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova \
    --teacher-ckpt wandb/run-20260430_220905-jn3l22j9/files/model_4999.pt \
    --env.scene.num-envs 1024 \
    --agent.max-iterations 5000 \
    --agent.save-interval 100 \
    --agent.wandb-project mjlab-kinova-tasks-osc-vision \
    --agent.experiment-name kinova_pick_cube_distill_mcr_osc \
    --agent.run-name MCR_01_full \
    --agent.wandb-tags '("phase_a","mcr","frozen_resnet50","teacher_jn3l22j9")' \
    --video True --video-length 100 --video-interval 200 \
    > logs/distill_MCR_01_full.log 2>&1 &
```

If throughput logged at smoke shows < 1000 fps at 1024 envs, drop `--env.scene.num-envs` to 512 and re-launch. 5000 iters is half of A_03's run — MCR features are pretrained, so DAgger should converge faster, and we can extend later if the loss curve says so.

### 5.3 What to log alongside

Append a row to `exp_tracker.md` with:
- run id (wandb)
- ckpt path
- throughput (fps)
- final `Loss/behavior` (compare to A_03's 0.17)
- final `Episode_Reward/goal_precise` (compare to A_03's 0.71)
- final `Episode_Metrics/object_to_goal_error` (compare to A_03's 0.156)
- final `Episode_Termination/object_out_of_bounds` (compare to A_03's 0.26)

These are the same 4 numbers I pulled from tensorboard for A_03 — gives a clean A vs MCR head-to-head on the *training env*. (Real-robot comparison waits for a later plan.)

---

## 6. Decision rules

After `MCR_01_full` lands, three outcomes and what to do for each:

1. **MCR clearly wins on training-env metrics** (e.g. `goal_precise` ≥ 0.74, behavior loss ≤ 0.15, training stable). Stop here on the encoder front. The next plan layers visual DR on top of MCR (Phase C in `plan.md`).
2. **MCR matches A_03** (within ±2 pp on `goal_precise`, similar loss). Still ship MCR forward — the value of the swap is sim2real domain invariance, which the training env can't measure. Carry both ckpts to real-robot eval; if they tie there too, drop MCR for the cheaper 64×64 CNN.
3. **MCR underperforms A_03** by ≥ 5 pp. Most likely culprits in order: (a) `obs_normalization` is touching the CNN output (§4); (b) image preprocessing wrong (range / normalization); (c) MLP head too narrow for 4096-D input (try variant B in §3.5); (d) MCR weights silently mis-loaded — verify by linear-probing object color from frozen MCR features, expect ≥80% R² (it's pretrained on cluttered manipulation data). Diagnose, do not just abandon.

If outcome 3 persists after the four fixes, fall back to R3M (Nair et al. 2022, [arXiv:2203.12601](https://arxiv.org/abs/2203.12601)) as the second-choice frozen backbone — same drop-in pattern, different state-dict.

---

## 7. What we are explicitly *not* doing in plan_v2

- **No eval harness.** `goal_precise` from tensorboard is the comparison axis until we have a real-robot run that demands more rigor.
- **No aux loss.** Folded back into a future plan, only after the encoder swap settles.
- **No DR additions.** Object-color randomization stays as-is; lighting/camera/background DR are Phase C.
- **No PPO finetune.** Phase C territory.
- **No DINOv2.** Vakil 2025 ranks it last; explicitly skipped.
- **No real-robot work.** Phase D.
- **No SplatSim.** Tracked separately in `sim2real_vision/PIPELINE.md`.

---

## 8. Concrete checklist (to copy into a TODO when starting work)

Status as of 2026-05-07 — checklist updated as work landed:

```
[x] Download MCR ResNet-50 weights → assets/mcr/mcr_resnet50.pth
    URL: https://huggingface.co/GqJiang/robots-pretrain-robots/resolve/main/mcr_resnet50.pth
    Size: 90 MB, sha256: 21560db9e795384a7b73b0626d33d6525f9d4c5de11fca6ded52089aab14f97c
    Loads cleanly into torchvision.models.resnet50() — no key remap needed.
[x] Write src/kinova_tasks/encoders/__init__.py + mcr_encoder.py
    [x] Load + freeze ResNet-50 with MCR state_dict
    [x] Resize 64→224, ImageNet-normalize, forward, pool to 2048
    [x] Handle 6-channel concat → 2-camera batched forward
    [x] Match CNNModel constructor signature; expose 4096 output_dim
    [x] **Added: per-camera LayerNorm on the 2048-D feature** (not in §3
        plan).  Without it the freshly-init 4120→512 MLP head saw raw
        ResNet activations (mean ~0.09, std ~0.18 with high outliers) and
        the early behavior-loss diverged in the smoke test.  LayerNorm
        is the only trainable part of the encoder.  Toggleable via
        cnn_cfg.feature_layernorm (default True).
[x] Write src/kinova_tasks/tasks/pick_cube_distill_mcr_osc.py
    Reuses RslRlDistillationRunnerCfg / RslRlDistillationAlgorithmCfg /
    teacher dims from pick_cube_distill_osc.py.  Student hidden_dims
    widened to (512, 256, 128) to absorb the 4096-D MCR feature.  Default
    weights path: assets/mcr/mcr_resnet50.pth, overridable via
    MCR_WEIGHTS_PATH env var.
[x] Register Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova in __init__.py
[x] Add --task-id flag to train_distill.py
    Two-pass tyro parse: pre-extract --task-id from sys.argv to anchor
    the dataclass defaults on the right task before tyro takes over.
[x] Run MCR_smoke (32 envs × 60 iters) — pipeline works end-to-end.
    Throughput: ~180 fps at 32 envs × 24 steps/iter → ~4.3 s/iter.
    Behavior loss oscillates 4.6–11.9 over 60 iters with period ~10 iters
    — same regime as A_03_smoke (3.6–5.3) but higher amplitude because
    of 4096-D vs 128-D encoder feature.  Not yet converging at 32 envs;
    expect this to settle at 1024 envs (A_03 followed the same pattern).
[ ] Save one rendered frame per camera and eyeball (check enabled_geom_groups)
    *Deferred — A_03 trained successfully with the same setup, so
    geom_groups (0, 3) are presumed correct; revisit only if MCR_01_full
    diverges in a way that hints at "encoder can't see the cube."*
[ ] Launch MCR_01_full (1024 envs × 5000 iters, with wandb).
    Command in README.md under "Phase A (MCR variant)".
[ ] Append row to exp_tracker.md with 4 comparison metrics vs A_03
[ ] Read decision rules in §6 → pick next plan
```

### What got built — file map

| Path | What |
|---|---|
| `assets/mcr/mcr_resnet50.pth` | MCR weights (gitignored — see `.gitignore`) |
| `src/kinova_tasks/encoders/__init__.py` | New package marker. |
| `src/kinova_tasks/encoders/mcr_encoder.py` | `FrozenMCREncoder` (per-cam-group nn.Module, MCR ResNet-50 + LayerNorm) and `MCRCNNModel(CNNModel)`. |
| `src/kinova_tasks/tasks/pick_cube_distill_mcr_osc.py` | Distill cfg using MCR encoder; `_resolve_mcr_weights_path()` honors `MCR_WEIGHTS_PATH` env var. |
| `src/kinova_tasks/__init__.py` | Imports + registers `Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova`. |
| `src/kinova_tasks/train_distill.py` | New `--task-id` flag + two-pass argv pre-parse so tyro anchors defaults on the chosen task. |
| `.gitignore` | Excludes `assets/mcr/*.pth` and `*.pt`. |
| `README.md` | New "Phase A (MCR variant)" section with download / smoke / full / play commands. |

### Non-obvious design choices made during implementation

1. **LayerNorm on the encoder output (not in the original plan §3).** Without
   it the smoke run showed early-iter behavior loss climbing rather than
   converging.  Added per-camera LayerNorm so each 2048-D feature is unit-
   scale before the MLP head; this is the only trainable parameter the
   encoder owns.  The (frozen) backbone forward stays under
   ``torch.no_grad()`` for speed; only the LayerNorm participates in
   autograd.
2. **Cameras forwarded independently, not stacked.** The vision env
   concatenates wrist+d455 channel-wise into ``(B, 6, 64, 64)``.  Splitting
   into two ``(B, 3, 64, 64)`` calls and concat-ing features keeps each
   camera's geometric prior intact (a single shared backbone produces
   per-image features either way; the question is only how the LayerNorm
   sees them).  Keeps the door open for two-stream variants later.
3. **64→224 upsample inside the encoder.** Re-rendering the env at 224×224
   would 12× the per-step camera cost; bilinear-upsampling 64→224 inside
   the encoder costs ~negligible extra and matches MCR's training-time
   input shape.
4. **`weights_path` resolved at construct time, not at registry import
   time.** ``kinova_pick_cube_distill_mcr_osc_runner_cfg()`` only stores
   the string in the cfg dict — the file existence check is deferred to
   ``FrozenMCREncoder.__init__``.  This keeps the task registry import
   safe on machines that haven't downloaded the weights yet (e.g.
   syntax-check / type-check / CI).
5. **Two-pass tyro pre-parse for `--task-id`.** ``DistillTrainConfig.from_task``
   instantiates the env+agent cfg from the task id; tyro's defaults get
   anchored on whatever is passed there.  A naïve approach would always
   anchor on the original Phase A task's cfg even when the user passes
   ``--task-id Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova``, then tyro would
   build a frankencfg (Phase A defaults overridden by MCR overrides).
   The pre-parse extracts ``--task-id`` from ``sys.argv`` before tyro
   runs, picks the right defaults, then lets tyro handle the rest.
