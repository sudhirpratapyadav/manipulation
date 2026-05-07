# Sim2Real Vision — Decoupling Geometry from Appearance

Companion to `sim2real.md` and `sim2real_vision_dr.md`. This file is organized around a specific research question that came out of a working teacher-student manipulation pipeline:

> **"My state-based teacher works on the real robot. My student (RGB wrist + env cam + proprio, basic CNN, standard DR) converges in sim but I'm worried the features are entangled appearance-junk that won't transfer. What's the actual evidence on how to decouple geometry from appearance — implicitly, without extracting explicit pose at deploy time?"**

---

## 0. Starting context (the user's setup)

- **Teacher:** PPO on low-dim state (object pose, gripper pose, goal). Transfers zero-shot to real with state tracking. Tasks: pick cube, open drawer, etc.
- **Student:** Distilled from teacher. Inputs = wrist camera + environment camera + proprio. Basic CNN encoder. Trained with standard DR (camera angles, textures, colors).
- **Status:** Converges in sim, success below teacher, not yet tested on real.
- **Open question:** Will it generalize to "almost any scene" for one task (e.g. pick-cube with new table, cube color, lighting), and is a basic-CNN encoder the right call?

The discussion below was the conversational scaffolding before this research dive — preserved here so the research can be evaluated against a real, concrete setup.

---

## 1. Will broad DR alone get you "any scene"?

Probably not for RGB with a from-scratch CNN. You get robustness *within* the DR support and weak/unpredictable behavior outside it. A from-scratch CNN trained on a manipulation reward has no incentive to learn "cube" — it learns whatever pixel statistics correlate with reward inside your DR distribution. That can be the cube, or it can be "the reddish blob near the gripper," or "the thing that casts this shadow pattern."

Two levers change the picture:

1. **Pretrained backbone with strong visual priors** (DINOv2, R3M, MCR, VC-1). These have seen huge amounts of natural imagery, so "object on a table under unusual lighting" is already in-distribution for the *features*, even if your sim never rendered it. Single biggest lever for "generalize to scenes I didn't randomize."
2. **Photorealistic rendering** (ray tracing, Gaussian splats of representative real scenes). Shrinks the sim-real gap so DR doesn't have to do all the work.

## 2. The "humans don't see millions of cubes" objection

Right that the "needs millions of examples" framing is wrong, but humans aren't doing what a from-scratch CNN is doing:

- Decades of multimodal experience — depth from stereo + parallax, touch, proprioception. The human "cube concept" is grounded in physics, not just pixels.
- Strong 3D scene prior — humans parse "object on surface" before "this specific object."
- Active perception — head motion, refocus, look from another angle when uncertain.
- Compositional reasoning — shape + affordance + task, not one entangled mapping.
- Massive transfer from related motor programs.

The real claim isn't "you need millions of cubes." It's: **either (a) use features that already encode object-ness from somewhere else, or (b) inject enough variation during training that the only stable solution is to actually represent the object.** DR is (b) done crudely. Pretrained encoders are (a) done cheaply. Humans use a much richer (a) built up over years of embodied multimodal experience.

## 3. Implicit vs explicit decoupling of vision and geometry

The user's instinct: "the policy needs geometry, not appearance — why not tell the encoder directly instead of hoping DR teaches it?" That's correct, and there's a spectrum.

- **Explicit decoupling** (what the state-based teacher does): vision → 6D pose → policy. Clean but needs a pose estimator that works on the real robot, which is the problem you're trying to avoid.
- **Fully entangled** (the current student): pixels → CNN → action. Features encode geometry, appearance, and task-relevance jumbled together.
- **Implicit decoupling:** train the encoder so its features factorize into "what's the geometry" vs "what's the surface look" — without ever extracting an explicit pose. The policy reads from the geometry-flavored features under task pressure.

### Patterns for implicit decoupling

Roughly ordered by infrastructure cost:

1. **Auxiliary geometric losses on the encoder.** Predict depth / segmentation / object keypoints / optical flow alongside the policy loss. Encoder forced to preserve geometry; appearance becomes nuisance. Cheap and often effective. Hora used keypoint prediction; lots of grasping work uses segmentation.
2. **Contrastive / invariance objectives.** Two views of the same scene with different appearance → force features to be similar. Encoder learns to throw away appearance. Much more sample-efficient than pure DR for the same goal.
3. **Foundation encoders that already factorize.** DINOv2 features encode geometry and semantics in roughly separable subspaces — depth, correspondence, segmentation can be linear-probed. "Use DINOv2" is soft implicit decoupling for free.
4. **RCAN-style canonicalization.** Image-translation net maps randomized renderings → canonical clean renderings. Policy sees only canonical inputs. 70% real success vs 35% for naive DR on QT-Opt.
5. **NeRF / 3D-aware encoders.** Bake 3D structure into the latent space. Heavy infra, strong prior.

### Concrete next steps for the user's setup

- **A. Auxiliary depth/keypoint head on the student encoder.** Sim has ground-truth depth and object pose. Add a head that predicts depth (or cube keypoints) from the encoder features, train jointly with BC. Encoder is pressured to preserve geometry.
- **B. Asymmetric encoder: feed depth + RGB during training, only RGB at deploy, with feature-matching between them.** Force RGB-only features to match depth-aware features. RGB encoder gets pulled toward geometry-flavored representation.

Both are small additions to the existing pipeline; either is likely to help more than adding DR axes.

### Why this isn't a free win

If the auxiliary task is too easy, the encoder satisfies it with a small subspace and the rest of the features can still be appearance-junk the policy latches onto. So in practice: auxiliary loss + DR + (ideally) pretrained encoder, layered.

---

## 4. Open research questions to chase

These are the things the deep-research pass below should answer with concrete papers, codebases, and ablation numbers:

1. **Auxiliary geometric losses for sim2real:** which auxiliary tasks (depth, segmentation, keypoints, flow, NOCS, normals) actually help sim2real transfer in manipulation, and by how much? Does it matter if the aux head is supervised vs self-supervised?
2. **Modality distillation / asymmetric encoders:** papers that train RGB+depth in sim and deploy RGB-only with a feature-matching loss. How much of the depth advantage transfers?
3. **Pretrained encoders for manipulation sim2real:** head-to-head numbers for DINOv2 vs R3M vs MCR vs VC-1 vs Theia vs from-scratch on manipulation sim2real. Frozen vs LoRA vs full fine-tune.
4. **Contrastive / invariance objectives:** explicit appearance-invariance losses (CycleGAN-style, multi-view contrastive, augmentation-contrastive) used inside a sim2real pipeline.
5. **Canonicalization / image translation (RCAN lineage):** what's been published since RCAN — is this idea still alive, has it been superseded by foundation encoders or photorealism?
6. **Object-centric representations:** slot attention, object-aware tokens, segmentation-conditioned policies. Does explicit object factorization help sim2real?
7. **3D-aware encoders without explicit pose:** NeRF features, Gaussian splat features, neural-field encoders, and lifted 2D-to-3D representations used as policy inputs.
8. **Empirical decoupling evidence:** are there papers that *probe* learned features (linear probe for object pose / depth on the policy's encoder) to actually demonstrate decoupling, rather than just reporting task success?
9. **Real-robot results, not just sim:** which of the above have been demonstrated on real hardware on manipulation, with reported success rates?

---

## 5. Deep research findings

The 9 questions in Section 4 are answered below, one subsection each, then a synthesis in 5.10. Entries marked `?` are not double-verified against the paper PDF — treat the specific number as approximate. Citations re-used from `sim2real.md` and `sim2real_vision_dr.md` are referenced by name rather than re-tabled.

### 5.1 Auxiliary geometric losses (Question 1)

What auxiliary tasks (depth, segmentation, keypoints, flow, NOCS, normals) actually move the needle on sim2real for manipulation?

| Paper | Year / Venue | Aux loss | Real result | Verified? |
|---|---|---|---|---|
| **DextrAH-G — Lum et al.** ([arXiv:2407.02274](https://arxiv.org/abs/2407.02274)) | CoRL 2024 | **Auxiliary object-position regression** (β=0.1) on top of action-imitation loss; depth student | 93.6% single-object grasp, 87% bin-pack across 256 attempts. Auth states aux head enables state-machine transitions and is part of the loss. | yes |
| **DextrAH-RGB — Lum et al.** ([arXiv:2412.01791](https://arxiv.org/abs/2412.01791)) | 2024 | **Same auxiliary object-position L2 head** on stereo-RGB student; ablations show fine-tuned ResNet > frozen > scratch encoders | 60-100% per-object grasp, 77% bin-pack normal lighting, 74% HDR. Explicitly attributes 3D inference ability to the aux head. | yes |
| **Hora — Qi et al.** ([arXiv:2210.04887](https://arxiv.org/abs/2210.04887), [PDF](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)) | CoRL 2022 | **Keypoint-based object representation** (8 keypoints) used by privileged teacher; student is proprioception-only with adaptation module. *Vision is not in the deployment loop, but keypoint repr beats pos+quat in the teacher.* | TriFinger ~83% z-axis rotation success zero-shot. Keypoints > pos+quat in convergence and final success (per `sim2real_vision_dr.md` summary). | partial |
| **kPAM — Manuelli et al.** ([arXiv:1903.06684](https://arxiv.org/abs/1903.06684)) | ISRR 2019 | **Semantic 3D keypoints as the explicit representation** between perception and action | Category-level transfer to never-seen mugs/shoes; foundational for keypoint-conditioned manipulation. No raw success-rate ablation against non-keypoint baseline in same paper. | yes |
| **HRP — Srirama et al.** ([arXiv:2407.18911](https://arxiv.org/abs/2407.18911)) | RSS 2024 | **Pre-train ViT to predict human-affordance labels** (future contact, hand pose, target object) via L2 regression on top of self-supervised encoder | Outperforms 6 SOTA encoders by ≥20% across 5 real tasks; +15% min on 5 real tasks across 3 robot morphologies. Affordance prediction = aux geometric/contact loss directly raises real-world success. | yes |
| **KOVIS — Puang et al.** ([DeepAI link](https://deepai.org/publication/kovis-keypoint-based-visual-servoing-with-zero-shot-sim-to-real-transfer-for-robotics-manipulation)) | IROS 2020 | **Keypoint autoencoder + decoder reconstructs depth and segmentation only from keypoints** — keypoint bottleneck forces geometry to be encoded | Zero-shot sim2real for fine manipulation (peg insertion, mug stacking) without real-world fine-tune. Concrete success rates per-task in the paper. | partial |
| **ATK — Zhang et al.** ([arXiv:2506.13867](https://arxiv.org/abs/2506.13867)) | CoRL 2025 | **Optimize a minimal task-driven keypoint set**, distill teacher into RGB policy that tracks selected keypoints | Improves robustness to visual disturbances on real robot; explicitly framed as solving the problem this section is about — find geometry-only state. | yes |
| **PlantTrack** ([arXiv:2407.16829](https://arxiv.org/html/2407.16829v1)) | 2024 | **Sim-trained keypoint heatmap predictor** with 20 synthetic images, then used by policy | Zero-shot sim2real for plant keypoint tracking. Niche but a useful data-efficiency anchor. | partial |
| **Learning to Augment Synthetic Images — Pashevich et al.** ([arXiv:1903.07740](https://arxiv.org/abs/1903.07740)) | IROS 2019 | **Object-aware augmentation** (boundary noise, object-erasing using sim seg masks) used as a proxy for object-localization | Object-aware augs gave the largest gains in their search; selection criterion = object localization on real images. (See `sim2real_vision_dr.md` Sec B.) | yes |
| **Learning to Navigate in Complex Environments — Mirowski et al.** (DeepMind) | ICLR 2017 | **Depth prediction + loop closure** as RL aux losses | Higher data efficiency and final task performance vs RL-only baseline. Foundational reference for "depth-as-aux-loss." | yes |

**Synthesis.** When the literature actually controls for it, **task-driven keypoints and object-position aux losses are the single most consistent winners** (DextrAH-G/RGB explicitly use a position-regression head, Hora uses keypoints over pos+quat, ATK/HRP show big real-world gains from affordance/keypoint pre-training). Generic "predict depth from sim ground truth" is positive but small (Mirowski-style) and has not been shown to be transformative for manipulation specifically. **Reconstruction-style auxiliary losses (predict raw RGB) are mildly counter-productive** — Tan et al. (RLJ 2025) and others find that reconstructing task-relevant masked images outperforms reconstructing raw RGB precisely because the latter pulls the encoder toward modeling distractor pixel statistics. The lesson for the user's situation: **use a small auxiliary head that regresses object 6D pose or a few task keypoints from sim ground truth**, not a generic depth or RGB reconstruction head.

### 5.2 Modality distillation / asymmetric encoders (Question 2)

Papers that train with depth (or other privileged modality) and deploy with RGB only, or that match features between RGB and depth encoders.

| Paper | Year / Venue | Mechanism | Real result | Verified? |
|---|---|---|---|---|
| **DextrAH-RGB → DextrAH-G** ([arXiv:2412.01791](https://arxiv.org/abs/2412.01791) and [arXiv:2407.02274](https://arxiv.org/abs/2407.02274)) | 2024 | Same fabric-guided privileged teacher; depth student vs stereo-RGB student. Auxiliary object-position regression in both. **Stereo > mono ablation: 89% vs 83%; finetuned ResNet > frozen > scratch.** | RGB student is 60-100% per-object, 77% bin-pack normal light, 74% HDR (vs depth student 87% bin-pack). RGB closes most of the gap but is ~10pp behind depth in bin-pack. | yes |
| **MonoLift** ([OpenReview](https://openreview.net/forum?id=wZzC5rpDY1)) | 2024 | **3D depth-guided teacher → monocular RGB student** with cross-modal distillation of spatial / temporal / action-level features | Reports cross-modal distillation gains for monocular RGB on manipulation. | partial |
| **TraKDis — Lin et al.** ([arXiv:2401.13362](https://arxiv.org/abs/2401.13362)) | 2024 | Privileged-state teacher (cloth state) → vision student (RGB) via knowledge distillation; transformer-based student | Cloth manipulation; reports significant improvement vs visual-RL baseline trained from scratch. | partial |
| **Lee et al. — Quadruped over Challenging Terrain** ([Science Robotics 2020](https://www.science.org/doi/10.1126/scirobotics.abc5986), [arXiv:2010.11251](https://arxiv.org/abs/2010.11251)) | 2020 | **Privileged teacher with terrain ground truth → proprioception-only TCN student** with latent-encoder reconstruction. The student matches the teacher's latent state from history. | Zero-shot transfer to deformable terrain, snow, mud. The asymmetric-encoder pattern that everything below copies. | yes |
| **Miki et al. — Robust Perceptive Locomotion** ([Science Robotics 2022](https://www.science.org/doi/10.1126/scirobotics.abk2822), [arXiv:2201.08117](https://arxiv.org/abs/2201.08117)) | 2022 | **Attention-based recurrent encoder fuses proprio + exteroceptive depth**; teacher trained with privileged height-map info, student reconstructs belief from noisy depth | 1700m underground without a fall; the canonical "asymmetric encoder + belief reconstruction" reference for vision policies. | yes |
| **Visual-Policy Multi-Camera → Single-Camera KD** ([arXiv:2303.07026](https://arxiv.org/abs/2303.07026)) | 2023 | Multi-cam privileged teacher → single-cam student with feature distillation | Demonstrates feature-matching across camera-count asymmetry on manipulation. | partial |
| **UniDexGrasp++** ([ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wan_UniDexGrasp_Improving_Dexterous_Grasping_Policy_Learning_via_Geometry-Aware_Curriculum_and_ICCV_2023_paper.pdf)) | 2023 | **State specialists → vision generalist**; iterative generalist-specialist cycle. Closer to state-vision asymmetric than depth-RGB. | 78% on test objects, +11% vs UniDexGrasp; from `sim2real.md`. | yes |
| **RialTo** ([arXiv:2403.03949](https://arxiv.org/abs/2403.03949)) | 2024 | **"Inverse distillation": vision policy → state policy → fine-tune in sim → distill back to vision**. Asymmetric in time, not in modality. | Strong real-world manipulation results, see `sim2real.md`. | yes |
| **TWIST** ([Project](https://twist-sim2real.github.io/)) | 2024 | World-model teacher on state → world-model student on DR images | Sim2real model-based RL; in `sim2real.md`. | partial |

**Synthesis.** The depth-teacher → RGB-student pattern is *the* canonical answer to this question, with **DextrAH-RGB the most direct manipulation example with reported numbers** (RGB student loses ~10pp vs depth student in bin-pack but matches it in normal-light single-object grasps). The legged-locomotion lineage (Lee 2020, Miki 2022) is where the encoder-side trick — **train the student to predict the teacher's latent or belief state** — was operationalized first, and this is the move that the user should copy. Specifically: train the RGB encoder so its features predict (a) the teacher's latent state and (b) ground-truth object 6D pose / keypoints; do not just match action distributions. The depth modality itself is not load-bearing — what is load-bearing is the auxiliary geometric supervision the depth-student happened to get from depth ground truth. You can replicate that with RGB + depth-aux-prediction without ever deploying depth.

### 5.3 Pretrained encoders for manipulation sim2real (Question 3)

Head-to-head numbers on which pretrained encoder is actually best for sim2real manipulation.

| Encoder | Paper / arXiv | Dataset / pretext | Real-robot evidence |
|---|---|---|---|
| **R3M** | Nair et al. ([arXiv:2203.12601](https://arxiv.org/abs/2203.12601), CoRL 2022) | Ego4D human video; time-contrastive + video-language alignment + L1 sparsity | +20% vs scratch and +10% vs CLIP/MoCo on 12 sim tasks; 5 real Franka tasks in cluttered apartment with 20 demos. |
| **VC-1** | Majumdar et al. ([arXiv:2303.18240](https://arxiv.org/abs/2303.18240), NeurIPS 2023) | 4000 hr egocentric + ImageNet, MAE pre-train, ViT | CortexBench (17 tasks, locomotion/nav/dex/mobile manip): VC-1 best on average but **does not universally dominate**; task-specific adaptation needed. Real-hardware: VC-1(adapted) > all prior PVRs. |
| **MCR** | Jiang et al. ([arXiv:2410.22325](https://arxiv.org/abs/2410.22325), 2024) | DROID robot dataset, action-prediction + time-contrastive + dynamics-alignment loss; **CNN backbone** | +14.8% over best baseline across 4 sim domains, 20 tasks. **+76.9% real-world success** on 3 manipulation tasks. |
| **Theia** | Shang et al. ([arXiv:2407.20179](https://arxiv.org/abs/2407.20179), CoRL 2024) | Distills CLIP+DINOv2+ViT into DeiT-Tiny via cosine + smooth-L1 | Outperforms its teacher VFMs on robot tasks with smaller model and less training data. Real-robot: in CoRL eval suite. |
| **VIP** | Ma et al. ([arXiv:2210.00030](https://arxiv.org/abs/2210.00030), ICLR 2023) | Ego4D, value-implicit goal-conditioned RL objective | Strong dense reward for sim+real tasks; outperforms prior PVRs as a frozen reward. |
| **Voltron (V-Cond / V-Dual / V-Gen)** | Karamcheti et al. ([arXiv:2302.12766](https://arxiv.org/abs/2302.12766), RSS 2023) | Sth-Sth + captions, language-conditioned reconstruction + caption generation | SOTA on Voltron-Eval (5 problems incl. real-robot LCBC). Better than R3M on language-grounded tasks. |
| **HRP** | Srirama et al. ([arXiv:2407.18911](https://arxiv.org/abs/2407.18911), RSS 2024) | Affordance fine-tune of ViT-B/16 on human videos | +20% over 6 SOTA; +15% min on 5 real tasks across 3 robot morphologies (incl. dexterous hand). |
| **DINOv2** | Oquab et al. ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193), TMLR 2024) | LVD-142M curated images, self-distillation | **As a robot encoder, has weaknesses**: in `Bridging the Sim2Real Gap` (below), DINOv2-B is the lowest-ranked of 23 encoders on both Action Score and Domain Invariance Score; CNN-based MCR wins. As a *probe target*: best in class for monocular depth and surface normals (Banani et al. CVPR 2024 below). |
| **Theia-tiny-cdiv** | (HF) `theaiinstitute/theia-tiny-patch16-224-cdiv` | CLIP+DINOv2+ViT distilled | Distilled small model with combined teachers — practical drop-in for compute-limited real robots. |

**Benchmark/study papers worth citing:**

- **Bridging the Sim2Real Gap — Vakil et al.** ([arXiv:2501.16389](https://arxiv.org/abs/2501.16389)) — evaluates 23 encoders on Action Score (linear-probe action prediction) and Domain Invariance Score (sim/real embedding alignment). **MCR wins both.** **DINOv2-B is bottom on both axes.** **CNNs > ViTs for domain invariance.** Already in `sim2real_vision_dr.md` Sec F.
- **An Unbiased Look at Datasets — Dasari et al.** ([arXiv:2310.09289](https://arxiv.org/abs/2310.09289), CoRL 2023) — pre-training **dataset distribution matters more than dataset size**; ImageNet/Kinetics are surprisingly strong; sim benchmarks systematically misrank methods relative to real-Franka results. Authors release `data4robotics` codebase as a clean baseline.
- **The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based RL** ([arXiv:2411.10175](https://arxiv.org/abs/2411.10175)) — counterpoint: in *model-based* visual RL, frozen PVRs often fail to beat from-scratch CNNs. Suggests the gain is regime-specific (BC > model-based RL > model-free RL).
- **When Pre-trained Visual Representations Fall Short** ([arXiv:2502.03270](https://arxiv.org/abs/2502.03270)) — limitations analysis, suggests that simple regularizations on top of frozen features can recover much of the gap.
- **Probing the 3D Awareness of VFMs — Banani et al.** (CVPR 2024, [arXiv:2404.08636](https://arxiv.org/abs/2404.08636)) — ranks VFMs on depth, surface-normal, multi-view correspondence probes. **DINOv2 best, Stable Diffusion close second; CLIP very poor on geometry.** Multi-view consistency is weak for *all* models, however — they capture semantic correspondence not 3D-consistent correspondence. Important caveat for sim2real.

**Synthesis.** Two consistent findings cut across the benchmarks: **(1) pre-training on robot/manipulation data wins** (MCR, R3M, HRP all beat generic encoders on robot tasks), and **(2) which architecture wins depends on what you measure** — DINOv2 wins on geometry probes but loses on sim2real domain invariance and downstream BC action score. For the user's setup: **frozen-backbone CNN pretrained on manipulation data (MCR is the current best published) is the safest first move**; if you need a ViT for some reason, prefer Theia (multi-teacher distilled) or VC-1 (adapted) over raw DINOv2. The "just slap DINOv2 on it" prior is wrong for sim2real specifically. Layer aux losses (Sec 5.1) on top regardless of which encoder you pick.

### 5.4 Contrastive / invariance objectives (Question 4)

Explicit appearance-invariance losses inside a sim2real pipeline.

| Paper | Year | Mechanism | Result |
|---|---|---|---|
| **Time-Contrastive Networks (TCN)** ([arXiv:1704.06888](https://arxiv.org/abs/1704.06888), Sermanet et al., ICRA 2018) | 2017 | Triplet loss across **simultaneous viewpoints** of the same observation, repelled from temporal neighbors. Discovers viewpoint/occlusion/lighting/background-invariant features. | First self-supervised end-to-end imitation of human motion by a real robot; viewpoint invariance is the explicit win. |
| **CURL — Srinivas/Laskin/Abbeel** ([arXiv:2004.04136](https://arxiv.org/abs/2004.04136), ICML 2020) | 2020 | InfoNCE between random crops of the same observation, jointly with off-policy RL | First image-RL alg to nearly match state-based sample efficiency on DM Control. **No real-robot results directly.** |
| **RAD — Laskin et al.** ([NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf)) | 2020 | Plain data augmentation (random shift / crop / color jitter / cutout / random conv) inside RL | SOTA on DM Control + ProcGen at the time. Data augmentation alone, no contrastive. |
| **DrQ / DrQ-v2 — Yarats et al.** ([arXiv:2004.13649](https://arxiv.org/abs/2004.13649); [arXiv:2107.09645](https://arxiv.org/abs/2107.09645)) | 2020/2021 | **Random shift only**, regularize Q-value across two augmented copies. No contrastive head. | DrQ-v2 dominates DM Control humanoid from pixels. **Random shift alone matched CURL's contrastive at fraction of complexity.** A real lesson: contrastive heads aren't always needed if the augmentation is right. |
| **MANGO** ([arXiv:2601.09605](https://arxiv.org/html/2601.09605)) | 2026? | **Sim2real image translation with segmentation-conditioned InfoNCE** + regularized discriminator | Improves IL policy robustness to camera shift; reports real success rate gains on 4 real tasks. |
| **VR-Goggles — Zhang et al.** ([arXiv:1802.00265](https://arxiv.org/abs/1802.00265), RA-L 2019) | 2018 | **Real → sim** image translation + shift loss (frame consistency) for visual control | Real-world indoor/outdoor robot navigation; the lineage that anticipates RCAN. |
| **RL-CycleGAN — Rao et al.** ([CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Rao_RL-CycleGAN_Reinforcement_Learning_Aware_Simulation-to-Real_CVPR_2020_paper.pdf)) | 2020 | CycleGAN sim→real with an RL-aware Q-consistency loss | Outperforms vanilla CycleGAN for sim2real grasping. |
| **RetinaGAN — Ho et al.** ([arXiv:2011.03148](https://arxiv.org/abs/2011.03148), ICRA 2021) | 2020 | **Object-detector-consistency-loss** between sim and adapted-sim images. CycleGAN with object-aware constraint. | +12% over prior sim2real on real grasp; 90% on push, 97% on door-opening with Ensemble-RetinaGAN. |
| **Sim-to-Real via Latent Prediction** (Frontiers 2022, [link](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1067502/full)) | 2022 | Cross-domain forward-dynamics latent alignment (CFD loss) sim ↔ real | Sim2real for visual non-prehensile manipulation without paired data. |
| **Inter-Token Contrast (ICon)** ([arXiv link in search](https://arxiv.org/html/2505.18487)) | 2025 | Contrastive loss on ViT tokens to **separate agent-tokens from environment-tokens** | Improves manipulation policy transfer across robots; agent-centric inductive bias in features. |

**Synthesis.** The "explicit contrastive" recipe has bifurcated. **Plain augmentation (DrQ-v2 / RAD) often matches or beats InfoNCE-style contrastive (CURL)** at much lower complexity for in-domain RL. For *sim2real specifically*, the strongest contrastive results come from **object-aware or task-aware variants** (RetinaGAN's detector consistency, MANGO's seg-conditioned InfoNCE, ICon's agent-vs-environment token split) rather than naive sim-real contrastive on whole frames. Generic CycleGAN sim→real (RL-CycleGAN) works but is dated. For the user's setup: if you want a contrastive objective, make it **object-conditioned** (e.g. anchor positives on the same object across appearance variations, negatives across object identity) — not whole-frame sim/real contrastive. Or just lean on aug+aux-loss and skip the contrastive head.

### 5.5 Canonicalization / image translation, RCAN lineage (Question 5)

Has the RCAN idea been superseded?

| Paper | Year | Approach | Result |
|---|---|---|---|
| **RCAN — James et al.** ([arXiv:1812.07252](https://arxiv.org/abs/1812.07252), CVPR 2019) | 2019 | U-Net translates randomized → canonical sim. Aux outputs: seg mask + depth. QT-Opt trained on canonical only. | **70% real grasp** zero-shot vs 35% naive-DR-on-QT-Opt; 91% with 5k real fine-tune (vs 580k from-scratch). |
| **VR-Goggles** ([arXiv:1802.00265](https://arxiv.org/abs/1802.00265)) | 2018 | Real → sim translation at deploy time | Predates and motivates RCAN; navigation tasks. |
| **RL-CycleGAN** (CVPR 2020) | 2020 | CycleGAN with RL-aware loss | Beats vanilla CycleGAN for sim2real grasp. |
| **RetinaGAN** ([arXiv:2011.03148](https://arxiv.org/abs/2011.03148), ICRA 2021) | 2020 | CycleGAN + object-detection consistency loss | +12% real grasp over prior sim2real. |
| **PNDR — Zakharov et al.** ([arXiv:2210.12682](https://arxiv.org/abs/2210.12682), ECCV 2022) | 2022 | Modular neural materials/lighting/ray-tracer with per-component DR | Photorealism + DR are complementary for object pose; in `sim2real_vision_dr.md`. |
| **MANGO** ([arXiv:2601.09605](https://arxiv.org/html/2601.09605)) | 2024-25 | Sim → real image translation with segmentation-conditioned InfoNCE; preserves viewpoint consistency | Improves IL policy robustness to camera shift on 6 sim + 4 real tasks. |
| **DiffusionRenderer** ([arXiv:2501.18590](https://arxiv.org/abs/2501.18590)) | 2025 | Video-diffusion-based inverse + forward renderer; estimate G-buffers from real, re-render under different conditions | Generic — not yet a robot-policy paper, but the right substrate for diffusion-based canonicalization. |
| **SplatSim** ([arXiv:2409.10161](https://arxiv.org/abs/2409.10161)) | 2024 | Replaces sim renderer with Gaussian splat reconstruction of real workspace. "DR" becomes photorealism + light augmentation. | 86.25% real success vs 21% without aug. (See `sim2real_vision_dr.md`.) |
| **DextrAH-RGB photorealistic tiled rendering** ([arXiv:2412.01791](https://arxiv.org/abs/2412.01791)) | 2024 | Photorealistic ray-traced rendering in Isaac Lab. Photorealism replacing canonicalization. | 60-100% per-object grasp under varied lighting. |
| **RealD²iff** ([arXiv:2511.22505](https://arxiv.org/abs/2511.22505)) | 2025 | Coarse-to-fine **depth diffusion** for real-to-sim canonicalization (clean real depth → match clean sim depth) | Zero-shot sim2real manipulation; flips the standard direction. |
| **Manipulation as in Simulation** ([arXiv:2509.02530](https://arxiv.org/abs/2509.02530)) | 2025 | Camera Depth Model trained to **clean real depth to match clean sim depth**. Same flip. | Long-horizon manipulation zero-shot transfer; in `sim2real_vision_dr.md`. |

**Synthesis.** The RCAN idea is alive and **partially superseded** by two things: (a) photorealism (SplatSim, DextrAH-RGB, PNDR) — if you can render realistically, you don't need to translate; and (b) real-to-sim canonicalization with diffusion (RealD²iff, Manipulation-as-in-Simulation) — clean the *real* sensor stream to match clean sim, instead of training the policy through DR. **No published RCAN-with-diffusion-canonicalizer for manipulation policies (as of mid-2025) that I could verify.** The closest direction is depth-cleaning diffusion. For the user's RGB setup, RCAN-style canonicalization would still work but is rarely the most practical move now — photorealism (Gaussian splats of your workspace) or pretrained encoders are usually higher-leverage.

### 5.6 Object-centric / slot attention representations (Question 6)

| Paper | Year / Venue | Approach | Manipulation result |
|---|---|---|---|
| **Slot Attention** ([arXiv:2006.15055](https://arxiv.org/abs/2006.15055)) | NeurIPS 2020 | Iterative attention to bind features to slots | Foundational; not robotic on its own. |
| **SAVi** ([Kipf et al.](https://slot-attention-video.github.io/)) | ICLR 2022 | Slot Attention extended to video; conditioning on object cues helps segmentation | Video; precursor to robotic uses. |
| **POCR — Shi et al.** ([arXiv:2404.13474](https://arxiv.org/abs/2404.13474)) | RSS 2024 | **Compose pre-trained "where" (SAM) + "what" (LIV / R3M / CLIP)** to build object-centric reps with no new training | "Better than prior pre-trained reps for robotics on sim + real manipulation"; systematic generalization on multi-object tasks. |
| **GROOT — Zhu et al.** ([arXiv:2310.14386](https://arxiv.org/abs/2310.14386), CoRL 2023) | 2023 | **Object-centric 3D point clouds + transformer policy**; segmentation-correspondence model for new objects | Robust to background changes, camera-view shifts, new object instances. Real-robot evaluation under "wild changes." |
| **SOLD** ([arXiv:2410.08822](https://arxiv.org/abs/2410.08822)) | 2024 | **Slot-attention object-centric latent dynamics model** for visual RL | Outperforms DreamerV3 on relational manipulation tasks. Sim only. |
| **Spotlighting Task-Relevant Features (SBOCR) — Chapin et al.** ([arXiv:2601.21416](https://arxiv.org/abs/2601.21416)) | 2026 | Slot-Based Object-Centric Reps group dense features into object-like entities | "Outperform dense and global representations in generalization settings, even without task-specific pretraining." |
| **DOCIR** ([arXiv:2503.11565](https://arxiv.org/abs/2503.11565)) | 2025 | Disentangled object-centric image rep for manipulation | Designed for distractor robustness. |
| **Is Object-Centric Beneficial?** ([arXiv:2506.19408](https://arxiv.org/abs/2506.19408)) | 2025 | **Empirical study** comparing OCR vs dense vs global feature backbones for manipulation | OCR-based policies outperform dense/global on generalization, even without pretraining. |
| **OP3 / Entity Abstraction — Veerapaneni et al.** ([arXiv:1910.12827](https://arxiv.org/abs/1910.12827), CoRL 2019) | 2019 | Symmetric per-entity dynamics in MBRL | Generalizes to unseen object configurations (build new block towers). Toy/sim only. |
| **CAGE** ([arXiv:2410.14974](https://arxiv.org/abs/2410.14974)) | 2024 | Causal-attention manipulation policy for distractor robustness | Real-world generalization gains. |

**Synthesis.** Object-centric is **consistently positive for distractor and viewpoint robustness** but **rarely tested in a clean sim2real ablation against a pure CNN baseline at matched compute**. POCR and GROOT are the strongest real-robot existence proofs that "factor visual input into objects, then policy" generalizes better than dense features. SOLD / OP3 are sim-only. The 2025 SBOCR study is the most direct "OCR vs dense" head-to-head — OCR wins on generalization. **The hidden cost** is that the segmentation/slot stage has to be reliable at deploy time; this is plausible now with SAM-2 / DINOv2-based segmentation, but it's an extra failure mode the entangled-CNN baseline doesn't have. For the user's setup, if you can stand SAM at deploy, POCR is the cheapest existing framework to plug in.

### 5.7 3D-aware encoders without explicit pose (Question 7)

NeRF / Gaussian-splat / neural-field policy inputs.

| Paper | Year / Venue | What | Result |
|---|---|---|---|
| **F3RM (Distilled Feature Fields) — Shen et al.** ([arXiv:2308.07931](https://arxiv.org/abs/2308.07931), CoRL 2023) | 2023 | NeRF that renders RGB **+ pre-trained 2D vision features (CLIP)**. Distilled Feature Field. | Few-shot 6-DoF grasp/place; in-the-wild generalization to unseen objects via free-text. Real-robot. |
| **GNFactor — Ze et al.** ([arXiv:2308.16891](https://arxiv.org/abs/2308.16891), CoRL 2023 Oral) | 2023 | **Generalizable Neural Feature Field as reconstruction module + Perceiver Transformer policy**, sharing a 3D voxel. Distills Stable Diffusion features in 3D. | 10 RLBench tasks + 3 real tasks. Substantial gains over SOTA. |
| **GraspNeRF** ([Project](https://pku-epic.github.io/GraspNeRF/)) | ICRA 2023 | Generalizable NeRF for 6-DoF grasp on transparent/specular objects from sparse RGB | Material-agnostic grasp in clutter; outperforms NeRF-VGN with 6 views vs 49. |
| **Dex-NeRF** ([arXiv:2110.14217](https://arxiv.org/abs/2110.14217)) | CoRL 2021 | NeRF-based depth pipeline for transparent objects | 90-100% grasp success on transparent objects (where depth fails). |
| **ManiGaussian — Lu et al.** ([arXiv:2403.08321](https://arxiv.org/abs/2403.08321), ECCV 2024) | 2024 | **Dynamic Gaussian Splatting** with semantic feature propagation + Gaussian world model | +13.1% avg success vs SOTA on RLBench; multi-task. |
| **GWM — Lu et al.** (ICCV 2025) | 2025 | Scalable Gaussian world model for manipulation | Follow-up scaling work. |
| **Splat-MOVER** ([arXiv:2405.04378](https://arxiv.org/abs/2405.04378)) | 2024 | Gaussian-splat scene with distilled semantic + grasp affordance features (ASK-Splat) | Open-vocabulary manipulation. |
| **Embodied Gaussians** ([Project](https://embodied-gaussians.github.io/)) | 2024 | Visually learnt + physically grounded 3D rep | Robotics-oriented 3D rep with physics grounding. |
| **RoboGSim** ([arXiv:2411.11839](https://arxiv.org/abs/2411.11839)) | 2024 | Real2Sim2Real robotic Gaussian-splat simulator | SplatSim follow-up with more real assets. |
| **DP3 (3D Diffusion Policy)** ([arXiv:2403.03954](https://arxiv.org/abs/2403.03954), RSS 2024) | 2024 | **Sparse point cloud + MLP** + diffusion policy | 85% on 4 real tasks with 40 demos each. Cleanest "PC > RGB at fixed everything else" comparison. |
| **GROOT** | CoRL 2023 | Object-centric 3D PC + transformer policy | Real-robot; under "wild" setup changes. |
| **RVT / RVT-2** ([arXiv:2306.14896](https://arxiv.org/abs/2306.14896); [arXiv:2406.08545](https://arxiv.org/abs/2406.08545)) | 2023/2024 | Multi-view virtual rendering + transformer for 3D manipulation | RVT-2 trains 6× faster than RVT and adds 19pp success; mm-precision tasks. |

**Synthesis.** 3D-aware encoders are the **strongest existence proof of "implicit decoupling"** — feature fields literally bake geometry into the representation, and policies on top do generalize to backgrounds/viewpoints they never saw. **F3RM and GNFactor are the closest thing to "use 2D foundation features in a 3D-aware shell"**. The deployment cost is not trivial: you need multi-view at deploy time (F3RM, GraspNeRF) or some splat-fitting step. For the user's wrist + env cam setup, **F3RM-style distillation of DINOv2 features into a small per-scene 3D representation** is the strongest aspirational direction — but it is more infra than aux-loss-on-CNN. 3D Gaussian splat policies (ManiGaussian/GWM) are research-grade as of late 2025 and not yet a turn-key drop-in. Pragmatic intermediate: GROOT-style "segment object, build object-centric 3D point cloud, transformer policy" is the path with the most documented real-world successes.

### 5.8 Empirical decoupling evidence — probing learned policy features (Question 8)

Papers that linear-probe a manipulation policy's encoder to actually demonstrate what's encoded.

| Paper | What is probed | Finding |
|---|---|---|
| **Probing the 3D Awareness of Visual Foundation Models — Banani et al.** ([CVPR 2024, arXiv:2404.08636](https://arxiv.org/abs/2404.08636)) | Linear probes for **monocular depth, surface normals, multi-view correspondence** on DINO, DINOv2, CLIP, MAE, Stable Diffusion, ViT, ResNet | DINOv2 best on depth + normals; Stable Diffusion close 2nd; CLIP very poor on geometry. **All models fail multi-view consistency** — they capture semantic correspondence not 3D-consistent correspondence. |
| **Lexicon3D** ([NeurIPS 2024](https://yunzeman.github.io/lexicon3d/static/Lexicon3D.pdf)) | Probes VFMs on complex 3D scene understanding tasks | Provides task-by-task ranking of which VFM encodes what 3D structure. |
| **A General Protocol to Probe Large Vision Models for 3D Physical Understanding** ([arXiv:2310.06836](https://arxiv.org/abs/2310.06836)) | DINO/DINOv2/SD probed on shape/depth/lighting/support | DINOv2 and SD lead; ranks support relations, lighting, depth as separable in features. |
| **Bridging the Sim2Real Gap — Vakil et al.** ([arXiv:2501.16389](https://arxiv.org/abs/2501.16389)) | **Action Score** = linear probe predicting actions from encoder features; **Domain Invariance Score** = sim/real centroid alignment | 23 encoders ranked. MCR top, DINOv2 bottom for sim2real. The clearest "linear probe of a manipulation encoder" study so far. |
| **State Representations as Incentives — Petropoulakis et al.** ([arXiv:2309.11984](https://arxiv.org/abs/2309.11984)) | Compares state representations along a continuum from numerical state → image embeddings; measures sim-vs-real transfer ability | "Incentivizing the state representation with task-specific knowledge facilitates faster convergence and increases sim2real success." Empirical sim2real ablation. |
| **An Unbiased Look at Datasets — Dasari et al.** ([arXiv:2310.09289](https://arxiv.org/abs/2310.09289)) | Linear-probe BC fine-tunes on multiple PVRs; sim vs real Franka | Sim benchmarks systematically misrank methods on real hardware. |
| **The Surprising Ineffectiveness of PVRs for MBRL** ([arXiv:2411.10175](https://arxiv.org/abs/2411.10175)) | Direct comparison of frozen PVRs vs scratch CNN inside a model-based RL agent | Frozen PVRs underperform scratch in MBRL; complicates the "DINOv2 is free win" story. |
| **When Pre-trained Visual Representations Fall Short** ([arXiv:2502.03270](https://arxiv.org/abs/2502.03270)) | Limitation analysis of PVRs in visuo-motor learning | Identifies failure modes for PVR-based policies; suggests light task-specific regularization recovers most. |

**Synthesis.** **Probing-of-policy-encoders specifically** is thinner than probing-of-foundation-models. The Banani et al. lineage tells you what DINOv2 *can* encode, but not what a *manipulation policy's* encoder ends up encoding after BC. The Bridging-Sim2Real-Gap framework (Vakil et al.) is the closest thing — it linear-probes encoders for action prediction and sim/real alignment. **There is a clear gap in the literature: nobody has linear-probed a sim-trained manipulation CNN encoder to show what fraction of its features predict object pose vs background statistics, especially before vs after adding aux losses.** This would be a high-value diagnostic for the user's project — probe your own encoder on cube pose / cube depth / table-color before and after adding an aux head and see whether the geometry-flavored axes actually grow.

### 5.9 Real-robot results (Question 9)

Subset of the above with verified real-hardware manipulation success rates. Re-tabling for convenience; some entries duplicate `sim2real.md` and `sim2real_vision_dr.md`.

| Method | Modality | Real result | Source |
|---|---|---|---|
| **DextrAH-G** | Depth + aux pos | 87% bin-pack across 256 attempts; 93.6% single-object | [arXiv:2407.02274](https://arxiv.org/abs/2407.02274) |
| **DextrAH-RGB** | Stereo RGB + aux pos | 77% bin-pack normal light, 74% HDR; 60-100% per-object | [arXiv:2412.01791](https://arxiv.org/abs/2412.01791) |
| **R3M (frozen, BC)** | RGB | Strong gains on 5 real Franka tasks with 20 demos vs scratch / CLIP / MoCo | [arXiv:2203.12601](https://arxiv.org/abs/2203.12601) |
| **MCR** | RGB | **+76.9%** real-world success on 3 manipulation tasks vs strongest baseline | [arXiv:2410.22325](https://arxiv.org/abs/2410.22325) |
| **HRP (ViT-B affordance ft)** | RGB | +15% min on 5 real tasks across 3 morphologies (incl. dexterous) | [arXiv:2407.18911](https://arxiv.org/abs/2407.18911) |
| **VC-1 (adapted)** | RGB | Beats prior PVRs on real hardware in the CortexBench eval | [arXiv:2303.18240](https://arxiv.org/abs/2303.18240) |
| **F3RM** | Multi-view RGB → DFF | Few-shot real-robot grasp/place, in-the-wild objects | [arXiv:2308.07931](https://arxiv.org/abs/2308.07931) |
| **GNFactor** | Multi-view RGB → 3D voxel | 3 real tasks; SOTA-level success | [arXiv:2308.16891](https://arxiv.org/abs/2308.16891) |
| **GROOT** | Object-centric 3D PC | Real Franka under wild background / viewpoint / object changes | [arXiv:2310.14386](https://arxiv.org/abs/2310.14386) |
| **DP3** | Sparse PC | 85% on 4 real Allegro/gripper tasks with 40 demos each | [arXiv:2403.03954](https://arxiv.org/abs/2403.03954) |
| **POCR** | OC RGB (SAM + LIV) | Better than R3M / VC-1 on real multi-object manipulation | [arXiv:2404.13474](https://arxiv.org/abs/2404.13474) |
| **kPAM** | Semantic 3D keypoints | Category-level real transfer to unseen mug/shoe instances | [arXiv:1903.06684](https://arxiv.org/abs/1903.06684) |
| **Hora** | Proprio + keypoint teacher | ~83% z-axis rotation real success zero-shot from cylinders-only training | [PDF](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf) |
| **TriFinger sim2real** | 8-keypoint object | 83% real success | [arXiv:2108.09779](https://arxiv.org/abs/2108.09779) |
| **DexPoint** | Single-view PC | 81% novel-object grasp on Allegro | [arXiv:2211.09423](https://arxiv.org/abs/2211.09423) |
| **UniDexGrasp++** | PC | 78% on test objects, +11% over UniDexGrasp | [arXiv:2304.00464](https://arxiv.org/abs/2304.00464) |
| **RCAN** | RGB w/ canonicalization | **70% real grasp** zero-shot vs ~35% naive DR; 91% with 5k real fine-tune | [arXiv:1812.07252](https://arxiv.org/abs/1812.07252) |
| **SplatSim** | RGB w/ splat photoreal | 86.25% real zero-shot, near-real-data 97.5% | [arXiv:2409.10161](https://arxiv.org/abs/2409.10161) |
| **Sim-and-Real Co-Training** | RGB + small real | +38% real success from sim co-training | [arXiv:2503.24361](https://arxiv.org/abs/2503.24361) |
| **TRANSIC** | Point cloud + online correction | Strong real-robot transfer (in `sim2real.md`) | [PDF](https://transic-robot.github.io/assets/pdf/transic_paper.pdf) |
| **ATK** | RGB w/ keypoint distill | Real-robot improvements on visual disturbance robustness | [arXiv:2506.13867](https://arxiv.org/abs/2506.13867) |
| **RetinaGAN** | RGB w/ object-aware GAN | +12% real grasp over prior; 90% push, 97% door-open | [arXiv:2011.03148](https://arxiv.org/abs/2011.03148) |

---

### 5.10 Cross-cutting takeaways for the user's setup

Given the user's situation (state-based teacher transferring fine, RGB student converges in sim with basic CNN + standard DR, not yet real-tested, worried about appearance entanglement), the evidence above suggests the following ordering of moves:

1. **Add an auxiliary object-pose / keypoint head before adding any more DR.** This is the single change with the most consistent positive signal in the literature: DextrAH-G and DextrAH-RGB both run an L2 object-position head with weight 0.1 alongside action distillation; Hora switched from pos+quat to keypoints and saw ~83% real success; HRP's affordance-prediction head adds ≥15% on every real task tested. This costs you almost nothing — sim has the ground truth — and **directly attacks the appearance-entanglement worry** by forcing the encoder to preserve geometry.

2. **Don't drop the basic CNN — but stop training it from scratch.** "Bridging the Sim2Real Gap" (Vakil et al. 2025) ranks 23 encoders and the winner is a **CNN pre-trained on robot manipulation data (MCR)**. ViTs are worse for domain invariance, and DINOv2 is the *worst* of the 23 by both metrics for sim2real BC despite being best on geometric probes. **Replace your scratch CNN with a frozen MCR backbone (or R3M as a close 2nd) and keep the rest of your pipeline.** This is roughly an hour of code and historically gives 10-20pp on real tasks.

3. **Asymmetric encoder beats more DR. Train RGB-student features to predict the teacher's latent state.** Lee 2020 / Miki 2022 / DextrAH lineage all do a version of this: student features must reconstruct privileged information available only in sim. This is more direct than DR — it's the user's "B" suggestion (depth+RGB at training, RGB at deploy with feature matching) generalized. The matching target should be **teacher latent + ground-truth object pose**, not raw teacher actions alone.

4. **Object-centric shells are the next step up if (1)-(3) underperform.** POCR's "SAM for where + R3M/LIV for what" is the cheapest existing object-centric framework with real-robot wins, and you can drop it on top of an existing BC pipeline. SBOCR (2026) shows OCR-based policies generalize better than dense features even without task-specific pretraining. The cost is a deploy-time segmentation step — workable now with SAM-2.

5. **Pure data-augmentation (DrQ-v2 random shift) is a freebie.** The DrQ-v2 lineage shows random-shift alone matches CURL-style contrastive at a fraction of the complexity. If you're doing online RL fine-tuning at any stage, add it.

6. **Real-to-sim canonicalization (RealD²iff, "Manipulation as in Sim") is the surprising 2025 inversion of RCAN.** Instead of training the policy through DR, train a model to clean real depth (or potentially RGB) at deploy. Worth tracking for your eventual real-robot deployment, but probably not a first move.

7. **Photorealism (SplatSim, DextrAH-RGB tiled ray-tracing) is the biggest "shrink the gap before DR" lever** but is high infra. If you have a budget for a one-time Gaussian-splat reconstruction of your real workspace, SplatSim's 21% → 86.25% delta is the upper bound on what photorealism alone buys. If you don't, skip.

8. **Probe your own encoder.** Nobody in the literature has done a clean linear-probe of a sim-trained manipulation CNN's features for object pose / depth / appearance, before vs after adding an aux-loss head. **You can do this for your own setup in a day** — train a linear classifier on frozen-encoder features for cube position, table color, lighting condition; ablate the aux loss; show that aux-loss-on raises pose-decode accuracy and lowers color-decode accuracy. That gives you direct evidence that decoupling worked, not just a final-task success number.

9. **Negative result worth knowing.** Frozen PVRs underperform scratch in *model-based* RL (`The Surprising Ineffectiveness of PVRs for MBRL`, [arXiv:2411.10175](https://arxiv.org/abs/2411.10175)). The "use a foundation encoder" advice in (2) is specific to BC and model-free RL. If you ever go to MBRL + visual, expect to retune.

10. **Frame-wise > episode-wise for any DR you do keep, and camera-pose DR is undersold.** Both are findings from Jin 2026 (`?` in `sim2real_vision_dr.md`); they apply orthogonally to everything above and are essentially free to add.

Concrete first-week plan: swap scratch CNN → frozen MCR backbone; add an aux head that L2-regresses cube + gripper 6D pose from the encoder features (weight 0.1); keep your existing DR; add random shift augmentation; linear-probe the encoder for cube-pose vs table-color before and after the aux head to confirm decoupling. Real-robot test before adding photorealism, OCR, or 3D representations.

---

## 6. Extreme visual adversity — what's actually been demonstrated

The previous sections cover the *sim2real* literature. This section addresses a different question: the recent crop of viral demos showing manipulation policies surviving lights flicked off mid-task, flashlights pointed at the camera, people walking through the scene, or hands snatching the object out of the gripper. These are mostly *real-data-heavy* systems — a fundamentally different regime from the user's sim-distilled student. The honest takeaway is parked in 6.4.

### 6.1 Company demos — the X/Twitter robustness videos

For each entry: what was shown, what's known about the method, how verifiable the claim is.

#### Skild AI — "Skild Brain" `[demo]`

- **What was shown.** July 2025 unveil ([X thread](https://x.com/SkildAI/status/1950232914454872118), [blog](https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain)) and a September 2025 "shattered limbs / jammed motors" follow-up ([X](https://x.com/SkildAI/status/1970940614234771579)) showed: humanoid loading a dishwasher across form factors, quadruped staying balanced after pushes/kicks, robots traversing slippery slopes and cluttered environments, recovery from broken legs and jammed wheels (2-3 second recovery for jammed wheels). The locomotion blog ([One Policy, All Scenarios](https://www.skild.ai/blogs/one-policy-all-scenarios)) reports recovery from "considerable pushes and pulls on stairs" and unseen urban environments.
- **What's known about the method.** Hierarchical: low-frequency high-level policy + high-frequency low-level policy. Per Brett Adcock's [recap](https://x.com/adcock_brett/status/1952037672957931628) and Skild's blog: "**simulation + internet videos for pre-training, post-trained on real robot data**." For locomotion, RoboHub claims "100,000 diverse simulated robots for 1,000 [sim-]years." Model size, encoder architecture, and data-mix proportions are not disclosed.
- **What they credit for robustness.** Massive sim-fleet diversity ("omni-bodied" → forced morphology invariance), internet video pre-training, real-data post-training.
- **Verifiability.** Low-medium. Blog posts plus demo videos; no paper, no architecture spec, no held-out evaluation protocol. The viral clips are cherry-picked. The high-level recipe (sim+video pretrain, real post-train) is plausible and matches the broader trend, but specific numbers do not exist publicly.

#### Physical Intelligence — π₀ and π₀.₅

- **What was shown.** The π₀.₅ blog ([pi.website/blog/pi05](https://www.pi.website/blog/pi05)) shows mobile manipulators cleaning kitchens and bedrooms in **homes that were never in training data**, with humans interfering mid-task. 10-15 minute multi-stage tasks. The earlier π₀ demos showed laundry folding, table bussing, grocery bagging.
- **What's known about the method.** Public paper ([arXiv:2410.24164](https://arxiv.org/abs/2410.24164) for π₀; [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) for π₀.₅). π₀.₅ is a VLA built on a PaliGemma-class backbone with a flow-matching action expert (~300M parameter action head). Co-trained on **~400 hours of mobile manipulation + cross-embodiment data from π₀ + multimodal web data + verbal instruction data**. Data is overwhelmingly real teleop; sim is not the centerpiece.
- **What they credit for robustness.** Co-training across heterogeneous data types — and explicitly: "**web data makes the biggest difference for generalizing to out-of-distribution objects**." Scale (~100+ training environments). High-level/low-level decomposition with discrete subtask tokens then continuous flow-matching control.
- **Verifiability.** Medium-high. There is a paper, public weights ([openpi](https://github.com/Physical-Intelligence/openpi)), and a third-party reproduction: Penn GRASP Lab's [Pi0-in-the-Wild](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/) ran 300+ trials and reports **42.3% average progress, ~20-50% success on simple OOD tasks**. Failure modes: blocking the wrist camera dropped success to 0%; blocking the side camera 50%; the model freezes on multi-step tasks; sensitive to prompt phrasing. So the home-tour demos cherry-pick favorable runs — the underlying robustness *exists* but is partial.

#### Figure — Helix and Helix 02

- **What was shown.** Helix ([VLA blog](https://www.figure.ai/news/helix), Feb 2025) showed two humanoids cooperating on grocery put-away with novel items. Helix 02 ([blog](https://www.figure.ai/news/helix-02), [logistics scaling](https://www.figure.ai/news/scaling-helix-logistics)) showed a 4-minute autonomous dishwasher unload-and-reload across a full kitchen, plus disturbance recovery in package handling — "if a package shifts or an attempted grasp doesn't land perfectly, Helix corrects mid-motion."
- **What's known about the method.** Three-tier System 0 / S1 / S2 architecture. S0 is a 10M-param network trained **entirely in simulation across 200,000 parallel environments with extensive domain randomization**, on 1000+ hours of joint-level retargeted human motion data. S1 is a transformer policy. S2 is a VLM. Helix VLA itself trained on **~500 hours of teleoperated demonstrations** plus auto-labeled hindsight instructions.
- **What they credit for robustness.** Stereo vision (60% throughput improvement vs mono), state history window, force-feedback closing the loop, scaled real demonstrations, S0's massive sim-DR pretraining.
- **Verifiability.** Low-medium. Engineering blog with throughput numbers (88.2% → 94.4% barcode, 58% throughput gain with more data) but no held-out generalization protocol, no paper, no architecture/weight release.

#### 1X — NEO with Redwood / 1X World Model

- **What was shown.** NEO doing autonomous mobile bi-manual manipulation in homes ([Redwood blog](https://www.therobotreport.com/1xs-neo-humanoid-gains-autonomy-with-new-redwood-ai-model/)) under "changing lighting, clutter, partially obstructed objects." A separate [1X World Model](https://www.1x.tech/discover/1x-world-model) (Jan 2026) is a video-pretrained generative model used as policy/simulator hybrid.
- **What's known about the method.** Redwood is trained on data from EVE and NEO. The 1X World Model is **trained on thousands of hours of real EVE manipulation data in homes/offices**, no synthetic content, no internet video. Public baselines are Llama-class and GENIE-class on HuggingFace.
- **What they credit for robustness.** Real-world data scale, learning the simulator from real data ("absorb the full complexity of the real world without manual asset creation"), hand-selection / retry policies.
- **Verifiability.** Low. Their own blog admits the world model has object-coherence and physics-consistency failure modes ("loses track of occluded items," inconsistent gravity, no self-recognition in mirrors). Marketing-grade demos.

#### Boston Dynamics + TRI — Atlas LBM

- **What was shown.** Aug 2025 [joint demo](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/) of Atlas doing long-horizon packing/sorting/organizing with whole-body manipulation. Researchers explicitly inject **mid-task disturbances** — closing the lid of a box mid-grasp, sliding it across the floor — and Atlas adjusts.
- **What's known about the method.** **450M-parameter Diffusion Transformer with flow-matching**, predicts 48-step (1.6s) action chunks at 30Hz. Conditioned on language. Co-trained on real teleop (VR-collected on Atlas hardware) + Atlas Manipulation Test Stand data + TRI Ramen data + simulation as a co-training source. The disturbance recovery is explicitly noted as data-driven: "**by demonstrating examples of the robot recovering from such disturbances and retraining**" — i.e., they teleop the recovery and retrain.
- **What they credit for robustness.** Diffusion + flow-matching, multi-task multi-embodiment co-training, **explicit teleoperated recovery demonstrations as data**, sim co-training. Inference-time speedup (1.5-2x) without retraining.
- **Verifiability.** Medium. The TRI LBM paper ([arXiv:2507.05331](https://arxiv.org/abs/2507.05331)) is public — 1700 hours total: 468h bimanual teleop + 45h sim teleop + 32h UMI + ~1150h Open X-Embodiment internet data. ViT vision-language encoder + transformer denoising head with AdaLN. The TRI paper reports **3-5× sample efficiency vs single-task on hard new tasks** and 1800 real rollouts + 47k sim rollouts. The Atlas-specific demo numbers are not paperized.

#### Toyota Research Institute — Large Behavior Models (standalone)

- **What was shown.** TRI's "[Careful Examination](https://toyotaresearchinstitute.github.io/lbm1/)" paper (CoRL 2025, [arXiv:2507.05331](https://arxiv.org/abs/2507.05331)) — multitask diffusion policy trained on heterogeneous teleop + sim + Open-X data, evaluated on 49 simulation tasks and real bimanual tasks.
- **What's known about the method.** Same 450M Diffusion Transformer / flow-matching as Atlas. Crucially, this is the *only* entry in this section with a real paper documenting **how multitask pre-training affects robustness** — they show 3-5× less data needed for new tasks "in challenging settings requiring robustness to a variety of environmental factors."
- **Verifiability.** High by the standards of this section: paper + code repo ([lbm_eval](https://github.com/ToyotaResearchInstitute/lbm_eval)) + 1800 real rollouts + 47k sim rollouts. Still not a clean lighting/occlusion/disturbance ablation though — robustness is reported as aggregate sample efficiency, not per-perturbation success.

#### Google DeepMind — Gemini Robotics / RT-2

- **What was shown.** Gemini Robotics tech report ([arXiv:2503.20020](https://arxiv.org/abs/2503.20020)) demos visual-generalization tasks across **background, lighting, distractor objects, novel object instances**, plus instruction generalization (typos, paraphrasing, novel languages) and action generalization (object size variation). Gemini Robotics 1.5 ([tech report PDF](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1-5-Tech-Report.pdf), [arXiv:2510.03342](https://arxiv.org/abs/2510.03342)) extends this with embodied reasoning and motion transfer.
- **What's known about the method.** VLA built on Gemini 2.0 vision encoder. Trained on "thousands of hours of real-world expert robot demonstrations" plus non-action data (web docs, code, multi-modal content, embodied reasoning, VQA).
- **What they credit for robustness.** Powerful Gemini VLM backbone + diverse web pretraining + extensive real demos.
- **Verifiability.** High for the *existence* of perturbation evaluations — Gemini Robotics actually defines a 85-task generalization benchmark (20% in-dist, 28% visual, 28% instruction, 24% action). But results are reported as progress scores rather than per-perturbation success rates with statistical comparison, and the model is closed-weight.

#### Tesla Optimus

- **What was shown.** Various Tesla demos (kung fu Oct 2025, factory data collection, household chores including egg cracking and laundry folding). The Miami "Autonomy Visualized" event had a robot fall, which raised teleoperation suspicions ([Fortune](https://fortune.com/2025/12/09/tesla-optimus-robots-fall-autonomous-demonstration-elon-musk/)).
- **What's known about the method.** Modified FSD-derived end-to-end neural network. No paper, no architecture details, no data composition.
- **Verifiability.** Very low. Mix of confirmed-AI and likely-teleoperated content. No published method.

#### Covariant — Brain / RFM-1

- **What was shown.** Industrial bin-picking under "chaotic" item arrangements ([Brain](https://covariant.ai/covariant-brain/)). RFM-1 is positioned as a "ChatGPT for robots" foundation model.
- **What's known about the method.** Massive in-warehouse data scale across customers; transformer policy. Few public details.
- **Verifiability.** Low method-wise, but real-deployment-throughput numbers exist with logistics customers — distinct from a viral clip.

#### Generalist AI — GEN-0 / GEN-1

- **What was shown.** GTC 2026 demo: multi-step box-packing with paper/cardboard components requiring tight motion + force tolerances on a UR7e + MiR mobile platform ([blog](https://generalistai.com/blog/mar-24-2026-gtc-demo)). One viral clip shows a person poking the robot mid-task with a hockey stick and the policy continuing — credited as evidence of "resilience." GEN-1 reports **99% success** on kitting, t-shirt folding, vacuum servicing, box folding, phone packing, ~3× faster than reported SOTA, and 10× less task-specific data ([blog](https://generalistai.com/blog/apr-02-2026-GEN-1)).
- **What's known about the method.** Founded by Pete Florence + Karol Hausman ex-Google/DeepMind (RT-2, PaLM-E lineage). **GEN-0** ([blog](https://generalistai.com/blog/nov-04-2025-GEN-0)): 1B-10B+ params, cross-embodiment (6/7/16+ DoF), trained on **270k hours of real-world manipulation data** (no sim), growing 10k hours/week. Inference uses "Harmonic Reasoning" — async continuous-time sensing/action token streams. Encoder, action head, attention details are **not disclosed**. **GEN-1**: half a million hours of human wearable-device data (no robot data in pretrain), still no architecture detail. ~5.6 hr task-specific post-training quoted as the floor.
- **Robustness conditions claimed.** Hockey-stick disturbance during the box-pack task; "deployment on entirely new robot hardware within days without collecting task-specific training data." No systematic perturbation eval (lighting on/off, occlusion sweeps, distractor axes) reported in any of the three blog posts.
- **Verifiability.** **Low to medium.** Real-data scale claims are auditable in principle but no third-party eval exists. The 99% numbers are on cherry-picked tasks with internal eval protocols. Architecture is fully undisclosed — closer to a marketing claim than a method paper. **No academic publication.** Most directly relevant to the user's question: this is **the most extreme example of "scale of real data substitutes for sim2real entirely"** — there is *no* sim2real story at Generalist AI; sim is explicitly excluded from the pretrain mix.

#### Dyna Robotics — DYNA-1

- **What was shown.** DYNA-1 folded 800+ napkins continuously over 24 hours at **99.4% success rate** ([press release](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-1-the-first-commercial-ready-robot-foundation-model-offering-fully-autonomous-round-the-clock-dexterity-302441437.html)).
- **What's known about the method.** Foundation model framing, "zero-shot environment generalization" claim. Method details not disclosed.
- **Verifiability.** Low — single-task narrow eval. The headline number (99.4%) is on a single repetitive task in a controlled commercial setting, not a robustness claim.

#### Agility Robotics — Digit / whole-body control foundation model

- **What was shown.** Digit at NVIDIA GTC shopping for groceries; "always-on" motor cortex safety layer.
- **What's known about the method.** Two-tier — motor cortex for low-level reactive control with learned dexterous behaviors on top.
- **Verifiability.** Low; engineering blog only.

### 6.2 Academic papers with explicit robustness evaluations

Numbers, not videos.

| Paper | What it tests | Headline robustness number |
|---|---|---|
| **THE COLOSSEUM — Pumacay et al.** ([arXiv:2402.08191](https://arxiv.org/abs/2402.08191), RSS 2024) | 14 perturbation axes (light color, table color/texture, distractors, background, camera pose, object friction/mass) on 20 RLBench tasks; 5 SOTA models | **30-50% drop per single perturbation; ≥75% drop when multiple perturbations combined**. Worst offenders: distractors, target color, lighting. **Sim↔real correlation R²=0.614**. 3D baselines (PerAct) more robust than 2D (RVT). |
| **LIBERO-Plus — Fu et al.** ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626)) | 7 perturbation dimensions × 21 sub-dimensions × 5 difficulty levels; 10,030 tasks | **VLA success drops from 95% to <30% under modest perturbations**. Camera viewpoints + robot init states are the worst; lighting + background are *relatively* survivable. |
| **Eva-VLA — Liu et al.** ([arXiv:2509.18953](https://arxiv.org/abs/2509.18953)) | CMA-ES adversarial search over 3D transforms, illumination, adversarial patches on LIBERO-Long | **OpenVLA average failure rate >90% across the three variation types**. Adversarial training on the discovered worst-cases recovers some robustness. |
| **"Is OpenVLA Truly Robust?"** ([ACL/IJCNLP 2025](https://aclanthology.org/2025.ijcnlp-short.1.pdf)) | Lighting brightness/angle + neutral distractors on real OpenVLA | **OpenVLA drops 33-56% even under random (non-adversarial) perturbations; OpenVLA-OFT 21-42%**. Wrist cameras consistently rescue lighting robustness vs third-person-only baselines. |
| **Diffusion Policy — Chi et al.** ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)) | Visual occlusion (camera blocked 3s by hand), physical disturbance during fine adjustment, physical disturbance during navigation — qualitative single-episode reports | Diffusion Policy "remains on-course" through 3-second wrist-camera blockage and re-plans when the T-block is shifted mid-task. **No quantitative perturbation table.** Pushed by the demo, not the eval. |
| **OpenVLA — Kim et al.** ([arXiv:2406.09246](https://arxiv.org/abs/2406.09246)) | 29-task generalist eval across visual / motion / physical / semantic axes | **+16.5% absolute success vs RT-2-X (55B) at 7× fewer params**; +20.4% over Diffusion Policy. Qualitative: approaches correct object with distractors, recovers from insecure grasps. No per-perturbation table. |
| **RT-2 — Brohan et al.** ([arXiv:2307.15818](https://arxiv.org/abs/2307.15818), CoRL 2023) | Unseen objects/backgrounds/environments + emergent semantic generalization | The original "internet-scale pretraining helps robotics" paper. Robustness shown qualitatively in unseen real scenes; closed weights. |
| **Open X-Embodiment / RT-X — RT-X authors** ([arXiv:2310.08864](https://arxiv.org/abs/2310.08864), ICRA 2024) | 22 robots × 527 skills × 160k tasks cross-embodiment transfer | Positive cross-robot transfer at scale. Not a perturbation benchmark — a *data-diversity* paper. |
| **DeepMind RoboCat** ([arXiv:2306.11706](https://arxiv.org/abs/2306.11706), TMLR 2023) | Cross-embodiment generalist + self-improvement | Adapts to new tasks/robots with 100-1000 demos. Authors *explicitly note* eval is in "controlled lab setting with visually-similar backgrounds" and that "next-generation foundation agents" should target visual-diversity-in-the-wild. |
| **Mobile ALOHA — Fu et al.** ([arXiv:2401.02117](https://arxiv.org/abs/2401.02117)) | Co-training mobile + static ALOHA bimanual data | **Co-training boosts mobile manipulation success up to 90% over real-only** with 50 demos/task. Robustness to real disturbances not the focus — co-training data efficiency is. |
| **ALOHA Unleashed — Zhao et al.** ([arXiv:2410.13126](https://arxiv.org/abs/2410.13126)) | Diffusion-Transformer policy on bimanual ALOHA, 5 real tasks | "Filtering by episode duration enhances robustness;" non-diffusion architectures fail outright on harder tasks. |
| **Sim-and-Real Co-Training** ([arXiv:2503.24361](https://arxiv.org/abs/2503.24361)) | Simulation-augmented real BC across robot arm + humanoid | **+38% real success on average from sim co-training even with sim-real misalignment.** Already in `sim2real.md`. |
| **DROID** ([arXiv:2403.12945](https://arxiv.org/abs/2403.12945)) | 76k demos / 350 hours / 564 scenes / 50 collectors / 3 continents | **+20% policy robustness/generalization** on average over prior large-data baselines, on 6 tasks across 4 real locations. The cleanest "real data diversity → real robustness" anchor. |
| **BridgeData V2** ([arXiv:2308.12952](https://arxiv.org/abs/2308.12952), CoRL 2023) | 60k trajectories, 24 environments | Skill-level positive transfer; cross-skill data improves pick-and-place robustness on unseen tasks. |
| **AutoEval** ([arXiv:2503.24278](https://arxiv.org/abs/2503.24278)) | Autonomous real-world eval of generalist policies | Methodology paper, not a robustness number — but the scaffolding for honest robustness benchmarking. |

A reusable observation across these benchmarks: **lighting is one of the easier perturbations to survive; camera-pose shift and robot-init perturbations are the hardest**. The viral demos that emphasize lights-on/off as the "wow" axis are picking the perturbation models actually do best on.

### 6.3 The recurring recipes — what these systems share

Six things keep recurring across the systems above. Ordered by how much explanatory weight they appear to carry.

1. **Massive real teleoperation data is the dominant lever.** π₀, π₀.₅, Helix, RT-2, Mobile ALOHA, ALOHA Unleashed, Atlas LBM, DROID, BridgeData — every one of these reports 100s-1000s of hours of *real* demonstrations across 10s-100s of scenes. **Generalist AI is the extreme version: 270k hours of real manipulation data in GEN-0 and 500k hours of human-wearable data in GEN-1, with sim explicitly excluded from the pretrain mix.** This is the most boring and most consistent finding: the easiest way to be robust to real adversity is to have seen lots of real adversity in training. Sim DR is mostly a *cheap supplement*, not a substitute.
2. **Internet-scale visual pretraining.** Gemini Robotics rides Gemini 2.0; RT-2 / OpenVLA bake in PaLM/Llama VLMs; π₀.₅ explicitly credits "web data makes the biggest difference for OOD generalization;" Skild claims "internet videos" in pre-training. The VLM backbone gives "person walking in scene" / "flashlight in frame" some semantic prior even when the robot has never seen them.
3. **Diffusion / flow-matching policy heads on top of VLM features.** Atlas LBM (450M Diffusion Transformer + flow-matching), TRI LBM (same), π₀ (flow-matching action expert), ALOHA Unleashed (diffusion). Diffusion appears to make policies more *re-plannable* under disturbance — when the world shifts mid-action, the policy is more willing to abandon and re-roll a new chunk than a deterministic regressor would be. Diffusion Policy's qualitative recovery from camera blockage is the cleanest existence proof.
4. **Wrist camera ≥ third-person camera for lighting.** Multiple robustness studies (LIBERO-Plus, Is-OpenVLA-Truly-Robust, the Penn π₀ eval) find that **wrist cameras provide illumination-invariant geometric cues** because the scale of the object in the wrist view is mostly fixed. This is one of the cheapest robustness wins.
5. **Explicit recovery-data teleoperation.** Boston Dynamics is explicit about it, π₀ implicitly relies on it (the model "can recover from failures" because failures-and-recoveries are in the data distribution). If you want a policy that recovers from a person grabbing the cube, you teleoperate someone grabbing the cube and the operator handling it.
6. **High-frequency reactive control loops.** Helix's S0 at 1 kHz, Atlas LBM at 30 Hz with 1.6s chunks, π₀'s 50-step chunks. Re-planning frequently is *the* mechanism that lets a system look "robust to disturbance" — slow open-loop policies can't.

Notable absences from the recurring recipes — things the user might have expected to be central but aren't:
- **Sim domain randomization is rarely the headline.** Helix S0 uses massive sim + DR for the *low-level controller*, but the manipulation policy itself is real-data. Skild claims sim+video pretrain but post-trains on real. None of the company demos credit DR for the high-level visual robustness.
- **Foundation visual encoders alone don't carry the day.** As Sec 5.3 already noted, DINOv2 is *worst* of 23 encoders for sim2real BC. The viral-demo systems that work are not "frozen DINOv2 + small head" — they're VLA-scale models with action-conditioned training.
- **Photorealistic rendering / RCAN-style canonicalization** — basically absent from the company demos. The companies skip the sim2real visual-gap problem by using real data, not by closing the gap.

### 6.4 Implications for the user's setup

Honest reading: **the kind of "robot survives a flashlight to the face" robustness shown in those clips is not currently achievable with a single sim-distilled student policy and no large real dataset**. Every system that demonstrably does this has *thousands of hours of real teleoperation data* and a VLM-class encoder behind it. The user has neither. So it is important to be precise about which robustness target is being aimed at.

Three targets, in order of difficulty for the user's setup:

**Target A — survive *modest* visual variation on a single task** (different table color, slightly different cube color, indoor lighting at a different time of day): **achievable** with the moves already in Sec 5.10 (frozen MCR backbone + aux pose head + DR + random shift + asymmetric encoder). This is the realistic Q4 goal.

**Target B — survive *interactive* disturbances** (hand occluding the cube, person walking through, cube moved during reach): **partially achievable, but only if you teleoperate or scripted-demo the disturbances into the training distribution**. Diffusion-style action heads help substantially because they re-plan. Cheap version: in sim, randomize the cube's position mid-episode (drop it 5cm in a random direction), randomize the gripper's joint state mid-episode, occlude the wrist camera for 200ms with a random patch. This forces the student to handle "the world doesn't match the last observation" which is the underlying skill the company demos are showing.

**Target C — survive *adversarial* visual conditions** (flashlight at camera, lights flicked off, robot pushed, table moved): **not realistically achievable** in the user's regime without (a) a VLM-class pretrained backbone, (b) hours of real demonstrations on the actual robot in those conditions, or (c) both. The published robustness numbers (Eva-VLA's >90% failure under adversarial illumination, LIBERO-Plus's 95% → <30% under modest perturbations) say even the SOTA VLAs are not robust to this; the viral demos are cherry-picked. Don't aim here; aim at A then B.

The cheapest first step toward *any* of these, layered on top of Sec 5.10's plan:

1. **Wrist camera bias.** If you have wrist + env cam, weight the wrist camera more in the policy (or train two heads and average with the wrist head dominant). Multiple robustness studies show wrist cams carry illumination invariance for free.
2. **Mid-episode perturbations in sim.** Cheap and big return. Drop the cube to a new pose mid-episode, occlude one camera for ~10 frames, randomize gripper state. This teaches re-planning. This is the closest sim analog to "person grabs the object out of the gripper."
3. **Diffusion policy head over the encoder, not a deterministic MLP.** The qualitative robustness gap between Diffusion Policy and earlier IL on the T-pushing demos is the strongest argument that the *policy head matters as much as the encoder*. Cost: ~1 day to swap in a small DiT or 1D U-Net diffusion head. Sec 5.10 implicitly takes encoder gains for granted; this is the action-side equivalent.
4. **Collect 10-50 real demos on the actual robot, even if you keep training in sim.** "Sim-and-Real Co-Training" ([arXiv:2503.24361](https://arxiv.org/abs/2503.24361)) reports +38% real success from co-training on small real datasets with sim. This is the smallest dose of "real-data robustness recipe" that produces a measurable bump and is achievable in a week.
5. **Lower the bar in your own marketing.** If a follow-up paper or demo of your own claims robustness, evaluate honestly along the COLOSSEUM/LIBERO-Plus axes (lighting, distractors, camera shift, object color, background, distractor count) and report drops, not cherry-picked clips. The literature has already shown that even SOTA VLAs collapse under these axes; a sim-distilled student getting 30-40% drop instead of 70% is a real result.

Bottom line: the viral-demo regime and the user's regime are *different problems*. The user's correct robustness target is "the policy works on the real robot under the lighting/background/object conditions of the lab" — and Sec 5.10's plan plus the five additions above is the right move. Treating "flashlight survival" as a target leads to over-engineering for a regime that requires data the user does not have.
