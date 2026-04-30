# Hyperparameter / algorithmic experiments

Tracks PPO + env-config variations on the **same task** (`baseline_dr`).
Distinct from `rl_experiments_log.md`, which tracks DR / reward / obs / action
variations — i.e. *what the policy is learning*. This file tracks *how* it
learns.

For each experiment: only one knob differs from the reference (single-axis
variation) so robustness/regression on each lever is causally attributable.

## Reference: `baseline_dr` (unchanged config)

- Task: `Mjlab-Open-Drawer-Osc-Kinova-BaselineDr` (Phase 1 init-pose floor +
  Phase 2 drawer DR + curriculum widening of drawer cube and init-pose ranges).
- See `rl_experiments_log.md` for the env/DR side; this file's "reference" is
  the algorithmic config below, with everything inherited from
  `kinova_ppo_runner_cfg`:

  | Field | Value |
  |---|---|
  | hidden_dims (actor & critic) | (512, 256, 128) |
  | activation | elu |
  | distribution | GaussianDistribution, init_std=1.0, scalar |
  | obs_normalization | True |
  | value_loss_coef | 1.0 |
  | use_clipped_value_loss | True |
  | clip_param | 0.2 |
  | entropy_coef | 0.005 |
  | num_learning_epochs | 5 |
  | num_mini_batches | 4 |
  | learning_rate | 1e-3 (adaptive schedule) |
  | gamma | 0.99 |
  | lam | 0.95 |
  | desired_kl | 0.01 |
  | max_grad_norm | 1.0 |
  | num_steps_per_env | 24 |
  | max_iterations | 5000 |
  | save_interval | 100 |
  | num_envs (env.scene.num-envs at launch) | 1024 |

- W&B reference run: [`cmxw5ysd`](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/cmxw5ysd)
  (run-name `baseline_dr`, started 2026-04-29 22:09).

## Experiments table

| ID | Knob varied | Value | vs reference | W&B run | Status | Best SR @ iter | Notes |
|---|---|---|---|---|---|---|---|
| ref | (none) | — | — | [cmxw5ysd](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/cmxw5ysd) | **done** | **0.657** (mean SR iter 4800-4999, model_4999.pt) | 10h35m wall, 7.6 s/iter, no NaN |
| A | `num_envs` | 8192 | 8× more parallel envs | [xzq0b2ft](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/xzq0b2ft) | running | iter 1005: 0.76 (curriculum 20% ramped) | 17.5s/iter, ETA ~24h |
| B | `num_steps_per_env` | 48 | 2× longer rollouts | [v69rmqca](https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-kinova-tasks-osc/runs/v69rmqca) | running | iter 177: 0.09 (still in warmup) | 13.5s/iter, ETA ~19h |

## Per-experiment notes

### ref — `baseline_dr` (running, see `rl_experiments_log.md` "baseline_dr" section for full details)

The reference algorithmic config + the curriculum-widening DR. Will be the
basis for comparison. SR at iter 2684 = 0.63 with curriculum at jdelta=28°,
drawer ±19cm, base ±4.6cm.

### A — `num_envs=8192`

- **Hypothesis:** With `num_envs=1024`, each PPO update sees 1024×24 = 24576
  env-step samples spread over the curriculum-widened distribution. As the
  distribution widens, the per-batch sample density on any single
  configuration drops, which can stall learning. Bumping `num_envs` to 8192
  raises the per-batch sample density 8× without changing the gradient
  algorithm.
- **Expected behavior:** Faster convergence per iter (more samples per
  update), at the cost of ~8× wall-clock per iter.
- **Risk:** GPU memory pressure. Mujoco_warp scales linearly with envs, so
  ~8× more state. May OOM on a single GPU.

### B — `num_steps_per_env=48`

- **Hypothesis:** With 24 steps/env at episode length 100, each env
  contributes ~24% of an episode per rollout. Most rollouts don't cross a
  reset, so the policy gets little terminal-reward signal per update. Doubling
  to 48 means ~half an episode per env per rollout — more rollouts cross a
  reset, more terminal `goal_precise` signal lands in each gradient.
- **Expected behavior:** Better dwell-precision plateau (higher final SR)
  due to stronger terminal-reward signal, especially under the wider
  init distribution where reaching the goal late is more common.
- **Risk:** Still well under one full episode (100 steps), so the per-env
  signal isn't dramatic. May not move the needle much vs cost.

## Decision criteria

When all three runs plateau (or hit max iterations), compare:

1. **Plateau SR** at the final ~500 iters (mean over those iters).
2. **OOD sweep robustness score** vs Phase 1 (0.746) and the ref `baseline_dr`.
3. **Wall-clock to SR=0.7** (proxy for sample efficiency).

Best variant becomes the new `baseline_dr` algorithmic config; the others get
documented here and dropped.
