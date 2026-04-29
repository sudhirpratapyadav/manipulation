"""End-to-end check that ``eval_sweep._run_setting`` returns numbers that match
a hand-computed ground truth on the same checkpoint and the same env.

Procedure:
1. Build the eval env at ``num_envs=16`` with the **nominal** override (so the
   env is identical to training nominal).
2. Run ``_run_setting`` on this env via the harness — this is what Lane A
   computes per (axis, value).
3. Independently run the same checkpoint and the same env manually for the
   same number of episodes, recording terminal ``object_to_goal_error`` per
   terminated episode and computing success_rate the same way.
4. Print both numbers and PASS/FAIL on |diff| < 0.05.

This complements the override-plumbing smoke test: it validates the *rollout
loop* end-to-end (vec-env reset bookkeeping, terminal-error attribution, max
steps termination, episode counting). See ``docs/AGENT.md`` § "Resume
sequence" step 3, second bullet.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab  # noqa: F401
import kinova_tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from kinova_tasks.eval_sweep import (
    EVAL_TASK_ID,
    SUCCESS_THRESHOLD_M,
    TRAIN_TASK_ID,
    _object_to_goal_error_per_env,
    _run_setting,
    default_axes,
)


def _hand_rollout(
    cfg: ManagerBasedRlEnvCfg,
    checkpoint_path: Path,
    num_envs: int,
    target_episodes: int,
    device: str,
    agent_cfg_dict: dict,
) -> tuple[float, int]:
    """Independent rollout: returns (success_rate, n_episodes_collected)."""
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = MjlabOnPolicyRunner(wrapped, dict(agent_cfg_dict), device=device)
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True,
                map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs, _ = wrapped.reset()
    successes: list[float] = []
    max_steps = int(env.max_episode_length) * (
        math.ceil(target_episodes / num_envs) + 2
    )
    for _ in range(max_steps):
        with torch.no_grad():
            actions = policy(obs)
        prev_err = _object_to_goal_error_per_env(env)
        obs, _r, dones, _i = wrapped.step(actions)
        idx = torch.nonzero(dones, as_tuple=False).flatten()
        if idx.numel() > 0:
            for i in idx.tolist():
                err_i = float(prev_err[i].item())
                successes.append(1.0 if err_i < SUCCESS_THRESHOLD_M else 0.0)
            if len(successes) >= target_episodes:
                break
    env.close()
    n = len(successes)
    sr = sum(successes) / n if n > 0 else float("nan")
    return sr, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--axis", default="goal_depth",
                        help="Axis whose nominal value to use (override is identity).")
    args = parser.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint).resolve()

    agent_cfg = load_rl_cfg(TRAIN_TASK_ID)
    agent_cfg_dict = asdict(agent_cfg)

    def factory() -> ManagerBasedRlEnvCfg:
        return load_env_cfg(EVAL_TASK_ID, play=True)

    axes = {a.name: a for a in default_axes()}
    axis = axes[args.axis]
    nominal_value = axis.nominal

    print(f"[handcheck] checkpoint={checkpoint_path}")
    print(f"[handcheck] axis={axis.name} value={nominal_value}  (nominal == identity override)")
    print(f"[handcheck] num_envs={args.num_envs}  target_eps={args.episodes}")

    import copy
    print("[handcheck] running harness …")
    sr_h, mean_err_h, mean_len_h, n_h = _run_setting(
        factory, axis.apply, nominal_value, checkpoint_path,
        args.num_envs, args.episodes, device, copy.deepcopy(agent_cfg_dict),
    )
    print(f"  harness: success_rate={sr_h:.3f}  mean_err={mean_err_h:.3f}  "
          f"mean_len={mean_len_h:.1f}  n={n_h}")

    print("[handcheck] running independent rollout …")
    cfg = factory()
    axis.apply(cfg, nominal_value)
    sr_g, n_g = _hand_rollout(
        cfg, checkpoint_path, args.num_envs, args.episodes,
        device, copy.deepcopy(agent_cfg_dict),
    )
    print(f"  hand:    success_rate={sr_g:.3f}  n={n_g}")

    diff = abs(sr_h - sr_g)
    tol = 0.05
    status = "OK" if diff < tol else "FAIL"
    print(f"[handcheck] |Δ success_rate| = {diff:.3f}  (tolerance {tol})  → {status}")


if __name__ == "__main__":
    main()
