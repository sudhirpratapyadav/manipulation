"""Trajectory-collecting evaluator for kinova_tasks.

Mirrors the play CLI but instead of opening a viewer it runs N parallel
envs, K trials per env, and saves every step's data + per-trial metadata
+ per-run config to two Parquet files for later analysis.

Usage:

    # Local checkpoint
    uv run eval Mjlab-Pick-Cube-Osc-Kinova \\
        --checkpoint-path logs/.../model_4999.pt \\
        --num-envs 64 --trials-per-env 16 \\
        --output-dir docs/pick_task/eval_results/run_xyz

    # W&B checkpoint
    WANDB_API_KEY=... uv run eval Mjlab-Pick-Cube-Osc-Kinova \\
        --wandb-run-path entity/project/run_id \\
        --num-envs 64 --trials-per-env 16 \\
        --output-dir docs/pick_task/eval_results/run_xyz

    # Dummy agents (no checkpoint required)
    uv run eval Mjlab-Pick-Cube-Osc-Kinova --agent zero \\
        --num-envs 16 --trials-per-env 4 \\
        --output-dir docs/pick_task/eval_results/zero_baseline

What's recorded
---------------
Three scopes (matches user's spec):

* **per-run** — task id, checkpoint provenance, RNG seed, num_envs,
  trials_per_env, code git SHA, env config snapshot. Saved as
  ``run.json``.
* **per-trial** — env_id, trial_idx, sampled init state at step 0
  (cube pose, joint pos, base pose, goal pos), the env's frozen
  startup-DR draws (cube friction/mass, fingertip frictions),
  episode length, terminal reason, terminal error, dwell-success
  (mean of step-wise indicator). Saved to ``runs.parquet``
  (1 row / trial).
* **per-step** — obs, action, raw reward + per-term reward, all
  metrics, joint state, ee pose, cube pose, gripper state, per-step
  termination flags. Saved to ``steps.parquet`` (1 row / step).

Auto-reset stays on. We let the env sample reset-time axes from the
configured ranges (cube spawn / goal / joint delta / base pose); we
just *capture* what was sampled at step 0 of each trial. Per-env
startup-DR axes (cube friction, cube mass, fingertip frictions) are
sampled once at scene build and read directly from the warp model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    """CLI config for `eval`."""

    output_dir: str
    """Directory to write run.json + runs.parquet + steps.parquet into."""

    num_envs: int = 64
    """Parallel sim envs. Each env is one frozen DR draw."""

    trials_per_env: int = 16
    """Episodes (trials) per env. Total trials = num_envs * trials_per_env."""

    agent: Literal["policy", "zero", "random"] = "policy"
    """`policy` loads from checkpoint; `zero`/`random` need no checkpoint."""

    wandb_run_path: str | None = None
    """W&B run path (entity/project/run_id) — checkpoint is downloaded."""

    checkpoint_path: str | None = None
    """Local path to model_*.pt. Mutually exclusive with --wandb-run-path."""

    seed: int = 0
    """RNG seed. Affects torch + python random for the policy/dummy agent;
    env DR draws come from the env's own RNG which is currently independent."""

    device: str | None = None
    """`cuda:0` if available else `cpu`. Override to pin a specific GPU."""

    record_obs: bool = True
    """If False, skip recording the 33-D obs tensor per step (saves disk)."""


# ---------------------------------------------------------------------------
# Resolve checkpoint (file > wandb)
# ---------------------------------------------------------------------------


def _resolve_checkpoint(cfg: EvalConfig, experiment_name: str) -> Path:
    if cfg.agent in {"zero", "random"}:
        raise RuntimeError(
            "Internal: _resolve_checkpoint should not be called for dummy agents."
        )
    if cfg.checkpoint_path is not None and cfg.wandb_run_path is not None:
        raise ValueError(
            "Pass exactly one of --checkpoint-path / --wandb-run-path, not both."
        )
    if cfg.checkpoint_path is not None:
        p = Path(cfg.checkpoint_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    if cfg.wandb_run_path is None:
        raise ValueError(
            "For --agent policy, pass --checkpoint-path or --wandb-run-path."
        )
    log_root = (Path("logs") / "rsl_rl" / experiment_name).resolve()
    resume_path, was_cached = get_wandb_checkpoint_path(
        log_root, Path(cfg.wandb_run_path)
    )
    print(
        f"[INFO] Loaded checkpoint {resume_path.name} "
        f"({'cached' if was_cached else 'downloaded'})"
    )
    return resume_path


# ---------------------------------------------------------------------------
# Build agent
# ---------------------------------------------------------------------------


def _build_policy(cfg: EvalConfig, env, agent_cfg, task_id: str, device: str):
    if cfg.agent == "zero":
        action_shape = env.unwrapped.action_space.shape

        def policy_zero(obs):  # noqa: ARG001
            return torch.zeros(action_shape, device=device)
        return policy_zero, None

    if cfg.agent == "random":
        action_shape = env.unwrapped.action_space.shape

        def policy_random(obs):  # noqa: ARG001
            return 2.0 * torch.rand(action_shape, device=device) - 1.0
        return policy_random, None

    # policy
    resume_path = _resolve_checkpoint(cfg, agent_cfg.experiment_name)
    # Prefer task-registered runner if any; else MjlabOnPolicyRunner (which
    # strips cnn_cfg=None so MLPModel constructs cleanly with the kinova
    # task's agent_cfg).
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    return runner.get_inference_policy(device=device), str(resume_path)


# ---------------------------------------------------------------------------
# Read per-env startup-DR draws
# ---------------------------------------------------------------------------


def _read_dr_draws(env: ManagerBasedRlEnv) -> dict[str, np.ndarray]:
    """Per-env values of the startup-DR'd fields, frozen for the eval.

    Reads warp-side (per-env) arrays of geom_friction and body_mass and
    pulls out the entries that correspond to the cube and fingertip pads.
    """
    sim = env.sim
    mj = sim.mj_model
    wp = sim.wp_model

    geom_names = [mj.geom(i).name for i in range(mj.ngeom)]
    body_names = [mj.body(i).name for i in range(mj.nbody)]

    def _gidx(suffix: str) -> int | None:
        for i, n in enumerate(geom_names):
            if n.endswith(suffix):
                return i
        return None

    def _bidx(suffix: str) -> int | None:
        for i, n in enumerate(body_names):
            if n.endswith(suffix):
                return i
        return None

    # warp arrays: geom_friction (num_envs, ngeom) of vec3f; body_mass (num_envs, nbody)
    gf = wp.geom_friction.numpy()  # (E, G, 3)
    bm = wp.body_mass.numpy()      # (E, B)

    out: dict[str, np.ndarray] = {}
    cube_g = _gidx("cube_geom")
    if cube_g is not None:
        out["dr_object_friction_slide"] = gf[:, cube_g, 0].astype(np.float32)
        out["dr_object_friction_spin"] = gf[:, cube_g, 1].astype(np.float32)
        out["dr_object_friction_roll"] = gf[:, cube_g, 2].astype(np.float32)
    cube_b = _bidx("cube")
    if cube_b is not None:
        out["dr_object_mass"] = bm[:, cube_b].astype(np.float32)

    # Fingertip pads: 4 geoms (left/right pad1/pad2). Friction was DR'd
    # with the same range across all 4; we read pad1 of each side as
    # representative.
    for side in ("right", "left"):
        gi = _gidx(f"{side}_pad1")
        if gi is not None:
            out[f"dr_{side}_pad_friction_slide"] = gf[:, gi, 0].astype(np.float32)
            out[f"dr_{side}_pad_friction_spin"] = gf[:, gi, 1].astype(np.float32)
            out[f"dr_{side}_pad_friction_roll"] = gf[:, gi, 2].astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Per-step recording — extract everything we want as float32 numpy
# ---------------------------------------------------------------------------


def _read_step_data(env: ManagerBasedRlEnv, action_t: torch.Tensor) -> dict[str, np.ndarray]:
    """Snapshot env state into numpy. Called *after* env.step()."""
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    site_ids = robot.find_sites("pinch_site")[0]

    joint_pos = robot.data.joint_pos[:, :7].detach().cpu().numpy()
    joint_vel = robot.data.joint_vel[:, :7].detach().cpu().numpy()

    ee_pos_w = robot.data.site_pos_w[:, site_ids].squeeze(1).detach().cpu().numpy()
    ee_quat_w = robot.data.site_quat_w[:, site_ids].squeeze(1).detach().cpu().numpy()
    cube_pos_w = cube.data.root_link_pos_w.detach().cpu().numpy()
    cube_quat_w = cube.data.root_link_quat_w.detach().cpu().numpy()

    # gripper state — same definition as the obs term
    rd_idx = robot.find_joints("right_driver_joint")[0]
    gripper = (robot.data.joint_pos[:, rd_idx] / 0.8).detach().cpu().numpy()
    if gripper.ndim == 2:
        gripper = gripper.squeeze(-1)

    # rewards: per-term and total
    rm = env.reward_manager
    step_rewards = rm._step_reward.detach().cpu().numpy()  # (E, num_terms)

    # metrics: latest per-step values (object_to_goal_error etc.)
    mm = env.metrics_manager
    metrics: dict[str, np.ndarray] = {}
    for term_name in mm.active_terms:
        idx = mm._term_names.index(term_name)
        v = mm._step_values[:, idx].detach().cpu().numpy()
        metrics[f"metric_{term_name}"] = v

    # termination flags
    tm = env.termination_manager
    term_flags: dict[str, np.ndarray] = {}
    for term_name in tm.active_terms:
        v = tm.get_term(term_name).detach().cpu().numpy()
        term_flags[f"term_{term_name}"] = v

    return {
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "ee_pos_w": ee_pos_w.astype(np.float32),
        "ee_quat_w": ee_quat_w.astype(np.float32),
        "cube_pos_w": cube_pos_w.astype(np.float32),
        "cube_quat_w": cube_quat_w.astype(np.float32),
        "gripper_state": gripper.astype(np.float32),
        "step_rewards_per_term": step_rewards.astype(np.float32),
        "action": action_t.detach().cpu().numpy().astype(np.float32),
        **metrics,
        **term_flags,
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def run_eval(task_id: str, cfg: EvalConfig) -> None:
    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    # Episode length is finite under play=True iff we override it; play=True
    # in pick_cube_osc sets episode_length_s=1e9, which we DON'T want for
    # eval — we need natural time-outs to delimit trials. Restore the
    # training episode length.
    env_cfg.episode_length_s = 10.0
    env_cfg.observations["actor"].enable_corruption = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    policy, ckpt_path = _build_policy(cfg, wrapped, agent_cfg, task_id, device)

    # Per-env DR draws: read once after build (frozen for the run).
    dr_draws = _read_dr_draws(env)

    # Episode horizon in env steps.
    dt_outer = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    horizon = int(round(env_cfg.episode_length_s / dt_outer))
    print(
        f"[INFO] num_envs={cfg.num_envs}, trials_per_env={cfg.trials_per_env}, "
        f"horizon={horizon} steps, total trials={cfg.num_envs * cfg.trials_per_env}"
    )

    # State.
    obs, _ = wrapped.reset()
    actor_obs = obs["actor"]
    obs_dim = actor_obs.shape[-1] if cfg.record_obs else None
    action_dim = wrapped.unwrapped.action_space.shape[-1]
    num_envs = cfg.num_envs
    num_trials_total = num_envs * cfg.trials_per_env

    # Per-env trial counter; eval done once every env hits trials_per_env.
    trial_idx_per_env = np.zeros(num_envs, dtype=np.int64)
    # Step counter within each env's current trial.
    step_in_trial = np.zeros(num_envs, dtype=np.int64)
    # Trial-id assigned in order of trial start. envs share a global trial
    # counter so we get unique global trial_ids.
    global_trial_id = np.arange(num_envs, dtype=np.int64).copy()
    next_global_id = num_envs

    # Per-env "init snapshot" — captured at step 0 of each trial. We update
    # this whenever a trial starts (at reset or auto-reset on time-out).
    init_snapshot = _read_step_data(env, torch.zeros(num_envs, action_dim, device=device))

    # Buffers — fill incrementally.
    # Per-step rows: built as a list of dicts, converted to DataFrame at end.
    step_rows: list[dict] = []
    # Per-trial rows similarly.
    trial_rows: list[dict] = []

    # We need to track success-dwell (mean of step-wise indicator) per
    # trial — accumulate during episode.
    success_threshold = 0.02
    dwell_running_sum = np.zeros(num_envs, dtype=np.float64)
    init_state_per_env: list[dict] = [{} for _ in range(num_envs)]
    for ei in range(num_envs):
        init_state_per_env[ei] = {k: v[ei].copy() for k, v in init_snapshot.items()}

    # Active mask: env that still has more trials to run. Once all trials
    # for an env are done we stop *recording* its steps (we still step the
    # env in lockstep — mjlab is vectorized — but skip its rows).
    env_active = np.ones(num_envs, dtype=bool)

    completed_trials = 0
    while completed_trials < num_trials_total:
        with torch.inference_mode():
            action = policy(obs)  # obs is the TensorDict; policy handles it
        # mjlab dummy agents return torch tensors of the right shape; trained
        # policy returns (E, A). Make sure clip_actions matches what agent_cfg
        # specifies (already handled by wrapper).

        # Step environment. RslRlVecEnvWrapper returns (obs, rew, dones, extras)
        # where dones = terminated | truncated and extras["time_outs"] = truncated.
        obs, _rew, _dones, _info = wrapped.step(action)
        actor_obs = obs["actor"]
        # The wrapper's reset behavior: time-out + termination cause auto
        # reset, so by the time we see obs back, the next-trial state is
        # already in place. We need to read step data *before* that auto
        # reset happens — but mjlab's standard auto-reset is what we want
        # to use. Workaround: read step data from env directly using the
        # post-step state, BUT identify which envs just reset via dones.

        # Done mask captures envs that just terminated this step (either
        # time_out or other termination). After termination, env_manager
        # has reset — so the data we read is the FIRST step of the new
        # trial, not the last step of the old one. To capture per-step
        # data correctly, we need to read it *before* the auto-reset.

        # Detour: read step data on the *unwrapped* env from its state
        # tensors which reflect post-physics-step state. The reset event
        # triggered by dones overwrites these mid-step. So read into
        # step_rows BEFORE checking dones (between policy's view and
        # next iteration).
        step_data = _read_step_data(env, action)

        # Per-step success indicator using metric_object_to_goal_error.
        if "metric_object_to_goal_error" in step_data:
            ind = (step_data["metric_object_to_goal_error"] < success_threshold).astype(
                np.float32
            )
        else:
            ind = np.zeros(num_envs, dtype=np.float32)
        dwell_running_sum += ind

        # Append per-step rows (only for envs still running their trial budget).
        for ei in range(num_envs):
            if not env_active[ei]:
                continue
            row: dict = {
                "trial_id": int(global_trial_id[ei]),
                "env_id": int(ei),
                "step": int(step_in_trial[ei]),
                "action": step_data["action"][ei].tolist(),
                "joint_pos": step_data["joint_pos"][ei].tolist(),
                "joint_vel": step_data["joint_vel"][ei].tolist(),
                "ee_pos_w": step_data["ee_pos_w"][ei].tolist(),
                "ee_quat_w": step_data["ee_quat_w"][ei].tolist(),
                "cube_pos_w": step_data["cube_pos_w"][ei].tolist(),
                "cube_quat_w": step_data["cube_quat_w"][ei].tolist(),
                "gripper_state": float(step_data["gripper_state"][ei]),
                "reward": float(step_data["step_rewards_per_term"][ei].sum()),
                "success_indicator": float(ind[ei]),
            }
            # per-term rewards
            for ti, tname in enumerate(env.reward_manager.active_terms):
                row[f"reward_{tname}"] = float(
                    step_data["step_rewards_per_term"][ei, ti]
                )
            # metrics + term flags
            for k, v in step_data.items():
                if k.startswith("metric_") or k.startswith("term_"):
                    row[k] = float(v[ei]) if v.dtype != np.bool_ else bool(v[ei])
            if cfg.record_obs:
                row["obs"] = actor_obs[ei].detach().cpu().numpy().astype(np.float32).tolist()
            step_rows.append(row)

        step_in_trial += 1

        # Detect trial completion: any termination flag true OR step_in_trial
        # >= horizon. We treat *any* termination as end-of-trial. Because
        # the env auto-resets, the very next step is already step 0 of the
        # next trial — we need to capture init state *now* (post-step
        # auto-reset has already happened by the time we observe the
        # returned obs).
        any_done = _dones.detach().cpu().numpy().astype(bool)
        # Only count "done" for envs still inside their trial budget.
        any_done = any_done & env_active

        for ei in np.where(any_done)[0]:
            # Decide terminal reason from the per-step term flags we
            # captured this step.
            terminal_reason = "time_out"
            for k in step_data:
                if k.startswith("term_") and k != "term_time_out":
                    if bool(step_data[k][ei]):
                        terminal_reason = k.replace("term_", "")
                        break

            # Episode length = step_in_trial (already incremented this
            # iteration), terminal is the step we just recorded.
            ep_len = int(step_in_trial[ei])
            terminal_err = float(
                step_data.get(
                    "metric_object_to_goal_error", np.full(num_envs, np.nan)
                )[ei]
            )
            success_terminal = bool(terminal_err < success_threshold)
            success_dwell = float(dwell_running_sum[ei] / max(1, ep_len))

            # Pull init snapshot for this env from when this trial started.
            init = init_state_per_env[ei]

            trial_row = {
                "trial_id": int(global_trial_id[ei]),
                "env_id": int(ei),
                "trial_idx_in_env": int(trial_idx_per_env[ei]),
                "episode_length": ep_len,
                "terminal_reason": terminal_reason,
                "terminal_object_to_goal_error": terminal_err,
                "success_terminal": success_terminal,
                "success_dwell": success_dwell,
                # frozen DR draws (constant across env's life)
                **{k: float(v[ei]) for k, v in dr_draws.items()},
                # init state of THIS trial (sampled by the env's reset event)
                "init_joint_pos": init["joint_pos"].tolist(),
                "init_ee_pos_w": init["ee_pos_w"].tolist(),
                "init_ee_quat_w": init["ee_quat_w"].tolist(),
                "init_cube_pos_w": init["cube_pos_w"].tolist(),
                "init_cube_quat_w": init["cube_quat_w"].tolist(),
                "init_gripper_state": float(init["gripper_state"]),
            }
            # init goal pos: read from command
            try:
                gp = env.command_manager.get_term("object_goal").goal_pos[ei]
                origin = env.scene.env_origins[ei]
                trial_row["init_goal_pos_local"] = (gp - origin).detach().cpu().numpy().tolist()
            except Exception:
                pass

            trial_rows.append(trial_row)

            # Reset for next trial in this env.
            trial_idx_per_env[ei] += 1
            step_in_trial[ei] = 0
            dwell_running_sum[ei] = 0.0
            completed_trials += 1
            if trial_idx_per_env[ei] < cfg.trials_per_env:
                # Assign a new global trial id for the next trial on this env.
                global_trial_id[ei] = next_global_id
                next_global_id += 1
            else:
                # This env has finished its budget. Stop recording its rows.
                env_active[ei] = False

            if completed_trials % max(1, num_trials_total // 20) == 0:
                print(
                    f"[INFO] {completed_trials}/{num_trials_total} trials complete"
                )

        # Capture init snapshot for env that *just* reset (next iter's step 0
        # already happened on those envs). The data we're about to record
        # next iteration is the first step of the new trial; the state
        # right now reflects the reset.
        if any_done.any():
            new_snap = _read_step_data(env, torch.zeros(num_envs, action_dim, device=device))
            for ei in np.where(any_done)[0]:
                if trial_idx_per_env[ei] < cfg.trials_per_env:
                    init_state_per_env[ei] = {k: v[ei].copy() for k, v in new_snap.items()}

        # Stop once every env has run trials_per_env. completed_trials counter
        # handles this; while-loop condition exits.

    # ----- save -----
    out_dir = Path(cfg.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # run.json
    git_sha = _git_sha()
    run_meta = {
        "task_id": task_id,
        "checkpoint_path": ckpt_path,
        "wandb_run_path": cfg.wandb_run_path,
        "agent": cfg.agent,
        "num_envs": cfg.num_envs,
        "trials_per_env": cfg.trials_per_env,
        "total_trials": num_trials_total,
        "horizon_steps": horizon,
        "episode_length_s": env_cfg.episode_length_s,
        "decimation": env_cfg.decimation,
        "physics_dt_s": env_cfg.sim.mujoco.timestep,
        "seed": cfg.seed,
        "device": device,
        "git_sha": git_sha,
        "success_threshold_m": success_threshold,
        "reward_term_names": list(env.reward_manager.active_terms),
        "metric_term_names": list(env.metrics_manager.active_terms),
        "termination_term_names": list(env.termination_manager.active_terms),
        "obs_dim": int(obs_dim) if obs_dim is not None else None,
        "action_dim": int(action_dim),
        "dr_axes": list(dr_draws.keys()),
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2))

    print(f"[INFO] writing {len(trial_rows)} trials, {len(step_rows)} steps to {out_dir}")
    pd.DataFrame(trial_rows).to_parquet(out_dir / "runs.parquet", index=False)
    pd.DataFrame(step_rows).to_parquet(out_dir / "steps.parquet", index=False)

    # Quick summary.
    df = pd.DataFrame(trial_rows)
    print()
    print(f"--- Eval summary ({task_id}) ---")
    print(f"  trials                 : {len(df)}")
    print(f"  success_terminal mean  : {df['success_terminal'].mean():.3f}")
    print(f"  success_dwell mean     : {df['success_dwell'].mean():.3f}")
    print(f"  terminal_error mean    : {df['terminal_object_to_goal_error'].mean():.4f} m")
    print(f"  episode_length mean    : {df['episode_length'].mean():.1f}")
    print()
    print(f"  written: {out_dir}/run.json")
    print(f"           {out_dir}/runs.parquet")
    print(f"           {out_dir}/steps.parquet")

    wrapped.close()


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    import mjlab.tasks  # noqa: F401 — populate task registry
    import kinova_tasks  # noqa: F401 — register kinova_tasks too

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        args=sys.argv[1:],
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        EvalConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_eval(chosen_task, cfg)


if __name__ == "__main__":
    main()
