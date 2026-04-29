"""OOD evaluation sweep driver for the open-drawer OSC task.

Per-axis sweeps that find the policy's breaking point along each axis
listed in docs/open_drawer_improvement_plan.md §7.2.

Usage (run from /ihub/homedirs/svs_ald/sudhir/manipulation):

    uv run python -m kinova_tasks.eval_sweep \
        --checkpoint-file logs/rsl_rl/open_drawer_osc_phase0/<run>/model_4900.pt \
        --output-dir docs/results/open_drawer_osc_phase0 \
        --num-envs 64 --episodes-per-setting 64

Each axis is swept independently (others held at training nominal). For each
setting we run ``episodes_per_setting`` episodes and record:
- ``success_rate`` — fraction of episodes whose final step (just before
  termination) has ``object_to_goal_error < threshold`` (default 2 cm).
- ``mean_error`` — mean final-step ``object_to_goal_error`` across episodes.
- ``mean_episode_length`` — mean steps until termination.

Results land in ``output_dir/sweep_summary.csv`` (one row per (axis, value))
plus ``breaking_points.md`` with the per-axis envelope summary.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch
import tyro

import mjlab  # noqa: F401  (sets up TYRO_FLAGS / warp config)
import kinova_tasks  # noqa: F401  (registers task IDs)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


EVAL_TASK_ID = "Mjlab-Open-Drawer-Osc-Kinova-Eval"
TRAIN_TASK_ID = "Mjlab-Open-Drawer-Osc-Kinova"
SUCCESS_THRESHOLD_M = 0.02

# Pass/degraded/fail thresholds on success_rate.
PASS_THRESHOLD = 0.80
DEGRADED_THRESHOLD = 0.50

# Drawer entity slide joint name (must match XML).
_DRAWER_SLIDE_JOINT = "drawer_slide"


# ---------------------------------------------------------------------------
# Sweep axis definitions
# ---------------------------------------------------------------------------


@dataclass
class SweepAxis:
    """One axis of variation to sweep."""

    name: str
    values: list[float]
    # Apply this override to the env_cfg before building the env.
    apply: Callable[[ManagerBasedRlEnvCfg, float], None]
    nominal: float
    units: str = ""


def _override_drawer_slide_friction(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    """Set drawer-slide joint frictionloss via a startup DR event."""
    import mjlab.envs.mdp as mdp

    cfg.events["sweep_drawer_friction"] = EventTermCfg(
        mode="startup",
        func=mdp.dr.dof_frictionloss,
        params={
            "asset_cfg": SceneEntityCfg("drawer", joint_names=(_DRAWER_SLIDE_JOINT,)),
            "operation": "abs",
            "distribution": "uniform",
            "ranges": (value, value),  # deterministic
        },
    )


def _override_drawer_slide_damping(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    import mjlab.envs.mdp as mdp

    cfg.events["sweep_drawer_damping"] = EventTermCfg(
        mode="startup",
        func=mdp.dr.dof_damping,
        params={
            "asset_cfg": SceneEntityCfg("drawer", joint_names=(_DRAWER_SLIDE_JOINT,)),
            "operation": "abs",
            "distribution": "uniform",
            "ranges": (value, value),
        },
    )


def _override_drawer_base_mass_scale(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    """Scale drawer base mass + inertia by ``value`` via pseudo_inertia(α=ln(scale)/2)."""
    import math as _math

    import mjlab.envs.mdp as mdp

    if value <= 0:
        raise ValueError("mass scale must be > 0")
    alpha = 0.5 * _math.log(value)
    cfg.events["sweep_drawer_mass"] = EventTermCfg(
        mode="startup",
        func=mdp.dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("drawer", body_names=("drawer_base",)),
            "alpha_range": (alpha, alpha),
            "distribution": "uniform",
        },
    )


def _override_goal_depth(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    """Force the drawer goal slide to a fixed value (deterministic)."""
    cmd_cfg = cfg.commands["drawer_goal"]
    cmd_cfg.slide_lo = value
    cmd_cfg.slide_hi = value


def _override_init_slide(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    """Start the drawer at a fixed slide value (partially open)."""
    reset = cfg.events["reset_drawer"]
    reset.params["init_slide_range"] = (value, value)


def _override_robot_base_xy(cfg: ManagerBasedRlEnvCfg, value_cm: float) -> None:
    """Robot base offset along a fixed direction (here: +x). Magnitude in cm."""
    m = value_cm * 0.01
    reset = cfg.events["reset_base"]
    reset.params["pose_range"] = {"x": (m, m)}  # deterministic shift along +x


def _override_init_joint_delta_deg(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    """Increase per-joint init delta (degrees)."""
    reset = cfg.events["reset_robot_joints"]
    reset.params["joint_delta_deg"] = value


def _override_action_scale(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    """Scale OSC delta_pos_scale (m/step)."""
    cfg.actions["osc_pose"].delta_pos_scale = value


def _override_arm_link_mass_pct(cfg: ManagerBasedRlEnvCfg, pct: float) -> None:
    """Perturb arm link masses by ±pct via pseudo_inertia (mass+inertia consistent)."""
    import math as _math

    import mjlab.envs.mdp as mdp

    if pct == 0.0:
        return
    # mass scale = (1 ± pct/100) → α = ln(scale)/2; symmetric about 0.
    alpha_max = 0.5 * _math.log(1.0 + pct / 100.0)
    cfg.events["sweep_arm_link_mass"] = EventTermCfg(
        mode="startup",
        func=mdp.dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=r".*_link"),
            "alpha_range": (-alpha_max, alpha_max),
            "distribution": "uniform",
        },
    )


def _override_fingertip_friction(cfg: ManagerBasedRlEnvCfg, value: float) -> None:
    """Override fingertip slide friction to a fixed value (replaces training DR)."""
    import mjlab.envs.mdp as mdp

    cfg.events["fingertip_friction_slide"] = EventTermCfg(
        mode="startup",
        func=mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=r"(left|right)_pad[12]"),
            "operation": "abs",
            "distribution": "uniform",
            "axes": [0],
            "ranges": (value, value),
        },
    )


def default_axes() -> list[SweepAxis]:
    """Sweep axes defined in plan §7.2."""
    return [
        SweepAxis(
            name="drawer_slide_friction",
            # nominal frictionloss is small (XML default ~0.01 — treat as nominal=0.01)
            values=[0.0025, 0.005, 0.01, 0.02, 0.04],
            apply=_override_drawer_slide_friction,
            nominal=0.01,
            units="N",
        ),
        SweepAxis(
            name="drawer_slide_damping",
            values=[0.25, 0.5, 1.0, 2.0, 4.0],
            apply=_override_drawer_slide_damping,
            nominal=1.0,
            units="N·s/m",
        ),
        SweepAxis(
            name="drawer_base_mass_scale",
            values=[0.5, 1.0, 2.0, 5.0],
            apply=_override_drawer_base_mass_scale,
            nominal=1.0,
            units="× nominal",
        ),
        SweepAxis(
            name="goal_depth",
            values=[-0.10, -0.15, -0.20, -0.25, -0.28],
            apply=_override_goal_depth,
            nominal=-0.20,
            units="m",
        ),
        SweepAxis(
            name="init_slide",
            values=[0.0, -0.05, -0.10],
            apply=_override_init_slide,
            nominal=0.0,
            units="m",
        ),
        SweepAxis(
            name="robot_base_x_offset",
            values=[0.0, 2.0, 5.0, 10.0],
            apply=_override_robot_base_xy,
            nominal=0.0,
            units="cm",
        ),
        SweepAxis(
            name="init_joint_delta_deg",
            values=[5.0, 10.0, 20.0, 30.0, 45.0],
            apply=_override_init_joint_delta_deg,
            nominal=5.0,
            units="deg",
        ),
        SweepAxis(
            name="action_scale",
            values=[0.005, 0.01, 0.02, 0.05],
            apply=_override_action_scale,
            nominal=0.01,
            units="m",
        ),
        SweepAxis(
            name="arm_link_mass_pct",
            values=[0.0, 10.0, 25.0, 50.0],
            apply=_override_arm_link_mass_pct,
            nominal=0.0,
            units="±%",
        ),
        SweepAxis(
            name="fingertip_friction_slide",
            values=[0.1, 0.3, 0.6, 1.0, 1.5],
            apply=_override_fingertip_friction,
            nominal=0.6,
            units="μ",
        ),
    ]


# ---------------------------------------------------------------------------
# Eval one (axis, value) setting
# ---------------------------------------------------------------------------


@dataclass
class SettingResult:
    axis: str
    value: float
    success_rate: float
    mean_error: float
    mean_episode_length: float
    n_episodes: int

    def status(self) -> str:
        if self.success_rate >= PASS_THRESHOLD:
            return "pass"
        if self.success_rate >= DEGRADED_THRESHOLD:
            return "degraded"
        return "fail"


def _build_env(cfg: ManagerBasedRlEnvCfg, num_envs: int, device: str) -> ManagerBasedRlEnv:
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def _object_to_goal_error_per_env(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Recompute the metric per-env directly (avoids relying on extras['log'])."""
    from kinova_tasks.tasks.open_drawer_osc import object_to_goal_error

    return object_to_goal_error(env, "drawer", "drawer_goal")


def _run_setting(
    base_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
    apply_override: Callable[[ManagerBasedRlEnvCfg, float], None],
    value: float,
    checkpoint_path: Path,
    num_envs: int,
    episodes_per_setting: int,
    device: str,
    agent_cfg_dict: dict,
) -> tuple[float, float, float, int]:
    """Build a fresh env with the override, roll out, return (success_rate, mean_err, mean_len, n_eps)."""
    cfg = base_cfg_factory()
    apply_override(cfg, value)

    env = _build_env(cfg, num_envs=num_envs, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    # Deep-copy: the runner mutates inner dicts (pops `class_name` etc.), so a
    # shallow copy would crash the 2nd setting in the loop.
    import copy as _copy
    runner = MjlabOnPolicyRunner(wrapped, _copy.deepcopy(agent_cfg_dict), device=device)
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Per-env episode bookkeeping.
    successes: list[float] = []
    errors: list[float] = []
    lengths: list[int] = []

    obs, _ = wrapped.reset()
    step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    target_episodes = episodes_per_setting
    # Hard cap on steps to avoid runaway when the policy never terminates an episode.
    max_steps = int(env.max_episode_length) * (math.ceil(target_episodes / num_envs) + 2)

    for _ in range(max_steps):
        with torch.no_grad():
            actions = policy(obs)
        # Snapshot pre-step error so we can sample at terminal step.
        prev_err = _object_to_goal_error_per_env(env)
        obs, _rew, dones, info = wrapped.step(actions)
        step_count += 1

        # `dones` is a bool tensor of shape (num_envs,). Anything non-zero = episode ended.
        done_idx = torch.nonzero(dones, as_tuple=False).flatten()
        if done_idx.numel() > 0:
            # Use the metric value computed *before* this step's reset (env.step
            # auto-resets terminated envs inside vecenv wrapper).
            for i in done_idx.tolist():
                err_i = float(prev_err[i].item())
                successes.append(1.0 if err_i < SUCCESS_THRESHOLD_M else 0.0)
                errors.append(err_i)
                lengths.append(int(step_count[i].item()))
            step_count[done_idx] = 0
            if len(successes) >= target_episodes:
                break

    env.close()

    n = len(successes)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        sum(successes) / n,
        sum(errors) / n,
        sum(lengths) / n,
        n,
    )


# ---------------------------------------------------------------------------
# Envelope width
# ---------------------------------------------------------------------------


def _envelope_width(axis: SweepAxis, results: list[SettingResult]) -> tuple[float, list[float]]:
    """Largest contiguous sorted range around nominal where success >= PASS, normalized.

    Returns (width_normalized in [0, 1], passing_values).
    """
    by_value = sorted(results, key=lambda r: r.value)
    values = [r.value for r in by_value]
    passes = [r.success_rate >= PASS_THRESHOLD for r in by_value]

    # Find nominal index — closest swept value to the declared nominal.
    nominal_idx = min(range(len(values)), key=lambda i: abs(values[i] - axis.nominal))
    if not passes[nominal_idx]:
        # Policy fails at nominal — envelope width is 0.
        return 0.0, []

    lo = nominal_idx
    while lo - 1 >= 0 and passes[lo - 1]:
        lo -= 1
    hi = nominal_idx
    while hi + 1 < len(values) and passes[hi + 1]:
        hi += 1

    if values[-1] == values[0]:
        return 1.0, [values[i] for i in range(lo, hi + 1)]
    width_abs = values[hi] - values[lo]
    total = values[-1] - values[0]
    return width_abs / total, [values[i] for i in range(lo, hi + 1)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    checkpoint_file: str
    """Absolute or relative path to the rsl_rl checkpoint .pt file."""
    output_dir: str
    """Directory to write sweep_summary.csv and breaking_points.md into."""
    num_envs: int = 64
    episodes_per_setting: int = 64
    device: str | None = None
    only_axes: tuple[str, ...] = ()
    """If non-empty, only run these axis names."""


def main(cfg: SweepConfig | None = None) -> None:
    cfg = cfg if cfg is not None else tyro.cli(SweepConfig, config=mjlab.TYRO_FLAGS)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(cfg.checkpoint_file).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve agent cfg from the *training* task so policy architecture matches.
    agent_cfg = load_rl_cfg(TRAIN_TASK_ID)
    agent_cfg_dict = asdict(agent_cfg)

    def base_cfg_factory() -> ManagerBasedRlEnvCfg:
        return load_env_cfg(EVAL_TASK_ID, play=True)

    axes = default_axes()
    if cfg.only_axes:
        axes = [a for a in axes if a.name in cfg.only_axes]

    rows: list[SettingResult] = []
    for axis in axes:
        print(f"[sweep] axis={axis.name} values={axis.values}")
        for v in axis.values:
            sr, me, ml, n = _run_setting(
                base_cfg_factory,
                axis.apply,
                v,
                checkpoint_path,
                cfg.num_envs,
                cfg.episodes_per_setting,
                device,
                agent_cfg_dict,
            )
            r = SettingResult(
                axis=axis.name,
                value=v,
                success_rate=sr,
                mean_error=me,
                mean_episode_length=ml,
                n_episodes=n,
            )
            print(f"    value={v:.4g}  success={sr:.3f}  err={me:.4f}  n={n}  [{r.status()}]")
            rows.append(r)

    # Write CSV.
    csv_path = output_dir / "sweep_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "value", "success_rate", "mean_error_m",
                    "mean_episode_length", "n_episodes", "status"])
        for r in rows:
            w.writerow([r.axis, f"{r.value:.6g}", f"{r.success_rate:.4f}",
                        f"{r.mean_error:.6f}", f"{r.mean_episode_length:.2f}",
                        r.n_episodes, r.status()])

    # Per-axis envelope and overall robustness score.
    md_lines = ["# OOD sweep — breaking points", "",
                f"Checkpoint: `{checkpoint_path}`",
                f"`success_rate` threshold: {SUCCESS_THRESHOLD_M:.3f} m",
                f"Pass: success ≥ {PASS_THRESHOLD:.0%}, "
                f"degraded: ≥ {DEGRADED_THRESHOLD:.0%}, fail: below.",
                "",
                "| Axis | Nominal | Envelope (passes) | Width norm. | Status |",
                "|---|---|---|---|---|"]
    widths: list[float] = []
    for axis in axes:
        axis_rows = [r for r in rows if r.axis == axis.name]
        width_norm, passing = _envelope_width(axis, axis_rows)
        widths.append(width_norm)
        passing_str = (
            f"[{min(passing):.4g} .. {max(passing):.4g}]" if passing else "—"
        )
        nominal_row = next(
            (r for r in axis_rows if math.isclose(r.value, axis.nominal, rel_tol=1e-6, abs_tol=1e-9)),
            None,
        )
        nominal_status = nominal_row.status() if nominal_row else "n/a"
        md_lines.append(
            f"| {axis.name} ({axis.units}) | {axis.nominal:.4g} | {passing_str} "
            f"| {width_norm:.2f} | {nominal_status} |"
        )

    robustness = sum(widths) / len(widths) if widths else 0.0
    md_lines += [
        "",
        f"**Robustness score** (mean normalized envelope width): **{robustness:.3f}**",
        "",
        "Per-axis CSV: `sweep_summary.csv`.",
    ]

    md_path = output_dir / "breaking_points.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[sweep] wrote {csv_path}")
    print(f"[sweep] wrote {md_path}")
    print(f"[sweep] robustness_score = {robustness:.3f}")


if __name__ == "__main__":
    main()
