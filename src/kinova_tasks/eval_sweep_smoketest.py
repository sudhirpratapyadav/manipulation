"""Smoke test for ``eval_sweep.py`` overrides.

For each override function in ``default_axes()``:
1. Build the eval env at small num_envs with the override applied at one
   off-nominal value.
2. Read back the affected MuJoCo / cfg field.
3. Print whether the override took effect.

No policy, no rollouts. The point is to confirm the harness's override
plumbing actually does what it claims before we trust any number it
produces. See ``docs/AGENT.md`` § "Resume sequence" step 3.

Run:
    uv run python -m kinova_tasks.eval_sweep_smoketest
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Callable

import torch

import mjlab  # noqa: F401
import kinova_tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from kinova_tasks.eval_sweep import (
    EVAL_TASK_ID,
    default_axes,
)


NUM_ENVS = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _build(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnv:
    cfg.scene.num_envs = NUM_ENVS
    return ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode=None)


def _model_field_at(env: ManagerBasedRlEnv, field: str, idx: int) -> float:
    """Read scalar at global ``idx`` from a (possibly per-env-expanded) model field.

    Returns env-0 value if expanded, otherwise the shared value.
    """
    arr = getattr(env.sim.model, field)
    t = arr.torch() if hasattr(arr, "torch") else arr
    if t.dim() == 1:
        return float(t[idx].item())
    return float(t[0, idx].item())


def _read_drawer_slide_friction(env: ManagerBasedRlEnv) -> float:
    drawer = env.scene["drawer"]
    jids, _ = drawer.find_joints("drawer_slide")
    mj_jnt_id = int(drawer.indexing.joint_ids[jids[0]])
    dof_adr = int(env.sim.mj_model.jnt_dofadr[mj_jnt_id])
    return _model_field_at(env, "dof_frictionloss", dof_adr)


def _read_drawer_slide_damping(env: ManagerBasedRlEnv) -> float:
    drawer = env.scene["drawer"]
    jids, _ = drawer.find_joints("drawer_slide")
    mj_jnt_id = int(drawer.indexing.joint_ids[jids[0]])
    dof_adr = int(env.sim.mj_model.jnt_dofadr[mj_jnt_id])
    return _model_field_at(env, "dof_damping", dof_adr)


def _read_drawer_base_mass(env: ManagerBasedRlEnv) -> float:
    drawer = env.scene["drawer"]
    bids, _ = drawer.find_bodies("drawer_base")
    mj_bid = int(drawer.indexing.body_ids[bids[0]])
    return _model_field_at(env, "body_mass", mj_bid)


def _read_command_slide_range(env: ManagerBasedRlEnv) -> tuple[float, float]:
    cmd = env.command_manager.get_term("drawer_goal")
    return float(cmd.cfg.slide_lo), float(cmd.cfg.slide_hi)


def _read_init_slide_range(env: ManagerBasedRlEnv) -> tuple[float, float]:
    return tuple(env.cfg.events["reset_drawer"].params["init_slide_range"])  # type: ignore[arg-type]


def _read_base_pose_range(env: ManagerBasedRlEnv) -> dict[str, tuple[float, float]]:
    return env.cfg.events["reset_base"].params["pose_range"]  # type: ignore[return-value]


def _read_init_joint_delta_deg(env: ManagerBasedRlEnv) -> float:
    return float(env.cfg.events["reset_robot_joints"].params["joint_delta_deg"])  # type: ignore[arg-type]


def _read_action_scale(env: ManagerBasedRlEnv) -> float:
    return float(env.cfg.actions["osc_pose"].delta_pos_scale)  # type: ignore[arg-type]


def _read_arm_link_mass_total(env: ManagerBasedRlEnv) -> float:
    """Sum of arm-link masses for env 0 (used to detect any DR perturbation)."""
    robot = env.scene["robot"]
    bids, _ = robot.find_bodies(r".*_link")
    arr = env.sim.model.body_mass
    t = arr.torch() if hasattr(arr, "torch") else arr
    total = 0.0
    for bid in bids:
        mj_bid = int(robot.indexing.body_ids[bid])
        total += _model_field_at(env, "body_mass", mj_bid)
    return total


def _read_fingertip_friction_slide(env: ManagerBasedRlEnv) -> float:
    """Read sliding friction (axis 0) of a fingertip pad geom for env 0."""
    robot = env.scene["robot"]
    gids, _ = robot.find_geoms(r"(left|right)_pad[12]")
    if not gids:
        return float("nan")
    mj_gid = int(robot.indexing.geom_ids[gids[0]])
    arr = env.sim.model.geom_friction
    t = arr.torch() if hasattr(arr, "torch") else arr
    if t.dim() == 2:
        return float(t[mj_gid, 0].item())
    return float(t[0, mj_gid, 0].item())


# Axis name → (off-nominal test value, reader, label-fn for expected vs got)
_READERS: dict[str, dict[str, Any]] = {
    "drawer_slide_friction": {
        "value": 0.04,
        "reader": _read_drawer_slide_friction,
        "expected": lambda v: v,
    },
    "drawer_slide_damping": {
        "value": 4.0,
        "reader": _read_drawer_slide_damping,
        "expected": lambda v: v,
    },
    "drawer_base_mass_scale": {
        "value": 5.0,
        "reader": _read_drawer_base_mass,
        # Without a baseline mass to compare to, just print observed.
        "expected": None,
    },
    "goal_depth": {
        "value": -0.10,
        "reader": _read_command_slide_range,
        "expected": lambda v: (v, v),
    },
    "init_slide": {
        "value": -0.10,
        "reader": _read_init_slide_range,
        "expected": lambda v: (v, v),
    },
    "robot_base_x_offset": {
        "value": 5.0,  # cm
        "reader": _read_base_pose_range,
        "expected": lambda v_cm: {"x": (v_cm * 0.01, v_cm * 0.01)},
    },
    "init_joint_delta_deg": {
        "value": 30.0,
        "reader": _read_init_joint_delta_deg,
        "expected": lambda v: v,
    },
    "action_scale": {
        "value": 0.05,
        "reader": _read_action_scale,
        "expected": lambda v: v,
    },
    "arm_link_mass_pct": {
        "value": 50.0,
        "reader": _read_arm_link_mass_total,
        "expected": None,
    },
    "fingertip_friction_slide": {
        "value": 1.5,
        "reader": _read_fingertip_friction_slide,
        # The override sets a deterministic friction in `axes=[0]`. We expect
        # the read-back to be near 1.5 at env 0.
        "expected": lambda v: v,
    },
}


def _check_axis(axis_name: str, apply_override: Callable[[ManagerBasedRlEnvCfg, float], None]) -> dict[str, Any]:
    spec = _READERS.get(axis_name)
    if spec is None:
        return {"axis": axis_name, "ok": False, "msg": "no reader registered"}
    val = spec["value"]
    reader = spec["reader"]
    expected_fn = spec["expected"]

    cfg = load_env_cfg(EVAL_TASK_ID, play=True)
    apply_override(cfg, val)

    try:
        env = _build(cfg)
    except Exception as e:
        return {
            "axis": axis_name,
            "ok": False,
            "msg": f"build failed: {type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }

    try:
        got = reader(env)
    except Exception as e:
        env.close()
        return {
            "axis": axis_name,
            "ok": False,
            "msg": f"reader failed: {type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }

    env.close()

    if expected_fn is None:
        return {"axis": axis_name, "ok": True, "msg": f"set value={val} → got {got!r} (no closed-form expected)"}

    expected = expected_fn(val)

    def _close(a: Any, b: Any) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                return False
            return all(_close(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                return False
            return all(_close(x, y) for x, y in zip(a, b))
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-6)
        return a == b

    ok = _close(got, expected)
    return {
        "axis": axis_name,
        "ok": ok,
        "msg": f"set value={val} → expected {expected!r}, got {got!r}",
    }


def main() -> None:
    configure_torch_backends()
    axes = default_axes()
    print(f"[smoke] num_envs={NUM_ENVS}  device={DEVICE}  axes={len(axes)}")
    print()

    results = []
    for axis in axes:
        print(f"[smoke] testing axis={axis.name} ...", flush=True)
        r = _check_axis(axis.name, axis.apply)
        results.append(r)
        status = "OK " if r["ok"] else "FAIL"
        print(f"  [{status}] {r['msg']}")
        if not r["ok"] and "trace" in r:
            print(r["trace"])
        print()

    n_ok = sum(1 for r in results if r["ok"])
    print(f"[smoke] {n_ok}/{len(results)} axes verified.")
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
