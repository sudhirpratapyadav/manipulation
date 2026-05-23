"""Force-probe for the snap-fit task.

Drives a single env with a constant +X EE delta and logs:
  - EE position vs time
  - peg-base position vs time
  - per-step axial force on the peg (cfrc_ext, world-X component)
  - per-step axial force on the EE pinch_site body (cfrc_ext, world-X)
  - prong hinge angles (top / bottom)
  - arm-joint actuator torques (qfrc_actuator on joint_[1-7])

Saves a CSV and a multi-panel PNG to ``tests/out/<tag>/``.

Usage:
    uv run python tests/snap_fit_force_probe.py --tag baseline --steps 200
    uv run python tests/snap_fit_force_probe.py --tag stiff_hinge --steps 200 --action 0.5

``--action`` is the +X component of the 6-D OSC action (1.0 = full
delta_pos_scale per step = 2 cm of commanded EE travel per env step).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import kinova_tasks  # noqa: F401  — register tasks  # noqa: E402
import mujoco_warp as mjwarp  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402


TASK_ID = "Mjlab-Snap-Fit-Osc-Kinova"


def run_probe(tag: str, steps: int, action_x: float, out_root: Path) -> None:
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = 1

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device)

    # Action: 6-D (dx, dy, dz, drx, dry, drz). Only push +X.
    action_dim = env.action_space.shape[1]
    assert action_dim == 6, f"Expected 6D OSC action, got {action_dim}"
    action = torch.zeros(1, action_dim, device=device)
    action[0, 0] = float(action_x)

    robot = env.scene["robot"]
    peg = env.scene["peg"]

    # Entity-local indices for entity-data accessors (site_pos_w etc.).
    pinch_site_local = robot.find_sites("pinch_site")[0][0]
    arm_joint_local = robot.find_joints("joint_[1-7]")[0]
    lip_top_local = peg.find_joints("lip_top_hinge")[0][0]
    lip_bot_local = peg.find_joints("lip_bot_hinge")[0][0]

    # Global body id for the peg root (for sim.data.cfrc_ext).
    peg_root_body = int(peg.indexing.root_body_id)

    obs, _ = env.reset()

    records: list[dict[str, float]] = []
    for step_i in range(steps):
        obs, rew, terminated, truncated, info = env.step(action)

        # Read state.
        ee_pos = robot.data.site_pos_w[0, pinch_site_local].cpu().numpy()
        peg_pos = peg.data.root_link_pos_w[0].cpu().numpy()

        # cfrc_ext: (nworld, nbody, 6) — spatial wrench in COM-aligned world frame,
        # layout [angular(3), linear(3)] per MuJoCo convention. The linear part
        # is the net external force on the body.
        # cfrc_ext is only filled when a sensor requires it; populate manually
        # via rne_postconstraint each step so we can read contact wrenches.
        mjwarp.rne_postconstraint(env.sim.wp_model, env.sim.wp_data)
        cfrc_ext = env.sim.data.cfrc_ext  # TorchArray view, already a tensor
        peg_wrench = cfrc_ext[0, peg_root_body].cpu().numpy()
        peg_force_xyz = peg_wrench[3:6]

        # Arm actuator torques in joint space.
        qfrc_act = robot.data.qfrc_actuator[0].cpu().numpy()
        arm_torques = qfrc_act[list(arm_joint_local)]

        # Hinge angles.
        joint_pos = peg.data.joint_pos[0].cpu().numpy()
        lip_top_angle = float(joint_pos[lip_top_local])
        lip_bot_angle = float(joint_pos[lip_bot_local])

        row = {
            "step": step_i,
            "ee_x": float(ee_pos[0]),
            "ee_y": float(ee_pos[1]),
            "ee_z": float(ee_pos[2]),
            "peg_x": float(peg_pos[0]),
            "peg_y": float(peg_pos[1]),
            "peg_z": float(peg_pos[2]),
            "peg_force_x": float(peg_force_xyz[0]),
            "peg_force_y": float(peg_force_xyz[1]),
            "peg_force_z": float(peg_force_xyz[2]),
            "peg_force_mag": float(np.linalg.norm(peg_force_xyz)),
            "lip_top_angle": lip_top_angle,
            "lip_bot_angle": lip_bot_angle,
        }
        for i, t in enumerate(arm_torques):
            row[f"arm_torque_{i+1}"] = float(t)
        records.append(row)

        if (step_i + 1) % 20 == 0:
            print(
                f"[step {step_i+1:4d}] ee_x={ee_pos[0]:.4f} peg_x={peg_pos[0]:.4f} "
                f"F_x={peg_force_xyz[0]:+.2f} |F|={np.linalg.norm(peg_force_xyz):.2f} "
                f"lips=({lip_top_angle:+.3f},{lip_bot_angle:+.3f})",
                flush=True,
            )

        if terminated.any() or truncated.any():
            print(f"Episode ended at step {step_i+1}")
            break

    env.close()

    # Write CSV.
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "probe.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV: {csv_path}")

    # Plot.
    steps_arr = np.array([r["step"] for r in records])
    ee_x = np.array([r["ee_x"] for r in records])
    peg_x = np.array([r["peg_x"] for r in records])
    f_x = np.array([r["peg_force_x"] for r in records])
    f_mag = np.array([r["peg_force_mag"] for r in records])
    lip_top = np.array([r["lip_top_angle"] for r in records])
    lip_bot = np.array([r["lip_bot_angle"] for r in records])

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(steps_arr, ee_x, label="EE x")
    axes[0].plot(steps_arr, peg_x, label="peg x")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"snap-fit probe — tag={tag} action_x={action_x}")

    axes[1].plot(steps_arr, f_x, label="peg F_x (axial)")
    axes[1].plot(steps_arr, f_mag, label="|peg F|", alpha=0.5)
    axes[1].set_ylabel("force (N)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="k", linewidth=0.5)

    axes[2].plot(steps_arr, lip_top, label="top hinge")
    axes[2].plot(steps_arr, lip_bot, label="bot hinge")
    axes[2].set_ylabel("hinge angle (rad)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    # Arm torque magnitude as a summary line.
    arm_keys = [k for k in records[0].keys() if k.startswith("arm_torque_")]
    arm_mat = np.array([[r[k] for k in arm_keys] for r in records])
    arm_norm = np.linalg.norm(arm_mat, axis=1)
    axes[3].plot(steps_arr, arm_norm, label="‖arm τ‖")
    axes[3].set_ylabel("‖arm torque‖ (N·m)")
    axes[3].set_xlabel("env step")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "probe.png"
    fig.savefig(png_path, dpi=130)
    print(f"PNG: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Output subdir name")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--action",
        type=float,
        default=1.0,
        help="+X OSC action component (1.0 = full delta_pos_scale per step)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).parent / "out",
    )
    args = parser.parse_args()
    run_probe(args.tag, args.steps, args.action, args.out_root)


if __name__ == "__main__":
    main()
