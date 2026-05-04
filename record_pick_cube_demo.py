"""Record rollout videos of the pick-cube vision OSC task with random actions.

Saves two MP4s per run:
  run_XX_scene.mp4  — 3rd-person scene view
  run_XX_wrist.mp4  — 32x32 wrist camera obs (upscaled to 256x256)

Usage:
    MUJOCO_GL=egl uv run python record_pick_cube_demo.py
    MUJOCO_GL=egl uv run python record_pick_cube_demo.py --num-runs 4 --steps 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

import kinova_tasks  # noqa: F401 — populate task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


TASK_ID = "Mjlab-Pick-Cube-Vision-Osc-Kinova"
OUT_DIR = Path("videos/pick_cube_vision_osc_random")
WRIST_UPSCALE = 256  # display size for the 32x32 wrist image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-runs", type=int, default=4)
    p.add_argument("--steps", type=int, default=300, help="Steps per episode")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default=None)
    return p.parse_args()


def cam_tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (1, 3, H, W) float [0,1] tensor → (H, W, 3) uint8."""
    img = t[0].permute(1, 2, 0).cpu().float().numpy()  # (H, W, 3)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def upscale_nearest(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upscale (H, W, 3) → (size, size, 3)."""
    h, w = img.shape[:2]
    scale_h = size // h
    scale_w = size // w
    return np.repeat(np.repeat(img, scale_h, axis=0), scale_w, axis=1)


def main() -> None:
    args = parse_args()
    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1

    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env_base, clip_actions=agent_cfg.clip_actions)

    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

    def policy(obs: torch.Tensor) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=device) - 1

    print(f"Using random actions on {TASK_ID}")
    print(f"Recording {args.num_runs} runs → {OUT_DIR.resolve()}")

    for run_idx in range(1, args.num_runs + 1):
        scene_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []

        obs, _ = env.reset()

        for _ in range(args.steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # Scene view
            scene = env_base.render()
            if scene is not None:
                scene_frames.append(scene)

            # Wrist camera obs: stored in env_base.obs_buf["camera"] as (1, 3, 32, 32)
            cam_buf = env_base.obs_buf.get("camera")
            if cam_buf is not None:
                wrist_img = cam_tensor_to_uint8(cam_buf)
                wrist_frames.append(upscale_nearest(wrist_img, WRIST_UPSCALE))

        scene_path = OUT_DIR / f"run_{run_idx:02d}_scene.mp4"
        wrist_path = OUT_DIR / f"run_{run_idx:02d}_wrist.mp4"

        imageio.mimwrite(str(scene_path), scene_frames, fps=args.fps, codec="libx264")
        if wrist_frames:
            imageio.mimwrite(str(wrist_path), wrist_frames, fps=args.fps, codec="libx264")

        print(f"  [{run_idx}/{args.num_runs}] scene → {scene_path.name}  |  wrist → {wrist_path.name}")

    env.close()
    print(f"\nDone. Videos saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
