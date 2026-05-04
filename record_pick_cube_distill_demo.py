"""Record rollout videos of the pick *distill* OSC task with random actions.

Runs `--num-envs` parallel envs (default 4) for one episode and saves:

  scene_envXX.mp4   — 3rd-person scene render (env 0 only; render() is shared)
  wrist_envXX.mp4   — wrist camera obs (upscaled), one per env
  d455_envXX.mp4    — static D455 camera obs (upscaled), one per env

The distill env (`pick_cube_vision_osc`) concatenates wrist + d455 along
width.  We split obs_buf["camera"] at the midpoint and save one video
per env per camera.

Usage:
    MUJOCO_GL=egl uv run python record_pick_cube_distill_demo.py
    MUJOCO_GL=egl uv run python record_pick_cube_distill_demo.py --num-envs 4 --steps 300
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


TASK_ID = "Mjlab-Pick-Cube-Distill-Osc-Kinova"
OUT_DIR = Path("videos/pick_cube_distill_osc_random")
CAM_UPSCALE = 256  # display size for the 32x32 cam images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=300, help="Steps per episode")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default=None)
    return p.parse_args()


def cam_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) float [0,1] tensor → (H, W, 3) uint8."""
    img = t.permute(1, 2, 0).cpu().float().numpy()
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def upscale_nearest(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upscale (H, W, 3) → (size, size, 3)."""
    h, w = img.shape[:2]
    return np.repeat(np.repeat(img, size // h, axis=0), size // w, axis=1)


def main() -> None:
    args = parse_args()
    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs

    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env_base, clip_actions=agent_cfg.clip_actions)

    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

    def policy(obs: torch.Tensor) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=device) - 1

    print(f"Random actions on {TASK_ID}, num_envs={args.num_envs}")
    print(f"Recording {args.steps} steps → {OUT_DIR.resolve()}")

    scene_frames: list[np.ndarray] = []
    wrist_frames: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
    d455_frames: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]

    obs, _ = env.reset()

    for _ in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        # Scene view (renders from env 0; mjlab.render() returns a single frame).
        scene = env_base.render()
        if scene is not None:
            scene_frames.append(scene)

        # Camera obs: wrist + d455 concatenated along width.
        # Shape is (N, 3, H, 2W); split at the midpoint.
        cam_buf = env_base.obs_buf.get("camera")
        if cam_buf is None:
            continue
        half = cam_buf.shape[-1] // 2
        wrist_buf = cam_buf[:, :, :, :half]
        d455_buf = cam_buf[:, :, :, half:]
        for i in range(args.num_envs):
            wrist_frames[i].append(upscale_nearest(cam_to_uint8(wrist_buf[i]), CAM_UPSCALE))
            d455_frames[i].append(upscale_nearest(cam_to_uint8(d455_buf[i]), CAM_UPSCALE))

    if scene_frames:
        scene_path = OUT_DIR / "scene_env00.mp4"
        imageio.mimwrite(str(scene_path), scene_frames, fps=args.fps, codec="libx264")
        print(f"  scene (env0) → {scene_path.name}")

    for i in range(args.num_envs):
        if wrist_frames[i]:
            p = OUT_DIR / f"wrist_env{i:02d}.mp4"
            imageio.mimwrite(str(p), wrist_frames[i], fps=args.fps, codec="libx264")
            print(f"  wrist  env{i} → {p.name}")
        if d455_frames[i]:
            p = OUT_DIR / f"d455_env{i:02d}.mp4"
            imageio.mimwrite(str(p), d455_frames[i], fps=args.fps, codec="libx264")
            print(f"  d455   env{i} → {p.name}")

    env.close()
    print(f"\nDone. Videos saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
