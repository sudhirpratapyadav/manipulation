"""Phase A: state -> vision distillation training entrypoint.

Mirrors `mjlab.scripts.train.run_train` but uses
`MjlabDistillationRunner` and accepts `--teacher-ckpt` (a local path
to the trained PPO checkpoint).  The PPO checkpoint stores
`actor_state_dict`; `Distillation.load()` auto-detects this and loads
it as the teacher only.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder

# Force task registry to populate.
import kinova_tasks  # noqa: F401


_TASK_ID = "Mjlab-Pick-Cube-Distill-Osc-Kinova"


@dataclass(frozen=True)
class DistillTrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    teacher_ckpt: str
    """Local path to the trained PPO checkpoint (the teacher)."""
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    log_root: str = "logs/rsl_rl"
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str, teacher_ckpt: str) -> "DistillTrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return DistillTrainConfig(
            env=env_cfg, agent=agent_cfg, teacher_ckpt=teacher_ckpt
        )


def run_distill(cfg: DistillTrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(
        f"[INFO] Distillation training: device={device}, seed={seed}, rank={rank}"
    )
    print(f"[INFO] Teacher checkpoint: {cfg.teacher_ckpt}")
    teacher_path = Path(cfg.teacher_ckpt).resolve()
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env,
        device=device,
        render_mode="rgb_array" if cfg.video else None,
    )

    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(_TASK_ID)
    if runner_cls is None:
        raise RuntimeError(
            f"Task '{_TASK_ID}' has no runner_cls; expected MjlabDistillationRunner."
        )

    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner = runner_cls(env, agent_cfg, str(log_dir), device)
    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)

    # Load teacher from PPO checkpoint.  Distillation.load() auto-detects
    # `actor_state_dict` and applies load_cfg={"teacher": True, "iteration": False}.
    print(f"[INFO] Loading teacher from: {teacher_path}")
    runner.load(str(teacher_path), load_cfg=None, strict=True)

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()


def launch_distill(args: DistillTrainConfig) -> None:
    log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        run_distill(args, log_dir)
    else:
        raise NotImplementedError(
            "Multi-GPU distillation not wired up yet — pass gpu_ids=[<n>]."
        )


def main():
    args = tyro.cli(
        DistillTrainConfig,
        default=DistillTrainConfig.from_task(
            _TASK_ID, teacher_ckpt="<REQUIRED>"
        ),
        prog=sys.argv[0],
        config=mjlab.TYRO_FLAGS,
    )
    if args.teacher_ckpt == "<REQUIRED>":
        raise SystemExit(
            "Pass --teacher-ckpt <path/to/model_4999.pt>."
        )
    launch_distill(args)


if __name__ == "__main__":
    main()
