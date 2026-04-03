"""Extended play script that adds ``--agent keyboard`` to the standard play command.

For all other agents (zero, random, trained) the call is forwarded unchanged to
mjlab's play.  Only ``--agent keyboard`` is handled here.

Key bindings (6D action: 3D position + 3D orientation):
    Position
        W / S   →  +X / -X
        A / D   →  +Y / -Y
        Q / E   →  +Z / -Z
    Orientation
        I / K   →  +roll  / -roll   (around X)
        J / L   →  +pitch / -pitch  (around Y)
        U / O   →  +yaw   / -yaw   (around Z)

Usage:
    uv run play Mjlab-Reach-Osc-Kinova --agent keyboard --viewer viser
    uv run play Mjlab-Reach-Osc-Kinova --agent zero          # still works
    uv run play Mjlab-Reach-Osc-Kinova --agent trained ...   # still works
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


# ---------------------------------------------------------------------------
# Key → (action_dim_index, sign)
# ---------------------------------------------------------------------------

_KEY_BINDINGS: dict[str, tuple[int, int]] = {
    "w": (0, +1),  # +X
    "s": (0, -1),  # -X
    "a": (1, +1),  # +Y
    "d": (1, -1),  # -Y
    "q": (2, +1),  # +Z
    "e": (2, -1),  # -Z
    "i": (3, +1),  # +roll
    "k": (3, -1),  # -roll
    "j": (4, +1),  # +pitch
    "l": (4, -1),  # -pitch
    "u": (5, +1),  # +yaw
    "o": (5, -1),  # -yaw
}


class KeyboardPolicy:
    """Policy that maps live keypresses to a fixed-magnitude action vector.

    Runs a pynput listener in a daemon thread; the main loop reads the current
    pressed-key set each step and builds the action tensor.
    """

    def __init__(self, action_shape: tuple[int, ...], device: str) -> None:
        self._action_shape = action_shape
        self._device = device
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._start_listener()
        print(
            "\n[Keyboard agent]\n"
            "  Position:    W/S=+X/-X   A/D=+Y/-Y   Q/E=+Z/-Z\n"
            "  Orientation: I/K=roll    J/L=pitch   U/O=yaw\n"
        )

    def _start_listener(self) -> None:
        from pynput import keyboard

        def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
            try:
                ch = key.char.lower()  # type: ignore[union-attr]
                with self._lock:
                    self._pressed.add(ch)
            except AttributeError:
                pass

        def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
            try:
                ch = key.char.lower()  # type: ignore[union-attr]
                with self._lock:
                    self._pressed.discard(ch)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        del obs
        action = torch.zeros(self._action_shape, device=self._device)
        action_dim = self._action_shape[-1]
        with self._lock:
            pressed = set(self._pressed)
        for ch, (dim, sign) in _KEY_BINDINGS.items():
            if ch in pressed and dim < action_dim:
                action[..., dim] += sign
        return action.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Keyboard-specific config and runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _KeyboardPlayConfig:
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    """Disable all termination conditions."""


def _run_keyboard(task_id: str, remaining_args: list[str]) -> None:
    cfg = tyro.cli(
        _KeyboardPlayConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {task_id} --agent keyboard",
        config=mjlab.TYRO_FLAGS,
    )

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")

    env_cfg.scene.num_envs = cfg.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    policy = KeyboardPolicy(action_shape, device)

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()

    env.close()


# ---------------------------------------------------------------------------
# Entry point — intercept keyboard, delegate everything else to mjlab
# ---------------------------------------------------------------------------


def main() -> None:
    import mjlab.tasks  # noqa: F401 — populate task registry

    # Intercept --agent keyboard before tyro sees it so mjlab's Literal check
    # doesn't reject the value.
    if "--agent" in sys.argv and "keyboard" in sys.argv:
        # Strip --agent keyboard from argv so the sub-parser doesn't see it.
        argv = list(sys.argv[1:])
        idx = argv.index("--agent")
        argv.pop(idx)      # remove --agent
        argv.pop(idx)      # remove keyboard

        all_tasks = list_tasks()
        chosen_task, remaining_args = tyro.cli(
            tyro.extras.literal_type_from_choices(all_tasks),
            args=argv,
            add_help=False,
            return_unknown_args=True,
            config=mjlab.TYRO_FLAGS,
        )
        _run_keyboard(chosen_task, remaining_args)
    else:
        # Delegate to mjlab's play unchanged.
        from mjlab.scripts.play import main as mjlab_play_main
        mjlab_play_main()
