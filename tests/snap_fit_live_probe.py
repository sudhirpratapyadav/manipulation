"""Live force-probe for the snap-fit task.

Opens a single-env viser viewer and drives the EE with the keyboard:

    w / s   : +X / -X   (insertion axis)
    space   : zero action (hold)
    r       : reset env
    q       : quit

A live uPlot panel in viser shows the last N steps of:
    - axial force on the peg (cfrc_ext, world-X)
    - |force| magnitude
    - top + bot hinge angle (right axis)
    - arm-actuator torque norm (right axis)

A slider in the GUI scales the keyboard action magnitude (0..3) — the OSC
controller still maps it through ``delta_pos_scale`` so the effective EE
delta per step is ``scale * delta_pos_scale``.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python tests/snap_fit_live_probe.py
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import torch
import viser
from viser import uplot

import kinova_tasks  # noqa: F401  — register tasks
import mujoco_warp as mjwarp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viser.viewer import ViserPlayViewer


TASK_ID = "Mjlab-Snap-Fit-Osc-Kinova"
ROLLING_WINDOW = 600  # ~12 s at 50 Hz env steps


class _KeyState:
    """Thread-safe key-press state + action-scale slider mirror."""

    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.scale: float = 1.0
        self.lock = threading.Lock()
        self._start_listener()

    def _start_listener(self) -> None:
        from pynput import keyboard

        def on_press(key) -> None:
            try:
                ch = key.char.lower()
            except AttributeError:
                return
            with self.lock:
                self.pressed.add(ch)

        def on_release(key) -> None:
            try:
                ch = key.char.lower()
            except AttributeError:
                return
            with self.lock:
                self.pressed.discard(ch)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def read(self) -> tuple[set[str], float]:
        with self.lock:
            return set(self.pressed), self.scale


class ProbePolicy:
    """Maps keyboard to a 6-D OSC action and records telemetry into a ring buffer."""

    def __init__(
        self,
        action_shape: tuple[int, ...],
        device: str,
        env: ManagerBasedRlEnv,
        keys: _KeyState,
        buffers: dict[str, deque],
    ) -> None:
        self._action_shape = action_shape
        self._device = device
        self._env = env
        self._keys = keys
        self._buf = buffers

        robot = env.scene["robot"]
        peg = env.scene["peg"]
        self._pinch_local = robot.find_sites("pinch_site")[0][0]
        self._arm_local = robot.find_joints("joint_[1-7]")[0]
        self._lip_top_local = peg.find_joints("lip_top_hinge")[0][0]
        self._lip_bot_local = peg.find_joints("lip_bot_hinge")[0][0]
        self._peg_root_body = int(peg.indexing.root_body_id)
        self._robot = robot
        self._peg = peg
        self._step = 0

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        # Telemetry first — uses *current* state (post previous step).
        # rne_postconstraint populates cfrc_ext.
        mjwarp.rne_postconstraint(self._env.sim.wp_model, self._env.sim.wp_data)
        cfrc_ext = self._env.sim.data.cfrc_ext
        peg_wrench = cfrc_ext[0, self._peg_root_body].cpu().numpy()
        peg_force_xyz = peg_wrench[3:6]
        f_x = float(peg_force_xyz[0])
        f_mag = float(np.linalg.norm(peg_force_xyz))

        ee_pos = self._robot.data.site_pos_w[0, self._pinch_local].cpu().numpy()
        joint_pos = self._peg.data.joint_pos[0].cpu().numpy()
        lip_top = float(joint_pos[self._lip_top_local])
        lip_bot = float(joint_pos[self._lip_bot_local])

        qfrc_act = self._robot.data.qfrc_actuator[0].cpu().numpy()
        arm_tau = float(np.linalg.norm(qfrc_act[list(self._arm_local)]))

        self._buf["step"].append(self._step)
        self._buf["ee_x"].append(float(ee_pos[0]))
        self._buf["f_x"].append(f_x)
        self._buf["f_mag"].append(f_mag)
        self._buf["lip_top"].append(lip_top)
        self._buf["lip_bot"].append(lip_bot)
        self._buf["arm_tau"].append(arm_tau)
        self._step += 1

        # Build action from keys.
        pressed, scale = self._keys.read()
        action = torch.zeros(self._action_shape, device=self._device)
        action_dim = self._action_shape[-1]
        x = 0.0
        if "w" in pressed:
            x += 1.0
        if "s" in pressed:
            x -= 1.0
        if " " in pressed or "space" in pressed:
            x = 0.0
        if action_dim >= 1:
            action[..., 0] = x * scale
        return action


def _make_ring(maxlen: int) -> deque:
    return deque(maxlen=maxlen)


def main() -> None:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    # Long horizon — let the user drive freely.
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = int(1e9)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    buffers: dict[str, deque] = {
        k: _make_ring(ROLLING_WINDOW)
        for k in ("step", "ee_x", "f_x", "f_mag", "lip_top", "lip_bot", "arm_tau")
    }

    keys = _KeyState()
    action_shape: tuple[int, ...] = env_wrapped.unwrapped.action_space.shape
    policy = ProbePolicy(action_shape, device, env, keys, buffers)

    server = viser.ViserServer(label="snap-fit-probe")

    plot_handles: dict[str, object] = {}

    class _ProbeViewer(ViserPlayViewer):
        def setup(self) -> None:
            super().setup()
            # Add our controls + plots after the viewer's own GUI is wired up.
            with self._server.gui.add_folder("Probe controls"):
                self._server.gui.add_markdown(
                    "**Keys:** `w` = +X, `s` = -X, `space` = hold, "
                    "`r` = reset, `q` = quit."
                )
                scale_slider = self._server.gui.add_slider(
                    "Action scale",
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    initial_value=1.0,
                )

                @scale_slider.on_update
                def _(_):
                    with keys.lock:
                        keys.scale = float(scale_slider.value)

            seed_x = np.array([0.0, 1.0], dtype=np.float64)
            seed_y = np.zeros(2, dtype=np.float64)
            with self._server.gui.add_folder("Force vs step"):
                plot_handles["force"] = self._server.gui.add_uplot(
                    data=(seed_x, seed_y, seed_y),
                    series=(
                        uplot.Series(label="step"),
                        uplot.Series(label="F_x (N)", stroke="#ef4444"),
                        uplot.Series(label="|F| (N)", stroke="#f59e0b"),
                    ),
                    aspect=1.6,
                    height=220,
                )
            with self._server.gui.add_folder("Hinge angles vs step"):
                plot_handles["hinge"] = self._server.gui.add_uplot(
                    data=(seed_x, seed_y, seed_y),
                    series=(
                        uplot.Series(label="step"),
                        uplot.Series(label="top hinge (rad)", stroke="#3b82f6"),
                        uplot.Series(label="bot hinge (rad)", stroke="#10b981"),
                    ),
                    aspect=1.6,
                    height=200,
                )
            with self._server.gui.add_folder("Arm torque norm vs step"):
                plot_handles["torque"] = self._server.gui.add_uplot(
                    data=(seed_x, seed_y),
                    series=(
                        uplot.Series(label="step"),
                        uplot.Series(label="‖arm τ‖ (N·m)", stroke="#8b5cf6"),
                    ),
                    aspect=1.6,
                    height=180,
                )

    viewer = _ProbeViewer(env_wrapped, policy, viser_server=server)

    # Background thread that pushes new buffer contents into the uplot handles.
    stop_event = threading.Event()

    def _refresh_plots() -> None:
        while not stop_event.is_set():
            if len(buffers["step"]) >= 2 and plot_handles:
                x = np.array(buffers["step"], dtype=np.float64)
                f_x = np.array(buffers["f_x"], dtype=np.float64)
                f_mag = np.array(buffers["f_mag"], dtype=np.float64)
                lip_top = np.array(buffers["lip_top"], dtype=np.float64)
                lip_bot = np.array(buffers["lip_bot"], dtype=np.float64)
                arm_tau = np.array(buffers["arm_tau"], dtype=np.float64)
                try:
                    plot_handles["force"].data = (x, f_x, f_mag)  # type: ignore[attr-defined]
                    plot_handles["hinge"].data = (x, lip_top, lip_bot)  # type: ignore[attr-defined]
                    plot_handles["torque"].data = (x, arm_tau)  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001
                    print(f"[plot refresh] {exc}", flush=True)
            stop_event.wait(0.1)

    plot_thread = threading.Thread(target=_refresh_plots, daemon=True)
    plot_thread.start()

    # Reset on `r`, quit on `q`.
    last_r = False
    last_q = False

    def _key_watchdog() -> None:
        nonlocal last_r, last_q
        while not stop_event.is_set():
            pressed, _ = keys.read()
            r_now = "r" in pressed
            q_now = "q" in pressed
            if r_now and not last_r:
                viewer.request_reset()
            last_r = r_now
            if q_now and not last_q:
                stop_event.set()
                viewer._interrupted = True  # type: ignore[attr-defined]
            last_q = q_now
            stop_event.wait(0.05)

    key_thread = threading.Thread(target=_key_watchdog, daemon=True)
    key_thread.start()

    print(
        "\n[snap_fit live probe]\n"
        "  Open the viser URL printed above.\n"
        "  Keys: w/s = +X/-X, space = hold, r = reset env, q = quit.\n"
        "  Action scale slider lives in the 'Probe controls' folder.\n",
        flush=True,
    )

    try:
        viewer.run()
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    main()
