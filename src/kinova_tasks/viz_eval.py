"""Visualize eval trajectories in viser.

Loads an eval-output directory (`run.json` + `runs.parquet` + `states.h5`
+ `model.mjb`) produced by `kinova_tasks.eval`, lets you filter trials
by metadata, pick one, and scrub through its physics-state trajectory.

The mjlab eval saves one ``mjSTATE_PHYSICS`` snapshot (qpos+qvel+act)
per outer policy step. We re-apply each frame via ``mj_setState`` and
``mj_forward`` to drive a viser scene, identical to how
``mjlab.scripts.nan_viz`` replays nan dumps.

Usage:

    uv run viz-eval /path/to/eval_output_dir

The directory must contain:
    run.json
    runs.parquet
    states.h5
    model.mjb
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pandas as pd
import tyro
import viser

import mjlab
from mjlab.viewer.viser import ViserMujocoScene


# ---------------------------------------------------------------------------
# Trial label / filtering helpers
# ---------------------------------------------------------------------------


def _trial_label(row: pd.Series) -> str:
    """Compact one-line label shown in the trial dropdown."""
    sr = "✓" if bool(row.get("success_terminal", False)) else "✗"
    err = float(row.get("terminal_object_to_goal_error", float("nan")))
    return (
        f"{sr} trial_{int(row.trial_id):04d} "
        f"env={int(row.env_id)} "
        f"len={int(row.episode_length):3d} "
        f"err={err:.3f}m "
        f"{row.terminal_reason}"
    )


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class EvalTrajectoryViewer:
    def __init__(self, eval_dir: str | Path):
        self.eval_dir = Path(eval_dir).expanduser().resolve()
        if not self.eval_dir.is_dir():
            raise NotADirectoryError(f"Not an eval dir: {self.eval_dir}")

        run_json = self.eval_dir / "run.json"
        runs_pq  = self.eval_dir / "runs.parquet"
        states_h5 = self.eval_dir / "states.h5"
        model_mjb = self.eval_dir / "model.mjb"
        for p in (run_json, runs_pq, states_h5, model_mjb):
            if not p.exists():
                raise FileNotFoundError(f"Missing in eval dir: {p.name}")

        self.run_meta = json.loads(run_json.read_text())
        self.runs_df = pd.read_parquet(runs_pq)
        # Pre-load all state datasets into memory (small for typical scales).
        with h5py.File(states_h5, "r") as h:
            self.nq = int(h.attrs["nq"])
            self.nv = int(h.attrs["nv"])
            self.na = int(h.attrs["na"])
            self.nmocap = int(h.attrs.get("nmocap", 0))
            self.state_size = int(h.attrs["state_size"])
            self.trial_states: dict[int, np.ndarray] = {}
            self.trial_mocap: dict[int, np.ndarray] = {}
            for k in h.keys():
                if not k.startswith("trial_"):
                    continue
                tid = int(k.replace("trial_", ""))
                grp = h[k]
                # New layout: /trial_<id>/states + optional /trial_<id>/mocap.
                # Old layout (no mocap): /trial_<id> is a dataset directly.
                if isinstance(grp, h5py.Group):
                    self.trial_states[tid] = grp["states"][:].astype(np.float64)
                    if "mocap" in grp:
                        self.trial_mocap[tid] = grp["mocap"][:].astype(np.float64)
                else:
                    self.trial_states[tid] = grp[:].astype(np.float64)

        print(f"[viz-eval] eval dir         : {self.eval_dir}")
        print(f"[viz-eval] task             : {self.run_meta.get('task_id')}")
        print(f"[viz-eval] agent            : {self.run_meta.get('agent')}")
        print(f"[viz-eval] checkpoint       : {self.run_meta.get('checkpoint_path')}")
        print(f"[viz-eval] trials in runs   : {len(self.runs_df)}")
        print(f"[viz-eval] trials w/ states : {len(self.trial_states)}")
        print(
            f"[viz-eval] success_terminal mean: "
            f"{self.runs_df.success_terminal.mean():.3f}"
        )

        self.model = mujoco.MjModel.from_binary_path(str(model_mjb))
        self.data  = mujoco.MjData(self.model)

        self.server = viser.ViserServer(label="Eval Trajectory Viewer")
        # mjlab's ViserMujocoScene API: direct construction, not .create().
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)

        # Active selection state.
        self.current_trial_id: int | None = None
        self.current_step: int = 0
        self.current_states: np.ndarray | None = None  # (T, state_size)
        self.current_trial_mocap: np.ndarray | None = None  # (T, nmocap, 7)

    # -------- GUI setup --------

    def setup(self) -> None:
        self.info_html = self.server.gui.add_html(self._info_html())

        # Filters folder
        with self.server.gui.add_folder("Filters"):
            reasons = ["(any)"] + sorted(self.runs_df.terminal_reason.unique().tolist())
            self.f_reason = self.server.gui.add_dropdown(
                "terminal_reason", options=reasons, initial_value="(any)",
                hint="Filter trials by termination reason.",
            )
            self.f_success = self.server.gui.add_dropdown(
                "success_terminal", options=("(any)", "True", "False"),
                initial_value="(any)",
                hint="Trials where cube was within 2cm of goal at terminal step.",
            )
            env_ids = sorted(self.runs_df.env_id.unique().tolist())
            self.f_env_min = self.server.gui.add_slider(
                "env_id min", min=int(env_ids[0]), max=int(env_ids[-1]),
                step=1, initial_value=int(env_ids[0]),
            )
            self.f_env_max = self.server.gui.add_slider(
                "env_id max", min=int(env_ids[0]), max=int(env_ids[-1]),
                step=1, initial_value=int(env_ids[-1]),
            )
            self.f_apply = self.server.gui.add_button("Apply filter")

            @self.f_apply.on_click
            def _(_) -> None:
                self._refresh_trial_dropdown()

        # Trial picker folder
        with self.server.gui.add_folder("Trial"):
            self.trial_dropdown = self.server.gui.add_dropdown(
                "Pick trial", options=self._trial_options(self.runs_df),
                initial_value=self._trial_options(self.runs_df)[0],
                hint="Filtered by 'Filters' above. Click a row to load it.",
            )

            @self.trial_dropdown.on_update
            def _(_) -> None:
                self._select_trial_from_label(self.trial_dropdown.value)

        # Playback folder
        with self.server.gui.add_folder("Playback"):
            self.step_slider = self.server.gui.add_slider(
                "Step", min=0, max=1, step=1, initial_value=0,
                hint="Scrub through the selected trial's frames.",
            )

            @self.step_slider.on_update
            def _(_) -> None:
                self.current_step = int(self.step_slider.value)
                self._update_state()

            self.play_button = self.server.gui.add_button("▶ Play")
            self._is_playing = False

            @self.play_button.on_click
            def _(_) -> None:
                self._is_playing = not self._is_playing
                self.play_button.label = "■ Stop" if self._is_playing else "▶ Play"

            self.fps_slider = self.server.gui.add_slider(
                "Playback FPS", min=1, max=60, step=1, initial_value=10,
                hint="Outer policy step rate is ~10 FPS in training.",
            )

        # Visualization options from mjlab's scene helper.
        self.scene.create_scene_gui(show_debug_viz_control=False)

        # Auto-load the first trial.
        self._select_trial_from_label(self.trial_dropdown.value)

    # -------- filtering --------

    def _filtered_df(self) -> pd.DataFrame:
        df = self.runs_df.copy()
        # Restrict to trials that actually have states saved.
        df = df[df.trial_id.isin(self.trial_states.keys())]
        if self.f_reason.value != "(any)":
            df = df[df.terminal_reason == self.f_reason.value]
        if self.f_success.value != "(any)":
            df = df[df.success_terminal == (self.f_success.value == "True")]
        df = df[
            (df.env_id >= int(self.f_env_min.value))
            & (df.env_id <= int(self.f_env_max.value))
        ]
        return df.sort_values(["env_id", "trial_idx_in_env"])

    def _trial_options(self, df: pd.DataFrame) -> list[str]:
        if df.empty:
            return ["(no matching trials)"]
        return [_trial_label(r) for _, r in df.iterrows()]

    def _refresh_trial_dropdown(self) -> None:
        df = self._filtered_df()
        opts = self._trial_options(df)
        self.trial_dropdown.options = opts
        self.trial_dropdown.value = opts[0]
        self._select_trial_from_label(opts[0])

    def _select_trial_from_label(self, label: str) -> None:
        if label.startswith("(no"):
            self.current_states = None
            self.info_html.content = self._info_html()
            return
        # Parse trial id from "✓ trial_0007 env=2 ..."
        token = next((t for t in label.split() if t.startswith("trial_")), None)
        if token is None:
            return
        tid = int(token.split("_")[-1])
        self.current_trial_id = tid
        self.current_states = self.trial_states.get(tid)
        self.current_trial_mocap = self.trial_mocap.get(tid)
        if self.current_states is None:
            return
        n_steps = self.current_states.shape[0]
        self.step_slider.max = max(1, n_steps - 1)
        self.step_slider.value = 0
        self.current_step = 0
        self._update_state()
        self.info_html.content = self._info_html()

    # -------- replay --------

    def _update_state(self) -> None:
        if self.current_states is None:
            return
        s = self.current_states[self.current_step]
        mujoco.mj_setState(
            self.model, self.data, s, mujoco.mjtState.mjSTATE_PHYSICS
        )
        # Apply mocap if available for this trial — mjSTATE_PHYSICS does not
        # cover mocap_pos/quat, but the robot base is mocap-rooted in this
        # task, so its pose lives there.
        if self.current_trial_mocap is not None:
            mocap = self.current_trial_mocap[self.current_step]  # (nmocap, 7)
            self.data.mocap_pos[:] = mocap[:, :3]
            self.data.mocap_quat[:] = mocap[:, 3:7]
        mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)
        self.info_html.content = self._info_html()

    # -------- info panel --------

    def _info_html(self) -> str:
        if self.current_trial_id is None or self.current_states is None:
            return (
                "<div style='font-size:0.85em; padding:0 1em 0.5em 1em;'>"
                "<em>Pick a trial above to start.</em></div>"
            )
        row = self.runs_df[self.runs_df.trial_id == self.current_trial_id].iloc[0]
        n = int(self.current_states.shape[0])
        return f"""
        <div style="font-size:0.85em; line-height:1.3; padding:0 1em 0.5em 1em;">
          <strong>Trial:</strong> {int(row.trial_id)}
            (env {int(row.env_id)}, idx_in_env {int(row.trial_idx_in_env)})<br/>
          <strong>Step:</strong> {self.current_step} / {n - 1}<br/>
          <strong>Episode length:</strong> {int(row.episode_length)}<br/>
          <strong>Terminal reason:</strong> {row.terminal_reason}<br/>
          <strong>Terminal error:</strong> {float(row.terminal_object_to_goal_error):.4f} m
            &nbsp;<strong>Success:</strong> {bool(row.success_terminal)}
            (dwell {float(row.success_dwell):.3f})<br/>
          <strong>DR draws:</strong>
            cube_friction={float(row.dr_object_friction_slide):.3f},
            cube_mass={float(row.dr_object_mass):.4f}<br/>
          <strong>Init cube:</strong>
            {[round(x, 3) for x in row.init_cube_pos_w]}<br/>
          <strong>Init goal (local):</strong>
            {[round(x, 3) for x in row.init_goal_pos_local]}
        </div>
        """

    # -------- main loop --------

    def run(self) -> None:
        print("\n[viz-eval] Open the URL above in a browser. Ctrl+C to exit.\n")
        try:
            last_play_t = time.monotonic()
            while True:
                if self.scene.needs_update:
                    self.scene.refresh_visualization()
                if (
                    self._is_playing
                    and self.current_states is not None
                ):
                    fps = float(self.fps_slider.value)
                    dt = 1.0 / fps if fps > 0 else 0.1
                    now = time.monotonic()
                    if now - last_play_t >= dt:
                        n = self.current_states.shape[0]
                        next_step = (self.current_step + 1) % n
                        self.step_slider.value = next_step
                        # on_update callback drives _update_state.
                        last_play_t = now
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[viz-eval] shutting down")
            self.server.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_viewer(eval_dir: tyro.conf.Positional[str]) -> None:
    """Open an eval output directory in the trajectory viewer."""
    viewer = EvalTrajectoryViewer(eval_dir)
    viewer.setup()
    viewer.run()


def main() -> None:
    tyro.cli(run_viewer, description=__doc__, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
    main()
