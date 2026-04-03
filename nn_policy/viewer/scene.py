"""Viser-based MuJoCo viewer backed by mjlab conversion utilities.

Architecture
============
ViserMujocoScene  — viser server wrapper + fixed world geometry + GUI
ViserRobotView    — one robot's mesh handles with optional per-robot color/alpha

Typical usage (single robot)::

    scene = ViserMujocoScene.create(server, mj_model)
    robot = scene.add_robot("robot")
    scene.create_visualization_gui()
    # in loop (mjlab entity):
    robot.update_from_entity(entity)
    # or (plain mj_data):
    robot.update(mj_data)

Typical usage (sim + real ghost)::

    scene = ViserMujocoScene.create(server, mj_model)
    sim_view  = scene.add_robot("sim",  color=(0.75, 0.75, 0.75, 1.0))
    real_view = scene.add_robot("real", color=(0.20, 0.55, 0.90, 0.65))
    # in viz thread:
    sim_view.update_from_entity(entity)
    real_view.update(real_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh
import trimesh.visual
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj

from mjlab.viewer.viser.conversions import (
    get_body_name,
    group_geoms_by_visual_compat,
    is_fixed_body,
    merge_geoms,
    merge_sites,
)

_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150
_DEFAULT_ENVIRONMENT_INTENSITY = 0.8

RGBA = tuple[float, float, float, float]


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _apply_color(mesh: trimesh.Trimesh, rgba: RGBA) -> trimesh.Trimesh:
    """Return a copy of mesh with all vertices recoloured to rgba (0-1 floats)."""
    rgba_u8 = (np.clip(np.array(rgba, dtype=np.float32), 0, 1) * 255).astype(np.uint8)
    mesh = mesh.copy()
    mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(rgba_u8, (len(mesh.vertices), 1))
    )
    return mesh


# ── ViserRobotView ─────────────────────────────────────────────────────────────

@dataclass
class ViserRobotView:
    """One robot instance in a viser scene.

    Created via :meth:`ViserMujocoScene.add_robot`. Each view has its own
    namespace and optional color override so multiple robots can be shown
    simultaneously in distinct colours.

    Update options each frame:
      - :meth:`update_from_entity` — reads body poses directly from an mjlab entity
      - :meth:`update_from_pose`   — sets all bodies to one world pose (peg/hole)
      - :meth:`update`             — legacy: reads from a plain ``mujoco.MjData``
    """

    server: viser.ViserServer
    mj_model: Any  # mujoco.MjModel (typed as Any to avoid hard mujoco import at call sites)
    namespace: str  # e.g. "sim" or "real"
    color: RGBA | None  # None → use model colours; tuple → override all geoms

    # Mesh handles: keyed by (body_id, group_id, sub_idx).
    _handles: dict[tuple[int, int, int], viser.SceneNodeHandle] = field(
        default_factory=dict, init=False
    )
    # Site handles: keyed by (body_id, group_id).
    _site_handles: dict[tuple[int, int], viser.SceneNodeHandle] = field(
        default_factory=dict, init=False
    )
    # Per-group visibility.
    geom_groups_visible: list[bool] = field(
        default_factory=lambda: [True, True, True, False, False, False]
    )
    site_groups_visible: list[bool] = field(
        default_factory=lambda: [True, True, True, False, False, False]
    )
    _visible: bool = field(default=True, init=False)

    # ── Construction ──────────────────────────────────────────────────────────

    @staticmethod
    def create(
        server: viser.ViserServer,
        mj_model: Any,
        namespace: str,
        color: RGBA | None = None,
    ) -> ViserRobotView:
        """Build mesh handles for all non-fixed bodies and sites.

        Args:
            server: Viser server instance.
            mj_model: MuJoCo model.
            namespace: Scene-graph prefix, e.g. ``"sim"`` or ``"real"``.
            color: Optional RGBA color override (values 0-1).
        """
        view = ViserRobotView(
            server=server, mj_model=mj_model, namespace=namespace, color=color
        )
        view._build_geom_handles()
        view._build_site_handles()
        return view

    def _mesh_for_geoms(self, geom_ids: list[int]) -> trimesh.Trimesh:
        mesh = merge_geoms(self.mj_model, geom_ids)
        if self.color is not None:
            mesh = _apply_color(mesh, self.color)
        return mesh

    def _mesh_for_sites(self, site_ids: list[int]) -> trimesh.Trimesh:
        mesh = merge_sites(self.mj_model, site_ids)
        if self.color is not None:
            mesh = _apply_color(mesh, self.color)
        return mesh

    def _build_geom_handles(self) -> None:
        body_group_geoms: dict[tuple[int, int], list[int]] = {}
        for i in range(self.mj_model.ngeom):
            body_id = self.mj_model.geom_bodyid[i]
            if is_fixed_body(self.mj_model, body_id):
                continue
            key = (body_id, int(self.mj_model.geom_group[i]))
            body_group_geoms.setdefault(key, []).append(i)

        with self.server.atomic():
            for (body_id, group_id), geom_ids in body_group_geoms.items():
                body_name = get_body_name(self.mj_model, body_id)
                visible = group_id < 6 and self.geom_groups_visible[group_id]
                subgroups = group_geoms_by_visual_compat(self.mj_model, geom_ids)

                for sub_idx, sub_ids in enumerate(subgroups):
                    mesh = self._mesh_for_geoms(sub_ids)
                    suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
                    handle = self.server.scene.add_mesh_trimesh(
                        f"/{self.namespace}/{body_name}/group{group_id}{suffix}",
                        mesh,
                        position=np.zeros(3),
                        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                        visible=visible,
                    )
                    self._handles[(body_id, group_id, sub_idx)] = handle

    def _build_site_handles(self) -> None:
        body_group_sites: dict[tuple[int, int], list[int]] = {}
        for site_id in range(self.mj_model.nsite):
            body_id = self.mj_model.site_bodyid[site_id]
            if is_fixed_body(self.mj_model, body_id):
                continue
            key = (body_id, int(self.mj_model.site_group[site_id]))
            body_group_sites.setdefault(key, []).append(site_id)

        with self.server.atomic():
            for (body_id, group_id), site_ids in body_group_sites.items():
                body_name = get_body_name(self.mj_model, body_id)
                visible = group_id < 6 and self.site_groups_visible[group_id]
                mesh = self._mesh_for_sites(site_ids)
                handle = self.server.scene.add_mesh_trimesh(
                    f"/{self.namespace}/{body_name}/sites_group{group_id}",
                    mesh,
                    position=np.zeros(3),
                    wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    visible=visible,
                )
                self._site_handles[(body_id, group_id)] = handle

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update_from_arrays(self, xpos: np.ndarray, xmat: np.ndarray) -> None:
        """Update from raw shared-memory arrays written by the sim process.

        Args:
            xpos: Body world positions, shape (nbody, 3).
            xmat: Body world rotation matrices, shape (nbody, 9) or (nbody, 3, 3).
        """
        xquat = vtf.SO3.from_matrix(xmat.reshape(-1, 3, 3)).wxyz  # (nbody, 4)
        with self.server.atomic():
            for (body_id, _group_id, _sub_idx), handle in self._handles.items():
                if not handle.visible:
                    continue
                handle.position = xpos[body_id]
                handle.wxyz     = xquat[body_id]
            for (body_id, _group_id), handle in self._site_handles.items():
                if not handle.visible:
                    continue
                handle.position = xpos[body_id]
                handle.wxyz     = xquat[body_id]

    def update_from_entity(self, entity: Any, env_idx: int = 0) -> None:
        """Update body transforms directly from an mjlab entity's simulation data.

        Reads ``xpos`` / ``xmat`` from the underlying warp data — no
        ``mujoco.MjData`` or ``mj_kinematics`` call needed.

        Args:
            entity: mjlab ``Entity`` instance (must have ``entity.data.data``
                    with ``xpos`` and ``xmat`` warp arrays).
            env_idx: Environment index (default 0 for single-env setups).
        """
        wp_data = entity.data.data
        xpos  = wp_data.xpos.cpu().numpy()[env_idx]    # (nbody, 3)
        xmat  = wp_data.xmat.cpu().numpy()[env_idx]    # (nbody, 3, 3)
        xquat = vtf.SO3.from_matrix(xmat).wxyz         # (nbody, 4) wxyz

        with self.server.atomic():
            for (body_id, _group_id, _sub_idx), handle in self._handles.items():
                if not handle.visible:
                    continue
                handle.position = xpos[body_id]
                handle.wxyz     = xquat[body_id]

            for (body_id, _group_id), handle in self._site_handles.items():
                if not handle.visible:
                    continue
                handle.position = xpos[body_id]
                handle.wxyz     = xquat[body_id]

    def update_from_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Set all mesh handles to a single world-frame pose.

        Useful for simple single-body objects (peg, hole) whose world pose is
        known directly without needing mj_kinematics.

        Args:
            pos: World-frame position (3,).
            quat_wxyz: World-frame orientation as wxyz quaternion (4,).
        """
        with self.server.atomic():
            for handle in self._handles.values():
                if handle.visible:
                    handle.position = pos
                    handle.wxyz     = quat_wxyz
            for handle in self._site_handles.values():
                if handle.visible:
                    handle.position = pos
                    handle.wxyz     = quat_wxyz

    def update(self, mj_data: Any) -> None:
        """Update from a plain ``mujoco.MjData`` (legacy path).

        Call this after ``mujoco.mj_kinematics(model, data)`` or
        ``mujoco.mj_forward(model, data)``.
        """
        xpos  = mj_data.xpos
        xquat = vtf.SO3.from_matrix(
            mj_data.xmat.reshape(-1, 3, 3)
        ).wxyz  # (nbody, 4)

        with self.server.atomic():
            for (body_id, _group_id, _sub_idx), handle in self._handles.items():
                if not handle.visible:
                    continue
                mocap_id = self.mj_model.body_mocapid[body_id]
                if mocap_id >= 0:
                    handle.position = mj_data.mocap_pos[mocap_id]
                    handle.wxyz     = mj_data.mocap_quat[mocap_id]
                else:
                    handle.position = xpos[body_id]
                    handle.wxyz     = xquat[body_id]

            for (body_id, _group_id), handle in self._site_handles.items():
                if not handle.visible:
                    continue
                handle.position = xpos[body_id]
                handle.wxyz     = xquat[body_id]

    # ── Visibility ────────────────────────────────────────────────────────────

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, v: bool) -> None:
        self._visible = v
        for handle in self._handles.values():
            handle.visible = v
        for handle in self._site_handles.values():
            handle.visible = v

    def set_geom_group_visible(self, group: int, visible: bool) -> None:
        if 0 <= group < 6:
            self.geom_groups_visible[group] = visible
        for (body_id, group_id, sub_idx), handle in self._handles.items():
            if group_id == group:
                handle.visible = visible and self._visible

    def set_site_group_visible(self, group: int, visible: bool) -> None:
        if 0 <= group < 6:
            self.site_groups_visible[group] = visible
        for (body_id, group_id), handle in self._site_handles.items():
            if group_id == group:
                handle.visible = visible and self._visible

    def set_color(self, rgba: RGBA) -> None:
        """Recolour all geoms live (rebuilds mesh handles)."""
        self.color = rgba
        for h in self._handles.values():
            h.remove()
        for h in self._site_handles.values():
            h.remove()
        self._handles.clear()
        self._site_handles.clear()
        self._build_geom_handles()
        self._build_site_handles()


# ── ViserMujocoScene ───────────────────────────────────────────────────────────

@dataclass
class ViserMujocoScene:
    """Fixed world geometry, GUI controls, and a registry of robot views."""

    server: viser.ViserServer
    mj_model: Any  # mujoco.MjModel

    _robot_views: list[ViserRobotView] = field(default_factory=list, init=False)
    _fixed_frame: viser.SceneNodeHandle = field(init=False)

    @staticmethod
    def create(
        server: viser.ViserServer,
        mj_model: Any,
    ) -> ViserMujocoScene:
        scene = ViserMujocoScene(server=server, mj_model=mj_model)
        server.scene.configure_environment_map(
            environment_intensity=_DEFAULT_ENVIRONMENT_INTENSITY
        )
        scene._fixed_frame = server.scene.add_frame("/world", show_axes=False)
        scene._add_fixed_geometry()
        return scene

    # ── Robot views ──────────────────────────────────────────────────────────

    def add_robot(
        self,
        namespace: str = "robot",
        color: RGBA | None = None,
    ) -> ViserRobotView:
        view = ViserRobotView.create(self.server, self.mj_model, namespace, color)
        self._robot_views.append(view)
        return view

    # ── Fixed geometry ────────────────────────────────────────────────────────

    def _add_fixed_geometry(self) -> None:
        _visible_groups = {0, 1, 2}

        body_geoms: dict[int, list[int]] = {}
        for i in range(self.mj_model.ngeom):
            body_id = self.mj_model.geom_bodyid[i]
            if not is_fixed_body(self.mj_model, body_id):
                continue
            if int(self.mj_model.geom_group[i]) not in _visible_groups:
                continue
            body_geoms.setdefault(body_id, []).append(i)

        for body_id, geom_ids in body_geoms.items():
            body_name = get_body_name(self.mj_model, body_id)
            body_pos  = self.mj_model.body(body_id).pos
            body_quat = self.mj_model.body(body_id).quat

            nonplane: list[int] = []
            for geom_id in geom_ids:
                if self.mj_model.geom_type[geom_id] == mjtGeom.mjGEOM_PLANE:
                    geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, geom_id)
                    self.server.scene.add_grid(
                        f"/world/{body_name}/{geom_name}",
                        width=2000.0, height=2000.0,
                        infinite_grid=True, fade_distance=50.0,
                        shadow_opacity=0.2,
                        position=self.mj_model.geom_pos[geom_id],
                        wxyz=self.mj_model.geom_quat[geom_id],
                    )
                else:
                    nonplane.append(geom_id)

            if nonplane:
                for sub_idx, sub_ids in enumerate(
                    group_geoms_by_visual_compat(self.mj_model, nonplane)
                ):
                    suffix = f"/sub{sub_idx}" if len(nonplane) > len(sub_ids) else ""
                    self.server.scene.add_mesh_trimesh(
                        f"/world/{body_name}{suffix}",
                        merge_geoms(self.mj_model, sub_ids),
                        cast_shadow=False, receive_shadow=0.2,
                        position=body_pos, wxyz=body_quat,
                    )

    # ── GUI ──────────────────────────────────────────────────────────────────

    def create_visualization_gui(
        self,
        camera_distance: float = 3.0,
        camera_azimuth: float = 45.0,
        camera_elevation: float = 30.0,
    ) -> None:
        with self.server.gui.add_folder("Visualization"):
            slider_fov = self.server.gui.add_slider(
                "FOV (°)", min=_DEFAULT_FOV_MIN, max=_DEFAULT_FOV_MAX,
                step=1, initial_value=_DEFAULT_FOV_DEGREES,
            )

            @slider_fov.on_update
            def _(_) -> None:
                for client in self.server.get_clients().values():
                    client.camera.fov = np.radians(slider_fov.value)

            @self.server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                client.camera.fov = np.radians(slider_fov.value)

            snap_btn = self.server.gui.add_button("Snap camera")

            @snap_btn.on_click
            def _(_) -> None:
                az  = np.deg2rad(camera_azimuth)
                el  = np.deg2rad(camera_elevation)
                fwd = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                ])
                pos = -fwd * camera_distance
                for client in self.server.get_clients().values():
                    client.camera.position = pos
                    client.camera.look_at  = np.zeros(3)

        with self.server.gui.add_folder("Robots", expand_by_default=False):
            for view in self._robot_views:
                with self.server.gui.add_folder(view.namespace):
                    cb_vis = self.server.gui.add_checkbox(
                        "Visible", initial_value=view.visible
                    )

                    @cb_vis.on_update
                    def _(event, v=view) -> None:
                        v.visible = event.target.value

                    with self.server.gui.add_folder("Geom groups"):
                        for i in range(6):
                            cb = self.server.gui.add_checkbox(
                                f"G{i}", initial_value=view.geom_groups_visible[i]
                            )

                            @cb.on_update
                            def _(event, v=view, g=i) -> None:
                                v.set_geom_group_visible(g, event.target.value)

                    with self.server.gui.add_folder("Site groups"):
                        for i in range(6):
                            cb = self.server.gui.add_checkbox(
                                f"S{i}", initial_value=view.site_groups_visible[i]
                            )

                            @cb.on_update
                            def _(event, v=view, g=i) -> None:
                                v.set_site_group_visible(g, event.target.value)
