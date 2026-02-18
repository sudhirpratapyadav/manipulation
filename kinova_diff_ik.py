"""Interactive IK control demo for Kinova Gen3.

Drag the 3D transform control in the viser viewer to move the Kinova end-effector.

Run with:
  MJLAB_WARP_QUIET=1 uv run scripts/demos/kinova_diff_ik.py
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
import viser

from kinova_tasks.assets.kinova_gen3.kinova_constants import (
  KINOVA_GRIPPER_ARTICULATION,
  get_spec,
)
from mjlab.entity import Entity, EntityCfg
from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.utils.lab_api.math import quat_from_matrix
from mjlab.viewer.viser import ViserMujocoScene

# Demo initial state - home pose from gen3.xml keyframe
DEMO_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint_1": 0.0,
    "joint_2": 0.26179939,  # 15°
    "joint_3": 3.14159265,  # 180°
    "joint_4": -2.26892803,  # -130°
    "joint_5": 0.0,
    "joint_6": 0.95993109,  # 55°
    "joint_7": 1.57079633,  # 90°
  },
  joint_vel={".*": 0.0},
)

IK_ITERATIONS = 10


def main() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Create Kinova Gen3 arm (no gripper)
  robot_cfg = EntityCfg(
    init_state=DEMO_INIT_STATE,
    collisions=(),
    spec_fn=get_spec,
    articulation=KINOVA_GRIPPER_ARTICULATION,
  )
  entity = Entity(robot_cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg(
    mujoco=MujocoCfg(
      timestep=0.002,  # 500 Hz physics
      gravity=(0, 0, -9.81),
    )
  )
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
  sim.forward()

  env = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity}, sim=sim)
  ik_cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint_.*",),  # Match all arm joints
    frame_name="pinch_site",
    frame_type="site",
    posture_weight=0.02,
    joint_limit_weight=1e-1,
    damping=1e-1,
    use_relative_mode=False,
  )
  ik_action: DifferentialIKAction = ik_cfg.build(env)  # type: ignore[arg-type]
  joint_ids = ik_action._joint_ids

  server = viser.ViserServer(label="Kinova Gen3 IK Control Demo")
  scene = ViserMujocoScene.create(server, sim.mj_model, num_envs=1)
  scene.create_visualization_gui(
    camera_distance=1.2,
    camera_azimuth=135.0,
    camera_elevation=30.0,
  )

  site_id = ik_action._frame_id
  pos = sim.data.site_xpos[0, site_id].cpu().numpy()
  xmat = sim.data.site_xmat[0, site_id]
  quat = quat_from_matrix(xmat).cpu().numpy()

  transform_ctrl = server.scene.add_transform_controls(
    "/ik_target",
    position=(float(pos[0]), float(pos[1]), float(pos[2])),
    wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    scale=0.15,
  )

  needs_reset = [False]

  PHYSICS_DT = 0.002  # 500 Hz physics
  IK_RATE = 100  # Hz - recompute joint targets from pose
  POSE_RATE = 30  # Hz - update pose target
  VIZ_RATE = 30  # Hz - update visualization

  IK_PERIOD = 1.0 / IK_RATE  # 0.01s
  POSE_PERIOD = 1.0 / POSE_RATE  # ~0.033s
  VIZ_PERIOD = 1.0 / VIZ_RATE  # ~0.033s

  SIM_STEPS_PER_IK = int(IK_PERIOD / PHYSICS_DT)  # 5 steps per IK update
  SIM_STEPS_PER_VIZ = int(VIZ_PERIOD / PHYSICS_DT)  # ~16-17 steps per viz frame

  with server.gui.add_folder("IK Control"):
    reset_button = server.gui.add_button("Reset")
    reset_button.on_click(lambda _: needs_reset.__setitem__(0, True))
    server.gui.add_text("Physics Rate", initial_value=f"{1/PHYSICS_DT:.0f} Hz")
    server.gui.add_text("IK Compute Rate", initial_value=f"{IK_RATE} Hz (joint targets)")
    server.gui.add_text("Pose Target Rate", initial_value=f"{POSE_RATE} Hz (task commands)")
    server.gui.add_text("Viz Rate", initial_value=f"{VIZ_RATE} Hz")

  with server.gui.add_folder("IK Weights"):
    damping_slider = server.gui.add_slider(
      "Damping (λ)",
      min=1e-2,
      max=1.0,
      step=1e-3,
      initial_value=ik_cfg.damping,
    )
    pos_w_slider = server.gui.add_slider(
      "Position Weight",
      min=0.0,
      max=10.0,
      step=0.1,
      initial_value=ik_cfg.position_weight,
    )
    ori_w_slider = server.gui.add_slider(
      "Orientation Weight",
      min=0.0,
      max=10.0,
      step=0.1,
      initial_value=ik_cfg.orientation_weight,
    )
    jlim_w_slider = server.gui.add_slider(
      "Joint Limit Weight",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=ik_cfg.joint_limit_weight,
    )
    posture_w_slider = server.gui.add_slider(
      "Posture Weight",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=ik_cfg.posture_weight,
    )

  print("=" * 60)
  print("Kinova Gen3 IK Control Demo")
  print("  Open the viser URL printed above")
  print("  Drag the 3D transform control to move the end-effector")
  print("=" * 60)

  target_action = torch.zeros(1, 7, device=device)

  def _reset() -> None:
    entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
    sim.forward()
    ik_action.reset()
    p = sim.data.site_xpos[0, site_id].cpu().numpy()
    q = quat_from_matrix(sim.data.site_xmat[0, site_id]).cpu().numpy()
    transform_ctrl.position = (float(p[0]), float(p[1]), float(p[2]))
    transform_ctrl.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

  try:
    sim_steps_since_ik = 0

    while True:
      if needs_reset[0]:
        needs_reset[0] = False
        _reset()
        sim_steps_since_ik = 0

      # Update IK weights from sliders
      ik_cfg.damping = max(damping_slider.value, 1e-2)
      ik_cfg.position_weight = max(pos_w_slider.value, 0.0)
      ik_cfg.orientation_weight = max(ori_w_slider.value, 0.0)
      ik_cfg.joint_limit_weight = max(jlim_w_slider.value, 0.0)
      ik_cfg.posture_weight = max(posture_w_slider.value, 0.0)

      # Update pose target every frame (30 Hz)
      p = transform_ctrl.position
      w = transform_ctrl.wxyz
      target_action[0, :3] = torch.tensor([p[0], p[1], p[2]], device=device)
      target_action[0, 3:] = torch.tensor([w[0], w[1], w[2], w[3]], device=device)
      ik_action.process_actions(target_action)

      # Step simulation and recompute IK at 100 Hz
      for _ in range(SIM_STEPS_PER_VIZ):
        # Recompute IK every 5 sim steps (100 Hz)
        if sim_steps_since_ik >= SIM_STEPS_PER_IK:
          dq = ik_action.compute_dq()
          q_target = entity.data.joint_pos[:, joint_ids] + dq
          entity.data.write_ctrl(q_target, ctrl_ids=joint_ids)
          sim_steps_since_ik = 0

        sim.step()
        sim_steps_since_ik += 1

      # Print tracking error every frame
      current_ee_pos, _ = ik_action._get_frame_pose()
      target_ee_pos = ik_action._desired_pos
      ee_error = torch.norm(current_ee_pos - target_ee_pos, dim=-1)
      print(f"EE tracking error: {ee_error[0].item()*1000:.1f}mm")

      # Update visualization at 30 Hz
      scene.update(sim.wp_data)
      if scene.needs_update:
        scene.refresh_visualization()

      time.sleep(VIZ_PERIOD)  # 30 Hz viz rate
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()


if __name__ == "__main__":
  main()
