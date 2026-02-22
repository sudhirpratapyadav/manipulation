"""Interactive OSC control demo for Kinova Gen3.

Drag the 3D transform control in the viser viewer to move the Kinova end-effector
using Operational Space Control (OSC) which computes torques directly.

Run with:
  MJLAB_WARP_QUIET=1 uv run scripts/demos/kinova_osc.py
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
from mjlab.envs.mdp.actions import OperationalSpaceAction, OperationalSpaceActionCfg
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


def main() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Create Kinova Gen3 arm with gripper
  robot_cfg = EntityCfg(
    init_state=DEMO_INIT_STATE,
    collisions=(),
    spec_fn=get_spec,
    articulation=KINOVA_GRIPPER_ARTICULATION,
  )
  entity = Entity(robot_cfg)
  model = entity.compile()

  PHYSICS_DT = 0.002  # 500 Hz physics
  CONTROL_RATE = 100  # Hz - OSC torque computation
  VIZ_RATE = 30  # Hz
  CONTROL_PERIOD = 1.0 / CONTROL_RATE
  VIZ_PERIOD = 1.0 / VIZ_RATE

  SIM_STEPS_PER_CONTROL = int(CONTROL_PERIOD / PHYSICS_DT)  # 5 steps per control update
  SIM_STEPS_PER_VIZ = int(VIZ_PERIOD / PHYSICS_DT)  # ~16-17 steps per viz frame

  sim_cfg = SimulationCfg(
    mujoco=MujocoCfg(
      timestep=PHYSICS_DT,
      gravity=(0, 0, -9.81),
    )
  )
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
  sim.forward()

  env = SimpleNamespace(num_envs=1, device=device, scene={"robot": entity}, sim=sim)
  osc_cfg = OperationalSpaceActionCfg(
    entity_name="robot",
    actuator_names=("joint_.*",),
    frame_name="pinch_site",
    frame_type="site",
    kp_pos=400.0,
    kp_ori=400.0,
    kd_pos=100.0,
    kd_ori=100.0,
    posture_weight=0.01,
    posture_kp=10.0,
    posture_kd=2.0,
    use_relative_mode=False,
  )
  osc_action: OperationalSpaceAction = osc_cfg.build(env)  # type: ignore[arg-type]

  server = viser.ViserServer(label="Kinova Gen3 OSC Control Demo")
  scene = ViserMujocoScene.create(server, sim.mj_model, num_envs=1)
  scene.create_visualization_gui(
    camera_distance=1.2,
    camera_azimuth=135.0,
    camera_elevation=30.0,
  )

  site_id = osc_action._frame_id
  pos = sim.data.site_xpos[0, site_id].cpu().numpy()
  xmat = sim.data.site_xmat[0, site_id]
  quat = quat_from_matrix(xmat).cpu().numpy()

  transform_ctrl = server.scene.add_transform_controls(
    "/osc_target",
    position=(float(pos[0]), float(pos[1]), float(pos[2])),
    wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    scale=0.15,
  )

  needs_reset = [False]

  with server.gui.add_folder("OSC Control"):
    reset_button = server.gui.add_button("Reset")
    reset_button.on_click(lambda _: needs_reset.__setitem__(0, True))
    server.gui.add_text("Physics Rate", initial_value=f"{1/PHYSICS_DT:.0f} Hz")
    server.gui.add_text("Control Rate", initial_value=f"{CONTROL_RATE} Hz (torque commands)")
    server.gui.add_text("Pose Target Rate", initial_value=f"{VIZ_RATE} Hz (task commands)")
    server.gui.add_text("Viz Rate", initial_value=f"{VIZ_RATE} Hz")

  with server.gui.add_folder("OSC Gains"):
    kp_pos_slider = server.gui.add_slider(
      "Position Kp", min=0.0, max=1000.0, step=1.0, initial_value=400.0,
    )
    kd_pos_slider = server.gui.add_slider(
      "Position Kd", min=0.0, max=1000.0, step=1.0, initial_value=100.0,
    )
    kp_ori_slider = server.gui.add_slider(
      "Orientation Kp", min=0.0, max=1000.0, step=1.0, initial_value=400.0,
    )
    kd_ori_slider = server.gui.add_slider(
      "Orientation Kd", min=0.0, max=1000.0, step=1.0, initial_value=100.0,
    )

  with server.gui.add_folder("OSC Weights"):
    pos_w_slider = server.gui.add_slider(
      "Position Weight", min=0.0, max=10.0, step=0.1, initial_value=osc_cfg.position_weight,
    )
    ori_w_slider = server.gui.add_slider(
      "Orientation Weight", min=0.0, max=10.0, step=0.1, initial_value=osc_cfg.orientation_weight,
    )
    posture_w_slider = server.gui.add_slider(
      "Posture Weight", min=0.0, max=1.0, step=0.01, initial_value=osc_cfg.posture_weight,
    )
    posture_kp_slider = server.gui.add_slider(
      "Posture Kp", min=0.1, max=50.0, step=0.1, initial_value=osc_cfg.posture_kp,
    )
    posture_kd_slider = server.gui.add_slider(
      "Posture Kd", min=0.1, max=20.0, step=0.1, initial_value=osc_cfg.posture_kd,
    )

  print("=" * 60)
  print("Kinova Gen3 OSC Control Demo")
  print("  Open the viser URL printed above")
  print("  Drag the 3D transform control to move the end-effector")
  print("  OSC computes torques directly using mass matrix and dynamics")
  print("=" * 60)

  target_action = torch.zeros(1, 7, device=device)

  def _reset() -> None:
    entity.write_joint_position_to_sim(entity.data.default_joint_pos, joint_ids=None)
    sim.forward()
    osc_action.reset()
    p = sim.data.site_xpos[0, site_id].cpu().numpy()
    q = quat_from_matrix(sim.data.site_xmat[0, site_id]).cpu().numpy()
    transform_ctrl.position = (float(p[0]), float(p[1]), float(p[2]))
    transform_ctrl.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

  try:
    sim_steps_since_control = 0

    while True:
      if needs_reset[0]:
        needs_reset[0] = False
        _reset()
        sim_steps_since_control = 0

      # Update OSC parameters from sliders
      osc_cfg.kp_pos = max(kp_pos_slider.value, 1.0)
      osc_cfg.kd_pos = max(kd_pos_slider.value, 0.1)
      osc_cfg.kp_ori = max(kp_ori_slider.value, 1.0)
      osc_cfg.kd_ori = max(kd_ori_slider.value, 0.1)
      osc_cfg.position_weight = max(pos_w_slider.value, 0.0)
      osc_cfg.orientation_weight = max(ori_w_slider.value, 0.0)
      osc_cfg.posture_weight = max(posture_w_slider.value, 0.0)
      osc_cfg.posture_kp = max(posture_kp_slider.value, 0.1)
      osc_cfg.posture_kd = max(posture_kd_slider.value, 0.1)

      # Update pose target every frame (30 Hz)
      p = transform_ctrl.position
      w = transform_ctrl.wxyz
      target_action[0, :3] = torch.tensor([p[0], p[1], p[2]], device=device)
      target_action[0, 3:] = torch.tensor([w[0], w[1], w[2], w[3]], device=device)
      osc_action.process_actions(target_action)

      # Step simulation and apply OSC at 100 Hz
      for _ in range(SIM_STEPS_PER_VIZ):
        if sim_steps_since_control >= SIM_STEPS_PER_CONTROL:
          osc_action.apply_actions()
          sim_steps_since_control = 0

        entity.write_data_to_sim()
        sim.step()
        sim_steps_since_control += 1

      # Print tracking error every frame
      current_ee_pos = sim.data.site_xpos[0, site_id]
      target_ee_pos = target_action[0, :3]
      ee_error = torch.norm(current_ee_pos - target_ee_pos)
      print(f"EE tracking error: {ee_error.item()*1000:.1f}mm")

      # Update visualization at 30 Hz
      scene.update(sim.wp_data)
      if scene.needs_update:
        scene.refresh_visualization()

      time.sleep(VIZ_PERIOD)
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()


if __name__ == "__main__":
  main()
