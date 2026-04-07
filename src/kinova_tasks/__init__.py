"""Kinova Gen3 manipulation tasks for mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from kinova_tasks.tasks.lift_joint import (
    kinova_lift_joint_env_cfg,
    kinova_lift_joint_ppo_cfg,
)
from kinova_tasks.tasks.lift_cartesian import (
    kinova_lift_cartesian_env_cfg,
    kinova_lift_cartesian_ppo_cfg,
)
from kinova_tasks.tasks.peg_in_hole import (
    kinova_peg_in_hole_env_cfg,
    kinova_peg_in_hole_ppo_cfg,
)
from kinova_tasks.tasks.peg_in_hole_osc import (
    kinova_peg_in_hole_osc_env_cfg,
    kinova_peg_in_hole_osc_ppo_cfg,
)
from kinova_tasks.tasks.reach_diff_ik import (
    kinova_reach_diff_ik_env_cfg,
    kinova_reach_diff_ik_ppo_cfg,
)
from kinova_tasks.tasks.reach_osc import (
    kinova_reach_osc_env_cfg,
    kinova_reach_osc_ppo_cfg,
)

# Joint-space lift task
register_mjlab_task(
    task_id="Mjlab-Lift-Cube-Kinova",
    env_cfg=kinova_lift_joint_env_cfg(),
    play_env_cfg=kinova_lift_joint_env_cfg(play=True),
    rl_cfg=kinova_lift_joint_ppo_cfg(),
)

# Cartesian-space lift task
register_mjlab_task(
    task_id="Mjlab-Lift-Cube-Kinova-Cartesian",
    env_cfg=kinova_lift_cartesian_env_cfg(),
    play_env_cfg=kinova_lift_cartesian_env_cfg(play=True),
    rl_cfg=kinova_lift_cartesian_ppo_cfg(),
)

# Peg-in-hole task (cartesian)
register_mjlab_task(
    task_id="Mjlab-Peg-In-Hole-Kinova",
    env_cfg=kinova_peg_in_hole_env_cfg(),
    play_env_cfg=kinova_peg_in_hole_env_cfg(play=True),
    rl_cfg=kinova_peg_in_hole_ppo_cfg(),
)

# Peg-in-hole task (OSC torque control)
register_mjlab_task(
    task_id="Mjlab-Peg-In-Hole-Osc-Kinova",
    env_cfg=kinova_peg_in_hole_osc_env_cfg(),
    play_env_cfg=kinova_peg_in_hole_osc_env_cfg(play=True),
    rl_cfg=kinova_peg_in_hole_osc_ppo_cfg(),
)

# Reach task (diff-IK, no gripper)
register_mjlab_task(
    task_id="Mjlab-Reach-Kinova",
    env_cfg=kinova_reach_diff_ik_env_cfg(),
    play_env_cfg=kinova_reach_diff_ik_env_cfg(play=True),
    rl_cfg=kinova_reach_diff_ik_ppo_cfg(),
)

# Reach task (OSC torque control, no gripper)
register_mjlab_task(
    task_id="Mjlab-Reach-Osc-Kinova",
    env_cfg=kinova_reach_osc_env_cfg(),
    play_env_cfg=kinova_reach_osc_env_cfg(play=True),
    rl_cfg=kinova_reach_osc_ppo_cfg(),
)
