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
from kinova_tasks.tasks.pick_cube_osc import (
    kinova_pick_cube_osc_env_cfg,
    kinova_pick_cube_osc_ppo_cfg,
)
from kinova_tasks.tasks.pick_cube_vision_osc import (
    kinova_pick_cube_vision_osc_env_cfg,
    kinova_pick_cube_vision_osc_ppo_cfg,
)
from kinova_tasks.tasks.pick_cube_distill_osc import (
    kinova_pick_cube_distill_osc_env_cfg,
    kinova_pick_cube_distill_osc_runner_cfg,
)
from kinova_tasks.tasks.pick_cube_distill_mcr_osc import (
    kinova_pick_cube_distill_mcr_osc_env_cfg,
    kinova_pick_cube_distill_mcr_osc_runner_cfg,
    kinova_pick_cube_distill_mcr_ss_osc_runner_cfg,
    kinova_pick_cube_distill_mcr_widehead_osc_runner_cfg,
    kinova_pick_cube_distill_mcr_ll4_osc_runner_cfg,
    kinova_pick_cube_distill_mcr_smallhead_osc_runner_cfg,
)
from kinova_tasks.distill_runner import MjlabDistillationRunner
from kinova_tasks.tasks.open_door_osc import (
    kinova_open_door_osc_env_cfg,
    kinova_open_door_osc_ppo_cfg,
)
from kinova_tasks.tasks.open_drawer_osc import (
    kinova_open_drawer_osc_env_cfg,
    kinova_open_drawer_osc_eval_env_cfg,
    kinova_open_drawer_osc_phase1_env_cfg,
    kinova_open_drawer_osc_phase2_env_cfg,
    kinova_open_drawer_osc_phase4_env_cfg,
    kinova_open_drawer_osc_baseline_dr_env_cfg,
    kinova_open_drawer_osc_baseline_dr_v2_env_cfg,
    kinova_open_drawer_osc_ppo_cfg,
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

# Pick cube task (OSC torque control + gripper)
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Osc-Kinova",
    env_cfg=kinova_pick_cube_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_osc_ppo_cfg(),
)

# Pick cube task (OSC + RGB wrist camera)
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Vision-Osc-Kinova",
    env_cfg=kinova_pick_cube_vision_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_vision_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_vision_osc_ppo_cfg(),
)

# Phase A: state -> vision distillation (DAgger, MSE on actions).
# Teacher = trained pick_cube_osc PPO actor (wandb run jn3l22j9).
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Distill-Osc-Kinova",
    env_cfg=kinova_pick_cube_distill_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_distill_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_distill_osc_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)

# Phase A (MCR variant): same env / teacher / DAgger, but the student's
# visual encoder is a frozen MCR ResNet-50 instead of the from-scratch
# spatial-softmax CNN.  See docs/sim2real/plan_v2.md.
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Distill-Mcr-Osc-Kinova",
    env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_distill_mcr_osc_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)

# Phase A (MCR + spatial-softmax variant): MCR ResNet-50 cut at layer4
# with spatial-softmax over the (2048, 7, 7) map → preserves spatial
# localization info that global avg-pool throws away.  8192-D feature.
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Distill-Mcr-Ss-Osc-Kinova",
    env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_distill_mcr_ss_osc_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)

# Phase A (MCR + wider MLP head): same avg-pool encoder as Mcr-Osc but
# hidden_dims (1024, 512, 256, 128) so the head has more capacity to
# project the 4096-D feature into 7-D actions.
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Distill-Mcr-Widehead-Osc-Kinova",
    env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_distill_mcr_widehead_osc_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)

# Phase A (MCR + last-stage unfreeze): avg-pool encoder with `layer4`
# unfrozen — partial fine-tuning of the last ResNet stage to let the
# encoder adapt to pick-cube geometry.  Works for either MCR or R3M
# weights via MCR_WEIGHTS_PATH env var.
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Distill-Mcr-Ll4-Osc-Kinova",
    env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_distill_mcr_ll4_osc_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)

# Phase A (MCR + small MLP head): avg-pool encoder with hidden_dims
# (256, 128) — tests whether the pretrained feature is so well-
# structured that a smaller head suffices.
register_mjlab_task(
    task_id="Mjlab-Pick-Cube-Distill-Mcr-Smallhead-Osc-Kinova",
    env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(),
    play_env_cfg=kinova_pick_cube_distill_mcr_osc_env_cfg(play=True),
    rl_cfg=kinova_pick_cube_distill_mcr_smallhead_osc_runner_cfg(),
    runner_cls=MjlabDistillationRunner,
)

# Open door task (OSC torque control + gripper)
register_mjlab_task(
    task_id="Mjlab-Open-Door-Osc-Kinova",
    env_cfg=kinova_open_door_osc_env_cfg(),
    play_env_cfg=kinova_open_door_osc_env_cfg(play=True),
    rl_cfg=kinova_open_door_osc_ppo_cfg(),
)

# Open drawer task (OSC torque control + gripper)
register_mjlab_task(
    task_id="Mjlab-Open-Drawer-Osc-Kinova",
    env_cfg=kinova_open_drawer_osc_env_cfg(),
    play_env_cfg=kinova_open_drawer_osc_env_cfg(play=True),
    rl_cfg=kinova_open_drawer_osc_ppo_cfg(),
)

# Open drawer task — OOD evaluation skeleton (no DR sweep CLI yet).
# play_env_cfg deliberately uses the *finite-episode* eval cfg, not the
# infinite-episode play cfg, so success_rate aggregates over real episodes.
register_mjlab_task(
    task_id="Mjlab-Open-Drawer-Osc-Kinova-Eval",
    env_cfg=kinova_open_drawer_osc_eval_env_cfg(),
    play_env_cfg=kinova_open_drawer_osc_eval_env_cfg(),
    rl_cfg=kinova_open_drawer_osc_ppo_cfg(),
)

# Phase 1 — wider initial-pose randomization (joint_delta_deg=15, base XY/yaw).
register_mjlab_task(
    task_id="Mjlab-Open-Drawer-Osc-Kinova-Phase1",
    env_cfg=kinova_open_drawer_osc_phase1_env_cfg(),
    play_env_cfg=kinova_open_drawer_osc_phase1_env_cfg(play=True),
    rl_cfg=kinova_open_drawer_osc_ppo_cfg(),
)

# Phase 2 — Phase 1 + targeted DR on drawer slide friction/damping/mass and arm link mass.
register_mjlab_task(
    task_id="Mjlab-Open-Drawer-Osc-Kinova-Phase2",
    env_cfg=kinova_open_drawer_osc_phase2_env_cfg(),
    play_env_cfg=kinova_open_drawer_osc_phase2_env_cfg(play=True),
    rl_cfg=kinova_open_drawer_osc_ppo_cfg(),
)

# Phase 4 — combined deploy run (Phase 2 + wider DR + Phase 3 keeper TBD at submit time).
register_mjlab_task(
    task_id="Mjlab-Open-Drawer-Osc-Kinova-Phase4",
    env_cfg=kinova_open_drawer_osc_phase4_env_cfg(),
    play_env_cfg=kinova_open_drawer_osc_phase4_env_cfg(play=True),
    rl_cfg=kinova_open_drawer_osc_ppo_cfg(),
)

# baseline_dr — Phase 1 init-pose floor + Phase 2 drawer DR (no arm_link_mass)
# + step-stepped curriculum widening of drawer cube and init-pose ranges.
register_mjlab_task(
    task_id="Mjlab-Open-Drawer-Osc-Kinova-BaselineDr",
    env_cfg=kinova_open_drawer_osc_baseline_dr_env_cfg(),
    play_env_cfg=kinova_open_drawer_osc_baseline_dr_env_cfg(play=True),
    rl_cfg=kinova_open_drawer_osc_ppo_cfg(),
)

# baseline_dr_v2 — baseline_dr + longer drawer (joint range -0.40 m, goal in
# [-0.35, -0.10]) + stepped impulse curriculum on drawer base body.
register_mjlab_task(
    task_id="Mjlab-Open-Drawer-Osc-Kinova-BaselineDrV2",
    env_cfg=kinova_open_drawer_osc_baseline_dr_v2_env_cfg(),
    play_env_cfg=kinova_open_drawer_osc_baseline_dr_v2_env_cfg(play=True),
    rl_cfg=kinova_open_drawer_osc_ppo_cfg(),
)
