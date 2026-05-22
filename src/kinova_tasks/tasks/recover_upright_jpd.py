"""Kinova Gen3 self-recovery task: lift collapsed arm to a target joint config.

Robot starts torque-free on the ground (joints settled under gravity). The policy
must drive the arm to a target joint configuration (sampled near a reference
"recovered" pose) with minimum energy expenditure and minimum contact impulse
("damage").

Action space (7D, all in [-1, 1]): normalized joint torques.
Controller: τ = a · τ_max + qfrc_bias
    a = 0   → just gravity comp → arm holds current pose (weightless).
    a = ±1  → ±τ_max additional torque on top of gravity comp.

qfrc_bias = gravity + Coriolis, read directly from MuJoCo data. Grav comp is
always on, so the policy never needs to "spend" torque counteracting gravity —
it only commands net joint accelerations.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import mjlab.envs.mdp as mdp
from kinova_tasks.assets.kinova_gen3 import get_kinova_robot_cfg_peginhole_osc
from kinova_tasks.tasks.base_rl_cfg import kinova_ppo_runner_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEG_TO_RAD = _math.pi / 180.0

# Reference "recovered / standing" pose. Goal is sampled in a small window
# around this each episode.
_RECOVERED_JOINT_POS = (
    0.0,            # joint_1
    0.5235987756,   # joint_2  30°
    0.0,            # joint_3
    1.5707963268,   # joint_4  90°
    0.0,            # joint_5
    1.0471975512,   # joint_6  60°
    0.0,            # joint_7
)
_GOAL_DELTA_DEG = 10.0  # ± per-joint window around _RECOVERED_JOINT_POS

# Hardcoded collapsed-on-ground joint config.
#
# Generated offline by `find_collapsed_pose.py`: drops the arm with zero
# torque from a flopped-out pose, steps physics 8 s under gravity onto a
# floor plane, prints settled q. Re-run it (and paste the result here) if
# the robot model, gripper config, or floor geometry changes.
_COLLAPSED_JOINT_POS = (
     0.04832226,   # joint_1    +2.77°
     1.57844746,   # joint_2   +90.44°
    -0.86009896,   # joint_3   -49.28°
     1.69101930,   # joint_4   +96.89°
    -3.84382772,   # joint_5  -220.24°
     2.09003830,   # joint_6  +119.75°
    -4.01472187,   # joint_7  -230.03°
)

_RESET_RESETTLE_STEPS = 50   # ~0.1 s at 500 Hz: damp out reset jitter
_RESET_JITTER_DEG = 5.0      # per-joint jitter on the collapsed pose

# Per-joint torque limits (Nm) — same as Kinova Gen3 spec used elsewhere.
_TAU_MAX = (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)


# ---------------------------------------------------------------------------
# Joint-target command (sampled near _RECOVERED_JOINT_POS each episode)
# ---------------------------------------------------------------------------


class JointTargetCommand(CommandTerm):
    """Per-episode 7D target joint configuration sampled around a reference pose."""

    cfg: JointTargetCommandCfg

    def __init__(self, cfg: JointTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._reference = torch.tensor(cfg.reference_pose, device=self.device)
        self._delta = cfg.delta_deg * _DEG_TO_RAD
        self.target = self._reference.unsqueeze(0).expand(self.num_envs, -1).clone()

    @property
    def command(self) -> torch.Tensor:
        return self.target

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        delta = sample_uniform(
            torch.full((7,), -self._delta, device=self.device),
            torch.full((7,),  self._delta, device=self.device),
            (n, 7), device=self.device,
        )
        self.target[env_ids] = self._reference.unsqueeze(0) + delta

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        pass


@dataclass(kw_only=True)
class JointTargetCommandCfg(CommandTermCfg):
    reference_pose: tuple[float, ...] = _RECOVERED_JOINT_POS
    delta_deg: float = _GOAL_DELTA_DEG

    def build(self, env: ManagerBasedRlEnv) -> JointTargetCommand:
        return JointTargetCommand(self, env)


def _get_joint_target(env: ManagerBasedRlEnv, name: str = "joint_goal") -> JointTargetCommand:
    term = env.command_manager.get_term(name)
    assert isinstance(term, JointTargetCommand)
    return term


# ---------------------------------------------------------------------------
# Joint-PD action term with gravity compensation
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TorqueGravityCompActionCfg(ActionTermCfg):
    """7D normalized-torque action with gravity compensation.

    τ_i = a_i · τ_max_i + qfrc_bias_i

    a_i ∈ [-1, 1] is clamped before scaling. qfrc_bias is gravity + Coriolis
    read from `env.sim.data.qfrc_bias` (same source as OSC's bias term).
    """

    actuator_names: tuple[str, ...] = ("joint_.*",)
    """Actuator name expressions to resolve the 7 controlled arm joints."""

    max_torque: tuple[float, ...] = _TAU_MAX
    """Per-joint torque scale: a=±1 maps to ±max_torque (Nm)."""

    def build(self, env: ManagerBasedRlEnv) -> TorqueGravityCompAction:
        return TorqueGravityCompAction(self, env)


class TorqueGravityCompAction(ActionTerm):
    cfg: TorqueGravityCompActionCfg
    _entity: Entity

    def __init__(self, cfg: TorqueGravityCompActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        joint_ids, _ = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._num_joints = len(joint_ids)
        assert self._num_joints == 7, f"expected 7 arm joints, got {self._num_joints}"
        self._joint_dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids]

        self._tau_max = torch.tensor(cfg.max_torque, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._last_tau = torch.zeros(self.num_envs, 7, device=self.device)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def last_tau(self) -> torch.Tensor:
        return self._last_tau

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)

    def apply_actions(self) -> None:
        bias = self._env.sim.data.qfrc_bias[:, self._joint_dof_ids]
        tau = self._raw_actions * self._tau_max + bias
        # Clamp final torque so policy + grav-comp can't exceed actuator limits
        # (e.g. when the arm is fully extended horizontally and grav-comp alone
        # is near tau_max, an additional positive action would saturate hardware).
        tau = torch.clamp(tau, -self._tau_max, self._tau_max)
        self._last_tau[:] = tau
        self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._last_tau[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Reset events
# ---------------------------------------------------------------------------


def reset_collapsed_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    entity_name: str = "robot",
    collapsed_pose: tuple[float, ...] = _COLLAPSED_JOINT_POS,
    resettle_steps: int = _RESET_RESETTLE_STEPS,
    jitter_deg: float = _RESET_JITTER_DEG,
) -> None:
    """Spawn arm at the (hardcoded) collapsed pose ± jitter, then settle briefly.

    The collapsed pose is generated offline by `find_collapsed_pose.py` and
    pasted into ``_COLLAPSED_JOINT_POS``. Each reset:
      1. write q = collapsed + Uniform(±jitter_deg), q̇ = 0
      2. zero the effort target
      3. step physics ``resettle_steps`` times so the small jitter dies out
         and the policy sees a static-equilibrium start state.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[entity_name]
    n = len(env_ids)
    arm_ids = torch.arange(7, device=env.device)
    jitter_rad = jitter_deg * _DEG_TO_RAD

    soft = robot.data.soft_joint_pos_limits  # (B, J, 2)
    lo = soft[env_ids, :7, 0]
    hi = soft[env_ids, :7, 1]

    base = torch.tensor(collapsed_pose, device=env.device).unsqueeze(0).expand(n, -1)
    jitter = sample_uniform(
        torch.full((7,), -jitter_rad, device=env.device),
        torch.full((7,),  jitter_rad, device=env.device),
        (n, 7), device=env.device,
    )
    q0 = torch.clamp(base + jitter, lo, hi)
    qd0 = torch.zeros_like(q0)
    robot.write_joint_state_to_sim(q0, qd0, joint_ids=arm_ids, env_ids=env_ids)

    # Zero arm effort across ALL envs so leftover targets from the previous
    # episode don't fight the brief settle. Then step physics so the jitter
    # damps out before the policy takes over.
    zero = torch.zeros(env.num_envs, 7, device=env.device)
    robot.set_joint_effort_target(zero, joint_ids=arm_ids)
    for _ in range(resettle_steps):
        env.sim.step()


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def joint_pos(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    return env.scene[entity_name].data.joint_pos[:, :7]


def joint_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    return env.scene[entity_name].data.joint_vel[:, :7]


def joint_pos_error(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    command_name: str = "joint_goal",
) -> torch.Tensor:
    """q_goal - q (7D)."""
    q = env.scene[entity_name].data.joint_pos[:, :7]
    return _get_joint_target(env, command_name).target - q


def joint_goal(
    env: ManagerBasedRlEnv,
    command_name: str = "joint_goal",
) -> torch.Tensor:
    return _get_joint_target(env, command_name).target


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def joint_pos_tracking_reward(
    env: ManagerBasedRlEnv,
    std: float,
    entity_name: str = "robot",
    command_name: str = "joint_goal",
) -> torch.Tensor:
    """Gaussian on ||q - q_goal||."""
    q = env.scene[entity_name].data.joint_pos[:, :7]
    goal = _get_joint_target(env, command_name).target
    err_sq = torch.sum(torch.square(q - goal), dim=-1)
    return torch.nan_to_num(torch.exp(-err_sq / std**2), nan=0.0)


def success_bonus(
    env: ManagerBasedRlEnv,
    threshold: float = 0.1,
    entity_name: str = "robot",
    command_name: str = "joint_goal",
) -> torch.Tensor:
    """1.0 when ||q - q_goal||₂ < threshold (rad), else 0."""
    q = env.scene[entity_name].data.joint_pos[:, :7]
    goal = _get_joint_target(env, command_name).target
    err = torch.norm(q - goal, dim=-1)
    return (err < threshold).float()


def energy_dissipated(
    env: ManagerBasedRlEnv,
    action_term_name: str = "torque",
    entity_name: str = "robot",
) -> torch.Tensor:
    """Σ |τ_i · q̇_i| · dt — positive, penalize with negative weight."""
    term = env.action_manager.get_term(action_term_name)
    assert isinstance(term, TorqueGravityCompAction)
    tau = term.last_tau
    qd = env.scene[entity_name].data.joint_vel[:, :7]
    return torch.sum(torch.abs(tau * qd), dim=-1) * env.step_dt


class contact_impulse_penalty:
    """Penalize first-difference of total contact-force magnitude (damage proxy).

    Tracks ‖f_t‖ from a body-vs-terrain contact sensor and penalizes
    |‖f_t‖ - ‖f_{t-1}‖|. Steady contact ≈ 0; slamming spikes the value.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._sensor_name: str = cfg.params.get("sensor_name", "robot_ground_collision")
        self._prev_force_mag = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_force_mag[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "robot_ground_collision",
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        force = sensor.data.force
        if force is None:
            return torch.zeros(env.num_envs, device=env.device)
        f_mag = torch.norm(force, dim=-1).max(dim=-1).values  # (B,)
        impulse = torch.abs(f_mag - self._prev_force_mag)
        self._prev_force_mag = f_mag.detach()
        return impulse


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def joint_pos_error_norm(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    command_name: str = "joint_goal",
) -> torch.Tensor:
    q = env.scene[entity_name].data.joint_pos[:, :7]
    goal = _get_joint_target(env, command_name).target
    return torch.norm(q - goal, dim=-1)


def cumulative_energy(
    env: ManagerBasedRlEnv,
    action_term_name: str = "torque",
    entity_name: str = "robot",
) -> torch.Tensor:
    """Per-step dissipated energy — same as the reward term, exposed as metric."""
    return energy_dissipated(env, action_term_name, entity_name)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def kinova_recover_upright_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # --- Observations (~38D) ---
    actor_terms = {
        "joint_pos":       ObservationTermCfg(func=joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel":       ObservationTermCfg(func=joint_vel, noise=Unoise(n_min=-1.5,  n_max=1.5)),
        "joint_pos_error": ObservationTermCfg(func=joint_pos_error, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_goal":      ObservationTermCfg(func=joint_goal, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "actions":         ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = {**actor_terms}

    observations = {
        "actor":  ObservationGroupCfg(actor_terms,  enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    # --- Actions ---
    actions = {
        "torque": TorqueGravityCompActionCfg(
            entity_name="robot",
            actuator_names=("joint_.*",),
        ),
    }

    # --- Events ---
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_collapsed_pose": EventTermCfg(
            func=reset_collapsed_pose,
            mode="reset",
            params={
                "entity_name": "robot",
                "collapsed_pose": _COLLAPSED_JOINT_POS,
                "resettle_steps": _RESET_RESETTLE_STEPS,
                "jitter_deg": _RESET_JITTER_DEG,
            },
        ),
    }

    # --- Rewards ---
    rewards = {
        # Three-tier Gaussian on ||q - q_goal||₂ (~7 rad full collapse → 0 rad
        # at goal). Wide std=3.0 keeps a dense gradient even at full collapse;
        # mid std=0.8 takes over once the arm is roughly upright; tight std=0.2
        # rewards precise alignment.
        "joint_tracking_wide": RewardTermCfg(
            func=joint_pos_tracking_reward,
            weight=1.0,
            params={"std": 3.0},
        ),
        "joint_tracking": RewardTermCfg(
            func=joint_pos_tracking_reward,
            weight=2.0,
            params={"std": 0.8},
        ),
        "joint_tracking_tight": RewardTermCfg(
            func=joint_pos_tracking_reward,
            weight=3.0,
            params={"std": 0.2},
        ),
        "success_bonus": RewardTermCfg(
            func=success_bonus,
            weight=5.0,
            params={"threshold": 0.15},
        ),
        # Energy: penalize dissipated mechanical power.
        "energy": RewardTermCfg(
            func=energy_dissipated,
            weight=-0.05,
            params={"action_term_name": "torque"},
        ),
        # Impulse: penalize Δ‖contact_force‖ (damage proxy).
        "contact_impulse": RewardTermCfg(
            func=contact_impulse_penalty,
            weight=-0.02,
            params={"sensor_name": "robot_ground_collision"},
        ),
        # Smoothing.
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
        "joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.001,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",))},
        ),
        # Soft joint-limit nudge. Weight is small because the *collapsed start
        # state* legitimately sits at joint_6's hard limit (~120°) — that's
        # where the physical arm rests on the floor. A large penalty here
        # would dominate early-step rewards before the policy has a chance
        # to lift away from the limit.
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]",))},
        ),
    }

    # --- Sensors: catch contact between any arm link and the ground ---
    # Subtree-vs-floor on a 7-DOF arm + gripper produces dozens of simultaneous
    # contact matches when fully collapsed. We only care about the strongest
    # contact (reduce=maxforce → top slot) so a single slot is enough; the
    # sensor maxmatch limit is bumped via SimulationCfg.nconmax below.
    robot_ground_collision_cfg = ContactSensorCfg(
        name="robot_ground_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        global_frame=False,
    )

    # --- Terminations ---
    terminations = {
        "time_out":      TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=mdp.nan_detection, time_out=False),
    }

    # --- Metrics ---
    metrics = {
        "joint_pos_error_norm": MetricsTermCfg(func=joint_pos_error_norm),
        "energy_per_step":      MetricsTermCfg(func=cumulative_energy),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=4096,
            env_spacing=1.0,
            entities={"robot": get_kinova_robot_cfg_peginhole_osc()},
            sensors=(robot_ground_collision_cfg,),
        ),
        observations=observations,
        actions=actions,
        commands={
            "joint_goal": JointTargetCommandCfg(
                resampling_time_range=(1e9, 1e9),
                debug_vis=False,
                reference_pose=_RECOVERED_JOINT_POS,
                delta_deg=_GOAL_DELTA_DEG,
            ),
        },
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=1.5,
            elevation=-10.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            # Fully-collapsed arm + gripper produces ~70+ simultaneous contacts
            # against the floor. Bump both the per-world contact buffer (nconmax)
            # and the contact-sensor maxmatch buffer well above the observed peak.
            nconmax=200,
            njmax=2000,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(
                timestep=0.002,    # 500 Hz physics
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=10,        # 50 Hz policy rate (higher than pick_cube; PD wants faster control)
        episode_length_s=6.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg


def kinova_recover_upright_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return kinova_ppo_runner_cfg(experiment_name="kinova_recover_upright_jpd")
