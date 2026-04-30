# mjlab pull audit — findings (2026-04-30)

After fast-forwarding `/ihub/homedirs/svs_ald/sudhir/mjlab` from `67b288fd` to `12dc0db8` (65 commits), audited the three production tasks against the new mjlab API:

- `src/kinova_tasks/tasks/open_drawer_osc.py`
- `src/kinova_tasks/tasks/open_door_osc.py`
- `src/kinova_tasks/tasks/pick_cube_osc.py`

## Static API check — all clear

Every import and `mdp.*` / `manipulation_mdp.*` reference in the three files resolves against the new mjlab tree. Specifically verified:

- `mjlab.managers.*`, `mjlab.scene`, `mjlab.sensor`, `mjlab.sim`, `mjlab.rl`, `mjlab.entity`, `mjlab.utils.lab_api.math`, `mjlab.utils.noise`.
- `manipulation_mdp.illegal_contact`, `manipulation_mdp.joint_velocity_hinge_penalty`.
- `mdp.{action_rate_l, joint_pos_limits, joint_vel_l, last_action, nan_detection, reset_root_state_uniform, reward_curriculum, time_out}`.
- `mdp.dr.{dof_damping, dof_frictionloss, geom_friction, pseudo_inertia}` — the two `dof_*` names are aliases for `joint_damping` / `joint_friction`, still re-exported.
- `ContactSensorCfg` / `ContactMatch` kwargs all match the new dataclass shape.

## Behavioural changes worth knowing

1. **`mdp.reward_curriculum` is now a class, not a function.** It validates stage `step` ordering at construction (raises if non-monotonic). Calling convention `CurriculumTermCfg(func=mdp.reward_curriculum, params={"reward_name": ..., "stages": [...]})` is unchanged. Our PPO-iter → step-counter conversion already produces ordered stages, so no change needed.
2. **`ActuatorCfg.armature` and `frictionloss` defaults changed `0.0` → `None`** (PR #890). `None` preserves XML values. `go1_constants.py` sets `armature` explicitly, so unaffected — but any other config that omitted these and relied on them being zeroed will now keep XML defaults.
3. **XML actuator classes collapsed** (PR #857): `XmlPositionActuatorCfg` / `XmlMotorActuatorCfg` / `XmlVelocityActuatorCfg` / `XmlMuscleActuatorCfg` → single `XmlActuatorCfg(command_field=...)`. Same commit removed `DelayedActuator`, `DelayedActuatorCfg`, `DelayedBuiltinActuatorGroup`, and `sync_actuator_delays`. **Zero references** to any of these in our tasks.
4. **`CommandTerm.create_gui`** got two new optional kwargs (`on_change`, `request_action`). Our three custom command terms (`PickGoalCommand`, `DoorGoalCommand`, `DrawerGoalCommand`) don't override `create_gui`, so backward compatible.

## Stale code (not in the three target files)

- `src/kinova_tasks/tasks/peg_in_hole copy.py:744,756` calls `manipulation_mdp.reward_weight`, which was removed by mjlab PR #791 (consolidated into `mdp.reward_curriculum`). Filename has ` copy` — looks like an unused backup. Safe to delete; will fail if ever imported.

## Runtime blocker — venv needs rebuild

Trying `import kinova_tasks` against the existing `.venv` fails with:

```
ModuleNotFoundError: No module named 'mjviser'
```

Upstream mjlab now factors out viser conversions into an external `mjviser` package (`mjviser>=0.0.13`, pinned to git rev `1bdfd6fe79066b847a5f430000fcfbb53ec31a6f`). Required before any play/train can run. On a GPU node:

```sh
cd /ihub/homedirs/svs_ald/sudhir/manipulation
rm -rf .venv && uv sync
uv run python -c "import kinova_tasks; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Kinova' in t])"
uv run play Mjlab-Open-Drawer-Osc-Kinova --agent zero --num-envs 1 --viewer native
```

## mjlab pull conflict resolution (for the record)

`pyproject.toml`: auto-merged; kept local `torch>=2.6.0` pin over upstream `>=2.7.0`.
`src/mjlab/scripts/play.py`: manual resolution — kept both upstream `log_root` and our `stochastic` fields on `PlayConfig`.
`differential_ik.py`, `go1/__init__.py`, `go1/env_cfgs.py`: clean auto-merge.
