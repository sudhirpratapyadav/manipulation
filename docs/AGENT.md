# AGENT.md — Autonomous research entrypoint

You are continuing an autonomous research effort on the Kinova Gen3
open-drawer manipulation task. This file is the entrypoint: read it, then
execute the project until done. The user is hands-off.

## Goal

Train and ship a **more robust** OSC-based open-drawer policy
(`Mjlab-Open-Drawer-Osc-Kinova`). "Robust" means: the policy succeeds
across a wide envelope of physical and initial-condition variations, not
just at the training nominal. Sim2real is the eventual target — the real
Kinova Gen3 + Robotiq 2F-85 already runs the existing checkpoint but
unreliably.

The deliverable is a final policy (Phase 4) plus a written record of
which interventions improved the operating envelope and which didn't.

## Autonomy contract

- **Make decisions.** The plan defines phases; within and between them, you
  decide. Borderline keep/drop calls, plan deviations, new DR axes, new
  evaluation ideas — all yours. Document each call under "Autonomous
  decisions" in `experiments_log.md`.
- **Push to `main` at phase milestones** (one commit per finished phase
  with a clear message). Don't push WIP between checkpoints.
- **Use the 8-GPU holder pattern** (see `cluster_workflow.md`) to run
  multiple experiments in parallel. Sequential single-GPU runs waste
  days.
- **Cancel doomed runs early** per the cancel-signals in
  `cluster_workflow.md`. Don't burn 24 h on a run that has already
  flat-lined or NaN'd.
- **W&B is the source of truth for live progress.** Don't tail the local
  Slurm log waiting for output — it's buffered. Pull state via
  `wandb.Api()` (snippet in `cluster_workflow.md`).
- **Don't touch upstream `mjlab/`** unless a kinova_tasks change is
  blocked on a clear mjlab bug. Anything in `kinova_tasks` is fair game.
- **Don't bypass safety hooks** (no `--no-verify`, no force-pushes).

You will *not* stop and ask the user for guidance. If something is
genuinely ambiguous, pick the more conservative interpretation, log the
call, and continue.

## Where things live

| Path | What |
|---|---|
| `docs/AGENT.md` | this file |
| `docs/open_drawer_improvement_plan.md` | the phased plan of record |
| `docs/cluster_workflow.md` | SLURM + W&B operational recipes |
| `docs/experiments_log.md` | per-phase journal — append, don't rewrite |
| `src/kinova_tasks/tasks/open_drawer_osc.py` | training task config + metrics |
| `src/kinova_tasks/eval_sweep.py` | Lane A: mjlab in-process OOD sweep |
| `nn_policy/sim2real_open_drawer_osc.py` | Reference for Lane B: open-drawer task-specific async + OSC plumbing (don't run as-is — has real-robot path) |
| `nn_policy/sim_osc.py` | Reference for Lane B: minimal sim-only async driver shape, no task-specific logic |
| `slurm/open_drawer_osc_phase0.sh` | baseline training slurm script |
| `slurm/eval_open_drawer_osc.sh` | mjlab eval slurm script |
| `slurm/holder_8gpu.sh` | 8-GPU holder for parallel job-steps |

Task IDs registered:
- `Mjlab-Open-Drawer-Osc-Kinova` — training
- `Mjlab-Open-Drawer-Osc-Kinova-Eval` — eval (finite episode, no obs corruption,
  no curriculum)

## What is already in place (and what to verify)

Each item below was set up in a prior session. Treat the *intent* as
fixed and the *implementation* as a starting point: if a piece is buggy,
incomplete, or insufficient for the sim2real-robustness goal, fix or
extend it. Don't preserve a broken implementation just because it's
already there. Log every non-trivial change under "Autonomous decisions"
in `experiments_log.md`.

- **`success_rate` metric** on the training task: per-step indicator
  `object_to_goal_error < 0.02 m`, averaged across the episode by
  `MetricsManager` → "dwell success" (fraction of steps within
  tolerance). The threshold (2 cm), the form (dwell vs. terminal), and
  even the existence of additional metrics are open. If you decide a
  *terminal* success is more meaningful for sim2real, add it (mjlab
  doesn't have `reduce="last"` natively but you can implement it as a
  custom metric class that holds the latest per-env value and emits it
  on episode end). Whatever you change, keep the metric definition
  fixed across the phases you compare.
- **Eval task variant** (`Mjlab-Open-Drawer-Osc-Kinova-Eval`)
  registered with the finite-episode config (corruption off, curriculum
  off). Confirm it actually behaves as you expect end-to-end.
- **Sweep harness** (`src/kinova_tasks/eval_sweep.py`) — drafted but
  **not end-to-end tested**. Several pieces need verification before
  trusting any number it produces:
  - Per-axis override functions actually take effect (e.g. does
    `dr.dof_frictionloss` with a single-value range produce a
    deterministic friction in the resulting `mjModel`? Confirm by
    reading the field after env build).
  - The vec-env reset/dones bookkeeping correctly attributes a final
    `object_to_goal_error` to each terminated episode (auto-reset
    timing in `RslRlVecEnvWrapper.step` may shift the meaning of
    "previous-step error").
  - The `episodes_per_setting=64` × `num_envs=64` loop terminates in
    bounded time — `max_steps` is a hand-rolled estimate that may be
    too generous or too tight.
  - Whether sweeping each axis *deterministically* (single-point
    range) is the right thing, vs. sampling within a band per env.
    Deterministic is simpler to read; sampled is more honest about the
    real-world distribution.
  - Whether 10 axes captures the failure modes that matter for *this
    robot's* sim2real gap. Add axes if you find new failure modes;
    drop or merge axes that are uninformative.
- **Robustness score** (mean normalized envelope width where
  `success_rate ≥ 0.80`) is the headline. The 0.80 threshold and the
  "pass / degraded / fail" cuts are first-pass defaults; tighten or
  loosen if results say so. Whatever you pick, keep it fixed across the
  phases you compare.
- **Slurm scripts** and **W&B project wiring** (`mjlab-kinova-tasks-osc`)
  — see `cluster_workflow.md`.
- **`nn_policy/` directory** — sim2real deployment scripts
  (real-robot process + sim process running asynchronously, OSC at
  500 Hz, policy at 50 Hz, viser viz, keyboard input). **No real
  robot is connected**, so the hardware path is out of scope. Use
  these scripts as **references** for how to wire async OSC + a
  trained policy outside mjlab; **don't run them as-is** for eval
  (they boot a real-robot process, viser GUI, and keyboard listener
  none of which a batch-eval agent should touch). The right thing
  is to write a **new headless sim-only driver** for open-drawer,
  modelled on `sim2real_open_drawer_osc.py` (same task) plus
  `sim_osc.py` (cleaner sim-only shape). Falling back to running the
  existing script with its `--sim-only` flag is acceptable only if
  writing a clean new driver is genuinely blocked. See "Eval lanes"
  below.

A baseline run was started 2026-04-28 (job 18266, W&B run `qifkjgn6`)
and cancelled at iter ~250 to switch to the parallel-experiment
workflow. No checkpoint kept. **Phase 0 needs to be re-run from scratch**
under the holder pattern, *after* you've verified the sweep harness on a
quick smoke test (low num_envs, one axis, few episodes).

## What "robust" actually means here

There is **no real robot connected to the host**. All training and
evaluation are simulated. The eventual deployment target is a real
Kinova Gen3 + Robotiq 2F-85, but that's the user's concern, not yours
in this loop. Your job is to make the policy *behave well in sim under
variations that plausibly model the real-world conditions the user
will face when they later deploy*.

That framing matters because it's easy to drift into "add DR to make
the metric move" — which is shallow and doesn't help the eventual
deployment. Instead, every variation you train against or evaluate on
should pass a sniff test: **could a real lab cabinet, a real Kinova,
and a real Robotiq gripper actually exhibit this?**

Variations that pass the sniff test (use these):
- Drawer base weight (cabinet contents differ; "empty kitchen drawer"
  vs. "drawer full of cutlery").
- Drawer slide friction / damping (manufacturing variance, dust, age,
  whether the user oiled the rails).
- Drawer goal depth (user wants the drawer pulled different amounts).
- Drawer initial position (cabinet not always closed flush).
- Robot base placement (technician sets the robot down at slightly
  different positions / yaw each run).
- Initial joint pose (Kinova homing isn't identical run to run; user
  might start from a slightly different "ready" pose).
- Fingertip friction (rubber pad wear; cleanliness).
- Arm-link mass calibration (URDF vs. real never matches exactly).

Variations that fail the sniff test (don't add):
- Things that require a different task (obstacles, multiple drawers,
  vision-based goal).
- Numerical perturbations with no real-world correlate (e.g. random
  Gaussian noise on internal MuJoCo solver tolerances).
- Penalties or DR added "to regularize" without a stated real-world
  failure mode.

Decision pattern: when adding or removing a sweep axis or DR term,
**state the real-world phenomenon it models** in the
`experiments_log.md` decision entry. If you can't write a one-line
real-world rationale, drop the axis.

If a phase change improves the robustness score but the per-axis
breaking points reveal it's all coming from one easy axis while a
sim2real-relevant axis got worse, **drop the change** even though the
headline number rose.

## Robustness — operational definition (default; revise if better)

For each sweep axis, the **passing envelope** is the largest contiguous
range of swept values around the training nominal where
`success_rate ≥ 0.80`. Normalize by the total swept range. Average
across the axes → one scalar in `[0, 1]`. Companion: per-axis
breaking-points table. This is what `eval_sweep.py` writes to
`breaking_points.md`.

This is the *default* definition. If you find a better aggregate (e.g.
weighting axes by sim2real relevance, geometric mean instead of
arithmetic, percentile of per-episode success instead of dwell-mean),
adopt it — keeping the new definition fixed across the phases you
compare. Document the change in `experiments_log.md` so a later session
knows what's being measured.

## Eval lanes (use both)

Two complementary ways to evaluate a checkpoint. The first is the
primary one for phase-to-phase comparison; the second is a periodic
sanity check.

**Lane A — mjlab in-process sweep (`eval_sweep.py`).**
Synchronous, fast, runs many configurations per minute. Source of the
robustness score and per-axis breaking-points. Run after every phase.
Strength: cheap, controllable, easy to script. Weakness: same physics
engine and same vec-env timing the policy was trained against, so it
won't catch policies that overfit to mjlab specifics.

**Lane B — async sim-only test, headless, write your own.**
The point of Lane B is to run the policy with the **same async timing
structure a real Kinova would use** — 50 Hz outer policy loop, 500 Hz
OSC inner loop, separate threads/processes, against MuJoCo-native
physics (not mujoco_warp). Same physics engine the real-robot drivers
use, different from the synchronous batched simulator the policy was
trained on. The combination of (a) different engine and (b) async
timing makes Lane B catch overfitting to mjlab specifics that Lane A
can't.

**Write your own driver.** Suggested name:
`nn_policy/sim_open_drawer_osc_async.py`. References:
- `nn_policy/sim2real_open_drawer_osc.py` — same task, has the right
  obs/action shapes, the right OSC compute, the right handle FK.
  Strip the real-robot process, the viser viz thread, the keyboard
  listener, the GUI/main loop. Keep the sim thread, the OSC thread,
  the policy thread, and the multi-process / shared-memory glue
  between them.
- `nn_policy/sim_osc.py` — cleaner sim-only shape (no real-robot
  path to factor out). Less task-specific so you'll need to bring
  the open-drawer details from the file above.

A clean Lane B driver is a small headless multi-process script that:
1. Loads the policy checkpoint.
2. Builds a MuJoCo-native sim of the open-drawer scene.
3. Runs N episodes; each episode applies one (axis, value) override
   from the same set the user listed in "What 'robust' means" (drawer
   mass, friction, init slide, base offset, etc.).
4. Records terminal success and final error per episode.
5. Prints / writes a small results JSON.

No GUI, no keyboard, no human in the loop.

**Falling back to running `sim2real_open_drawer_osc.py --sim-only`**
is acceptable only if writing a new driver is genuinely blocked
(e.g. an mjlab/MuJoCo API change broke something subtle). In that
case, document the fall-back as an autonomous decision in
`experiments_log.md`. Otherwise, write the clean driver — it'll
also serve as the reference for the user's eventual hardware work.

Don't modify the existing `sim2real_*_osc.py` scripts; they belong
to the user's hardware path.

Use Lane B occasionally, not after every phase: at minimum after
Phase 0 (to confirm the trained policy works outside mjlab at all)
and after Phase 4 (the deployment candidate). Optional after any
phase that touches OSC parameters or actions.

Lane B is a **pass/fail signal**, not a sweep: a small set of episodes
under nominal conditions plus maybe 2–3 sim2real-relevant variations
(drawer mass, friction). If the policy that achieves robustness 0.7
in Lane A drops to 0 in Lane B, it's overfitting to mjlab —
investigate before continuing.

## Plan summary (read `open_drawer_improvement_plan.md` for details)

Each phase = one change → train → in-dist eval → OOD sweep → keep/drop
decision. Phases:

- **Phase 0**: baseline (no model change). Reference numbers.
- **Phase 1**: bigger init-pose randomization (joint delta + base XY/yaw).
- **Phase 2**: targeted DR — drawer slide friction/damping, drawer mass,
  arm-link mass.
- **Phase 3**: slow execution. Try 3a→3e in order, stop at first that
  works.
- **Phase 4**: combined deploy run with all keepers, possibly wider DR.

Keep rule per phase: **success_rate must not drop AND** at least one of
{robustness score, speed metric} must improve. Otherwise drop.

## Resume sequence

When starting a fresh session, do this in order:

1. `git status` — confirm a clean tree on `main`.
2. Read `docs/experiments_log.md` to see what state we're in.
3. **Verify the sweep harness before trusting any number from it.**
   - Quick smoke test: build the eval task at `num_envs=4` on a single
     GPU, run one axis with two values and a tiny number of episodes,
     and confirm the override actually changed the relevant `mjModel`
     field (read it back after `env = ManagerBasedRlEnv(...)`).
   - Confirm episode-level success_rate matches a hand-computed value
     on the same trajectories. If they disagree, fix the harness.
   - Improve / extend / replace the harness as needed — see "What is
     already in place" for known soft spots. The point is to have
     something whose numbers you trust before burning compute on
     phases.
4. `squeue -u $USER` — see if a holder allocation is already alive.
5. If no holder: `sbatch slurm/holder_8gpu.sh` (recipe in
   `cluster_workflow.md`).
6. Run Phase 0 baseline as a job-step inside the holder. While it
   trains, you can in parallel: refine the sweep, write Phase 1's
   slurm script, dry-run env builds at small num_envs, etc.
7. After Phase 0 finishes: in-dist eval, OOD sweep (Lane A), and a
   Lane B async-sim sanity check (write the open-drawer driver under
   `nn_policy/` if it doesn't exist yet). Log results, move to Phase
   1. Repeat through Phase 4. Lane B is required at Phase 0 and Phase
   4; optional in between.

## When to stop

Stop only when:
- All 5 phases (0–4) are complete and recorded in `experiments_log.md`,
  *or*
- A hard blocker that's not in your control (cluster outage, repeated
  unexplained training failures, mjlab API change you can't work around).

In either case, leave a clear "Next action for the user" entry at the top
of `experiments_log.md`.
