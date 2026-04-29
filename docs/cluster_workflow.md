# Cluster & W&B operational recipes

Verified 2026-04-29. Re-verify with the cluster-state cheatsheet at the
bottom if anything seems off.

## Cluster shape

- Single partition `1gpu` covering both GPU nodes (`dgx1`, `dgx2` — 8
  GPUs each). Partition `TIMELIMIT` is 20 days per job. No QoS-level
  wall caps.
- QoS tiers cap user GPUs:
  - `1gpu` → 1 GPU max
  - `2gpu` → 2 GPUs
  - `3gpu` → 3 GPUs
  - `8gpu` → 8 GPUs (`OverPartQOS` flag)

## "Once a job is running you can't add more" — partly true

A `--qos=NgpU` job reserves up to N GPUs for that user. Submitting any
second job that pushes the user over their highest active QoS cap → PD
with reason `QOSMaxGRESPerUser`. Caps are summed across allocations.

So: don't sequence single-GPU sbatches. Hold one big allocation and run
job-steps inside it.

## Holder pattern (use this by default)

`slurm/holder_8gpu.sh` reserves all 8 GPUs on dgx2 for 20 days and
sleeps. Everything else runs as `srun --jobid=<holder>` inside it.

Recipe:

```bash
# Acquire the holder (returns immediately; the job sits in the queue).
sbatch slurm/holder_8gpu.sh

# Find its id.
HOLDER=$(squeue -u $USER -h -o "%i" -n holder_8gpu)
echo "holder: $HOLDER"

# Wait for it to start (or just check `squeue -j $HOLDER` until ST=R).

# Launch a training run as a 1-GPU step inside the holder:
srun --jobid=$HOLDER --gres=gpu:1 -n1 --exclusive --pty \
     bash slurm/open_drawer_osc_phase0.sh &

# Different GPU step in parallel — pin via CUDA_VISIBLE_DEVICES if needed:
CUDA_VISIBLE_DEVICES=1 srun --jobid=$HOLDER --gres=gpu:1 -n1 --exclusive \
     bash slurm/open_drawer_osc_phase1.sh &

# Cancel the holder when all phases are done:
scancel $HOLDER
```

Notes:
- The holder `sbatch` script just `sleep`s, so the allocation is yours
  for the whole 20-day window.
- If `dgx2` has fewer than 8 free GPUs, the holder will sit PD until
  enough free up. Check `sinfo -N -o "%N %P %T %G %C"`.
- If `dgx1` is back up (was drained 2026-04-29 with `Duplicate jobid`,
  needs `scontrol update nodename=dgx1 state=resume`), edit
  `holder_8gpu.sh` to allow either node.

## W&B is the live source of truth

Slurm stdout buffers heavily under sbatch redirect. The local
`logs/*.log` can show only the wandb startup banner while W&B has
hundreds of iterations recorded. **Don't tail the local log to gauge
progress** — pull state via the W&B API:

```python
import wandb
api = wandb.Api()
run = api.run("sudhirpratapyadav-indian-institute-of-technology-jodhpur/"
              "mjlab-kinova-tasks-osc/<run_id>")

print(run.state, run.summary["Episode_Metrics/success_rate"])

hist = run.history(samples=200,
                   keys=["_step", "Train/mean_reward",
                         "Episode_Metrics/success_rate",
                         "Episode_Metrics/object_to_goal_error",
                         "Episode_Termination/nan_detection"],
                   pandas=False)
for row in hist[-5:]:
    print(row)
```

`WANDB_API_KEY` is in the slurm scripts:

```bash
export WANDB_API_KEY=$(grep -oP 'WANDB_API_KEY=\K\S+' \
                       slurm/open_drawer_osc_phase0.sh)
```

## When to cancel a run early

Cancel signals:
- `Episode_Termination/nan_detection > 0` and rising. Tolerable: rare
  bursts that return to 0 within a few iterations.
- `Train/mean_reward` flat or decreasing for ≥ 200 iterations after the
  first ~50 (warmup).
- `Episode_Metrics/success_rate` not rising after iter ~1000 in a config
  that previously trained to ≥ 80 % at the same iter (regression).
- `Train/mean_episode_length` collapsed to a small value (< 30) due to
  immediate termination — usually a reward function blew up.

Don't cancel on:
- A single iteration where one metric flickers — noise.
- Slow start in the first ~50 iters — JIT compile + entropy still high.
- NaN counts that *decrease* across iterations.

A healthy early-iter trajectory looks like: `success_rate` rising,
`object_to_goal_error` falling, `mean_reward` rising,
`nan_detection = 0`.

## Run naming & tagging

- W&B project: `mjlab-kinova-tasks-osc`
- Entity: `sudhirpratapyadav-indian-institute-of-technology-jodhpur`
- One W&B run per phase; tags `phase=N` plus a short variant tag.
- `--agent.experiment-name`: `open_drawer_osc_phaseN[_variant]`.
- `--agent.run-name`: short description (`baseline`, `init_pose_dr`,
  `drawer_friction_dr`).

## Cluster-state cheatsheet

```bash
sinfo -N -o "%N %P %T %G %C"           # nodes, GPUs, CPU usage
squeue -u $USER                         # my queue
squeue -p 1gpu -o "%.7i %.20j %.8u %R %b"  # partition queue with reasons
sacctmgr show qos format=Name,MaxTRESPerUser,Flags
scontrol show job <jobid>               # full job detail incl. exit code
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode  # post-mortem
```
