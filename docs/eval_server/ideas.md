# Eval analysis server — design notes

This is a working design doc for the eval-analysis tool we want to
build on top of the data the `eval` script already produces. It's not
a final spec; it's the shape we're committing to so the next person to
touch this (you or me) doesn't waste time re-litigating decisions.

## What we're trying to build

A **trial analysis IDE**: a single web app where you can

- Load one or more eval-output directories
  (`run.json + runs.parquet + steps.parquet + states.h5 + model.mjb`)
- Filter / query the trial population (SQL-style)
- See aggregate plots (success vs DR axes, error histograms, etc.)
- Click a point on a scatter → load *that* trial's 3D replay
- Scrub the trajectory in 3D with linked time-series plots beside it
- Eventually: compare multiple eval runs side-by-side and detect
  regressions

The data shape is **populations of independent trials** with
**per-step time-series inside each trial**. The tool needs to be good
at both granularities.

This is explicitly *not* a "live MuJoCo viewer" — that's mjviser's
job. We're building the analysis layer on top.

## Non-goals (for v1)

- Replacing the eval script. `eval` keeps writing the same files; the
  server only reads.
- Live training visualization. wandb does that.
- Hosting / multi-user. Local single-user web app, run via
  `uv run eval-server <dir>`.
- Auth / accounts.
- Cross-machine cluster integration.

## What we already have (working today)

- `eval.py` writes per-eval-run dirs:
  ```
  <run_dir>/
      run.json        # task, ckpt, seed, axes, term names, code SHA
      runs.parquet    # 1 row per trial: ids, sampled init, frozen DR
                      # draws, episode_length, terminal_reason,
                      # success_terminal, success_dwell, ...
      steps.parquet   # 1 row per step: obs, action, reward+per-term,
                      # metrics, joint/ee/cube state, gripper, term flags
      model.mjb       # MuJoCo binary model (template — DR not baked in)
      states.h5       # /trial_<id>/states (T, nq+nv+na) float32
                      # /trial_<id>/mocap (T, nmocap, 7) float32
                      # both already in env-local frame
  ```
- `viz_eval.py` already proves the replay pattern: load model.mjb +
  states.h5, scrub frames via `mj_setState` + mocap write +
  `mj_forward`, drive a viser scene through `ViserMujocoScene`
  (mjlab's subclass of mjviser's class). Filter widgets are stock
  viser dropdowns/sliders.

So the **3D replay layer is solved**. What's missing is the population-
analysis layer (filtering, plots, click-to-select, multi-run).

## Architecture (the picks)

Three tiers, each with a single straight answer that we commit to:

### 1. Backend — FastAPI + DuckDB on parquet

- **FastAPI** for HTTP + WebSocket. Standard async Python web stack;
  good docs; integrates cleanly with the eval Python env.
- **DuckDB** queries `runs.parquet` and `steps.parquet` directly off
  disk. Sub-100 ms filter+aggregate at 1M rows on a laptop. SQL is
  the query language — maps cleanly to a textbox in the UI and to
  JSON-serialized filter expressions in the URL.
- **HDF5 random-access** for trial states. We already store one group
  per trial; loading a single trial's `(T, state_size)` array is
  O(trial_size), not O(file_size).
- **No ETL step.** Parquet on disk is the database. Adding a new eval
  run = dropping a new directory under `evals/`. The server scans for
  `run.json` files at startup and rescans on demand.

Why not Polars in-memory: rigid query API, no SQL out of the box;
DuckDB wins here because the URL/JSON filter format becomes a SQL
`WHERE` clause directly.

Why not browser-side DuckDB-WASM: tempting (no server) but the
MuJoCo replay needs server-side mjviser anyway, so we already need
a backend process. One server, one stack.

### 2. 3D replay — mjviser via the viewer we already have

- We **depend on mjviser** (via mjlab's `ViserMujocoScene` subclass)
  for the mesh/texture/heightfield conversion code that turns an
  `mj_model` into a Three.js scene. That's ~2k LOC of polished
  rendering we are not going to rewrite.
- We do **not fork mjviser**. Reasons:
  - Its UI is built from viser's stock Python widget API (folders,
    sliders, dropdowns). That API is fine for a single-scene viewer
    but is the wrong abstraction for an analytics dashboard
    (no plotly, no linked brushing, no real tables, no async data
    fetching).
  - Its maintainer explicitly pushes app-specific concerns to
    subclasses (mjlab is the documented pattern). Upstream PRs adding
    plotly/filtering/multi-trial would correctly be rejected.
  - One-person project, ~2k LOC. Fork divergence and merge pain are
    real costs.
- Concrete integration: replace the bespoke `viz_eval.py` viser scene
  setup with the `ViserMujocoScene` we already use, plus our own
  scrub/select logic on top via a thin wrapper. **That's the level of
  integration. Don't go deeper.**
- The viser server runs alongside FastAPI in the same process. The
  frontend embeds it via iframe at a known port. FastAPI routes
  `POST /select_trial/{id}` etc. into an in-process call that updates
  the viser scene's current frame.

### 3. Frontend — plain HTML + Plotly.js + a tiny JS state machine

- v0 is **one `index.html`** served by FastAPI. No build step. No
  React. ~500 lines of vanilla JS + a few Plotly charts + an iframe
  for the viser panel.
- **Plotly.js** for charts. We get `plotly_click`, `plotly_selected`,
  `plotly_relayout` events out of the box — exactly the
  scatter-click → load-trial UX we want.
- A **minimal client-side state object** (~50 lines): current
  filter expression, currently-selected trial id, currently-selected
  step. Plot click handlers update state; state changes fire HTTP
  calls to FastAPI; FastAPI updates DuckDB query results and the
  viser scene.
- A SQL `WHERE` textbox plus a few sugar dropdowns
  (`terminal_reason`, `success_terminal`, env_id range slider) that
  prefill the textbox. SQL for power, dropdowns for ergonomics.

We **rejected**:
- Streamlit — every interaction = full Python re-render; you
  already tried it and didn't like it. Confirmed: not the right tool.
- Dash — better than Streamlit but still server-reactive; not
  fast enough for the click-scatter-→-update-3D loop we want.
- Bokeh — best Python-native linked brushing, but server-side
  rendering and we'd still want plotly-level click events.
- Vega-Lite — declarative is elegant for linked brushing but the
  custom-data-fetch story is rougher than Plotly.js.
- React from day one — overkill for v0, will revisit when we hit
  the complexity wall.

## Data flow (concrete)

```
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI process (single)                                         │
│                                                                  │
│  HTTP routes                Viser server (in-process thread)     │
│  ┌──────────────────┐       ┌────────────────────────────────┐   │
│  │ GET /runs        │       │ ViserMujocoScene (mjviser)     │   │
│  │ GET /trials?...  │       │ - loaded on first /select_run  │   │
│  │ GET /steps?...   │       │ - one model.mjb at a time      │   │
│  │ POST /select_run │──────▶│ - listens at :8081 for iframe  │   │
│  │ POST /select_trl │       └────────────────────────────────┘   │
│  └──────┬───────────┘                                            │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                            │
│  │ DuckDB           │                                            │
│  │ - reads parquet  │                                            │
│  │   off disk       │                                            │
│  │ - h5py for state │                                            │
│  │   tensors        │                                            │
│  └──────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘
                       ▲
                       │  HTTP + iframe
                       │
┌──────────────────────┴───────────────────────────────────────────┐
│ Browser (single static HTML page)                                │
│                                                                  │
│  ┌────────────────┬──────────────────────┬───────────────────┐   │
│  │ Filter sidebar │ Scatter / histogram  │ Iframe to :8081   │   │
│  │ - SQL textbox  │ - Plotly.js          │ (viser scene)     │   │
│  │ - dropdown     │ - click → select     │                   │   │
│  │   sugars       │ - lasso → subset     │                   │   │
│  ├────────────────┴──────────────────────┤                   │   │
│  │ Trial detail: time-series stack,      │                   │   │
│  │ step slider linked to scene scrub     │                   │   │
│  └───────────────────────────────────────┴───────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

Single process, single server, single page. No build step. No
microservices. No Docker.

## Storage layout

Today, each eval run is its own directory. We keep that. Add a
top-level discovery convention:

```
docs/<task>/eval_results/
    <eval_run_name>/
        run.json
        runs.parquet
        steps.parquet
        states.h5
        model.mjb
    <another_eval_run_name>/
        ...
```

The server scans `--root` (default `docs/`) for any subdirectory
containing all five files. Each becomes a "run" in the runs list. No
sidecar index file; the filesystem is the index. (We can add a
SQLite index later if scanning gets slow.)

## API surface (v1)

Concrete enough that the frontend and backend can be developed in
parallel:

```
GET  /api/runs
       → list of available eval runs (name, task_id, agent, num
         trials, success_terminal mean, ckpt path)

GET  /api/run/<name>/meta
       → contents of run.json

GET  /api/run/<name>/trials?where=<sql>&limit=N&offset=M
       → JSON rows from runs.parquet matching the WHERE clause
       → empty WHERE = all trials

GET  /api/run/<name>/aggregate?groupby=<col>&metric=<col>&where=<sql>
       → group-by aggregation for plots (e.g. success_terminal
         mean by dr_object_mass bucketed)

GET  /api/run/<name>/trial/<id>/steps
       → all step rows for that trial (per-step time-series)

GET  /api/run/<name>/trial/<id>/states
       → raw bytes of states.h5 /trial_<id>/states + /mocap

POST /api/run/<name>/select
       → tells the viser-side scene to load model.mjb (only matters
         when switching runs)

POST /api/run/<name>/trial/<id>/replay
       → tells viser to load this trial's states+mocap; scrub
         controlled by subsequent step messages

POST /api/scrub?step=<n>
       → set current step on the active replay; scene updates
```

WebSocket equivalents will be added once we hit the latency limits
of HTTP polling for the scrub case.

## Milestones

### v0.1 — single-run, scatter-to-replay loop
- One eval run loaded at startup (CLI arg).
- One scatter plot (e.g. `terminal_object_to_goal_error` vs
  `dr_object_mass`, color by `terminal_reason`).
- SQL WHERE textbox.
- Click scatter point → viser iframe shows that trial; step slider
  scrubs.

This is the smallest thing that beats the current `viz-eval`
viser-only viewer. ~1 week of focused work.

### v0.2 — multiple plots + linked selection
- Add per-step time-series stack: cube_to_goal_error, joint_vel L2,
  reward over time. Linked to the same step cursor as the 3D scrub.
- Histogram of `episode_length` colored by `terminal_reason`.
- Lasso select on scatter → table of selected trials → "play all"
  cycles through them.

### v0.3 — multi-run comparison
- Sidebar: list of eval runs in `--root`. Toggle to load 2+ at
  once.
- Color encode by run name in scatters; plotly facets for
  histograms.
- "Compare to baseline" view: side-by-side success rate by axis
  bucket, with delta highlighting.

### v0.4 — regression detection
- Define a "regression" formally (e.g. confidence interval on
  success rate per (run, axis-bucket); flag drops > threshold).
- A "regressions" tab that lists where run B got worse than run A.
- Export as Markdown for `experiments_log.md`.

### v0.5 — polish + scale
- DuckDB-backed cross-run aggregations.
- React migration if v0.x became unwieldy.
- Time-synced trajectory scrubbing across multiple loaded trials
  (the only thing Rerun would have given us — we get it natively
  here).

## What we're explicitly *not* doing

- Forking mjviser. Use as a dependency only.
- Adding plotly/filtering to the viser side of the world.
  Viser handles the 3D scene; the dashboard is a separate web page.
- Embedding the eval runner. The server is read-only; running new
  evals stays a CLI step.
- Any auth, multi-user, hosting concerns. Local-only.
- Rerun. Defer at minimum to v0.5; the time-synced replay it offers
  we can build natively with linked plotly + viser scrub.

## Open questions

These should resolve as we build, not block design:

1. **How big does `steps.parquet` get?** At
   `1024 envs × 16 trials × 100 steps × ~20 cols ≈ 33M rows` per
   eval run, DuckDB will be fine, but loading all step rows into the
   browser for time-series plots is not. We'll need server-side
   downsampling (every Nth step) or LTTB.
2. **How do we let the user save a filter / a selection / a view as
   a "report"?** Markdown export with embedded plot images? A
   shareable URL with the filter expression? Defer to v0.3.
3. **Do we want screenshots / video export of replays?** mjviser
   inherits whatever viser supports. Defer to v0.4+.
4. **Per-env vs per-trial aggregation.** Current eval freezes DR
   per env (8 trials/env share a friction draw). A "by env"
   aggregation is meaningful. Make sure the API and UI both expose
   `env_id` as a first-class group-by axis.
5. **How does this share state with `experiments_log.md`?** Probably
   a one-way export: dashboard generates a "results table" snippet
   that gets pasted into the journal. Don't try to make the dashboard
   *write* the log; that's a recipe for stale diffs.

## Where this lives

- `src/kinova_tasks/eval_server/` — the server package
  - `__init__.py`
  - `app.py`         — FastAPI app + routes
  - `db.py`          — DuckDB queries against parquet
  - `replay.py`      — viser scene mgmt + state loader
  - `static/`        — index.html + JS + CSS
  - `cli.py`         — `eval-server <root>` entry point
- `pyproject.toml` — `eval-server = "kinova_tasks.eval_server.cli:main"`
- New deps: `fastapi`, `uvicorn`, `duckdb`. (mjviser is already a
  transitive dep via mjlab.)
