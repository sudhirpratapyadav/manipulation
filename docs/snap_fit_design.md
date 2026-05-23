# Snap-Fit Mechanism Design Notes

Goal: produce a *hard* snap (bottle-cap / USB feel) — insertion force rises
monotonically with almost no motion, hits a peak, then collapses to ~0 in one
or two timesteps. The actor should feel a clear "wall, wall, wall, **gone**"
profile, not a smooth Gaussian hump.

Current geometry (committed alongside this doc): peg base + two hinged
cantilever prongs (cuboid arms, half-embedded latch spheres on the outer
face); socket has top/bot flaps with half-embedded blocker spheres at the
back of the flap. Hinge spring `stiffness=0.5`, `damping=0.02`, inward range
`-0.45..0` rad with stiff soft-limit (`solimplimit="0.99 0.999"`).

This gives a *soft* bag-buckle-style snap because both contact surfaces are
spherical → contact normal rotates smoothly through the engagement, and the
hinge spring is linear in θ. Force profile is a smooth hump, not a wall.

## Idea A — Ridge + stiff hinge (sphere → edge)
Replace the socket blocker spheres with thin **ridges** (a long thin box
across the channel, raised on the flap inner face). The prong sphere has to
climb over a sharp edge. The axial-force component stays large until the
sphere center crosses the ridge crest, then drops abruptly.

Also tighten the inward hinge range (`-0.05..0` instead of `-0.45..0`) and
bump `stiffness` from 0.5 → 5–10. The prong becomes almost rigid against
inward bending, so force rises very high before any displacement happens.

- **Pros:** small geometry change, predictable, one tuning knob (ridge height
  + hinge stiffness).
- **Cons:** still no over-center, so once past the ridge the prong is mostly
  relaxed already. Energy release is modest.

## Idea B — Over-center toggle (best feel, hardest to tune)
Shape each prong outer face as a **wedge/ramp** that ends in a sharp
shoulder; matching ramp on the socket flap inner face. While the ramps slide
the contact normal is angled forward (resists motion) and the prong is being
deflected. When the prong shoulder clears the socket ramp crest, contact
transfers to the **back face** of the prong shoulder — the normal flips to
*pushing the prong forward*. Combined with the strain energy released from
the hinge, this is the true bottle-cap jerk.

- **Pros:** real over-center action; energy storage is geometric not just
  spring-y.
- **Cons:** needs careful 2D geometry on prong + socket; more contact pairs
  to tune; failure modes (prong getting stuck on the ramp) are subtle.

## Idea C — Hinge-asymmetric breakaway
Don't change geometry at all. Just tighten the hinge limit hard
(`range="-0.05 0"`) and crank `solimplimit` so the prong is almost rigid
until the geometric clearance is reached. The "force rises with no movement"
phrasing matches this literally.

- **Pros:** zero geometry change. Matches the verbal description exactly.
- **Cons:** very stiff limits may ring/destabilize in MJX; may need smaller
  timestep or more `iterations`. Release is still smoothed by the sphere
  contact arc.

## Idea D — Pure box-on-box, no hinge (user's idea)
Remove the hinge entirely. Prong = rigid box attached directly to peg base.
Socket blocker = rigid box on flap inner face. Push the peg in; MuJoCo's
contact solver resists penetration via `solref` / `solimp`. Force grows
with applied effort; once it exceeds the saturation impedance the boxes
interpenetrate enough that the contact normal flips and the peg pops through
in 1–2 timesteps.

The "wall then sudden release" feel is real, but the mechanism is **not**
"MuJoCo decides to let it pass" — it's that the soft constraint saturates,
and once exceeded the commanded displacement happens almost instantly.

- **Pros:** dead-simple geometry. Contact normal is purely axial → all
  resistance translates to actor-felt force. No spring or armature to tune.
- **Cons:**
  - Box-on-box stacked contacts can jitter in MJX; needs careful
    `cone="elliptic"` and `iterations` budget.
  - **No latching.** Once past, pulling −X back through the blocker takes
    the same threshold force. Snap doesn't retain the peg.
  - Visible interpenetration during the loading phase (cosmetic).
  - Tuning is via `solimp` / `solref` — less intuitive than spring stiffness
    or ridge height.

## Idea E — Hinge + flat boxes (Idea D with latching)
Keep the prong hinge so the prong can deflect inward under load and spring
back after release (this is what gives latching). But replace the
**contact features** at both ends — prong tip and socket blocker — with
**flat boxes** instead of spheres. The contact normal becomes purely axial
during loading (rising-force feel), the soft-constraint saturation gives the
cliff release, and the hinge spring snaps the prong back to its rest splay
so it sits on the far side of the blocker = latched.

- **Pros:** combines Idea D's axial-force feel with Idea A's latching.
  Smallest delta from current geometry (swap two sphere geoms for two box
  geoms per side, keep everything else).
- **Cons:** still tuning `solimp` + hinge stiffness together. Need to watch
  for solver chatter from flat-on-flat contact.

## Recommended exploration order
Before committing to any of these, build a force-probe test (see
`tests/snap_fit_force_probe.py`) that drives the EE +X at constant rate and
records force vs. time / displacement. Run it on:

1. Current geometry (baseline soft hump).
2. Idea C (just stiffen hinge — cheapest change to try first).
3. Idea E (hinge + flat boxes — best expected feel/effort tradeoff).
4. Idea D (no hinge, pure box-on-box — if E doesn't latch right).
5. Idea B (over-center) — only if E + D both feel wrong.

Each iteration: change geometry/params → run probe → save plot → compare to
the bottle-cap-shaped target curve.

## Force direction notes
The OSC controller commands EE pose deltas; the gripper holds the peg. When
the snap releases, **most of the felt jerk is "the wall went away,"** not
"something pushed me forward." So the perceived snap quality scales with
*how high force was just before release*. Hence the recommendation to make
the loading regime as stiff as possible (either via hinge stiffness, ridge
geometry, or saturating contact constraints) — whichever Idea wins, it
should max out resistance just before the cliff.
