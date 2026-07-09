# generality-atlas

> Measure **generality** — skill-acquisition efficiency on genuinely novel tasks — contamination-free,
> with the coverage denominator as the headline and planted failures the instrument must catch before
> it is allowed to measure anything.

This is an instrument, not an intelligence. It never claims to measure AGI (the one formal definition
of general intelligence, Legg–Hutter universal intelligence, is provably uncomputable — that is a named
barrier, not an engineering obstacle). What it measures, exactly: how fast an agent's performance
climbs on task instances it has **never seen**, within a declared experience budget, across five
diagnostic families that each isolate one capability. Nothing beyond that declared universe is licensed.

## The design (reuse, not invention)
- **Diagnostic families** (bsuite's move — each isolates ONE capability): k-armed bandits
  (exploration) · seeded grid mazes, BFS-verified solvable (credit assignment) · hidden symbol
  bijections (rule induction) · cue/distractor/recall episodes (memory) · 4-bit parity (systematic
  computation).
- **Procedural freshness** (Procgen's move): every instance is generated from a seed at run time.
  There is nothing to memorize; a memorizer is structurally reduced to its true learning ability.
- **Metric names from the transfer-RL literature** (arXiv:2009.07888): jumpstart, AULC (area under
  the learning curve), final competence — normalized per family against an **analytic oracle** and a
  **random floor** (one floor, gridnav's, is sampled with its own fixed seed and labeled as such).
  Normalization is deliberately **unclipped**: clipping per-episode scores at zero turns symmetric
  noise into upward bias (the selftest caught this in the first cut — the random control read 0.26
  instead of ~0).
- **The aggregate is the profile + the GENERALITY FLOOR** — the minimum AULC across families, never a
  bare mean. A high mean with a low floor is narrowness, not generality, and the built-in narrow agent
  demonstrates exactly that signature (bandit spike 0.82, floor negative).

## The instrument must catch its planted failures before it may measure
`--selftest` gates every run (the same fail-closed discipline as the sibling tools):
- a **random control** must read ~0 everywhere, two-sided;
- a **tabular-Q learner** must separate clearly from the control where learning is possible, and the
  **memory family must floor it** — a memoryless learner cannot represent the cue, and the instrument's
  job is to display that gap, not hide it;
- a **planted narrow agent** (ignores observations) must show a spiky profile and a low floor;
- a **planted memorizer** (a frozen learner pre-trained on ONE leaked instance — the honest minimal
  contamination model, since instance identity is not recoverable from observations here) must ace the
  leaked instance (AULC 1.0) and floor on fresh ones (0.05);
- identical master seed ⇒ **bit-identical report** (reproducibility);
- with the optional `coverage-attestation` sibling present, each run can emit a scope-bound
  certificate: measured cells attest ok, declared-but-unmeasured cells DEFER, fail-closed.

## v0.5 — within-family difficulty transfer (controls-first)
`measure_transfer` compares learning a HARD instance from scratch vs after pretraining on EASY
instances of the same family (jumpstart delta, AULC delta — the literature's transfer metrics).
Difficulty axes preserve the action space (bandit: thinner best-arm margin; gridnav: larger grid;
memory: more distractors; parity: more bits); the sequence family is excluded from difficulty
transfer because its natural axes change the action space — declared, with the reason.

**No transfer number is reported unless both built-in controls pass in the same run (fail-closed):**
the POSITIVE control (parity's rule is instance-invariant, so same-difficulty transfer must be
strongly positive) and the NEGATIVE control (a random agent learns nothing and carries nothing, so
any nonzero delta means the harness itself is asymmetric). The negative control was originally
mis-specified as a bandit assumption and FAILED — the diagnosis found a real, unanticipated channel
(process-state transfer: a pretrained learner carries its annealed exploration schedule even where
instance knowledge cannot carry), which is now a named finding rather than a control. Transfer has
more channels than instance knowledge; each control isolates one.

First controls-gated findings on the built-in agents: transfer is REAL and STRUCTURED — strongly
positive where the learned representation is difficulty-invariant, near zero where observation keys
are disjoint across difficulties, and genuinely NEGATIVE where stale values plus an annealed
exploration schedule interfere on harder variants. Negative transfer is reported as measured, never
hidden.

## v0.6 — generality-per-budget (does compute buy breadth or depth?)
`measure_efficiency` sweeps the experience budget (×0.5 / ×1 / ×2) and reports how the **floor** moves,
because generality (Chollet) is skill-acquisition *efficiency*, not accumulated skill. The load-bearing
metric is the **breadth-vs-depth split**: when you spend marginal budget, does the FLOOR rise (the
weakest family improves = breadth) or only the MEAN (the strong families get stronger = depth)?

- `breadth_ratio` = Δfloor / Δmean across the sweep. ~1 = budget bought breadth; ~0 = budget bought only
  depth. It is reported **only when overall competence actually rose** (Δmean past a threshold); a
  saturated or non-learning agent gained nothing, so the split is `None` ("undefined"), never a
  fabricated number.
- `budget_to_floor` = the experience needed for the floor to reach a target (default 0.30), or `None`
  when it is never reached within the swept budgets — **fail-closed, not a zero.**

This makes budget an *assurance* axis, not a bare metric: it catches a "more compute → more general"
claim that is really "more compute → narrower but deeper." The standing demonstration is the **same**
tabular-Q learner shown two ways — on a universe it can cover (the four non-memory families) budget buys
**breadth** (floor rises with the mean, `breadth_ratio` ≈ 0.9, target floor reached); on the **full**
universe, where the memory capability is structurally unlearnable for it, the same budget buys **depth
only** (mean rises, floor memory-pinned near zero, `breadth_ratio` ≈ 0.2, `budget_to_floor` = `None`).
More compute did not make it general — and the axis says so.

## v0.7 — extension families (the universe grows deliberately, not accidentally)
`EXTENSION_FAMILIES` holds families that are **built, controlled, and measurable** (via
`measure(families=["changepoint"])`) but not yet in the default universe — because adding a family
re-baselines every existing number (the floor is a min; a new weakest family rewrites the headline).
Promoting an extension into `FAMILIES` is an explicit re-versioning of the declared universe, made in
the open.

First extension: **changepoint** (capability: ADAPTATION). A hidden per-instance bijection silently
switches to an everywhere-different bijection partway through 40 episodes — the switch is unsignaled;
detecting and recovering from it *is* the capability. The switch episode is **randomized per
instance** (16–24, drawn from the instance seed): the moment an optimizer trains against this family,
a fixed switch episode becomes an exploitable schedule regularity — a referee's incidental
regularities are attack surface, so they are randomized. The capability-isolating signal is the
**post-switch slice** of the curve: whole-window AULC can be padded by strong pre-switch acquisition
while adaptation is literally zero, so gates use conservative shared slices valid for every instance
(`CHANGEPOINT_PRE_SLICE`/`CHANGEPOINT_POST_SLICE`: pre = episodes below 16, post = 28 and beyond). Controls, planted before
any measurement is trusted: the random agent must read ~0 across the switch (harness symmetry);
fixed-step tabular-Q must visibly recover (its constant α forgets — post-switch ≈ +0.3); and a
planted **non-adapter** (competent, then frozen at the switch) must score *worse than random*
post-switch (≈ −0.16 — it confidently does the old right thing). First finding worth stating: an
agent's acquisition strength says nothing about its adaptation — a running-mean learner that tops
every stationary family can sit at random on the post-switch slice, because means that never forget
and exploration that has decayed to nothing are exactly the wrong equipment for a changed world.

## Run (zero dependencies — Python standard library)
```
python3 generality_atlas.py --selftest      # the gate; must pass before anything is measured
python3 generality_atlas.py --run           # the atlas for the built-in baselines
python3 generality_atlas.py --transfer      # controls-gated difficulty-transfer measurements
python3 generality_atlas.py --efficiency    # generality-per-budget: the breadth-vs-depth split
python3 generality_atlas.py --run --json    # machine-readable report (+ attestation if available)
```

Any agent that speaks the protocol can be measured: `act(obs) -> action`,
`update(obs, action, reward, next_obs, done)`, `episode_end()`.

## Claim-licensing ledger (how claims scale with evidence — open-ended by design)
1. **Claims are licensed by the declared universe and the measured evidence — nothing more, at any
   given time.** Today's universe (five toy families, these budgets) licenses no claim about general
   intelligence. That is a statement about today's evidence, not a ceiling on the project.
2. **The path to bigger claims is declared expansion.** Grow the family universe, the budgets, the
   difficulty axes, the transfer measurements — and what may honestly be claimed grows exactly as
   fast as the evidence does, and no faster. The ambition is unbounded; the wording is always bound
   to the current reference.
3. **The floor is the headline.** A mean without its floor misrepresents — the built-in narrow agent
   is the standing demonstration (respectable mean, negative floor).
4. **v0 defers transfer** — cross-family transfer is ill-defined for tabular agents (different
   observation/action spaces). Deferred with this reason, not hidden; v0.5 = within-family difficulty
   transfer, and transfer is exactly where the interesting generality evidence will live. v0.6 =
   generality-per-budget (the breadth-vs-depth split), which fail-closes `budget_to_floor` to `None`
   when a capability is unlearnable at any swept budget rather than reporting a flattering low number.
5. **The contamination-free protocol is the one permanent rule** — fresh instances per run, oracle
   grading, no judge in the verdict path. It is permanent because it protects measurement validity,
   not because it limits ambition: whatever this project one day claims, it will be able to prove it
   on instances nothing could have memorized.

## Lineage + what this is for
Sibling of [honesty-atlas](https://github.com/MonongahelaHellbender/honesty-atlas) (the same
contamination-free, oracle-graded, classified-failure discipline, pointed at generality instead of
calibration). The companion seed agent — an observe/plan/act/learn runtime this instrument will
eventually measure — lives in a separate WIP repo. The instrument came first, deliberately: audit
before capability. If a garage-scale system ever makes real progress toward general learning, an
instrument like this is the only way its builder would honestly KNOW; and if none does, the instrument
is the durable artifact.

MIT © Melissa Ellison. Standard library only. Nothing leaves your machine.
