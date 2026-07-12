# generality-atlas

> Measure **generality** — skill-acquisition efficiency on genuinely novel tasks — contamination-free,
> with the coverage denominator as the headline and planted failures the instrument must catch before
> it is allowed to measure anything.

This is an instrument, not an intelligence. It never claims to measure AGI (the one formal definition
of general intelligence, Legg–Hutter universal intelligence, is provably uncomputable — that is a named
barrier, not an engineering obstacle). What it measures, exactly: how fast an agent's performance
climbs on task instances it has **never seen**, within a declared experience budget, across eleven
diagnostic families that each isolate one capability. Nothing beyond that declared universe is licensed.

## The design (reuse, not invention)
- **Diagnostic families** (bsuite's move — each isolates ONE capability): k-armed bandits
  (exploration) · seeded grid mazes, BFS-verified solvable (credit assignment) · hidden symbol
  bijections (rule induction) · cue/distractor/recall episodes (memory) · 4-bit parity (systematic
  computation) · unsignaled mid-stream rule switches (adaptation) · repeated regime changes at a
  jittered cadence (sustained adaptation) · three conflicting rules over shared symbols
  (interference resistance) · an N-step chain with reward only at the far end (deep exploration) ·
  hidden primitive bijections whose held-out compositions must be answered without relearning
  (compositionality) · opaque codes with a hidden functional dependency between question types that
  must be discovered from behavior alone (factored-structure discovery).
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
  demonstrates exactly that signature (bandit spike 0.84, floor negative).

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
tabular-Q learner shown two ways — on a universe it can cover (bandit/gridnav/sequence/parity) budget
buys **breadth** (floor rises with the mean, `breadth_ratio` ≈ 0.85, target floor reached); on the
**full** universe, where two capabilities are structurally out of reach for it (memory — no cue
representation; deep exploration — eps-greedy fails exponentially), the same budget buys **depth
only** (mean rises, floor pinned near zero by the unlearnable capabilities, `breadth_ratio` ≈ 0.1,
`budget_to_floor` = `None`). More compute did not make it general — and the axis says so.

## v1.4 — the eleven-family universe (fifth promotion: at a measured price of exactly zero)
**factored** was promoted at a floor cost of **0.0000** — its whole-window score clears every
holdout master's current floor binder, so the eleven-family floor equals the ten-family floor to
four decimals. This is the discipline's cleanest arc end-to-end: the family was built as a
referee; its verdict named the missing capability; the mechanism it prescribed was built and
validated until the subject agent matched the family's own reference solver per draw; and only
then was the family promoted — hold-as-referee-until-strength, then free expansion. (Compare
v1.3's openly priced trade and v1.2's hold-then-promote: the promotion decision has now run in
all three honest modes.)

## v1.3 — the ten-family universe (fourth promotion: priced in the open, not free)
**compose** was promoted the same day it was built — the fastest referee-to-denominator arc yet,
because the mechanism it existed to referee arrived within hours of its verdict. And unlike the
prior promotions, this one is an **openly priced floor trade**: the ten-family floor is lower than
the nine-family one (compose binds on a majority of holdout seeds), traded deliberately for a tenth
capability — combinatorial generalization, the one ARC-style evaluation targets — on which the
subject agent now reads within reach of the family's own reference composer. The claim-licensing
rule is unchanged: the headline moved DOWN and says so; a universe that only ever grows when it
flatters the number would be a marketing instrument, not a measurement one.

## v1.2 — the nine-family universe (third promotion: the hold-then-promote arc completes)
**deepchain** joined the default universe on 2026-07-11 — and its path there is the discipline's
best demonstration, because it ran in **both directions**. Built as an extension on 2026-07-10, it
was **deliberately held**: the fresh-family test showed the strongest measured agent reading far
below the family's own positive-control solver, so promotion then would have crashed the headline
floor without informing anything — the family's verdict was "this capability is still missing," and
the family's job was to referee the missing mechanism when it arrived. It arrived: a measured
decision brief showed the nine-family floor is no longer deepchain-bound (the refereed mechanism now
clears the family's own positive-control solver), so the expansion strengthens the claim instead of
trading the headline for completeness. Universe growth stays deliberate in both directions —
promoted when it strengthens the claim, held when the verdict is "still missing."

## v1.1 — the eight-family universe (second promotion, same day)
**multirule** joined the default universe the evening of the same day, by the same discipline: built
as an extension, validated against its own planted failure class (a phase-blind learner that
generalizes over the rule signal and collapses to majority-consistent mush), promoted on a measured
brief — the eight-family floor for a genuinely general agent came out *higher* than the seven-family
one, so the expansion strictly strengthens the claim. (Family nine, deepchain, followed two days
later via the hold-then-promote arc above — numbers in this README are stated against the current
nine-family universe unless a section says otherwise.)

## v1.0 — the seven-family universe (re-versioned 2026-07-09)
Both adaptation families below were built as **extensions**, validated against their own planted
failure classes, and then **promoted into the default universe in one deliberate re-versioning
commit** — after a measured decision brief showed the expansion strictly strengthens the claim (a
seven-capability floor subsumes the five-capability one) and before any earlier numbers had been
published (retraction-free timing). (Superseded the same evening by v1.1, and by v1.2 two days
later — see those sections for the current universe.)

## v0.7 — extension families (the universe grows deliberately, not accidentally)
`EXTENSION_FAMILIES` holds families that are **built, controlled, and measurable** but not yet in
the default universe (all six extensions to date were promoted — one after a deliberate hold, one
as an openly priced floor trade, one at exactly zero price; the slot is currently empty) — because adding a family
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

Second extension: **changepoint_sustained** (capability: SUSTAINED ADAPTATION — a different
capability from single-shock adaptation, and the distinction is measured, not asserted: a mechanism
can recover from one switch and still collapse over six). The regime re-randomizes every 10–12
episodes (cadence drawn per instance — jittered so an optimizer cannot learn the schedule) across 60
episodes; the signal is the **late slice** (`CHANGEPOINT_SUSTAINED_SLICE`, past ≥3 switches for every
draw): steady-state readaptation. The planted failure is the plasticity-loss class itself — a
running-mean learner (α=1/n) whose never-forgetting values deadlock under repeated change (late
slice below random, −0.07, vs fixed-step tabular-Q's steady +0.22).

## Family nine: deepchain (deep exploration) — held first, promoted when its referee job completed
**deepchain** (bsuite's deep-sea design, reused not reinvented): an N-step chain where one
per-instance action advances and the other resets, with reward only at the far end — the random
floor is analytic and essentially zero (0.5^N). The positive control is an **optimistic-init
bootstrapped Q** (untried paths look promising; value propagates backward — the classic solver);
the displayed contrast is eps-greedy tabular-Q failing exponentially (0.49 vs 0.03, the
myopic-exploration split made visible). It sat in the extension slot **deliberately** while the
strongest measured agent read well below the solver — promoting it then would have crashed the
headline floor without informing anything — and was promoted (v1.2) only after the deep-exploration
mechanism it existed to referee arrived and cleared the solver itself. A family can be a referee
before it is a denominator; the promotion decision is part of the measurement discipline, not
housekeeping.

## Family ten: compose (compositionality) — promoted the day it was built (v1.3)
**compose**: per-instance hidden primitive bijections over a shared alphabet; rounds ask
either a single hop (apply g_i) or a composition (apply g_i then g_j). The pair space is split
per-instance — the training phase mixes single hops with compositions from one half; then, at a
per-instance randomized switch, every round draws from the **held-out half: compositions never seen
before**. The capability-isolating signal is the post-switch slice, and the design makes the split
arithmetic explicit: m·k primitive facts generate all m²·k composed answers, so for a compositional
learner the held-out half is **free** (the built-in reference answers it at ~1.0 immediately, having
learned only the hops), while a pair-memorizer starts from scratch on every unseen pair (tabular-Q
reads ~0.28 on the same slice — the combinatorial-generalization gap made visible, the same gap
ARC-style evaluation targets). Unlike changepoint, the post slice grants **no relearn burn-in**:
the capability is answering without relearning, so relearn time would erase the signal being
measured. Its referee verdict landed within the hour (the subject agent read as a pair-memorizer,
at half the reference composer) and prescribed the compositional-licensing mechanism that then
cleared it — after which it was promoted as an openly priced trade (v1.3). The extension slot is
empty; the natural next candidate is a family where factored structure itself must be discovered.

## Family eleven: factored (factored-structure discovery) — promoted at price zero (v1.4)
**factored** — the representation frontier the compose arc pointed at. In compose,
the factored structure was handed to the agent syntactically (the observation's slots name the
primitives being composed); here there are only **opaque codes** and three question types, and the
structure exists purely in behavior: question B's answer is determined by the code's A and S
answers through a hidden per-instance relation (the relation is drawn per instance — a fixed rule
would be a hardcodable regularity). The training phase asks A and S everywhere and B only on train
codes, whose answer pairs tile the relation's whole domain (identifiability by construction);
after a randomized switch, every round is a B question on a code where **B was never asked**. The
capability-isolating signal is an **EARLY post slice** — and that slice was itself found by
measurement, sequence recorded: the first cut gated the whole post window, and a fast tabular
relearner scored 0.7+ on a capability defined as answering *without* relearning (the whole-window
padding failure class, one level finer), so the isolating criterion was re-derived from the
controls — the reference must read ~1.0 before relearning could pay, the memorizer near its
starting point. On that slice the built-in relational reference (confirm answers by elimination,
fill the relation table from confirmed triples, derive held-out B answers) reads ~0.97 and
tabular-Q ~0.35. Functional-dependency discovery is a classic problem (reuse, not invention); the
packaging is the referee protocol: control-derived isolating slice, planted memorizer display,
controls gated before any measurement is trusted. It holds in the extension slot pending its referee verdict
on the subject agent — promotion is a separate, deliberate re-versioning, as always.

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
   given time.** Today's universe (eleven toy families, these budgets) licenses no claim about general
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
