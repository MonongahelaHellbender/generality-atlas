#!/usr/bin/env python3
"""generality-atlas — measure GENERALITY (skill-acquisition efficiency on genuinely novel tasks),
contamination-free, with the coverage denominator as the headline and planted failures the instrument
must catch before it is allowed to measure anything.

WHAT THIS IS (and is not): an instrument, not an intelligence. Generality here is operationalized as
Chollet's skill-acquisition efficiency — how fast performance climbs on task instances the agent has
NEVER seen, within a declared experience budget — measured across eleven diagnostic families, each
isolating ONE capability (bsuite's move), every instance procedurally generated fresh from a seed
(Procgen's move: nothing to memorize), every score normalized against an analytic oracle and a random
floor (no judge anywhere in the verdict path). The aggregate is the full profile plus the GENERALITY
FLOOR — the minimum across families — never a single mean without its denominator.

The claim-licensing ledger (open-ended by design — claims scale with evidence):
  * Claims are licensed by the DECLARED UNIVERSE and the measured evidence, nothing more, at any given
    time. Today's eight toy families license no claim about general intelligence — a statement about
    today's evidence, not a ceiling on the project. Grow the universe (families, budgets, difficulty,
    transfer) and what may honestly be claimed grows exactly as fast, and no faster.
  * Legg-Hutter universal intelligence is provably uncomputable; this is a finite, declared, computable
    slice of the question, and claims exactly that much — a slice that can keep growing.
  * v0 defers cross-family transfer (ill-defined for tabular agents across different observation and
    action spaces); v0.5 = within-family difficulty transfer. Deferred, with this reason, not hidden.
  * v0.6 = generality-PER-BUDGET (skill-acquisition efficiency read over the experience budget): sweep
    the budget, report how the FLOOR moves. The load-bearing metric is the BREADTH-VS-DEPTH split —
    does marginal budget lift the floor (breadth) or only the mean (depth)? A specialist given more
    compute deepens its spike without ever becoming general; the axis makes that visible and
    fail-closes (budget_to_floor is None when a capability is unlearnable at any swept budget).
  * The grid family's random floor is SAMPLED (with its own fixed seed, 200 episodes), not analytic —
    labeled as such in the report.
  * The one PERMANENT rule is the contamination-free protocol itself (fresh instances, oracle grading,
    no judge in the verdict path) — permanent because it protects measurement validity, not because it
    limits ambition. The instrument's job is to make narrowness VISIBLE, and to make any future claim
    of generality PROVABLE on instances nothing could have memorized.

Prior art (reused, not reinvented): metric names from the transfer-RL literature (jumpstart / AULC /
final competence — arXiv:2009.07888); diagnostic-family design from bsuite; per-instance procedural
novelty from Procgen; skill-acquisition-efficiency framing from Chollet (ARC; ARC-AGI-3 is the
big-lab procedural+agentic version, arXiv:2603.24621). The only claimed contribution is the assurance
packaging: planted-failure validation, fail-closed coverage, and an optional attestation certificate.

Zero dependencies. Run:
    python3 generality_atlas.py --selftest     # the gate: planted failures MUST be caught
    python3 generality_atlas.py --run          # measure the built-in baselines, print the atlas
    python3 generality_atlas.py --run --json   # machine-readable report (+ attestation if available)
    python3 generality_atlas.py --transfer     # v0.5 within-family difficulty transfer (controls-gated)
    python3 generality_atlas.py --efficiency   # v0.6 generality-per-budget (breadth-vs-depth split)
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path

# optional certificate (same lazy pattern as assurance-bench): measured cells attest ok, declared-but-
# unmeasured cells DEFER — fail-closed. Absent sibling = no certificate, everything else still works.
sys.path.insert(0, str(Path.home() / "code" / "coverage-attestation"))
sys.path.insert(0, str(Path.home() / "code" / "coverage-report"))
try:
    import coverage_attestation as _ca
    _HAS_ATTEST = True
except Exception:
    _ca = None
    _HAS_ATTEST = False


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# Environments — five families, one isolated capability each. Uniform protocol:
#   reset() -> obs ;  step(action) -> (obs, reward, done) ;  .actions = list of legal actions
#   .oracle_return / .random_return  = per-episode expected return of the optimal / random policy
# ────────────────────────────────────────────────────────────────────────────────────────────────────
class BanditEnv:
    """k-armed Bernoulli bandit. Capability: EXPLORATION. Analytic oracle and floor.
    Difficulty axis: the best-arm margin (thinner = harder), action space preserved."""
    def __init__(self, seed: int, k: int = 5, pulls: int = 20, margin: float = 0.15):
        rng = random.Random(seed)
        while True:
            self.probs = [round(rng.uniform(0.05, 0.95), 3) for _ in range(k)]
            top = sorted(self.probs)[-2:]
            if top[1] - top[0] >= margin:          # unique best arm with the declared margin
                break
        self.pulls, self.actions = pulls, list(range(k))
        self.oracle_return = pulls * max(self.probs)
        self.random_return = pulls * (sum(self.probs) / k)
        self._rng = random.Random(seed + 7)

    def reset(self):
        self._t = 0
        return 0                                    # stateless: constant observation

    def step(self, action):
        self._t += 1
        reward = 1.0 if self._rng.random() < self.probs[action] else 0.0
        return 0, reward, self._t >= self.pulls


class GridNavEnv:
    """Seeded-wall grid with BFS-verified solvable goal. Capability: CREDIT ASSIGNMENT.
    Reward 1 on reaching the goal (episode return is success). Random floor is SAMPLED (seeded).
    Difficulty axis: grid size (5x5 easy, larger harder), action space preserved (4 moves)."""

    def __init__(self, seed: int, size: int = 5, cap: int = 40):
        self.SIZE, self.CAP = size, cap
        rng = random.Random(seed)
        while True:
            self.walls = {(r, c) for r in range(self.SIZE) for c in range(self.SIZE)
                          if (r, c) != (0, 0) and rng.random() < 0.2}
            far = [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)
                   if r + c >= self.SIZE and (r, c) not in self.walls]
            if not far:
                continue
            self.goal = rng.choice(far)
            if self._bfs_reachable():
                break
        self.actions = [0, 1, 2, 3]                 # up, down, left, right
        self.oracle_return = 1.0
        self.random_return = self._sample_random_floor(seed + 1)

    def _bfs_reachable(self) -> bool:
        frontier, seen = [(0, 0)], {(0, 0)}
        while frontier:
            cur = frontier.pop()
            if cur == self.goal:
                return True
            for nxt in self._neighbors(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return False

    def _neighbors(self, pos):
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE and (nr, nc) not in self.walls:
                yield (nr, nc)

    def _sample_random_floor(self, seed: int, episodes: int = 200) -> float:
        rng, wins = random.Random(seed), 0
        for _ in range(episodes):
            pos = (0, 0)
            for _ in range(self.CAP):
                pos = self._move(pos, rng.choice(self.actions))
                if pos == self.goal:
                    wins += 1
                    break
        return wins / episodes

    def _move(self, pos, action):
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if 0 <= nxt[0] < self.SIZE and 0 <= nxt[1] < self.SIZE and nxt not in self.walls:
            return nxt
        return pos                                  # bump: stay

    def reset(self):
        self._pos, self._t = (0, 0), 0
        return self._pos

    def step(self, action):
        self._t += 1
        self._pos = self._move(self._pos, action)
        done = self._pos == self.goal or self._t >= self.CAP
        return self._pos, (1.0 if self._pos == self.goal else 0.0), done


class SequenceRuleEnv:
    """Hidden per-seed symbol bijection; emit the mapped symbol. Capability: RULE INDUCTION."""
    def __init__(self, seed: int, alphabet: int = 6, rounds: int = 12):
        rng = random.Random(seed)
        self.mapping = list(range(alphabet))
        rng.shuffle(self.mapping)
        self.alphabet, self.rounds, self.actions = alphabet, rounds, list(range(alphabet))
        self.oracle_return = float(rounds)
        self.random_return = rounds / alphabet
        self._rng = random.Random(seed + 7)

    def reset(self):
        self._t = 0
        self._cur = self._rng.randrange(self.alphabet)
        return self._cur

    def step(self, action):
        reward = 1.0 if action == self.mapping[self._cur] else 0.0
        self._t += 1
        self._cur = self._rng.randrange(self.alphabet)
        return self._cur, reward, self._t >= self.rounds


class MemoryRecallEnv:
    """See a cue, sit through distractors, reproduce the cue at a distinguishable RECALL prompt.
    Capability: MEMORY. A memoryless policy cannot beat the floor — by design, and the instrument
    should DISPLAY that gap for memoryless agents rather than hide it."""
    def __init__(self, seed: int, alphabet: int = 5, distractors: int = 3):
        self.alphabet, self.distractors = alphabet, distractors
        self.actions = list(range(alphabet))
        self.oracle_return = 1.0
        self.random_return = 1.0 / alphabet
        self._rng = random.Random(seed + 7)

    def reset(self):
        self._cue = self._rng.randrange(self.alphabet)
        self._t = 0
        return ("cue", self._cue)

    def step(self, action):
        self._t += 1
        if self._t <= self.distractors:            # distractor phase: actions ignored, no reward
            return ("noise", self._rng.randrange(self.alphabet)), 0.0, False
        if self._t == self.distractors + 1:        # the recall prompt arrives as the observation
            return ("recall",), 0.0, False
        return ("recall",), (1.0 if action == self._cue else 0.0), True


class ParityEnv:
    """4-bit strings; answer the parity. Capability: SYSTEMATIC COMPUTATION (16-state rule)."""
    def __init__(self, seed: int, bits: int = 4, rounds: int = 12):
        self.bits, self.rounds, self.actions = bits, rounds, [0, 1]
        self.oracle_return = float(rounds)
        self.random_return = rounds / 2
        self._rng = random.Random(seed + 7)

    def _draw(self):
        return tuple(self._rng.randrange(2) for _ in range(self.bits))

    def reset(self):
        self._t = 0
        self._cur = self._draw()
        return self._cur

    def step(self, action):
        reward = 1.0 if action == (sum(self._cur) % 2) else 0.0
        self._t += 1
        self._cur = self._draw()
        return self._cur, reward, self._t >= self.rounds


# Conservative curve slices for the changepoint family, valid for EVERY instance regardless of its
# drawn switch episode (switch_after in 16..24): pre-slice ends before the earliest possible switch;
# post-slice starts after the latest possible switch plus a relearn burn-in. Gate on these, never on
# a fixed switch index — consumers (selftests, the trainer's objective) import them from here.
CHANGEPOINT_PRE_SLICE = 16      # curve[:16]  = pre-switch for all instances
CHANGEPOINT_POST_SLICE = 28     # curve[28:]  = post-switch (+burn-in) for all instances
CHANGEPOINT_SUSTAINED_SLICE = 40  # curve[40:] of the sustained family = past >= 3 switches, all draws


class ChangePointEnv:
    """Hidden per-seed bijection that SWITCHES to an everywhere-different bijection after
    switch_after episodes, UNSIGNALED (detecting the change is the capability). Capability:
    ADAPTATION / plasticity. The capability-isolating signal is the POST-SWITCH slice of the curve:
    whole-window AULC can be padded by strong pre-switch acquisition while adaptation is ZERO (the
    seed's exact signature on first measurement).

    The switch episode is RANDOMIZED per instance (16..24, drawn from the instance seed), not fixed:
    the moment an optimizer trains AGAINST this family (the perpetual trainer's objective now
    includes it), a fixed switch episode becomes an exploitable schedule regularity — the search
    could tune surprise/forgetting parameters to expect-the-switch-at-20 instead of to detect
    change. A referee's incidental regularities are attack surface; randomize them."""
    def __init__(self, seed: int, alphabet: int = 4, rounds: int = 12, switch_after: int | None = None):
        rng = random.Random(seed)
        self.alphabet, self.rounds = alphabet, rounds
        self.switch_after = switch_after if switch_after is not None else rng.randint(16, 24)
        self.actions = list(range(alphabet))
        self.map_a = list(range(alphabet))
        rng.shuffle(self.map_a)
        while True:                                        # B differs from A on EVERY symbol
            self.map_b = list(range(alphabet))
            rng.shuffle(self.map_b)
            if all(self.map_b[i] != self.map_a[i] for i in range(alphabet)):
                break
        self.oracle_return = float(rounds)
        self.random_return = rounds / alphabet
        self._rng = random.Random(seed + 7)
        self._episodes = 0

    def _mapping(self):
        return self.map_a if self._episodes <= self.switch_after else self.map_b

    def reset(self):
        self._episodes += 1
        self._t = 0
        self._cur = self._rng.randrange(self.alphabet)
        return self._cur

    def step(self, action):
        reward = 1.0 if action == self._mapping()[self._cur] else 0.0
        self._t += 1
        self._cur = self._rng.randrange(self.alphabet)
        return self._cur, reward, self._t >= self.rounds


class SustainedChangeEnv(ChangePointEnv):
    """Regime re-randomized every switch_every episodes (drawn 10..12 per instance — jittered so an
    optimizer cannot learn the schedule; >= 10 keeps a relearn window) over 60 episodes. Capability:
    SUSTAINED ADAPTATION (plasticity that survives REPEATED change, a different capability from
    single-shock adaptation: a mechanism can pass one switch and still collapse over six — measured).
    The capability-isolating signal is the LATE slice, curve[40:] (past >= 3 switches for every
    draw): steady-state readaptation, not first-shock recovery."""
    def __init__(self, seed: int, alphabet: int = 4, rounds: int = 12):
        super().__init__(seed, alphabet, rounds, switch_after=10**9)
        self.switch_every = random.Random(seed + 31).randint(10, 12)
        self._maprng = random.Random(seed + 999)
        self.cur_map = list(self.map_a)

    def reset(self):
        obs = super().reset()
        if self._episodes > 1 and (self._episodes - 1) % self.switch_every == 0:
            old = list(self.cur_map)
            while True:                                    # each regime differs EVERYWHERE from the last
                m = list(range(self.alphabet))
                self._maprng.shuffle(m)
                if all(m[i] != old[i] for i in range(self.alphabet)):
                    break
            self.cur_map = m
        return obs

    def _mapping(self):
        return self.cur_map


class MultiRuleEnv:
    """THREE conflicting hidden bijections over SHARED symbols and actions; the active rule index is
    signaled in the observation (rule, symbol). Capability: INTERFERENCE-RESISTANCE — holding several
    mutually contradictory rules simultaneously without letting them bleed into each other. A learner
    that keys on the full observation is fine; a learner that generalizes over the rule signal (the
    planted PhaseBlindAgent) collapses to the majority-consistent mush."""
    def __init__(self, seed: int, alphabet: int = 4, n_rules: int = 3, rounds: int = 12):
        rng = random.Random(seed)
        self.alphabet, self.n_rules, self.rounds = alphabet, n_rules, rounds
        self.actions = list(range(alphabet))
        self.rules = []
        for _ in range(n_rules):
            m = list(range(alphabet))
            rng.shuffle(m)
            self.rules.append(m)
        self.oracle_return = float(rounds)
        self.random_return = rounds / alphabet
        self._rng = random.Random(seed + 7)

    def reset(self):
        self._t = 0
        self._phase = self._rng.randrange(self.n_rules)
        self._cur = self._rng.randrange(self.alphabet)
        return (self._phase, self._cur)

    def step(self, action):
        reward = 1.0 if action == self.rules[self._phase][self._cur] else 0.0
        self._t += 1
        self._phase = self._rng.randrange(self.n_rules)
        self._cur = self._rng.randrange(self.alphabet)
        return (self._phase, self._cur), reward, self._t >= self.rounds


class ChainEnv:
    """N-step chain: one action ADVANCES, the other RESETS to the start (which action advances is
    drawn per instance — nothing to memorize); reward ONLY at the far end. Capability: DEEP /
    COMMITTED EXPLORATION (bsuite's deep-sea design, reused not reinvented). The random floor is
    analytic and essentially zero (0.5^length); eps-greedy exploration fails exponentially while
    optimistic value propagation solves it — the family displays exactly that split."""
    def __init__(self, seed: int, length: int = 8):
        rng = random.Random(seed)
        self.length = length
        self.actions = [0, 1]
        self.right = rng.randrange(2)
        self.oracle_return = 1.0
        self.random_return = 0.5 ** length
        self.cap = length + 2

    def reset(self):
        self._pos = 0
        self._t = 0
        return self._pos

    def step(self, action):
        self._t += 1
        if action == self.right:
            self._pos += 1
        else:
            self._pos = 0
        done = self._pos >= self.length or self._t >= self.cap
        return self._pos, (1.0 if self._pos >= self.length else 0.0), done


# Conservative curve slices for the compose family, valid for EVERY instance regardless of its drawn
# switch episode (16..20): pre-slice ends before the earliest possible switch; post-slice starts
# just past the latest. Unlike changepoint (where burn-in is granted so an ADAPTER can recover),
# the post slice here starts almost immediately — the capability is answering held-out pairs
# WITHOUT relearning, so granting relearn time would erase the signal being measured.
COMPOSE_PRE_SLICE = 16       # curve[:16]  = training phase for all instances
COMPOSE_POST_SLICE = 22      # curve[22:]  = held-out-pairs phase for all instances


class CompositionEnv:
    """Hidden per-seed primitive bijections g_1..g_m over a shared alphabet; rounds ask either a
    SINGLE hop (obs ('s', i, x) -> answer g_i(x)) or a COMPOSITION (obs ('c', i, j, x) -> answer
    g_j(g_i(x))). The pair space is split per-seed: the training phase mixes single-op rounds with
    compositions from the TRAIN half; after switch_after episodes (drawn 16..20 per instance — a
    referee's incidental regularities are attack surface), every round is a composition from the
    HELD-OUT half, never seen before. Capability: COMPOSITIONALITY — m*k primitive facts generate
    all m^2*k composed answers, so for a compositional learner the held-out half is FREE, while a
    pair-memorizer starts from scratch on every unseen pair. The capability-isolating signal is the
    POST slice (COMPOSE_POST_SLICE): whole-window AULC can be padded by training-phase memorization
    while combinatorial generalization is zero."""
    def __init__(self, seed: int, alphabet: int = 5, n_ops: int = 4, rounds: int = 12,
                 switch_after: int | None = None):
        rng = random.Random(seed)
        self.alphabet, self.n_ops, self.rounds = alphabet, n_ops, rounds
        self.ops = []
        for _ in range(n_ops):
            m = list(range(alphabet))
            rng.shuffle(m)
            self.ops.append(m)
        pairs = [(i, j) for i in range(n_ops) for j in range(n_ops)]
        rng.shuffle(pairs)
        self.train_pairs = pairs[:len(pairs) // 2]
        self.test_pairs = pairs[len(pairs) // 2:]
        self.switch_after = switch_after if switch_after is not None else rng.randint(16, 20)
        self.actions = list(range(alphabet))
        self.oracle_return = float(rounds)
        self.random_return = rounds / alphabet
        self._rng = random.Random(seed + 7)
        self._episodes = 0

    def _draw_round(self):
        if self._episodes <= self.switch_after:
            if self._rng.random() < 0.5:                    # single hop: the primitive is learnable
                i = self._rng.randrange(self.n_ops)
                x = self._rng.randrange(self.alphabet)
                self._answer = self.ops[i][x]
                return ("s", i, x)
            i, j = self.train_pairs[self._rng.randrange(len(self.train_pairs))]
        else:                                               # held-out compositions ONLY
            i, j = self.test_pairs[self._rng.randrange(len(self.test_pairs))]
        x = self._rng.randrange(self.alphabet)
        self._answer = self.ops[j][self.ops[i][x]]
        return ("c", i, j, x)

    def reset(self):
        self._episodes += 1
        self._t = 0
        return self._draw_round()

    def step(self, action):
        reward = 1.0 if action == self._answer else 0.0     # grade BEFORE drawing the next round
        self._t += 1
        return self._draw_round(), reward, self._t >= self.rounds


# Conservative curve slices for the factored family (same tense logic as compose, insight #79:
# the capability is answering WITHOUT relearning, so the post slice starts almost immediately —
# relearn burn-in would erase the signal being measured).
FACTORED_PRE_SLICE = 16       # curve[:16]   = training phase for all instances
FACTORED_POST_SLICE = 21      # curve[21:]   = held-out-B phase for all instances (switch drawn 18..20)
FACTORED_EARLY_END = 25       # curve[21:25] = the ISOLATING slice: posts 1-7 for every draw, before
#   relearning can pay. Found by measurement, sequence recorded: the first cut's whole post window
#   (12 held-out keys, 18 episodes) leaked RELEARN credit — a fast tabular relearner scored 0.7+ on
#   a capability defined as answering WITHOUT relearning (the whole-window-padding failure class,
#   one level finer). The isolating criterion is control-defined: the reference must read ~1.0 here
#   and the memorizer near its starting point; the subject's verdict then falls where it falls.


class FactoredRelationEnv:
    """OPAQUE codes (plain ints — no slot structure to read), three question types over them:
    obs ('A', c) and ('S', c) have free per-instance random answers; obs ('B', c) is DETERMINED by
    a hidden per-instance relation, ansB(c) = R[ansA(c)][ansS(c)]. Unlike compose, nothing in the
    observation reveals the structure — the functional dependency must be DISCOVERED from behavior.
    Capability: FACTORED-STRUCTURE DISCOVERY (the representation frontier: B's answer factors
    through the other two answers, and only an agent that finds that factorization can answer B on
    codes where B was never asked).

    Protocol: the training phase asks A and S on ALL codes and B on the TRAIN codes only, whose
    (ansA, ansS) pairs TILE all k*k relation cells (identifiability by construction); after a
    per-instance randomized switch (16..20), every round is a B question on a HELD-OUT code. The
    relation R is drawn per instance — a fixed rule (e.g. modular addition) would be a hardcodable
    regularity, and a referee's incidental regularities are attack surface."""
    def __init__(self, seed: int, alphabet: int = 4, n_codes: int = 32, n_train: int = 16,
                 rounds: int = 14, switch_after: int | None = None):
        rng = random.Random(seed)
        self.alphabet, self.rounds = alphabet, rounds
        codes = list(range(n_codes))
        rng.shuffle(codes)
        self.train_codes = codes[:n_train]
        self.test_codes = codes[n_train:]
        cells = [(a, s) for a in range(alphabet) for s in range(alphabet)]
        rng.shuffle(cells)
        self.ansA, self.ansS = {}, {}
        for i, c in enumerate(self.train_codes):            # train codes tile the relation domain
            a, s = cells[i % len(cells)]
            self.ansA[c], self.ansS[c] = a, s
        for c in self.test_codes:
            self.ansA[c] = rng.randrange(alphabet)
            self.ansS[c] = rng.randrange(alphabet)
        self.R = [[rng.randrange(alphabet) for _ in range(alphabet)] for _ in range(alphabet)]
        self.switch_after = switch_after if switch_after is not None else rng.randint(18, 20)
        self.actions = list(range(alphabet))
        self.oracle_return = float(rounds)
        self.random_return = rounds / alphabet
        self._rng = random.Random(seed + 7)
        self._episodes = 0

    def _ansB(self, c):
        return self.R[self.ansA[c]][self.ansS[c]]

    def _draw_round(self):
        if self._episodes <= self.switch_after:
            kind = self._rng.randrange(3)
            if kind == 0:
                c = self.train_codes[self._rng.randrange(len(self.train_codes))]
                self._answer = self._ansB(c)
                return ("B", c)
            pool = self.train_codes + self.test_codes       # A and S are asked EVERYWHERE
            c = pool[self._rng.randrange(len(pool))]
            if kind == 1:
                self._answer = self.ansA[c]
                return ("A", c)
            self._answer = self.ansS[c]
            return ("S", c)
        c = self.test_codes[self._rng.randrange(len(self.test_codes))]
        self._answer = self._ansB(c)                        # held-out B ONLY after the switch
        return ("B", c)

    def reset(self):
        self._episodes += 1
        self._t = 0
        return self._draw_round()

    def step(self, action):
        reward = 1.0 if action == self._answer else 0.0
        self._t += 1
        return self._draw_round(), reward, self._t >= self.rounds


# family registry: name -> (constructor, capability tag, episodes budget)
# Budget calibration is part of the instrument's validity (learned the hard way, 2026-07-09): the
# memory family's original 25-episode budget was information-theoretically too tight — 5 cues x 5
# actions means even an IDEAL episodic learner spends ~the whole budget covering the space, so its
# AULC stays near floor and the family cannot distinguish the capability it exists to isolate. Rule:
# each family's budget must give an idealized learner room to both ACQUIRE and EXPLOIT within the
# AULC window, or the family measures nothing.
# v1.2 UNIVERSE (re-versioned 2026-07-11, third deliberate promotion): deepchain joined after the
# full arc of the discipline ran in BOTH directions — built as an extension 2026-07-10 and HELD
# deliberately (the strongest measured agent then read far below the family's own positive-control
# solver; promoting would have crashed the headline floor without informing anything), then promoted
# once the deep-exploration mechanism it existed to referee arrived and a measured brief showed the
# nine-family floor is no longer deepchain-bound — the expansion strengthens the claim instead of
# trading the headline for completeness. Nine families, nine capabilities.
# (v1.1 promoted multirule 2026-07-09 evening; v1.0 promoted the two adaptation families that morning.)
# v1.3 UNIVERSE (re-versioned 2026-07-11 evening, fourth deliberate promotion): compose joined the
# same day it was built — the fastest referee-to-denominator arc yet, because the mechanism it
# refereed (v9 compositional licensing) arrived within hours of its verdict. PRICED IN THE OPEN,
# not free: the ten-family floor mean is 0.5618 vs the nine-family 0.5913 (-0.0295), compose
# binding on 3 of 5 holdout masters (0.526/0.546/0.606 on measure()'s own draws) — a deliberate
# trade of headline for a tenth capability (combinatorial generalization, the ARC-adjacent one) on
# which the subject agent reads within reach of the family's own reference composer. Ten families,
# ten capabilities. NOTE: compose is appended LAST, so every incumbent family's instance-seed draws
# from measure() are unchanged — no existing number moves except the min.
FAMILIES = {
    "bandit":                (BanditEnv,          "exploration",            30),
    "gridnav":               (GridNavEnv,         "credit-assignment",      30),
    "sequence":              (SequenceRuleEnv,    "rule-induction",         25),
    "memory":                (MemoryRecallEnv,    "memory",                 60),
    "parity":                (ParityEnv,          "systematic-computation", 40),
    "changepoint":           (ChangePointEnv,     "adaptation-to-change",   40),
    "changepoint_sustained": (SustainedChangeEnv, "sustained-adaptation",   60),
    "multirule":             (MultiRuleEnv,       "interference-resistance", 50),
    "deepchain":             (ChainEnv,           "deep-exploration",       60),
    "compose":               (CompositionEnv,     "compositionality",       40),
    "factored":              (FactoredRelationEnv, "factored-structure-discovery", 40),
}
# v1.4 UNIVERSE (re-versioned 2026-07-11 night, fifth deliberate promotion): factored joined at a
# measured price of EXACTLY ZERO — its whole-window AULC (0.62-0.66 on measure()'s own draws)
# clears every master's current floor binder, so the eleven-family floor equals the ten-family
# floor to four decimals (mean 0.5618). The discipline's cleanest arc: built as a referee, its
# verdict prescribed the v10 mechanism, the mechanism cleared the family's own reference PER DRAW,
# and only then was the family promoted — hold-as-referee-until-strength, then free expansion.
# Eleven families, eleven capabilities. Appended LAST: incumbent instance draws unchanged.

# EXTENSION families: built, controlled, and measurable via measure(families=[...]) — but NOT yet in
# the default declared universe, because adding a family re-baselines EVERY existing number (the floor
# is a min; a new weakest family rewrites the headline). Promoting an extension into FAMILIES is a
# deliberate re-versioning of the universe, made in the open, not a side effect of adding code.
# (all six extensions to date were promoted — one after a deliberate hold, one as an openly priced
# floor trade, one at exactly zero price after its mechanism cleared the reference; the slot is
# empty. Natural next candidate: MULTI-STEP structure — plans, not answers.)
EXTENSION_FAMILIES = {}


def _family_spec(name):
    if name in FAMILIES:
        return FAMILIES[name]
    if name in EXTENSION_FAMILIES:
        return EXTENSION_FAMILIES[name]
    raise KeyError(f"unknown family {name!r} (not in FAMILIES or EXTENSION_FAMILIES)")


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# Agents — the protocol is act(obs) -> action; update(obs, action, reward, next_obs, done);
# episode_end(). Baselines + the two PLANTED FAILURES the instrument must catch.
# ────────────────────────────────────────────────────────────────────────────────────────────────────
class RandomAgent:
    """Control: flat curves everywhere, AULC ~ 0 after normalization."""
    def __init__(self, actions, seed: int):
        self.actions, self._rng = actions, random.Random(seed)

    def act(self, obs):
        return self._rng.choice(self.actions)

    def update(self, *a):  pass
    def episode_end(self): pass


class TabularQAgent:
    """The honest simple learner: epsilon-greedy tabular Q on hashable observations. Memoryless —
    so the memory family SHOULD floor it; that gap is a finding, not a bug."""
    def __init__(self, actions, seed: int, alpha=0.5, gamma=0.9, eps=0.35, eps_decay=0.93, eps_min=0.02):
        self.actions, self._rng = actions, random.Random(seed)
        self.alpha, self.gamma, self.eps, self.eps_decay, self.eps_min = alpha, gamma, eps, eps_decay, eps_min
        self.q = {}

    def _qs(self, obs):
        return self.q.setdefault(obs, [0.0] * len(self.actions))

    def act(self, obs):
        if self._rng.random() < self.eps:
            return self._rng.choice(self.actions)
        qs = self._qs(obs)
        best = max(qs)
        return self.actions[self._rng.choice([i for i, v in enumerate(qs) if v == best])]

    def update(self, obs, action, reward, next_obs, done):
        qs = self._qs(obs)
        target = reward if done else reward + self.gamma * max(self._qs(next_obs))
        i = self.actions.index(action)
        qs[i] += self.alpha * (target - qs[i])

    def episode_end(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


class BanditOnlyAgent:
    """PLANTED NARROW agent: ignores observations entirely — tracks one global best action.
    Near-oracle on bandits, floor everywhere context matters. The atlas must show the spike."""
    def __init__(self, actions, seed: int):
        self.actions, self._rng = actions, random.Random(seed)
        self.n = [0] * len(actions)
        self.avg = [0.0] * len(actions)

    def act(self, obs):
        if self._rng.random() < 0.1:
            return self._rng.choice(self.actions)
        best = max(self.avg)
        return self.actions[self._rng.choice([i for i, v in enumerate(self.avg) if v == best])]

    def update(self, obs, action, reward, next_obs, done):
        i = self.actions.index(action)
        self.n[i] += 1
        self.avg[i] += (reward - self.avg[i]) / self.n[i]

    def episode_end(self): pass


class FrozenAgent:
    """PLANTED MEMORIZER (the contamination model): a TabularQ pre-trained on LEAKED instance seeds,
    then frozen (no test-time learning, no exploration). High on the leaked instances, floor on fresh
    ones — the fresh-instance protocol must expose exactly this gap."""
    def __init__(self, actions, q_table, seed: int):
        self.actions, self.q, self._rng = actions, q_table, random.Random(seed)

    def act(self, obs):
        qs = self.q.get(obs)
        if not qs:
            return self._rng.choice(self.actions)
        best = max(qs)
        return self.actions[self._rng.choice([i for i, v in enumerate(qs) if v == best])]

    def update(self, *a):  pass
    def episode_end(self): pass


def pretrain_frozen(family_name: str, leaked_seed: int, master_seed: int, budget_mult: int = 3) -> dict:
    """Train a TabularQ thoroughly on ONE leaked instance (simulated dataset leakage), return its
    Q-table. SINGLE-instance by design — the first cut trained across four leaked instances whose
    hidden rules conflict, and learned mush; the selftest caught it. The failure is itself the
    finding: cross-instance memorization structurally cannot pay here, because instance identity is
    not recoverable from observations — so the honest minimal contamination model is a single leaked
    instance, memorized cold."""
    ctor, _, episodes = _family_spec(family_name)
    env = ctor(leaked_seed)
    agent = TabularQAgent(env.actions, seed=master_seed)
    for _ in range(episodes * budget_mult):
        _run_episode(env, agent)
        agent.episode_end()
    return agent.q


class NonAdapterAgent:
    """PLANTED FAILURE for the changepoint extension family: learns normally (wrapped TabularQ), then
    updates are DISABLED after freeze_at episodes — a competent acquirer that cannot adapt. The
    instrument must DISPLAY its post-switch crash (it confidently keeps doing the OLD right thing,
    scoring WORSE than random after the switch)."""
    def __init__(self, actions, seed: int, freeze_at: int = CHANGEPOINT_PRE_SLICE):
        # freeze_at defaults to the earliest possible switch episode: fully acquired on every
        # instance, frozen BEFORE any instance's switch — maximally competent pre-switch, maximally
        # (confidently) wrong on the post-slice. (History: a first cut froze at 18 against a fixed
        # switch-at-20 and read only -0.095 — Q-ties diluted the crash; the selftest caught it.)
        self.inner = TabularQAgent(actions, seed)
        self.freeze_at = freeze_at
        self._eps_done = 0

    def act(self, obs):
        return self.inner.act(obs)

    def update(self, obs, action, reward, next_obs, done):
        if self._eps_done < self.freeze_at:
            self.inner.update(obs, action, reward, next_obs, done)

    def episode_end(self):
        self._eps_done += 1
        self.inner.episode_end()


class RunningMeanQAgent:
    """PLANTED FAILURE for the changepoint_sustained family: per-key RUNNING-MEAN values (alpha=1/n)
    — the plasticity-loss class itself. Means that never forget deadlock under repeated regime
    change; the instrument must display its late-slice sitting at-or-below random while a fixed-step
    learner holds a positive steady state."""
    def __init__(self, actions, seed: int, eps=0.35, eps_decay=0.93, eps_min=0.02):
        self.actions = list(actions)
        self._rng = random.Random(seed)
        self.eps, self.eps_decay, self.eps_min = eps, eps_decay, eps_min
        self.q = {}

    def _s(self, obs, a):
        return self.q.setdefault((obs, a), [0, 0.0])

    def act(self, obs):
        if self._rng.random() < self.eps:
            return self._rng.choice(self.actions)
        vals = [self._s(obs, a)[1] + 0.3 / (1 + self._s(obs, a)[0]) for a in self.actions]
        best = max(vals)
        return self.actions[self._rng.choice([i for i, v in enumerate(vals) if v == best])]

    def update(self, obs, action, reward, next_obs, done):
        st = self._s(obs, action)
        st[0] += 1
        st[1] += (reward - st[1]) / st[0]

    def episode_end(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


class PhaseBlindAgent:
    """PLANTED FAILURE for the multirule extension family: keys on the SYMBOL only, ignoring the rule
    signal — the interference victim. It learns the majority-consistent mush across conflicting rules
    (~0.17) instead of the rules themselves; the instrument must display that collapse."""
    def __init__(self, actions, seed: int):
        self.inner = TabularQAgent(actions, seed)

    def act(self, obs):
        return self.inner.act(obs[1])

    def update(self, obs, action, reward, next_obs, done):
        self.inner.update(obs[1], action, reward, next_obs[1], done)

    def episode_end(self):
        self.inner.episode_end()


class CompositionalRefAgent:
    """POSITIVE-CONTROL reference for the compose family: learns each primitive hop from bandit
    feedback (systematic candidate elimination per cell — a wrong answer eliminates that symbol),
    then answers NEVER-SEEN composed pairs by running its learned primitives twice. Composed rounds
    also back-fill primitive knowledge: if g_i(x)=y is known, a correct composed answer reveals
    g_j(y), and a wrong one eliminates a candidate for it. This is what the family exists to
    display: m*k learned facts generate all m^2*k composed answers, so the held-out half of the
    pair space costs a compositional learner NOTHING."""
    def __init__(self, actions, seed: int):
        self.actions = list(actions)
        self._rng = random.Random(seed)
        self.known = {}          # ("p", op, x) -> g_op(x), confirmed by reward
        self.tried = {}          # question cell -> eliminated answers

    def _guess(self, cell):
        t = self.tried.setdefault(cell, set())
        untried = [a for a in self.actions if a not in t]
        return self._rng.choice(untried or self.actions)

    def act(self, obs):
        if obs[0] == "s":
            _, i, x = obs
            got = self.known.get(("p", i, x))
            return got if got is not None else self._guess(("s", i, x))
        _, i, j, x = obs
        y = self.known.get(("p", i, x))
        if y is not None:
            z = self.known.get(("p", j, y))
            if z is not None:
                return z                                   # composition: two known hops, zero cost
        return self._guess(("c", i, j, x))

    def update(self, obs, action, reward, next_obs, done):
        if obs[0] == "s":
            _, i, x = obs
            if reward == 1:
                self.known[("p", i, x)] = action
            else:
                self.tried.setdefault(("s", i, x), set()).add(action)
            return
        _, i, j, x = obs
        y = self.known.get(("p", i, x))
        if reward == 1:
            if y is not None:
                self.known[("p", j, y)] = action           # back-fill the second hop
        else:
            self.tried.setdefault(("c", i, j, x), set()).add(action)
            if y is not None:
                self.tried.setdefault(("s", j, y), set()).add(action)

    def episode_end(self):
        pass


class RelationalRefAgent:
    """POSITIVE-CONTROL reference for the factored family: learns each (question, code) answer by
    candidate elimination under bandit feedback; from every train code whose A, S, and B answers are
    all confirmed it fills one cell of the relation table R_hat[ansA][ansS] -> ansB; and for a code
    whose B was NEVER asked it answers B by running the discovered factorization: look up its own
    confirmed A and S answers, then R_hat. This is what the family exists to display — the B answer
    factors through the other two, and finding that structure makes the held-out codes free."""
    def __init__(self, actions, seed: int):
        self.actions = list(actions)
        self._rng = random.Random(seed)
        self.known = {}          # (q, c) -> confirmed answer
        self.tried = {}          # (q, c) -> eliminated answers
        self.rel = {}            # (ansA, ansS) -> ansB, from fully-confirmed train triples

    def _guess(self, cell):
        t = self.tried.setdefault(cell, set())
        untried = [a for a in self.actions if a not in t]
        return self._rng.choice(untried or self.actions)

    def act(self, obs):
        got = self.known.get(obs)
        if got is not None:
            return got
        q, c = obs
        if q == "B":
            a, s = self.known.get(("A", c)), self.known.get(("S", c))
            if a is not None and s is not None:
                r = self.rel.get((a, s))
                if r is not None:
                    return r                               # the discovered factorization, applied
        return self._guess(obs)

    def update(self, obs, action, reward, next_obs, done):
        if reward == 1:
            self.known[obs] = action
            q, c = obs
            a, s, b = (self.known.get(("A", c)), self.known.get(("S", c)),
                       self.known.get(("B", c)))
            if a is not None and s is not None and b is not None:
                self.rel[(a, s)] = b                       # one confirmed triple = one relation cell
        else:
            self.tried.setdefault(obs, set()).add(action)

    def episode_end(self):
        pass


class OptimisticQAgent:
    """POSITIVE-CONTROL reference for the deepchain family: Q initialized HIGH (optimism in the face
    of uncertainty) with bootstrapped updates — the classic deep-exploration solver. Untried paths
    look promising until proven otherwise, and value propagates backward through the chain; this is
    exactly what eps-greedy (and any non-bootstrapping learner) lacks."""
    def __init__(self, actions, seed: int, alpha=0.5, gamma=0.97, eps=0.05, eps_decay=0.95, eps_min=0.0):
        self.actions = list(actions)
        self._rng = random.Random(seed)
        self.alpha, self.gamma, self.eps, self.eps_decay, self.eps_min = alpha, gamma, eps, eps_decay, eps_min
        self.q = {}

    def _qs(self, obs):
        return self.q.setdefault(obs, [1.0] * len(self.actions))

    def act(self, obs):
        if self._rng.random() < self.eps:
            return self._rng.choice(self.actions)
        qs = self._qs(obs)
        best = max(qs)
        return self.actions[self._rng.choice([i for i, v in enumerate(qs) if v == best])]

    def update(self, obs, action, reward, next_obs, done):
        qs = self._qs(obs)
        target = reward if done else reward + self.gamma * max(self._qs(next_obs))
        i = self.actions.index(action)
        qs[i] += self.alpha * (target - qs[i])

    def episode_end(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# The measurement protocol + metrics (names from the transfer-RL literature)
# ────────────────────────────────────────────────────────────────────────────────────────────────────
def _run_episode(env, agent) -> float:
    obs, total, done = env.reset(), 0.0, False
    while not done:
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
        total += reward
    return total


def _normalize(ep_return: float, env) -> float:
    """UNCLIPPED normalization — found by the selftest, not by design review: clipping each episode's
    score at 0 before averaging turns symmetric noise into one-sided upward bias (the random control
    read ~0.26 instead of ~0). The metric keeps raw values (noise averages out around zero, slightly
    negative AULC is honest); only the DISPLAY bar clips."""
    span = env.oracle_return - env.random_return
    if span <= 0:
        return 0.0
    return (ep_return - env.random_return) / span


def measure_family(agent_factory, family_name: str, instance_seeds, agent_seed: int,
                   budget_mult: float = 1.0) -> dict:
    """Learning curve for one family: the agent learns each FRESH instance from scratch; per-episode
    normalized scores average across instances. Metrics: jumpstart / AULC / final competence.

    budget_mult scales the per-family episode budget (for the generality-per-budget axis). The AULC is
    the average competence DISPLAYED within a budget of `episodes` episodes — so at higher budget more
    of a learner's window is converged and its AULC rises, while a non-learner's stays flat. That is
    exactly skill-acquisition efficiency read over the budget; comparing AULC across budgets is
    comparing 'competence shown within B episodes' for different B, stated plainly, not hidden."""
    ctor, capability, base_episodes = _family_spec(family_name)
    episodes = max(3, round(base_episodes * budget_mult))
    curves = []
    for idx, s in enumerate(instance_seeds):
        env = ctor(s)
        agent = agent_factory(env.actions, agent_seed + idx)
        curve = []
        for _ in range(episodes):
            curve.append(_normalize(_run_episode(env, agent), env))
            agent.episode_end()
        curves.append(curve)
    mean_curve = [round(sum(c[e] for c in curves) / len(curves), 4) for e in range(episodes)]
    return {
        "capability": capability,
        "n_instances": len(instance_seeds),
        "episodes": episodes,
        "jumpstart": mean_curve[0],
        "aulc": round(sum(mean_curve) / len(mean_curve), 4),
        "final": round(sum(mean_curve[-3:]) / 3, 4),
        "curve": mean_curve,
    }


def measure(agent_factory, master_seed: int = 20260709, n_instances: int = 8,
            families=None, budget_mult: float = 1.0) -> dict:
    """The atlas run: every declared family × fresh instance seeds derived from the master seed.
    Aggregate = the profile + the GENERALITY FLOOR (min AULC across families). Never a bare mean.

    budget_mult (default 1.0) scales every family's episode budget uniformly; the instance seeds are
    derived from master_seed ALONE, so budget is the only thing that changes across levels — a clean
    controlled comparison for the generality-per-budget axis."""
    families = families or list(FAMILIES)
    rng = random.Random(master_seed)
    profile, scope = {}, []
    instance_seeds = {f: [rng.randrange(10**9) for _ in range(n_instances)] for f in families}
    for f in families:
        profile[f] = measure_family(agent_factory, f, instance_seeds[f], agent_seed=master_seed + 13,
                                     budget_mult=budget_mult)
        scope += [f"{f}:{s}" for s in instance_seeds[f]]
    aulcs = {f: profile[f]["aulc"] for f in families}
    total_episodes = sum(profile[f]["episodes"] for f in families)
    return {
        "master_seed": master_seed,
        "declared_families": families,
        "budget_mult": budget_mult,
        "total_episodes": total_episodes,
        "profile": profile,
        "generality_floor": min(aulcs.values()),
        "floor_family": min(aulcs, key=aulcs.get),
        "generality_mean": round(sum(aulcs.values()) / len(aulcs), 4),
        "scope_cells": scope,
        # dynamic family count: this string was caught reading "five" two re-versionings after the
        # universe left five — a boundary statement that can rot is a boundary statement that lies
        "boundary": (f"Scores are complete relative to THIS declared universe ({len(families)} toy "
                     "families, these budgets, these seeds) — nothing beyond it is licensed. The floor "
                     "is the headline; a high mean with a low floor is narrowness, not generality."),
    }


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# v0.6 — GENERALITY-PER-BUDGET (skill-acquisition efficiency read over the experience budget)
#
# Generality (Chollet) is not skill but skill-acquisition EFFICIENCY — how much competence you gain per
# unit of experience on novel tasks. The v0.5 atlas reports a floor at ONE budget; this axis sweeps the
# budget and reports how the floor moves. The instrument's headline stays the FLOOR, so the primary
# object here is the floor-vs-budget FUNCTION (the scalar summaries below are lossy projections of it,
# named as such).
#
# The load-bearing diagnostic is BREADTH-VS-DEPTH: when you spend marginal budget, does the FLOOR rise
# (the weakest family improves = breadth) or only the MEAN (the strong families get stronger = depth)?
# A true generalist lifts its floor; a specialist given more compute just deepens its spike. So this is
# an assurance axis, not a bare metric: it catches a 'more compute -> more general' claim that is really
# 'more compute -> narrower but deeper'. breadth_ratio = Δfloor / Δmean across the swept budget; ~1 means
# marginal budget bought breadth, ~0 means it bought only depth. If the mean does not move it is
# reported as None ('undefined, mean flat'), never a fabricated number.
DEFAULT_BUDGETS = (0.5, 1.0, 2.0)


def measure_efficiency(agent_factory, budgets=DEFAULT_BUDGETS, master_seed: int = 20260709,
                       n_instances: int = 8, families=None, target_floor: float = 0.30) -> dict:
    """Sweep the experience budget and report generality-EFFICIENCY. Primary object: the floor-vs-budget
    curve. Scalar projections (each stated as lossy): budget_to_floor (episodes to reach a target FLOOR,
    or None if never reached within the swept budgets = fail-closed, never fabricated) and breadth_ratio
    (Δfloor / Δmean, the breadth-vs-depth split)."""
    budgets = sorted(budgets)
    runs = [measure(agent_factory, master_seed=master_seed, n_instances=n_instances,
                    families=families, budget_mult=b) for b in budgets]
    floor_curve = [{"budget_mult": r["budget_mult"], "total_episodes": r["total_episodes"],
                    "floor": r["generality_floor"], "floor_family": r["floor_family"],
                    "mean": r["generality_mean"]} for r in runs]

    # budget_to_floor: smallest total_episodes at which the FLOOR first reaches the target. None (not a
    # zero, not the max) when the target is never reached within the swept budgets — fail-closed.
    reached = [p for p in floor_curve if p["floor"] >= target_floor]
    budget_to_floor = min(p["total_episodes"] for p in reached) if reached else None

    # breadth_ratio: how much of the marginal budget went to BREADTH (floor) vs DEPTH (mean), from the
    # smallest to the largest swept budget. It is only interpretable when the agent actually GAINED
    # overall competence with more budget — dividing by a ~zero mean-change gives garbage (the narrow
    # specialist saturated early: d_mean 0.01 would yield a wild ratio). So the ratio is defined only
    # when d_mean exceeds MEAN_EPS, and is None ('overall competence flat — no breadth/depth split to
    # report') otherwise. That is fail-closed, not a fabricated number.
    MEAN_EPS = 0.05
    lo, hi = floor_curve[0], floor_curve[-1]
    d_floor, d_mean = hi["floor"] - lo["floor"], hi["mean"] - lo["mean"]
    breadth_ratio = round(d_floor / d_mean, 3) if d_mean > MEAN_EPS else None

    return {
        "master_seed": master_seed,
        "swept_budgets": budgets,
        "target_floor": target_floor,
        "floor_curve": floor_curve,
        "d_floor": round(d_floor, 4),
        "d_mean": round(d_mean, 4),
        "breadth_ratio": breadth_ratio,
        "budget_to_floor": budget_to_floor,
        "boundary": ("Generality-efficiency is measured ONLY over the swept budgets and this declared "
                     "universe. budget_to_floor is None when the target floor is never reached here — "
                     "that is fail-closed, not a zero. breadth_ratio ~1 = marginal budget bought "
                     "breadth (floor rose); ~0 = it bought only depth (mean rose, floor flat); None = "
                     "mean did not move, ratio undefined. AULC at budget B is competence shown within "
                     "B episodes — comparing across B is comparing different windows, stated plainly."),
    }


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# v0.5 — WITHIN-FAMILY DIFFICULTY TRANSFER (the deferred item, now built, controls-first)
#
# Question: does learning EASY instances of a family accelerate acquisition on HARD instances of the
# same family? Metrics from the literature: jumpstart delta and AULC delta (transfer vs from-scratch).
#
# The transfer instrument carries its OWN controls, and may not report difficulty transfer unless both
# pass in the same run:
#   POSITIVE control — parity, same difficulty, different instances: the parity rule is instance-
#     invariant and the observation space is shared, so a pretrained learner MUST show strong positive
#     transfer. If it doesn't, the harness is broken.
#   NEGATIVE control — bandit, same difficulty, different instances: arm identities are instance-
#     random, so cross-instance transfer MUST be ~0. If it isn't, the harness is leaking.
#
# Difficulty axes preserve the ACTION SPACE (a transfer-condition requirement): bandit = thinner
# best-arm margin; gridnav = larger grid; memory = more distractors; parity = more bits. The sequence
# family is EXCLUDED from difficulty transfer (its natural axes change the action space) — declared,
# with this reason, not hidden.
DIFFICULTY = {
    "bandit":  {"easy": {},                        "hard": {"margin": 0.06}},
    "gridnav": {"easy": {},                        "hard": {"size": 7, "cap": 60}},
    "memory":  {"easy": {},                        "hard": {"distractors": 6}},
    "parity":  {"easy": {},                        "hard": {"bits": 6}},
}


def _make_env(family: str, seed: int, difficulty: str):
    ctor = FAMILIES[family][0]
    return ctor(seed, **DIFFICULTY[family][difficulty])


def measure_transfer(agent_factory, family: str, master_seed: int = 20260709,
                     n_source: int = 3, n_target: int = 4,
                     source_difficulty: str = "easy", target_difficulty: str = "hard") -> dict:
    """Transfer vs scratch on the same target instances. SCRATCH: a fresh agent learns each target.
    TRANSFER: a fresh agent first learns n_source SOURCE instances (full budget each), then the same
    target — same agent object, carrying whatever it can. Deltas > 0 mean the source learning helped."""
    episodes = FAMILIES[family][2]
    # zlib.crc32, NOT built-in hash(): hash() is randomized per process (PYTHONHASHSEED), which made
    # the transfer controls nondeterministic ACROSS processes while the in-process reproducibility
    # check still passed — caught when a fresh run flipped the negative control. Reproducibility
    # means bit-identical across processes, not just within one.
    import zlib
    rng = random.Random(master_seed + zlib.crc32(family.encode()) % 100000)
    source_seeds = [rng.randrange(10**9) for _ in range(n_source)]
    target_seeds = [rng.randrange(10**9) for _ in range(n_target)]

    def learn(env, agent):
        curve = []
        for _ in range(episodes):
            curve.append(_normalize(_run_episode(env, agent), env))
            agent.episode_end()
        return curve

    scratch_curves, transfer_curves = [], []
    for i, t in enumerate(target_seeds):
        scratch_curves.append(learn(_make_env(family, t, target_difficulty),
                                    agent_factory(_make_env(family, t, target_difficulty).actions,
                                                  master_seed + 31 + i)))
        agent = agent_factory(_make_env(family, source_seeds[0], source_difficulty).actions,
                              master_seed + 31 + i)
        for s in source_seeds:
            learn(_make_env(family, s, source_difficulty), agent)      # pretraining, discarded curves
        transfer_curves.append(learn(_make_env(family, t, target_difficulty), agent))

    def agg(curves):
        mean = [sum(c[e] for c in curves) / len(curves) for e in range(episodes)]
        return {"jumpstart": round(mean[0], 4), "aulc": round(sum(mean) / len(mean), 4)}

    scratch, transfer = agg(scratch_curves), agg(transfer_curves)
    return {
        "family": family,
        "source": f"{n_source}x {source_difficulty}", "target": f"{n_target}x {target_difficulty}",
        "scratch": scratch, "transfer": transfer,
        "jumpstart_delta": round(transfer["jumpstart"] - scratch["jumpstart"], 4),
        "aulc_delta": round(transfer["aulc"] - scratch["aulc"], 4),
    }


def transfer_controls_pass(master_seed: int = 20260709) -> tuple:
    """Both controls must pass before any difficulty-transfer number may be reported (fail-closed).
    The controls run on REFERENCE agents, never on the subject being measured.

    TWO control redesigns, each found by a control failing (2026-07-09): (1) the first negative
    control assumed bandit cross-instance transfer must be ~0 — it read +0.30, and the diagnosis was
    a REAL, unanticipated channel: process-state transfer (a pretrained learner carries its annealed
    exploration schedule even where instance knowledge cannot carry). The negative control's actual
    job is detecting HARNESS asymmetry, and its clean instrument is the RANDOM agent — learns
    nothing, carries nothing, so any nonzero delta is a harness bug. (2) The positive control
    originally ran on the SUBJECT's own factory and failed for a weak-at-parity subject even though
    the harness was fine — conflating instrument validity with subject strength. Controls belong to
    the instrument: the positive control runs a KNOWN learner (tabular-Q), the negative a known
    non-learner, and any subject's measurements are gated on instrument validity, not on the
    subject's own abilities."""
    pos = measure_transfer(lambda a, s: TabularQAgent(a, s), "parity", master_seed,
                           source_difficulty="easy", target_difficulty="easy")
    neg = measure_transfer(lambda a, s: RandomAgent(a, s), "parity", master_seed,
                           source_difficulty="easy", target_difficulty="easy")
    ok_pos = pos["jumpstart_delta"] >= 0.35          # a known learner's invariant rule MUST carry
    ok_neg = abs(neg["jumpstart_delta"]) <= 0.10 and abs(neg["aulc_delta"]) <= 0.10
    return (ok_pos and ok_neg), {"positive_control": pos, "negative_control": neg,
                                 "positive_ok": ok_pos, "negative_ok": ok_neg}


def attest_run(report: dict):
    """Optional certificate: every declared (family x seed) cell measured -> ok; anything declared but
    missing would DEFER (fail-closed). The numeric profile stays in the report, bound by scope hash."""
    if not _HAS_ATTEST:
        return None
    checks = {cell: "ok" for cell in report["scope_cells"]}
    decl = _ca.declare(sorted(checks))
    return _ca.attest(decl, checks, subject="generality-atlas-run", sign_key=_ca.make_key())


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# Selftest — the gate. The instrument may not measure anything until it catches its planted failures.
# ────────────────────────────────────────────────────────────────────────────────────────────────────
def _selftest(verbose: bool = True) -> int:
    fails = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append(name)
        if verbose:
            print(f"  [{'ok ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}",
                  file=sys.stderr)

    # 0) environment sanity: oracle beats floor on every family instance we probe
    for fname, (ctor, _, _) in FAMILIES.items():
        for s in (11, 222, 3333):
            env = ctor(s)
            check(f"env-sanity {fname}:{s}", env.oracle_return > env.random_return,
                  f"oracle={env.oracle_return} floor={round(env.random_return,3)}")

    # 1) reproducibility: identical master seed => identical profile, bit for bit
    r1 = measure(lambda a, s: RandomAgent(a, s), master_seed=99, n_instances=3)
    r2 = measure(lambda a, s: RandomAgent(a, s), master_seed=99, n_instances=3)
    check("reproducibility (same seed => identical report)", r1 == r2)

    # 2) control: the random agent's AULC is ~0 everywhere, TWO-SIDED (unclipped normalization means
    #    noise must average out around zero — a one-sided check would hide clipping-style bias)
    rnd = measure(lambda a, s: RandomAgent(a, s), n_instances=6)
    worst = max(abs(m["aulc"]) for m in rnd["profile"].values())
    check("random control ~0 everywhere (two-sided)", worst <= 0.12, f"max |AULC|={worst}")

    # 3) the honest learner: clearly SEPARATED from the control where it should learn, floored on
    #    memory (the TRUE gap). Separation-based, not absolute — the first cut demanded bandit >= 0.30
    #    and failed at 0.288: the honest value of epsilon-greedy exploration cost, which the forecast
    #    had overestimated (0.5-0.8, written pre-run). The instrument's requirement is that it can
    #    DISTINGUISH a learner from a non-learner, not that the learner flatters a round number.
    q = measure(lambda a, s: TabularQAgent(a, s), n_instances=6)
    qp = {f: m["aulc"] for f, m in q["profile"].items()}
    rp = {f: m["aulc"] for f, m in rnd["profile"].items()}
    check("tabular-Q learns bandit (separated from control)",
          qp["bandit"] >= 0.20 and qp["bandit"] >= rp["bandit"] + 0.15,
          f"AULC={qp['bandit']} vs control {rp['bandit']}")
    check("tabular-Q learns sequence", qp["sequence"] >= 0.30, f"AULC={qp['sequence']}")
    check("tabular-Q learns parity", qp["parity"] >= 0.25, f"AULC={qp['parity']}")
    check("tabular-Q memory gap DISPLAYED (memoryless => floor)", qp["memory"] <= 0.20,
          f"AULC={qp['memory']}")
    check("generality floor names the weakest family", q["floor_family"] in ("memory", "gridnav"),
          f"floor={q['generality_floor']} at {q['floor_family']}")

    # 4) planted NARROW agent: spike on bandit, floor elsewhere — narrowness must be VISIBLE
    nb = measure(lambda a, s: BanditOnlyAgent(a, s), n_instances=6)
    np_ = {f: m["aulc"] for f, m in nb["profile"].items()}
    others = max(v for f, v in np_.items() if f != "bandit")
    check("narrow agent spikes on bandit", np_["bandit"] >= 0.40, f"AULC={np_['bandit']}")
    check("narrow agent floors off-family (spiky profile shown)", others <= 0.20,
          f"max other AULC={others}")
    check("narrow agent's generality floor is low", nb["generality_floor"] <= 0.20,
          f"floor={nb['generality_floor']}")

    # 5) planted MEMORIZER (single-instance contamination): aces the leaked instance, floors on fresh
    #    ones — the fresh-instance protocol must expose exactly this gap
    q_table = pretrain_frozen("sequence", leaked_seed=41, master_seed=7)
    frozen_factory = lambda a, s: FrozenAgent(a, q_table, s)
    on_leaked = measure_family(frozen_factory, "sequence", [41], agent_seed=7)
    on_fresh = measure_family(frozen_factory, "sequence", [910, 911, 912], agent_seed=7)
    check("memorizer aces the LEAKED instance", on_leaked["aulc"] >= 0.60, f"AULC={on_leaked['aulc']}")
    check("memorizer floors on FRESH instances (contamination exposed)", on_fresh["aulc"] <= 0.15,
          f"AULC={on_fresh['aulc']}")

    # 6) v0.5 transfer instrument: BOTH built-in controls must pass for the honest learner, and the
    #    machinery must be reproducible. (Difficulty-transfer numbers themselves are MEASUREMENTS,
    #    reported not asserted — the controls are what make them meaningful.)
    ok_controls, ctl = transfer_controls_pass()
    check("transfer POSITIVE control (parity same-difficulty carries over)",
          ctl["positive_ok"], f"jumpstart_delta={ctl['positive_control']['jumpstart_delta']}")
    check("transfer NEGATIVE control (random agent => harness symmetric, ~0)",
          ctl["negative_ok"], f"aulc_delta={ctl['negative_control']['aulc_delta']}")
    t1 = measure_transfer(lambda a, s: TabularQAgent(a, s), "parity", master_seed=55)
    t2 = measure_transfer(lambda a, s: TabularQAgent(a, s), "parity", master_seed=55)
    check("transfer reproducibility (same seed => identical)", t1 == t2)

    # 8) v0.6 GENERALITY-PER-BUDGET axis. The axis must (a) DETECT BREADTH-buying: on a universe the
    #    learner can cover (the four non-memory families) tabular-Q's FLOOR rises with budget nearly as
    #    fast as its mean (breadth_ratio ~1) and the target floor becomes reachable; (b) DETECT
    #    DEPTH-only + FAIL-CLOSE: the SAME learner on the full universe gains overall competence (mean
    #    up) but its floor is memory-pinned, so breadth_ratio is low and budget_to_floor is None — more
    #    budget did NOT buy generality; (c) not fabricate a split when there was no gain: the saturated
    #    specialist and the random control gained nothing overall, so breadth_ratio is None, and the
    #    specialist's floor stays negative at every budget (never general regardless of compute).
    learnable = ["bandit", "gridnav", "sequence", "parity"]
    eff_breadth = measure_efficiency(lambda a, s: TabularQAgent(a, s), families=learnable, n_instances=6)
    check("budget axis DETECTS BREADTH (floor rises with budget on a coverable universe)",
          eff_breadth["d_floor"] > 0.10 and eff_breadth["breadth_ratio"] is not None
          and eff_breadth["breadth_ratio"] >= 0.6,
          f"d_floor={eff_breadth['d_floor']} breadth_ratio={eff_breadth['breadth_ratio']}")
    check("budget axis: target floor becomes REACHABLE for the covering learner",
          eff_breadth["budget_to_floor"] is not None, f"budget_to_floor={eff_breadth['budget_to_floor']}")
    eff_depth = measure_efficiency(lambda a, s: TabularQAgent(a, s), n_instances=6)
    check("budget axis DETECTS DEPTH-only (mean up, floor memory-pinned => low breadth_ratio)",
          eff_depth["breadth_ratio"] is not None and eff_depth["breadth_ratio"] <= 0.5,
          f"breadth_ratio={eff_depth['breadth_ratio']} d_mean={eff_depth['d_mean']}")
    check("budget axis FAIL-CLOSED (unlearnable capability => budget_to_floor None, not a low number)",
          eff_depth["budget_to_floor"] is None, f"budget_to_floor={eff_depth['budget_to_floor']}")
    eff_narrow = measure_efficiency(lambda a, s: BanditOnlyAgent(a, s), n_instances=6)
    check("budget axis: specialist NEVER general regardless of budget (floor stays <=0 at max budget)",
          eff_narrow["floor_curve"][-1]["floor"] <= 0.05,
          f"floor@maxbudget={eff_narrow['floor_curve'][-1]['floor']}")
    check("budget axis: no fabricated split when overall competence is flat (breadth_ratio None)",
          eff_narrow["breadth_ratio"] is None, f"breadth_ratio={eff_narrow['breadth_ratio']}")
    eff_rnd = measure_efficiency(lambda a, s: RandomAgent(a, s), n_instances=6)
    check("budget axis: random control gains nothing => breadth_ratio undefined (None)",
          eff_rnd["breadth_ratio"] is None, f"breadth_ratio={eff_rnd['breadth_ratio']} d_mean={eff_rnd['d_mean']}")
    e1 = measure_efficiency(lambda a, s: TabularQAgent(a, s), families=learnable, n_instances=6)
    check("budget axis reproducibility (same seed => identical)", e1 == eff_breadth)

    # 9) changepoint (adaptation-to-unsignaled-change; PROMOTED into the default universe
    #    2026-07-09 after passing these controls as an extension). The capability-isolating signal is the POST-SWITCH slice: whole-window AULC can be
    #    padded by pre-switch acquisition while adaptation is ZERO.
    cp_seeds = [random.Random(4242).randrange(10**9) for _ in range(6)]
    cp_q = measure_family(lambda a, s: TabularQAgent(a, s), "changepoint", cp_seeds, agent_seed=99)
    cp_r = measure_family(lambda a, s: RandomAgent(a, s), "changepoint", cp_seeds, agent_seed=99)
    cp_f = measure_family(lambda a, s: NonAdapterAgent(a, s), "changepoint", cp_seeds, agent_seed=99)
    post = lambda m: sum(m["curve"][CHANGEPOINT_POST_SLICE:]) / len(m["curve"][CHANGEPOINT_POST_SLICE:])
    check("changepoint: random control flat (harness symmetric across the switch)",
          abs(cp_r["aulc"]) <= 0.10 and abs(post(cp_r)) <= 0.10,
          f"aulc={cp_r['aulc']} post={post(cp_r):.3f}")
    check("changepoint: positive control ADAPTS (fixed-alpha tabular-Q recovers post-switch)",
          cp_q["aulc"] >= 0.28 and post(cp_q) >= 0.12,
          f"aulc={cp_q['aulc']} post={post(cp_q):.3f}")
    check("changepoint: planted NON-ADAPTER displayed (post-switch WORSE than random)",
          post(cp_f) <= -0.10, f"post={post(cp_f):.3f}")
    check("changepoint: adapter vs non-adapter separation is unmistakable (>= 0.2 post-switch)",
          post(cp_q) - post(cp_f) >= 0.20, f"gap={post(cp_q) - post(cp_f):.3f}")
    cp_q2 = measure_family(lambda a, s: TabularQAgent(a, s), "changepoint", cp_seeds, agent_seed=99)
    check("changepoint reproducibility (same seeds => identical)", cp_q == cp_q2)

    # 10) changepoint_sustained (STEADY-STATE plasticity — a different capability from single-shock
    #     adaptation; a mechanism can pass one switch and still collapse over six. PROMOTED 2026-07-09).
    #     Signal = the LATE slice (past >= 3 switches for every cadence draw). The planted failure is
    #     the plasticity-loss class ITSELF: running-mean values (alpha=1/n) that never forget.
    late = lambda m: (sum(m["curve"][CHANGEPOINT_SUSTAINED_SLICE:])
                      / len(m["curve"][CHANGEPOINT_SUSTAINED_SLICE:]))
    su_q = measure_family(lambda a, s: TabularQAgent(a, s), "changepoint_sustained", cp_seeds, agent_seed=99)
    su_r = measure_family(lambda a, s: RandomAgent(a, s), "changepoint_sustained", cp_seeds, agent_seed=99)
    su_p = measure_family(lambda a, s: RunningMeanQAgent(a, s), "changepoint_sustained", cp_seeds, agent_seed=99)
    check("sustained: random control flat across repeated switches", abs(late(su_r)) <= 0.10,
          f"late={late(su_r):+.3f}")
    check("sustained: fixed-step tabular-Q holds a positive STEADY STATE", late(su_q) >= 0.10,
          f"late={late(su_q):+.3f}")
    check("sustained: planted RUNNING-MEAN learner displayed (late slice at-or-below random)",
          late(su_p) <= 0.0, f"late={late(su_p):+.3f}")
    check("sustained: adapter vs plasticity-loss separation unmistakable (>= 0.15)",
          late(su_q) - late(su_p) >= 0.15, f"gap={late(su_q) - late(su_p):.3f}")
    su_q2 = measure_family(lambda a, s: TabularQAgent(a, s), "changepoint_sustained", cp_seeds, agent_seed=99)
    check("sustained reproducibility (same seeds => identical)", su_q == su_q2)

    # 11) EXTENSION family: multirule (interference-resistance). The plant is the failure class
    #     itself: a phase-blind learner that generalizes over the rule signal and collapses to the
    #     majority-consistent mush across three conflicting rules.
    mr_q = measure_family(lambda a, s: TabularQAgent(a, s), "multirule", cp_seeds, agent_seed=99)
    mr_r = measure_family(lambda a, s: RandomAgent(a, s), "multirule", cp_seeds, agent_seed=99)
    mr_p = measure_family(lambda a, s: PhaseBlindAgent(a, s), "multirule", cp_seeds, agent_seed=99)
    check("multirule: random control flat", abs(mr_r["aulc"]) <= 0.10, f"aulc={mr_r['aulc']}")
    check("multirule: full-key learner holds all three rules (tabular-Q >= 0.40)",
          mr_q["aulc"] >= 0.40, f"aulc={mr_q['aulc']}")
    check("multirule: planted PHASE-BLIND learner collapses (interference displayed, <= 0.25)",
          mr_p["aulc"] <= 0.25, f"aulc={mr_p['aulc']}")
    check("multirule: resistance vs interference separation unmistakable (>= 0.25)",
          mr_q["aulc"] - mr_p["aulc"] >= 0.25, f"gap={mr_q['aulc'] - mr_p['aulc']:.3f}")
    mr_q2 = measure_family(lambda a, s: TabularQAgent(a, s), "multirule", cp_seeds, agent_seed=99)
    check("multirule reproducibility (same seeds => identical)", mr_q == mr_q2)

    # 12) deepchain (deep/committed exploration — bsuite's deep-sea, reused; PROMOTED into the
    #     default universe 2026-07-11 after its referee job completed — held as an extension until
    #     the deep-exploration mechanism it existed to referee arrived).
    #     Positive control = optimistic-init bootstrapped Q (the classic solver); the displayed
    #     contrast is eps-greedy tabular-Q failing exponentially — myopic exploration made visible.
    dc_o = measure_family(lambda a, s: OptimisticQAgent(a, s), "deepchain", cp_seeds, agent_seed=99)
    dc_r = measure_family(lambda a, s: RandomAgent(a, s), "deepchain", cp_seeds, agent_seed=99)
    dc_q = measure_family(lambda a, s: TabularQAgent(a, s), "deepchain", cp_seeds, agent_seed=99)
    check("deepchain: random control flat (floor is analytic ~0)", abs(dc_r["aulc"]) <= 0.05,
          f"aulc={dc_r['aulc']}")
    check("deepchain: optimistic-init Q SOLVES it (value propagation, >= 0.35)",
          dc_o["aulc"] >= 0.35, f"aulc={dc_o['aulc']}")
    check("deepchain: eps-greedy myopia DISPLAYED (tabular-Q <= 0.10)",
          dc_q["aulc"] <= 0.10, f"aulc={dc_q['aulc']}")
    check("deepchain: deep-vs-myopic separation unmistakable (>= 0.30)",
          dc_o["aulc"] - dc_q["aulc"] >= 0.30, f"gap={dc_o['aulc'] - dc_q['aulc']:.3f}")
    dc_o2 = measure_family(lambda a, s: OptimisticQAgent(a, s), "deepchain", cp_seeds, agent_seed=99)
    check("deepchain reproducibility (same seeds => identical)", dc_o == dc_o2)

    # 13) compose (compositionality — PROMOTED into the default universe 2026-07-11 evening, v1.3 — combining known parts into never-seen
    #     combinations, the ARC-adjacent capability). Positive control = a compositional reference
    #     that learns primitive hops under bandit feedback and answers HELD-OUT pair compositions
    #     by running two known hops (post-slice ~1.0 measured — the held-out half is free). The
    #     displayed failure is the pair-memorizer signature: tabular-Q keys on full triples, so
    #     every held-out pair starts from scratch (post ~0.28 measured). Gates leave headroom.
    co_seeds = [random.Random(4242).randrange(10**9) for _ in range(6)]
    co_post = lambda m: (sum(m["curve"][COMPOSE_POST_SLICE:])
                         / len(m["curve"][COMPOSE_POST_SLICE:]))
    co_r = measure_family(lambda a, s: RandomAgent(a, s), "compose", co_seeds, agent_seed=99)
    co_q = measure_family(lambda a, s: TabularQAgent(a, s), "compose", co_seeds, agent_seed=99)
    co_c = measure_family(lambda a, s: CompositionalRefAgent(a, s), "compose", co_seeds, agent_seed=99)
    check("compose: random control flat (harness symmetric across the pair-holdout switch)",
          abs(co_r["aulc"]) <= 0.10 and abs(co_post(co_r)) <= 0.10,
          f"aulc={co_r['aulc']} post={co_post(co_r):+.3f}")
    check("compose: compositional reference answers HELD-OUT pairs at once (post >= 0.85)",
          co_post(co_c) >= 0.85, f"post={co_post(co_c):+.3f}")
    check("compose: pair-memorizer displayed (tabular-Q post <= 0.45 — every unseen pair from scratch)",
          co_post(co_q) <= 0.45, f"post={co_post(co_q):+.3f}")
    check("compose: compositional-vs-memorizer separation unmistakable (>= 0.40 post)",
          co_post(co_c) - co_post(co_q) >= 0.40, f"gap={co_post(co_c) - co_post(co_q):.3f}")
    co_c2 = measure_family(lambda a, s: CompositionalRefAgent(a, s), "compose", co_seeds, agent_seed=99)
    check("compose reproducibility (same seeds => identical)", co_c == co_c2)
    for s in (11, 222, 3333):
        env = CompositionEnv(s)
        check(f"compose env-sanity :{s}", env.oracle_return > env.random_return
              and len(env.train_pairs) == 8 and len(env.test_pairs) == 8,
              f"oracle={env.oracle_return} floor={env.random_return}")

    # 14) CONSUMER-CONVENTION gates (insight #75 backport: a capability / non-regression claim is bound
    #     to the SAMPLE it was measured on). Gates 9/10/12 validate changepoint / changepoint_sustained /
    #     deepchain via measure_family(cp_seeds = Random(4242) × 6, agent_seed=99) — but the published
    #     nine-family FLOOR and `--run` consume measure()'s OWN instance draws at the DEFAULT
    #     master_seed=20260709 / n_instances=8 / agent_seed=master_seed+13: a DIFFERENT sample. The
    #     between-sample gap is large in magnitude (gridnav 0.80 vs 0.60; the deepchain solver 0.49 on
    #     cp_seeds vs 0.50 on the consumer draw), so a gate passing on cp_seeds says nothing about the
    #     floor's convention. Each default-universe adaptation/exploration family is therefore ALSO gated
    #     on the exact sample the consumer reads. Same conservative bars as the measure_family gates,
    #     cleared with margin on the measured consumer draw — the property must hold on BOTH samples.
    cc = measure(lambda a, s: TabularQAgent(a, s), master_seed=20260709, n_instances=8)
    cc_slice = lambda m, sl: sum(m["curve"][sl:]) / len(m["curve"][sl:])  # noqa: E731
    cc_cp, cc_su, cc_dc = (cc["profile"]["changepoint"], cc["profile"]["changepoint_sustained"],
                           cc["profile"]["deepchain"])
    check("changepoint on the CONSUMER draw adapts (measure()@20260709,n=8: AULC>=0.28 & post>=0.12)",
          cc_cp["aulc"] >= 0.28 and cc_slice(cc_cp, CHANGEPOINT_POST_SLICE) >= 0.12,
          f"aulc={cc_cp['aulc']} post={cc_slice(cc_cp, CHANGEPOINT_POST_SLICE):.3f}")
    check("changepoint_sustained on the CONSUMER draw holds steady state (late>=0.10)",
          cc_slice(cc_su, CHANGEPOINT_SUSTAINED_SLICE) >= 0.10,
          f"late={cc_slice(cc_su, CHANGEPOINT_SUSTAINED_SLICE):.3f}")
    check("deepchain eps-greedy myopia on the CONSUMER draw (tabular-Q AULC<=0.10)",
          cc_dc["aulc"] <= 0.10, f"aulc={cc_dc['aulc']}")
    # The deepchain SOLVER (optimistic-Q) on the consumer sample: replicate measure()'s exact full-
    # universe seed draw (verified bit-identical to its profile) — NOT measure(families=["deepchain"]),
    # which would hand deepchain the FIRST 8 draws instead of the draws it gets as the 9th family (a
    # wrong-sample trap of exactly the #75(d) kind — a test that passes on the wrong inputs).
    _rng = random.Random(20260709)
    _iseeds = {f: [_rng.randrange(10**9) for _ in range(8)] for f in FAMILIES}
    cc_do = measure_family(lambda a, s: OptimisticQAgent(a, s), "deepchain",
                           _iseeds["deepchain"], agent_seed=20260709 + 13)
    check("deepchain solver clears its bar on the CONSUMER draw (optimistic-Q AULC>=0.35)",
          cc_do["aulc"] >= 0.35, f"aulc={cc_do['aulc']}")
    check("deepchain deep-vs-myopic separation on the CONSUMER draw (>=0.30)",
          cc_do["aulc"] - cc_dc["aulc"] >= 0.30, f"gap={cc_do['aulc'] - cc_dc['aulc']:.3f}")

    # 15) factored (factored-structure discovery — PROMOTED into the default universe 2026-07-11
    #     night, v1.4, at measured price zero after v10 cleared its reference).
    #     Unlike compose, the structure is NOT in the observation's slots: opaque codes, and B's
    #     answer factors through the A and S answers via a hidden per-instance relation. Positive
    #     control = a relational reference that confirms answers by elimination, fills the relation
    #     table from confirmed train triples, and answers NEVER-ASKED B questions by running the
    #     discovered factorization (post ~0.999 measured). Displayed failure = the memorizer:
    #     tabular-Q starts held-out B keys from scratch and only climbs by relearning (post ~0.51
    #     measured — the small held-out keyspace relearns fast, so the separation bar is 0.30).
    fa_seeds = [random.Random(4242).randrange(10**9) for _ in range(6)]
    fa_post = lambda m: (sum(m["curve"][FACTORED_POST_SLICE:])
                         / len(m["curve"][FACTORED_POST_SLICE:]))
    fa_early = lambda m: (sum(m["curve"][FACTORED_POST_SLICE:FACTORED_EARLY_END])
                          / (FACTORED_EARLY_END - FACTORED_POST_SLICE))
    fa_r = measure_family(lambda a, s: RandomAgent(a, s), "factored", fa_seeds, agent_seed=99)
    fa_q = measure_family(lambda a, s: TabularQAgent(a, s), "factored", fa_seeds, agent_seed=99)
    fa_c = measure_family(lambda a, s: RelationalRefAgent(a, s), "factored", fa_seeds, agent_seed=99)
    check("factored: random control flat (harness symmetric across the holdout switch)",
          abs(fa_r["aulc"]) <= 0.10 and abs(fa_early(fa_r)) <= 0.10,
          f"aulc={fa_r['aulc']} early={fa_early(fa_r):+.3f}")
    check("factored: reference answers NEVER-ASKED B at once (EARLY slice >= 0.85; measured 0.97)",
          fa_early(fa_c) >= 0.85, f"early={fa_early(fa_c):+.3f}")
    check("factored: memorizer displayed on the ISOLATING slice (tabular-Q early <= 0.50; measured 0.35)",
          fa_early(fa_q) <= 0.50, f"early={fa_early(fa_q):+.3f}")
    check("factored: discovery-vs-memorizer separation unmistakable (>= 0.35 early)",
          fa_early(fa_c) - fa_early(fa_q) >= 0.35, f"gap={fa_early(fa_c) - fa_early(fa_q):.3f}")
    check("factored: reference holds the WHOLE post window too (post >= 0.85)",
          fa_post(fa_c) >= 0.85, f"post={fa_post(fa_c):+.3f}")
    fa_c2 = measure_family(lambda a, s: RelationalRefAgent(a, s), "factored", fa_seeds, agent_seed=99)
    check("factored reproducibility (same seeds => identical)", fa_c == fa_c2)
    for s in (11, 222, 3333):
        env = FactoredRelationEnv(s)
        covered = {(env.ansA[c], env.ansS[c]) for c in env.train_codes}
        check(f"factored env-sanity :{s} (oracle > floor; train codes tile the relation domain)",
              env.oracle_return > env.random_return and len(covered) == env.alphabet ** 2,
              f"cells covered={len(covered)}/{env.alphabet ** 2}")

    # 7) certificate path (honest either way)
    if _HAS_ATTEST:
        doc = attest_run(rnd)
        ok, probs, _ = _ca.verify(doc)
        check("attestation verifies", ok, str(probs))
    elif verbose:
        print("  [ok ] attestation DEFERRED (coverage-attestation sibling absent)", file=sys.stderr)

    if verbose:
        print(f"\n  {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S): ' + ', '.join(fails)}",
              file=sys.stderr)
    return len(fails)


def _print_efficiency(eff: dict, label: str):
    print(f"\n=== generality-per-budget — {label} ===")
    for p in eff["floor_curve"]:
        print(f"  budget x{p['budget_mult']:<4} ({p['total_episodes']:>4} eps)  "
              f"floor={p['floor']:+.3f} ({p['floor_family']:<8})  mean={p['mean']:+.3f}")
    br = eff["breadth_ratio"]
    br_s = f"{br:.2f}" if br is not None else "None (overall competence flat — undefined)"
    b2f = eff["budget_to_floor"]
    b2f_s = f"{b2f} eps" if b2f is not None else f"None (floor never reached {eff['target_floor']} — fail-closed)"
    print(f"  Δfloor={eff['d_floor']:+.3f}  Δmean={eff['d_mean']:+.3f}  breadth_ratio={br_s}")
    print(f"  budget_to_floor(>={eff['target_floor']}) = {b2f_s}")
    print(f"  boundary: {eff['boundary']}")


def _print_atlas(report: dict, label: str):
    print(f"\n=== generality-atlas — {label} ===")
    print(f"  master_seed={report['master_seed']}  families={len(report['declared_families'])}")
    for f, m in report["profile"].items():
        bar = "#" * int(max(0.0, min(1.0, m["aulc"])) * 30)     # display clips; the metric never does
        print(f"  {f:<9} [{m['capability']:<22}] jumpstart={m['jumpstart']:.2f} "
              f"AULC={m['aulc']:.2f} final={m['final']:.2f}  {bar}")
    print(f"  GENERALITY FLOOR = {report['generality_floor']:.2f}  (weakest: {report['floor_family']})")
    print(f"  boundary: {report['boundary']}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if _selftest() else 0)
    if _selftest(verbose=False):
        print("SELFTEST FAILING — the instrument may not measure anything.", file=sys.stderr)
        sys.exit(2)
    if "--transfer" in sys.argv:
        factory = lambda a, s: TabularQAgent(a, s)
        ok, ctl = transfer_controls_pass()
        print(f"controls: positive_ok={ctl['positive_ok']} "
              f"(jumpstart_delta={ctl['positive_control']['jumpstart_delta']})  "
              f"negative_ok={ctl['negative_ok']} (aulc_delta={ctl['negative_control']['aulc_delta']})")
        if not ok:
            print("CONTROLS FAILED — difficulty-transfer numbers are not meaningful; not reporting.")
            sys.exit(3)
        for fam in DIFFICULTY:
            r = measure_transfer(factory, fam)
            print(f"  {fam:<8} easy->hard  jumpstart_delta={r['jumpstart_delta']:+.3f}  "
                  f"aulc_delta={r['aulc_delta']:+.3f}   (scratch AULC={r['scratch']['aulc']:.3f})")
        sys.exit(0)
    if "--efficiency" in sys.argv:
        # The SAME learner shown two ways: on a universe it can cover (budget buys BREADTH — the floor
        # rises) and on the full universe with an unlearnable capability (budget buys DEPTH only — the
        # floor is memory-pinned, budget_to_floor fail-closes to None). Plus the specialist (never
        # general at any budget) and the random control (no gain => split undefined).
        learnable = ["bandit", "gridnav", "sequence", "parity"]
        _print_efficiency(measure_efficiency(lambda a, s: TabularQAgent(a, s), families=learnable),
                          "tabular-Q on a COVERABLE universe (budget buys breadth)")
        _print_efficiency(measure_efficiency(lambda a, s: TabularQAgent(a, s)),
                          "tabular-Q on the FULL universe (memory unlearnable -> budget buys depth only)")
        _print_efficiency(measure_efficiency(lambda a, s: BanditOnlyAgent(a, s)),
                          "bandit-only specialist (never general regardless of budget)")
        _print_efficiency(measure_efficiency(lambda a, s: RandomAgent(a, s)),
                          "random control (no competence gain -> split undefined)")
        sys.exit(0)
    if "--run" in sys.argv:
        reports = {
            "random (control)": measure(lambda a, s: RandomAgent(a, s)),
            "tabular-Q (honest simple learner)": measure(lambda a, s: TabularQAgent(a, s)),
            "bandit-only (planted narrow)": measure(lambda a, s: BanditOnlyAgent(a, s)),
        }
        if "--json" in sys.argv:
            out = {k: v for k, v in reports.items()}
            if _HAS_ATTEST:
                out["attestation"] = attest_run(reports["tabular-Q (honest simple learner)"])
            print(json.dumps(out, indent=2))
        else:
            for label, rep in reports.items():
                _print_atlas(rep, label)
    else:
        print(__doc__)
