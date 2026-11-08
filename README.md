# BrainAI

**A spiking, multimodal cognitive architecture with continual learning.**
Built by Mohammed Zahid.

BrainAI is one model that perceives text, images, video, audio and touch;
routes each input through a reflex arc, a subconscious pathway and a gated
conscious pathway; keeps five kinds of memory; **predicts what happens next and
learns from being wrong**; weighs new claims against evidence before believing
them; pursues goals it can be interrupted out of and resume; plans by
imagination before acting; updates its own weights while running; and
consolidates what it learned during an offline sleep cycle. It emits both
language and actuator commands — through a safety controller it cannot
modify — so the same core drives speech and motion.

The default deployment name is `BrainAI`. Any instance can be renamed —
`model.rename("Javri")` — and authorship travels with the weights regardless.

```bash
pip install torch
python train.py --steps 400          # pretrain, one domain (CPU, ~10 min)
python train_multidomain.py          # 64 domains, per-domain eval (CPU, ~9 h)
python finetune_author.py            # boost author recall (continuation, ~20 min)
python evaluate_trained.py           # every benchmark on the trained weights
python continual.py --steps 120      # the anti-forgetting measurement
python demo.py                       # every capability, end to end
python -m pytest tests -q            # 114 tests

python benchmarks/cognition_bench.py # world model, curiosity, grounding, planner
python benchmarks/amsa_continual.py --sweep   # AMSA on loss (it loses)
python benchmarks/amsa_sparsity.py --seeds 3  # AMSA on efficiency (it wins)
python benchmarks/worldmodel_jobs.py # the world model's actual jobs
python benchmarks/gating_bench.py    # is the conscious pathway really skipped?
python benchmarks/belief_bench.py    # Reality Cortex vs a naive baseline
python benchmarks/stability_bench.py --control   # 400 online updates
```

---

## Evidence summary

Every row is produced by a script in this repo. Rows that came out badly are
still here.

| Subsystem | Benchmark | Control | Result |
|---|---|---|---|
| Multi-domain breadth | 64 domains, nano | start of training | **5.49 → 2.25 bits/byte**; generation still character-level |
| Author recall | 20× boost, EWC | pre-fine-tune | **3.32 → 2.59 bits/byte**, breadth held |
| Pathway gating | wall-clock | always-on | **2.16× faster**, 53.7% of forward time |
| World model | one-step MSE | linear map / RSSM | beats RSSM, **loses to linear** |
| World model | rollout, planning | action-conditioned linear | **ties by 3 steps**; only one with calibrated uncertainty |
| Curiosity | noisy-TV | pure novelty | **403×** preference for learnable |
| Trust grounding | broken vs. good sensor | fixed prior | **0.974 vs 0.104** from identical priors |
| Reality Cortex | 5 belief scenarios | naive vote counting | wins **4 of 5** |
| Collusion defence | 600-puppet sybil | no clustering | **1.0000 vs 0.0001** |
| Continual learning | EN → FR forgetting | no protection | **74% less** |
| Online stability | 400 updates | no guards | **3.7× less** degradation |
| Planner | step fails repeatedly | flat planner | `remethod` vs `failed` |
| Executive | deadline pressure | static priority | overtakes at the right tick |
| Safety | envelope + e-stop | — | 2 live hazards found and fixed |
| Memory index | recall@4 | exact search | 94–100% recall, **but slower** |
| **AMSA** | perplexity, speed, continual | SwiGLU | **loses on loss** (all three) |
| AMSA scale sweep | width 96 → 256 | — | loss gap **widens**, not shrinks |
| **AMSA** | sparsity / energy, 3 seeds | SwiGLU | **wins**: sparser 3/3, ~14-pt lower FFN density at equal loss |

Two of those are negative and both are load-bearing. AMSA loses on the problem
it was designed for, and the world model loses to a linear baseline. Neither is
buried.

---

## What is actually novel here

Spiking networks, transformers, retrieval memory, EWC, MoE and sleep replay all
exist separately. What is assembled here and what is not:

**Implemented and measured**

- **Latency-tiered pathways** with a gate that genuinely skips computation at
  inference rather than masking it.
- **A memory split along the differentiability line** — neural state for
  reasoning, non-differentiable stores for facts — so post-training knowledge
  cannot corrupt a weight an older skill depends on.
- **A predictive world model** whose error is simultaneously the dynamics
  signal, the curiosity signal, and the thing that grounds source trust without
  an oracle. Beats an RSSM-family baseline, loses to a linear one; 403×
  preference for learnable states over a noisy TV.
- **Typed sensors and actuators** — rate, latency, drift and chemical recovery
  time per sensor; per-joint safety envelopes derived from whether a joint is a
  compliant pneumatic gripper or a stiff hydraulic hip.
- **A Reality Cortex** that separates *important* from *true*, with
  correlation-discounted evidence pooling, credibility shrinkage for unvetted
  sources, behavioural collusion detection, and protected axioms. Beats a naive
  counting baseline on every scenario tested.
- **An executive with goals** — priority × urgency arbitration, interrupts that
  preserve and restore working memory, conflict suspension, and pressure-driven
  sleep scheduling.
- **A hierarchical planner** that scores plans by imagining them, imports
  corroborated procedures as executable methods, and distinguishes retry from
  replan from failure.
- **Confidence that propagates** from perception to actuator, where it can veto
  a motor command rather than merely being logged.
- **Bounded inference-time weight updates** with per-parameter plasticity, a
  trust region, and automatic rollback.
- **Sleep** that validates, replays raw tokens, merges redundant memories, and
  re-anchors.
- **A non-learned safety controller** that online learning provably cannot
  reach, between the motor cortex and the joints.
- **AMSA**, an activation whose threshold and gain are driven by each channel's
  learned importance rather than by learned constants alone.

**Proposed, mixed evidence.** AMSA **loses to SwiGLU on loss** (perplexity,
continual learning) and on wall-clock speed, but **wins on sparsity/energy** —
at comparable loss it fires fewer spikes and lights up ~14 points less of the
feed-forward layer, across 3 seeds. It is a quality-vs-efficiency tradeoff, and
it is the default in this spiking architecture for that reason; see
[the numbers](#does-amsa-actually-help). The anti-forgetting result is measured
on two tasks, which is the smallest honest test rather than a thorough one.
Sybil attacks are made expensive, not impossible. `full` has never been
instantiated. Nothing here has been run on a robot.

Every number in this README is produced by a script in this repo, including the
ones that came out badly. Several claims here had no measurement behind them
until they were tested — one of them (EWC) was doing literally nothing while
looking like it worked, and the first LSH configuration traded 81% of its recall
for a speedup it did not need.

---

## Architecture

```
    text   image   video   audio   touch
      └───────┴───────┴───────┴──────┘
                     │  per-modality spiking encoders
                     ▼
            shared d_model spike bus
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              │
   REFLEX      SUBCONSCIOUS         │   reflex answers immediately;
  (shallow,     (always on,         │   subconscious scores salience
   always)       sparse)            │
      │              │              │
      │              ├── salience ──┴──► gate
      │              ▼
      │        CONSCIOUS  ◄──── working memory (read/write)
      │      (wide, MoE,   ◄──── episodic + semantic recall
      │       gated)             via the LTM interface
      │              │
      └──────►  ARBITER  ◄───────┘   mixes fast vs. deliberate
                     │
             ┌───────┴───────┐
             ▼               ▼
        language head    motor cortex
                             │
                    SAFETY CONTROLLER      not learned, not reachable
                             │             by online learning
                             ▼
                          joints

    control path (which of these should be running at all):

    EXECUTIVE ──► goal stack, interrupts, task switching, sleep scheduling
        │
        └──► PLANNER ──► subgoals ──► WORLD MODEL (imagine) ──► action

    knowledge path (offline, not per-token):

    episodic ──► REALITY CORTEX ──► semantic ──► sleep ──► weights
                      ▲            evidence, source trust,
                      │            contradiction, promotion
                 WORLD MODEL
              prediction error grounds
              source trust with no oracle
```

Confidence is carried alongside every stage and composed end to end, so the
safety controller can refuse to act on a decision the pipeline was never sure
about.

### The three pathways

| Pathway | Runs | Cost | Job |
|---|---|---|---|
| Reflex | always | 1 shallow stack | Have *something* correct-enough available now. Never consults memory. |
| Subconscious | always | sparse | Filter, predict, and decide whether thinking is worth it. |
| Conscious | when salience clears threshold | wide, expert-routed | Reasoning, planning, memory integration. |

The gate is **soft during training** so the salience head learns *when*
deliberation pays, and **hard at inference** so skipping is a real saving.
`run_gated` selects the batch rows that need it and runs only those.

"Genuinely skipped" is a wall-clock claim, so it gets a wall-clock measurement —
a gate that computes everything and multiplies by zero looks identical from the
outside (`benchmarks/gating_bench.py`, batch 8 × seq 128, CPU):

| | sec/forward | vs. always-on |
|---|---|---|
| all rows deliberate | 0.2446 | 1.00× |
| no rows deliberate | 0.1133 | **2.16×** |

**53.7% of forward time** is in the conscious pathway, and skipping it recovers
all of it. It is the widest part of the model and the only expert-routed one, so
this is the largest single lever on inference cost.

### The five memories

| Memory | Backing | Differentiable | Holds |
|---|---|---|---|
| Sensory | deque | no | Last few frames, for temporal integration |
| Short-term | spiking LSTM | **yes** | The recent event stream, pooled |
| Working | attention over slots | **yes** | The reasoning scratchpad |
| Episodic | `VectorStore` buffer | no | Experiences, importance-ranked |
| Semantic | `VectorStore` + graph | no | Concepts and the edges between them |

The differentiability split is the anti-forgetting strategy. Facts learned
after training go into the *stores*, where writing them cannot damage a weight
some older skill depends on. Only slow, importance-gated consolidation ever
moves knowledge from a store into the weights.

Because the stores are registered buffers, they are saved and restored with the
checkpoint — a robot's episodic history survives a restart.

Two design decisions in there are load-bearing and easy to get wrong:

- **The key/value storage projections are frozen random projections, not
  learned.** Every key already in the store was written with those weights. If
  they were trained, each update would silently invalidate the entire store,
  because old keys and new queries would stop living in the same space. A
  random projection preserves the cosine geometry retrieval needs.
- **Retrieval is non-differentiable on purpose.** A bad memory can mislead the
  model but can never corrupt a gradient. The cost is that the importance
  scorer gets no signal from the language loss, so it is trained during sleep
  against a prioritised-replay target instead: an episode is important to the
  degree the model still gets it wrong.

---

## The world model — imagine first, learn from being wrong

Everything else here is reactive: something arrives, the pathways respond,
memory records it. `WorldModel` runs *ahead* of the world — given a latent state
and a candidate action it predicts the next state, and the gap between that and
what actually happened is a learning signal that needs no labels and no human.

Three things fall out of that one mechanism, which is why it is worth more than
its size:

**Grounding.** `PredictionLedger` lets a belief stake a claim on the future.
When the observation lands, the ledger resolves it and tells the Reality Cortex
which sources were right. This is what removes the oracle described below.

**Curiosity.** Prediction error, normalised against the model's own *expected*
error. It goes to zero both for mastered states and for irreducibly random ones,
so the drive points at what is learnable rather than merely surprising.

**Planning.** `imagine()` rolls the dynamics forward without touching the
environment, so a bad plan costs a few matmuls instead of a gripper.

Measured (`benchmarks/cognition_bench.py`):

| System | persistence | linear map | RSSM | **BrainAI** |
|---|---|---|---|---|
| linear (rotation) | 2.096 | **0.006** | 0.244 | 0.153 |
| nonlinear (saturating) | 1.905 | **0.249** | 0.406 | 0.354 |

**It beats an RSSM-family recurrent model on both systems and loses to a plain
linear map on both.** On the linear system that is expected — a linear map is
*optimal* there, so losing 24× measures overhead, not capability. On the
saturating nonlinear system the gap narrows to 1.4×, and that one is a genuine
loss: a 16k-parameter linear layer predicts better than a 119k-parameter
dynamics head.

Getting to that number took two attempts, and the first one was the useful
lesson. A persistence baseline made the model look 11× better than it is, and a
"nonlinear" system built as `tanh(s @ mix)` at unit variance is still nearly
linear — so it also measured almost nothing. The gain had to be pushed to ×3 to
put tanh into saturation before the comparison meant anything.

### But one-step MSE is not the job

That result is real and it stays. It also measures the wrong thing. Nothing in
this architecture asks the world model for a one-step point prediction — it asks
it to imagine multiple steps, rank candidate plans, know how far ahead it can be
trusted, and separate learnable surprise from noise. A model can be worse at one
step and fine at all of those, because errors compound differently and a point
prediction carries no uncertainty at all.

`benchmarks/worldmodel_jobs.py`, against an **action-conditioned** linear model
of the same form, 4 seeds, on a saturating action-driven system:

| Job | linear | BrainAI | winner |
|---|---|---|---|
| rollout MSE @ 1 step | **0.2355** ±0.003 | 0.2570 ±0.006 | linear |
| rollout MSE @ 3 steps | 0.3340 ±0.004 | 0.3404 ±0.007 | tie |
| rollout MSE @ 6 steps | 0.3852 ±0.004 | 0.3849 ±0.005 | tie |
| rollout MSE @ 12 steps | 0.4073 ±0.006 | 0.4055 ±0.005 | tie |
| planning: picks best of 8 | 38.3% ±4.8 | 39.4% ±2.5 | tie (random = 12.5%) |
| uncertainty tracks real error | **n/a** | **0.565** ±0.036 | BrainAI |
| surprise: structure vs. noise | **n/a** | −0.52 / 7.99 | BrainAI |

**The one-step deficit does not propagate.** It is real and separated at one
step, and gone by three — both models degrade to the same place over a horizon,
and both plan equally well and far above chance.

The last two rows say `n/a` for a structural reason, not a missing experiment: a
point prediction cannot report its own reliability. Without that, the planner
cannot tell a trustworthy rollout from a guess, and curiosity cannot tell
learnable surprise from noise. Those are the two things the world model exists
to feed.

Two methodological notes, because both changed the answer:

- A **single seed said BrainAI won** at 3, 6 and 12 steps and at planning. Four
  seeds with error bars showed those margins were inside noise. They are
  reported as ties.
- Calibration first came out at **−0.615**, an *anti*-correlation. The cause was
  mine: the head was trained at unit variance and scored across 0.3–2.3× state
  magnitude, so it was being measured off-distribution. Training across the same
  range it is tested on gives +0.565. A robot meets states of varying magnitude,
  so the varied-difficulty setup is the representative one — but the earlier
  number was a measurement artefact, not a discovery.

| | result |
|---|---|
| rollout confidence at horizon 1 / 4 / 16 | 0.623 / 0.145 / 0.000 |
| curiosity: learnable state vs. a noisy TV | **403×** preference |
| pure-novelty control on the same pair | 1.0× — cannot tell them apart |

The noisy-TV row is the one that matters. A pure novelty drive parks in front of
a noise source forever because every frame is unseen. Gating on *expected* error
is what walks away from it. The `error_head` exists solely for that: it
separates "I do not know" from "this is inherently random".

Rollout confidence compounds multiplicatively, so a 16-step plan through
uncertain dynamics correctly scores ~0. The alternative — trusting a long plan
as much as a short one — is how imagined planning usually fails.

The same idea turned inward is `BodyModel`: predict the proprioceptive result of
a motor command, and the residual is a calibration signal for the robot's real
dynamics — backlash, a sticking joint, a drifting encoder — learned from
operation rather than from a datasheet.

---

## Removing the oracle

`observe_outcome(claim_id, was_correct)` was the weakest thing in the previous
version. Source trust is what keeps beliefs anchored to reality, and it only
moved when *something told it* a prediction was right. In tests that caller was
the test. On a real robot nothing supplies it, so trust would never leave its
operator-supplied prior.

Now the environment supplies the verdict. Two sensors, both handed prior **0.8**
by the operator, one honest and one broken, with nothing labelling either:

| Source | Operator prior | After 40 rounds of grounding |
|---|---|---|
| good_sensor | 0.800 | **0.974** |
| broken_sensor | 0.800 | **0.104** |

Separation **0.870**, from predictions resolving against observations. A fixed
prior keeps both at 0.800 forever and can never tell them apart.

Predictions that are never checked **expire without penalty**. Scoring an
unobserved prediction as wrong would make trust decay merely for talking about
things the robot never got around to verifying.

---

## The executive — what to work on, and what to drop

The old executive was one `nn.Linear` mixing the reflex answer with the
deliberate one. That is an output arbiter, not an executive: no notion of a
goal, cannot be interrupted, cannot resume anything. The arbiter still does the
mixing and is now honestly named; `Executive` is the part that decides what the
model is *for* at any moment.

Goals are scored `priority × urgency × feasibility`:

```
  tick        winner   deep_work(p=0.80)   deliver_report(p=0.55, due@30)
     0     deep_work               0.400                            0.344
    10     deep_work               0.400                            0.367
    25  deliver_report             0.400                            0.458
    30  deliver_report             0.400                            0.550
```

The lower-priority job overtakes as its deadline closes, and only then. A static
priority scheme cannot express that.

One bug worth recording because it inverted the intended behaviour: urgency was
`1/(1 + k·remaining)`, which sends a *distant* deadline toward zero — so
attaching a deadline to a goal made it less likely to run than having none at
all. Urgency is now bounded below by the no-deadline baseline and rises to 1.0.

**Interruption preserves work.** On a switch the outgoing goal's working-memory
contents are saved into the goal and restored when it resumes — the difference
between an interruption and an abort. Conflicting goals are *suspended*, not
failed, because "not now" and "never" are different outcomes.

**Sleep scheduling** lives here, because deciding when to stop acting is an
executive decision. Four independent pressures — replay-buffer fill, unpromoted
claims, weight drift, time since last sleep — so no single one can force a
consolidation. Idleness is a multiplier rather than a term: the right moment to
consolidate is when nothing is being asked of you. `sleep()` with no arguments
now sizes its own cycle from that pressure.

---

## The planner — decompose, imagine, commit, repair

`SpikingStack("planner")` was a transformer stack with "planner" in its name.
`HierarchicalPlanner` is HTN-style: a method library maps a goal to ordered
subgoals, recursively, until every leaf is primitive. Candidate plans are scored
by rolling them through the world model rather than by executing them.

**Methods can be learned.** A procedure that survives the Reality Cortex —
corroborated by independent sources, not contradicted — is imported as a
planning method. That is where evidence and capability meet: a procedure the
robot picked up by watching becomes one it can carry out.

```
  two mechanics describe replacing a belt  ->  confidence 0.99
  imported as a method                     ->  plan: loosen_tensioner
                                                     slip_belt_off
                                                     fit_new_belt
```

A procedure asserted once by a low-trust source imports as **0 methods**.

Failure handling distinguishes three outcomes, and the distinction is the point:

| | flat planner | hierarchical |
|---|---|---|
| step fails repeatedly | `failed`, `failed` | `retry`, **`remethod`** |

A second bug found here: `remethod` originally checked only the *immediate*
parent for alternatives. Alternatives are normally registered against the
high-level task, not the leaf that broke, so every recoverable failure was
reported as fatal. It now walks up the ancestors.

---

## Uncertainty that reaches the actuators

The Reality Cortex had calibrated confidence and nothing else did — a blurry
frame and a clear one produced equally assertive motor commands.

Every stage now carries a confidence, composed **multiplicatively with a floor**
(a chain is no more trustworthy than its weakest link; averaging lets a
confident planner paper over a perception failure it cannot see). Sources,
cheapest first: predictive entropy, spike-rate deviation from the layer's usual
sparsity, expert disagreement, and learned heads where a supervision signal
exists.

```
perception=0.99  reflex=0.49  salience=0.47  language=0.05
overall 0.326    bottleneck: language
```

The end-to-end number goes to the safety controller, where
`requires_confidence` will veto a motor command outright. That is the point of
propagating it: uncertainty stops being a diagnostic and becomes a brake.

**Experts are allowed to disagree.** A mixture of experts routes tokens and then
*averages* the winners — so when two experts propose contradictory things, the
fact that the model was internally torn is discarded at exactly the moment it
was most informative. `InternalDebate` runs them separately, has a judge pick,
and reports the spread. That disagreement is a measurement rather than a
self-report.

`SelfCritique` closes the outer loop — generate, criticise, revise — and returns
the **best** round rather than the last, because revision is not monotone.

---

## The Reality Cortex — salience is not truth

A system that learns from whatever is around it will, without something like
this, simply adopt whatever it hears most. Importance answers *"is this worth
attending to"*; a claim shouted by a thousand people scores high on that and can
still be false. Everything upstream of here — episodic writes, importance
scoring, replay — is truth-agnostic by design.

`RealityCortex` sits on the **knowledge path**, not the token path:

```
episodic ──► Reality Cortex ──► semantic ──► sleep ──► weights
```

Not in the forward pass: contradiction detection over a belief graph is the
wrong thing to run per token, and ruinously slow if you do. Beliefs re-enter the
forward pass through retrieval, where confidence weights what gets recalled.

**Log-odds evidence pooling.** Each observation contributes `log(t / (1 − t))`
for its source's reliability `t` — the naive-Bayes likelihood ratio. Refutation
is the same quantity negated, so support and doubt accumulate on one scale.

**Per-source diminishing returns.** The *n*th claim from the same source counts
`1/n` as much as the first. This is the mechanism that matters most for a robot
that lives somewhere. Repetition from one voice grows harmonically and
saturates; agreement between *independent* sources compounds.

**Rival-relative confidence.** For functional relations — single-valued ones
like `capital_of` — confidence is `sigmoid(logit − best_rival_logit)`. Two
equally supported alternatives sit at 0.5 and neither is promoted, which is the
honest answer.

**Protected axioms.** `assert_axiom` seeds a belief with large prior log-odds,
so overturning it takes overwhelming independent evidence rather than one
confident stranger.

**Correlated-evidence discounting.** The *k*th distinct source counts
`1/sqrt(k)`. Naive pooling assumes independence, and 200 unvetted accounts
agreeing is far better explained by one actor than by 200 witnesses. Paired with
**credibility shrinkage** — a source's likelihood ratio is scaled by
`n_eff / (n_eff + 4)`, so an identity that has never once been checked against
reality does not arrive at full strength.

### Measured against a baseline

`benchmarks/belief_bench.py` compares it to `NaiveCounter` — confidence =
supports / (supports + refutes), the obvious implementation:

| Scenario | Reality Cortex | Naive |
|---|---|---|
| spam: 100 repeats vs. 2 good sources | **0.9986** ✓ | fails |
| crowd: the majority is right | **0.9857** ✓ | ✓ |
| sybil: 200 identities voting **once each** | 0.0145 ✗ | fails |
| sybil vs. a protected axiom | **0.9956** ✓ | fails |
| sybil: 200 identities **with a voting record** | **0.8636** ✓ | fails |

The two sybil rows are the interesting pair. A puppet that votes exactly once is
indistinguishable from a genuine one-time witness — correctly so, since a single
data point cannot support the accusation — and gets through unless the target is
a protected axiom. The moment a ring participates in several claims its shared
pattern gives it away, the whole ring collapses to one vote, and a *single*
trusted source then outweighs all 200 of them.

**Collusion detection.** The correlation discount alone still assumes distinct
names are distinct witnesses, which is exactly the assumption a sybil attack
violates. A puppet ring is visible in its *behaviour* — the accounts appear on
the same claims with the same polarity — so sources whose voting records match
are clustered and cast **one vote between them**:

| Puppets | Clustering off | Clustering on |
|---|---|---|
| 50 | 0.9981 | 1.0000 |
| 200 | 0.4867 | 1.0000 |
| 600 | **0.0001** | **1.0000** |

Detection is two passes: exact voting-signature grouping (O(S), catches a
scripted ring) then Jaccard refinement on a bounded sample (catches a ring with
jitter). Rebuilt on a counter rather than per submission, so an attack is caught
within a bounded number of votes instead of making every write O(S²).

Attack cost without clustering, for reference — identities needed to flip a
belief:

| Target | `discount="sqrt"` | `discount="none"` |
|---|---|---|
| single trusted source | 50 | 10 |
| protected axiom | **650** | 50 |

**This is not solved, only made expensive.** Clustering catches identities that
behave identically. A patient adversary that varies its puppets' voting patterns,
or one that compromises genuinely independent sources, still gets through.

One detail worth stating because it made the first version of this benchmark
meaningless: sockpuppets are given prior **0.62**, not 0.2. Below 0.5 a source's
*support* is already evidence against — `log(t/(1−t))` goes negative — so a
flood of visibly-untrusted accounts defeats itself and tests nothing. The attack
that matters uses identities that each look individually plausible.

### Other measured behaviour

| Scenario | Result |
|---|---|
| "M10 bolt = 40 Nm" from 3 independent sources | **0.9904** |
| `paris capital_of germany` vs. two reliable sources for `france` | **0.0007**, retired |
| Source wrong 4× via `observe_outcome` | trust 0.30 → **0.18**, and every claim resting on it falls with it |

### Scaling

| Claims | submit | confidence | promote | report |
|---|---|---|---|---|
| 1,000 | 4.5 µs | 1.5 µs | 1.4 ms | 0.1 ms |
| 10,000 | 8.3 µs | 1.5 µs | 14.7 ms | 0.9 ms |
| 100,000 | 3.0 µs | 1.9 µs | 152 ms | 9.7 ms |

`submit` and `confidence` are **O(1) in store size** — both are index lookups
over one `(subject, relation)` bucket. `promote` and `report` are full scans and
grow linearly: at 10M beliefs, expect ~15 s per sleep cycle. That is tolerable
for an offline phase and would need a dirty-set index before it is not.

### Memory retrieval at scale

`VectorStore` has an LSH index with multi-probe, and the honest finding is that
**it should almost never be used.** Exact search is a single BLAS matmul:

| Entries | Exact search | Dense size (d=128) |
|---|---|---|
| 50,000 | 12 ms | 26 MB |
| 200,000 | 62 ms | 102 MB |
| 1,000,000 | 290 ms | 512 MB |
| 4,000,000 | 1,535 ms | 2 GB |

Multi-probe LSH reaches **94–100% recall@4** on both clustered and unstructured
data, but Python-side bucket assembly cannot beat a matmul below roughly 500k
entries — an earlier 12-bit single-probe configuration managed 3.4× speedup at
only **19% recall**, which is the worst of both. `index_after` defaults to
500,000 for that reason.

The wall that actually forces an external index at full scale is **memory, not
latency**: these buffers are dense, so 50M entries costs ~8 GB at d=128 and
~400 GB at d=2048. That store belongs on disk behind FAISS or DiskANN;
`VectorStore` keeps the same API so the swap is local.

Promotion needs **both** enough confidence *and* enough independent sources — a
trusted source repeating itself is not corroboration. Beaten beliefs are marked
`obsolete` with `superseded_by`, never deleted, so the history stays auditable.
`rc.explain(claim_id)` prints the full evidence trail.

### Beyond triples

Triples handle facts well and sequences badly, so procedures and causal links
carry structured payloads alongside the triple rather than being forced into it:

```python
rc.submit_procedure("replace_belt", ["loosen", "slip_off", "fit_new"], source)
rc.submit_causal("low_coolant", "overheating", source, strength=0.8)
rc.procedure("replace_belt")     # best-supported ordered steps
rc.causes_of("overheating")      # believed causes, strongest first
```

Keeping the triple form means the same evidence, contradiction and promotion
machinery applies unchanged — two conflicting procedures for one task compete
like any other rival. `causes` is deliberately many-valued: an effect usually
has several contributing causes and forcing them to compete would delete real
ones.

Relation arity is a **declared modelling decision**, not a guess:
`declare_functional("state_of")` / `declare_multivalued("has_landmark")`.
Getting it wrong fails in both directions — mark a many-valued relation
functional and real facts suppress each other; leave a single-valued one out and
the system will happily believe a door is locked *and* unlocked.

Beliefs are serialised into the checkpoint. What the robot believes survives a
restart, or every reboot is an amnesia event.

---

## Sensors and actuators — the hardware is not interchangeable

A robot does not have "touch" and "motors". It has an IMU at 200 Hz, wheel
encoders, a LiDAR returning a range array, pressure mats on the fingertips, and
maybe a gas array that takes thirty seconds to recover after an exposure — and
it moves with hydraulic hips, brushless arms and a pneumatic gripper.

Collapsing that into one 64-channel touch vector and one uniform joint limit
throws away every property that changes how the hardware must be used.

### Sensors

`brainai/sensors.py` carries the fields that actually matter, not just channel
counts:

| Sensor | Modality | Ch | Rate | Latency | Confidence |
|---|---|---|---|---|---|
| rgb_camera / depth | spatial | 3 / 1 | 30 Hz | 33 ms | 0.86 / 0.81 |
| lidar_2d | spatial | 360 | 10 Hz | 100 ms | **0.93** |
| ultrasonic | spatial | 1 | 20 Hz | 50 ms | 0.71 |
| imu | proprio | 9 | 200 Hz | 5 ms | 0.86 |
| joint_encoders | proprio | 32 | 500 Hz | 2 ms | 0.99 |
| force_torque / tactile_skin | contact | 6 / 64 | 500 / 100 Hz | 2 / 10 ms | 0.93 / 0.81 |
| **mos_gas_array** | chemical | 8 | 1 Hz | 2 s | **0.51** |
| **electrochemical_gas** | chemical | 4 | 1 Hz | 15 s | 0.80 |
| **qcm_olfactory** | chemical | 16 | 2 Hz | 5 s | 0.68 |

Four properties change behaviour rather than just describing it:

- **`rate_hz`** — fusion runs at the *slowest* present stream. Fusing at the
  fastest rate means repeating a two-second-old gas reading two hundred times
  and letting it outvote the IMU.
- **`noise` and `drift`** — a MOS element drifts badly with temperature; an
  encoder does not. Per-sensor confidence flows into the uncertainty chain, so
  the model knows which inputs to lean on.
- **`latency_ms`** — a reflex arc that ignores this reacts to a reading that
  describes the past.
- **`recovery_s`** — unique to chemical sensing. After a strong exposure a MOS
  element is saturated and reads high regardless of what is in front of it. A
  robot following a gas plume by driving up the gradient will chase its own
  saturation and stop. `mark_saturated()` drops that sensor's confidence to 0.05
  until it recovers.

Olfaction is included because nothing else in a sensor suite finds a leak behind
a wall, and because it is genuinely absent from most robot stacks.

### Actuators

`brainai/actuators.py`. One uniform `JointLimits` is wrong in **both**
directions on mixed hardware — the rate cap that protects a fingertip servo
makes a hydraulic hip useless, and the effort ceiling that lets the hip stand up
will crush what the gripper is holding.

| Type | Control | Max step | Max effort | Compliance | Bandwidth |
|---|---|---|---|---|---|
| dc_brushed | velocity | 0.25 | 0.60 | 0.50 | 100 Hz |
| dc_brushless | velocity | 0.30 | 0.85 | 0.35 | 500 Hz |
| stepper | position | 0.10 | 0.50 | 0.15 | 50 Hz (200 steps) |
| servo | position | 0.15 | 0.70 | 0.25 | 100 Hz |
| **hydraulic** | force | **0.05** | **1.00** | **0.05** | 20 Hz |
| **pneumatic** | force | **0.35** | **0.45** | **0.85** | 30 Hz |

```python
BrainConfig.for_scale("base", actuator_profile="humanoid")
# hydraulic hips, brushless arms, pneumatic hands, servo neck
# -> per-joint limits, and a hazardous_joint_ceiling constraint added
```

Three consequences fall out automatically:

**Per-joint envelopes.** `ActuatorBank.limits()` gives the safety controller a
different bound per joint instead of the same wrong one everywhere.

**Quantisation.** A stepper cannot sit between steps. Commands are snapped to
reachable positions *before* the safety check, because letting the driver round
silently makes commanded and achieved disagree — which surfaces later as
unexplained body-model error.

**`compliance` is the safety-critical field.** A compliant actuator gives way
against an unexpected obstacle. Hydraulic joints do not, and they have the force
budget to keep going through one. `hazardous()` flags stiff-and-strong joints
and `hazardous_joint_ceiling` holds them to a tighter effort cap than the rest
of the body.

Adding a sensor is a `SensorSpec` and a list entry — not a new encoder class.

---

## Subject knowledge — one graph, tagged, not one database per subject

A humanoid will be asked about maths, physics, chemistry, biology, programming,
history, geography and the arts. The tempting design is a database per subject.
That is the wrong move, and the reason is visible in one example: *Einstein* is
physics **and** history **and** politics; a titration is chemistry **and**
procedure **and** safety. Filing a concept under one heading cuts exactly the
cross-domain edges that make a knowledge graph worth having.

So domain is a **tag on one graph**, and `knowledge(domain)` is a filter, not a
store:

```python
rc.learn_fact("einstein", "worked_on", "relativity", "textbook", domain="physics")
rc.learn_fact("einstein", "emigrated_in", "1933", "textbook", domain="history")
rc.cross_domain_links()   # {'physics': {'history'}, 'history': {'physics'}}
```

**Rigorous domains carry a higher bar.** Maths, physics, chemistry, biology,
medicine and engineering need higher confidence *and* an extra independent
source. A mathematical result is not more true for being widely repeated, and a
wrong chemistry fact can hurt someone — social consensus should not promote
either:

```
physics,  2 sources -> 0 promoted   (bar is 0.90 confidence + 3 sources)
physics,  3 sources -> 1 promoted
everyday, 2 sources -> 1 promoted   (bar is 0.75 + 2 sources)
```

What this does *not* do is answer the questions. A 5M-parameter byte-level model
cannot do physics. The architecture's position is that such knowledge belongs in
the stores rather than compiled into weights — this is the machinery for holding
it, checking it, and refusing to believe it too easily. Filling it is a separate
problem.

---

## Safety — the last thing before an actuator

A policy that rewrites its own weights at runtime must not also decide what it
is allowed to do. `SafetyController` is deliberately **not** learned, holds no
parameters, and is unreachable from `PlasticityController` — online learning
cannot widen its own operating envelope. There is a test asserting exactly that.

Three checks, cheapest first:

1. **Envelope** — position, per-step velocity and effort clamped to declared
   limits.
2. **Constraints** — named predicates returning allow / clamp / veto.
   Ships with `human_in_envelope`, `effort_ceiling`, `requires_confidence`.
3. **Interlock** — a hard e-stop that holds position and zeroes effort.

```
requested +5.000 -> commanded +0.150   32 joints clamped, 32 rate-limited
human at 0.20m   -> VETOED by human_in_envelope, effort 0.0
e-stop           -> holds position, effort 0.0
ramp             -> 0.15 0.30 0.45 0.60 0.75 0.90 1.00
```

`act()` returns both `joint_targets` (what will be sent) and
`requested_joint_targets` (what the network asked for). When they differ the
policy tried to leave the envelope — worth surfacing, not silently correcting. A
controller that clamps quietly is nearly as dangerous as none, because the
operator never learns the policy is asking for something impossible.

Two bugs found by writing the tests, both live hazards, both now regression-tested:

- After a veto the controller kept no reference pose, so the *next* command
  skipped rate limiting entirely and the joints could slam.
- The rate-limit guard compared `_last_command.shape == command.shape`, but the
  reference is `(n,)` and commands arrive `(1, n)` — so rate limiting was
  silently disabled on every single call.

---

## Multi-domain training: breadth at nano scale

The single-book runs prove the mechanism; they say nothing about breadth. So
`nano` (5.36M parameters) was trained across **64 domains, 70.7 MB** — code in
seven languages, physics/maths/chemistry/biology, history/economics/psychology,
five non-Latin languages (Urdu, Hindi, Marathi, Konkani, Arabic), Morse, binary,
bytecode, assembly, geography, and an author/model knowledge corpus. Domains are
sampled at temperature 0.5 and reported separately, because one averaged number
cannot tell a broadly-mediocre model from one that learned code and forgot
chemistry.

Held-out loss fell from **5.49 → 2.25 bits/byte** (mean). Per domain:

| Domain group | bits/byte | Note |
|---|---|---|
| Mathematics (KG–12) | **0.58** | largest, cleanest corpus |
| IT / CS reference | 0.93 | |
| Non-Latin languages (large) | **1.3 – 1.8** | Marathi 1.31, Hindi 1.54, Konkani 1.55, Urdu 1.73, Arabic 1.81 — learned from scratch at byte level |
| Python / C++ / C / C# | 2.24 – 2.38 | |
| geography, chemistry, physics | 1.9 – 2.0 | |
| Morse, assembly (tiny corpora) | 3.0 – 3.8 | little data, little learned |

The five languages began as 60–90 KB fragments and were rebuilt to 1.2–2.7 M
tokens each from Tanzil and Wikipedia; the byte tokenizer handles Arabic and
Devanagari with no vocabulary change, which is why a multilingual corpus cost
nothing but the download.

**Author knowledge** is a deliberate test of whether a fact injected as a small
corpus can be recalled. A 20×-boosted, EWC-protected fine-tune
(`finetune_author.py`, a continuation — not a retrain) moved the author corpus
from 3.32 → **2.59 bits/byte** while the 64-domain mean held steady (1.539 →
1.516, EWC doing its job).

### The honest ceiling

Loss fell everywhere, but **generation is still character-level fragments**:

```
'Q: Who built BrainAI?'  ->  ' Mass hand poncer. A: Forme hase struct...'
'To solve the equation'  ->  '...a sedor Zahisal stee of...'
```

You can see associations *trying* to form — "Zahisal" is a mangled "Zahid" —
but a 5.36M-parameter byte-level model cannot emit fluent text no matter how
much data or sampling weight it is given. Lower per-domain loss is real and
measurable; readable output is not reachable at this size.

**This is expected to improve with scale, and the improvement is the whole
point of the size ladder.** The bottleneck here is parameters, not data or
method: at ~1–2 bits/byte the model has learned each domain's *statistics* but
lacks the capacity to compose them into sentences. The `compact` (104M) preset
exists precisely for this step — roughly 20× the parameters, the smallest size
where byte-level generation typically becomes legible — and `base` (~3.8B)
beyond it. Those are GPU jobs; the method, corpus and evaluation here carry over
to them unchanged, only the parameter count and the compute change.

---

## Continual learning: the measurement

The claim is "learns new things without forgetting old ones." That is testable,
so it is tested. `continual.py` pretrains on English, then trains on French
three ways from identical weights, and measures what happened to English.

English and French at byte level have clearly different statistics, so
forgetting has an unambiguous number attached.

```
arm       EN before  EN after   forgot  FR before  FR after  learned    both    sec
-----------------------------------------------------------------------------------
naive        2.4521    2.9197  +0.4675     3.2158    2.4567  +0.7591  2.6882    211
ewc          2.4521    2.6201  +0.1679     3.2158    2.7615  +0.4543  2.6908    224
full         2.4521    2.5737  +0.1215     3.2158    2.7778  +0.4380  2.6758    348
```

| Arm | Forgetting | vs. naive |
|---|---|---|
| naive | +0.4675 nats (+0.675 bits/byte) | — |
| ewc | +0.1679 nats (+0.242 bits/byte) | **64% less** |
| full | +0.1215 nats (+0.175 bits/byte) | **74% less** |

**Read the `both` column before celebrating that.** It is the mean final loss
over the two tasks: 2.6882 / 2.6908 / 2.6758. Protecting old weights also slows
learning the new task — French improves by 0.76 nats under `naive` but only 0.44
under `full`. The methods buy retention by spending plasticity, close to 1:1.

`full` does come out ahead on `both` here, by 0.012 nats. That is a small enough
margin on a single seed that it should be read as "no worse overall while
forgetting 74% less", not as a free lunch. A forgetting number reported without
its stability-plasticity counterpart is close to meaningless.

Every arm sees the same number of French sequences over the same number of
optimizer steps. The `full` arm additionally replays English from its buffer,
which costs ~1.8× wall-clock per step plus the memory to hold the buffer. That
cost is the price of the method and is reported, not hidden.

One calibration bug is worth recording, because it produced a plausible-looking
result that was entirely fake. On the first run `ewc` forgot **+0.5384** against
naive's **+0.5385** — a 0% improvement — because the raw Fisher at convergence
is ~1e-8, so `lambda · Σ F·Δw²` came to 1e-4 against a task loss of 2.5 and the
penalty did nothing. It looks exactly like a working implementation until you
run the no-EWC control. The fix was to stop guessing lambda and calibrate it:
`consolidate()` now sets a scale such that a 1% relative drift in every weight
costs `lambda / 2` nats, so lambda has units. Measured after the fix — 0.1%
drift → 0.06, 1% → 6.0, 5% → 150.

---

## AMSA — Adaptive Memory Spike Activation

```
f(x) = z · σ(α·z) · (1 − e^(−β|z|)) · (1 + γ·m),    z = x − θ

θ = θ₀ + λ·p        p = per-channel plasticity
α = softplus(α₀ + δ·c)   c = per-channel confidence
β = softplus(β₀)
```

| Term | Effect |
|---|---|
| `z = x − θ` | Adaptive threshold — the neuron responds once its own learned threshold is exceeded. |
| `σ(α·z)` | Smooth gate; keeps gradients well-conditioned where a hard cutoff would not. |
| `1 − e^(−β|z|)` | Suppresses small activations, passes strong ones. This is what makes it sparse. |
| `1 + γ·m` | Memory modulation — channels carrying important information amplify. |

`θ₀, α₀, β₀, γ, λ, δ` are learned per channel. `m`, `p`, `c` are **state**,
written by the plasticity controller, never by gradient descent — which is what
makes the activation's *shape* follow the network's own estimate of which
channels are load-bearing. Confident channels sharpen; plastic channels stay
easy to move.

`1 − e^(−β|z|)` is computed as `-expm1(-β|z|)` to stay accurate for small `β|z|`.

### Does AMSA actually help?

On loss, no; on efficiency, yes — it is a tradeoff, and the two halves are
below. First the loss half. Same model, same data, same seeds, matched parameter
budget — only the feed-forward nonlinearity differs
(`benchmarks/activation_bench.py`, 300 steps, 2 seeds):

| FFN | Params | Val loss | bits/byte | Train sec |
|---|---|---|---|---|
| gelu | 1,124,480 | 2.3562 | 3.399 | 13.2 |
| **swiglu** | 1,124,480 | **2.3323** | **3.365** | **15.1** |
| amsa | 1,136,768 | 2.3466 | 3.385 | 26.2 |
| amsa+state | 1,136,768 | 2.3465 | 3.385 | 22.5 |

**SwiGLU wins on quality and is ~40% faster than AMSA.** AMSA lands between
GELU and SwiGLU and costs the most.

The `amsa+state` row exists because AMSA's adaptive terms multiply state buffers
that are zero unless something writes them — benchmarking only the static form
would test a different function than the architecture uses. Driving that state
changes the loss by **0.0001**. The adaptive machinery contributes nothing
measurable here.

### The test that actually matters

The objection to the benchmark above is fair: static language modelling is not
what AMSA was designed for. Its state buffers are **inert in a single-task run**
— plasticity and confidence only carry information once there is a previous task
whose importance has been estimated. So the honest test is continual learning.

`benchmarks/amsa_continual.py` — English → consolidate → French, identical
wiring, 3 seeds, with `amsa+state` having plasticity and confidence driven from
the task-A Fisher:

| FFN | EN forgotten | FR learned | both | vs. SwiGLU |
|---|---|---|---|---|
| **swiglu** | **+0.8062** | +1.1540 | **2.6875** | — |
| amsa | +0.8204 | +1.1476 | 2.7071 | +0.0141 worse |
| amsa+state | +0.8224 | +1.1464 | 2.7087 | +0.0161 worse |

Seed-to-seed spread on forgetting is ±0.005–0.008, so both gaps are **outside
seed noise**. And `amsa+state` is *worse than plain `amsa`* — driving the
adaptive state is marginally harmful, not merely neutral.

**So AMSA loses on all three axes: perplexity, speed, and the continual-learning
problem it was built for.**

### Is it just too small?

The obvious defence is that nano is too small and the runs too short for the
adaptive state to matter. That is a reasonable hypothesis and it is testable, so
`--sweep` tests it:

| width | layers | steps | swiglu | amsa | gap |
|---|---|---|---|---|---|
| 96 | 3 | 100 | +0.6475 | **+0.6206** | **−4.2%** (AMSA better) |
| 128 | 4 | 200 | +0.7937 | +0.7898 | −0.5% |
| 192 | 4 | 200 | +0.8313 | +0.8257 | −0.7% |
| 256 | 6 | 400 | **+0.9890** | +1.0483 | **+6.0%** (AMSA worse) |

**The gap widens with scale rather than shrinking**, which is the opposite of
what the hypothesis predicts. AMSA is genuinely *better* than SwiGLU at the
smallest setting and worse at the largest, so there is a crossover — just
pointing the wrong way for "it needs more room to show its value".

Read this carefully in both directions. It does not prove AMSA fails at 1B
parameters: the range here spans width 96→256, which is nothing, the two middle
rows are inside seed noise, and width and training length move together rather
than being varied independently. What it does do is remove the *support* for the
low-data explanation in the range that has been measured, and put the burden on
a larger experiment rather than on an assumption.

On loss, SwiGLU is the better choice, and that is a config flag, not an edit:

```python
BrainConfig.for_scale("nano", ffn="swiglu")    # lower loss, faster
BrainConfig.for_scale("nano", ffn="amsa")      # the default: sparser, brain-inspired
```

### But loss is the wrong axis for AMSA

Everything above measures loss/perplexity, which is not what AMSA was designed
for. Its purpose is *efficiency* — a brain does the same work with fewer neurons
firing, and AMSA's `(1 − e^(−β|z|))` factor is literally a small-activation
suppressor. The fair question is not "which reaches lower loss" but "at
comparable loss, which lights up less of the network." That is measured in
`benchmarks/amsa_sparsity.py` — the real spiking model on the multi-domain
corpus, identical init and batch stream, run over 3 seeds. A representative
seed:

| FFN | loss (bits/byte) | spike rate | FFN active |
|---|---|---|---|
| swiglu | 3.851 | 0.0291 | 73.7% |
| **amsa** | 3.861 | **0.0282** | **59.9%** |

Across all three seeds, at **comparable loss** (Δ ≤ 0.021 nats, within noise),
AMSA is **sparser in 3 of 3**: ~1–23% fewer spikes (Δ spike −0.0009, −0.0009,
−0.0066), and a **~14-point lower feed-forward activation density** (≈60% vs
≈74%). The density gap is the robust part — it is structural, a direct
consequence of the activation shape — while the exact spike-rate margin varies
seed to seed.

In a spiking network *spikes are the energy*, so this is a genuine win on the
axis AMSA was built for. It reframes AMSA from "loses on everything" to a real
**quality-vs-efficiency tradeoff**: SwiGLU is marginally lower-loss and faster on
today's dense hardware; AMSA reaches the same loss while firing less, which is
worth nothing on a CPU/GPU and is exactly what neuromorphic hardware charges
for. That is a defensible reason to keep it the default in a spiking,
brain-inspired architecture.

The honest boundary stays: it is a small model, a modest budget, and the
spike-rate margin (though 3/3 consistent) is small. AMSA does **not** win on
loss, and the earlier scale sweep showed the loss gap widening — but "same
accuracy, fewer neurons firing" is the brain-efficiency claim, and that one now
has evidence behind it rather than only intent.

---

## Spiking core

Spikes are real Heaviside steps — the forward pass is genuinely binary, so the
reported spike rate is a true sparsity number and not a soft approximation.
Training uses an arctan surrogate on the backward pass only.

Neurons are **adaptive** LIF: each spike raises that neuron's own firing
threshold, which decays over `tau_adapt`. A neuron that just fired is harder to
re-fire, giving the layer temporal memory across timesteps with no recurrent
weights.

Two implementation facts worth knowing before changing anything:

- **`timesteps >= 3` in every preset.** Adaptation is `a[t] = ρ·a[t−1] + s[t−1]`
  with `a[0] = s[0] = 0`, so at `T = 2` the decay only ever multiplies zeros and
  threshold adaptation is inert. `T = 3` is the smallest useful SNN. (Tested:
  `test_alif_needs_three_timesteps_for_adaptation`.)
- **The residual stream stays analogue.** Only the sublayer interiors are
  binary. Making the residual stream itself spiking destroys the gradient path
  and the model does not train.

Attention offers `softmax` (default) or `sparsemax`, which produces exactly-zero
attention weights and keeps the attention pattern sparse too.

---

## Scale presets

`BrainConfig.for_scale(...)` — the architecture is described once and
instantiated at six sizes.

| Scale | Layers | d_model | T | Total params | Active/token | Memory | Runs on |
|---|---|---|---|---|---|---|---|
| `nano` | 17 | 128 | 3 | 5.4M (actual) | 4.1M | 2k | a laptop CPU |
| `small` | 35 | 256 | 3 | ~64M | ~35M | 16k | a laptop CPU, slowly |
| `compact` | 36 | 352 | 3 | **104M (actual)** | 82M | 66k | an RTX 3050 (4 GB) |
| `base` | 87 | 768 | 4 | ~3.8B | ~1.0B | 262k | one GPU |
| `full` | 392 | 2048 | 8 | **~331B** | **~61B** | **50M** | a cluster |
| `full_wide` | 392 | 4096 | 8 | ~82T | ~13T | 10M | nothing |

Only `nano` has actually been trained. `compact` is the smallest size at which
generation is expected to read as language rather than domain-flavoured
character soup; it is built and verified (forward + backward, 103.8M params) but
its training is a GPU or multi-day-CPU job, not the overnight run `nano` gets.

Because the conscious section is expert-routed, only the top-2 experts run per
token — hence the gap between the two columns. That gap is the point of routing.

**`full` was rescaled, and the reason is the architecture's own argument.** As
originally specified — 96 conscious layers at width 65,536 — the estimator puts
it at ~82T parameters, roughly 50× the largest models ever trained, for one
robot. Transformer cost grows with the *square* of width, so that single number
dominated everything else: the conscious section alone was most of the total.

Keeping the layer counts and cutting the widths preserves the depth and the
entire cognitive structure at **1/248th the parameters**, and raises memory
capacity 5×. If lifelong knowledge belongs in episodic and semantic memory
rather than compiled into weights — which is the claim this whole design rests
on — then spending 82T parameters on weights contradicts it. A store entry costs
`d_model` floats against a weight matrix's `d_model²`.

`full_wide` keeps the original numbers so the comparison stays reproducible, not
because it should be built.

```
python -c "from brainai import BrainConfig; print(BrainConfig.for_scale('full').describe())"
```

---

## Learning while running

```python
model.eval()
model.learn_online(tokens, reward=1.0)   # weights change, in eval mode
model.sleep(cycles=8)                    # consolidate offline
```

`learn_online` is deliberately conservative, because unbounded online updates
destabilise a model within a few hundred steps:

1. Gradients are scaled per-parameter by **plasticity** = `1 − normalised
   Fisher`. Weights that mattered to past tasks barely move.
2. The applied update is clipped to a **trust region** — a fraction of the
   parameter norm, measured after plasticity scaling.
3. The loss is **re-evaluated**; if the step made things worse it is rolled
   back.

The Fisher is normalised by a **high quantile, not the max**. The Fisher is
heavy-tailed, so `f / f.max()` is near zero for almost every weight and nothing
ends up protected — a bug that looks like a working implementation until you
print the protected fraction.

### Does it stay stable?

400 inference-time updates against a live stream, with a control arm that has
plasticity gating, the trust region and rollback all disabled
(`benchmarks/stability_bench.py`):

| | old-task loss | drift | rollbacks | spike rate |
|---|---|---|---|---|
| guarded | 2.4882 → 2.5052 (**+0.0170**) | 0.2119 | 3% | 0.031 → 0.028 |
| control | 2.4880 → 2.5503 (**+0.0624**) | 0.3151 | 0% | stable |

The guards cut degradation **3.7×** and drift by a third. Note what this does
*not* show: the control degrades faster, it does not diverge. At this learning
rate over 400 updates the trust region is buying a slower slide, not preventing
a collapse. Both arms drift roughly linearly, so the gap widens with time — but
400 updates is what was measured and the reviewer's "millions of updates" remains
untested. The spike rate holding at 0.028 matters independently: a spiking
network that quietly goes dense under online updates has thrown away the reason
to build it.

`sleep()` then replays stored token sequences against current weights with the
EWC penalty active, merges near-duplicate memories, lets unimportant ones decay,
and re-anchors. Replay stores **raw tokens, not embeddings** — replaying an
embedding only rehearses the layers above where it was captured.

---

## Distilling from Qwen

`distill.py`. One constraint determines the design: **you cannot KL-match two
models with different tokenizers.** BrainAI's byte tokenizer has 260 symbols;
Qwen's BPE has ~150k. Their output distributions live in different spaces.

```bash
python distill.py --mode sequence --dry-run          # validate the pipeline offline
python distill.py --teacher Qwen/Qwen2.5-0.5B --mode sequence
python distill.py --teacher Qwen/Qwen2.5-0.5B --mode logits --share-tokenizer
```

- **`sequence`** (default) — the teacher generates text, the student trains on
  it. Tokenizer-agnostic, works with any teacher. What it cannot transfer is the
  teacher's uncertainty: the student sees the sampled token, not the
  distribution it came from.
- **`logits`** — rebuilds the student on the teacher's vocabulary, then matches
  full distributions with temperature-scaled KL. Transfers much more, and costs
  you the byte tokenizer, which for a robot also handling tactile and telemetry
  byte streams is a real loss.

Distillation gives the student language knowledge. Perception, planning and
motor control need their own training; language distillation alone will not
produce a robot.

---

## Naming and attribution

```python
model.rename("Javri")     # or anything else
model.identity.author     # always "Mohammed Zahid" - a read-only property
```

Authorship is stamped into every module, every config, and every checkpoint.
`BrainAI.load` verifies it and **refuses to load weights whose attribution has
been stripped or altered**, while renaming passes cleanly.

---

## Repository layout

```
brainai/
  identity.py       attribution, renaming, tamper checking
  config.py         scale presets, section specs
  activation.py     AMSA, GatedAMSA
  neurons.py        ALIF, surrogate gradients, spiking LSTM
  attention.py      spike attention, sparsemax, RMSNorm
  blocks.py         spiking blocks, MoE routing, expert growth
  encoders.py       text / image / video / audio / touch
  memory.py         the five memories, hippocampal controller, LSH index
  pathways.py       reflex, subconscious, conscious, arbiter, motor
  sensors.py        typed sensors incl. olfaction; rate/latency/drift fusion
  actuators.py      DC / stepper / servo / hydraulic / pneumatic envelopes
  world.py          dynamics, imagination, curiosity, body model, ledger
  reality.py        evidence, source trust, collusion, belief promotion
  executive.py      goals, interrupts, task switching, sleep scheduling
  planner.py        HTN decomposition, imagined scoring, replanning
  uncertainty.py    per-stage confidence and its composition
  critique.py       self-critique loop, internal expert debate
  safety.py         joint envelope, constraints, e-stop  (not learned)
  plasticity.py     Fisher, EWC, online updates, growth
  consolidation.py  replay buffer, sleep
  tokenizer.py      byte-level
  data.py           single-domain corpora, evaluation
  corpus.py         64-domain corpus: disk auto-discovery, temperature sampling
  prompts.py        domain prompt bank for teacher distillation
  model.py          the assembled model
train.py                pretraining, one domain
train_multidomain.py    64-domain training, per-domain eval
finetune_author.py      author-recall fine-tune (continuation, EWC-protected)
evaluate_trained.py     every weight-dependent benchmark on a real checkpoint
continual.py            the anti-forgetting benchmark
distill.py              two-phase teacher distillation (generate, then train)
demo.py                 every capability, end to end
tools/
  build_author_corpus.py   author/model knowledge corpus
  build_code_corpus.py     real Java/C#/C++/Go/Rust + local Apache C
  build_technical_corpus.py  Morse, binary, bytecode, NASM assembly
  build_language_corpus.py   Urdu/Hindi/Marathi/Konkani/Arabic (Tanzil + Wikipedia)
benchmarks/
  activation_bench.py  AMSA vs GELU vs SwiGLU, static
  amsa_continual.py    AMSA on loss / continual learning (loses)
  amsa_sparsity.py     AMSA on spike-rate & activation sparsity (wins)
  amsa_multidomain.py  AMSA vs SwiGLU in the real spiking model
  belief_bench.py      Reality Cortex vs naive counting; attack cost; scaling
  cognition_bench.py   world model vs linear/RSSM, curiosity, grounding, planner
  worldmodel_jobs.py   rollout, planning, calibration - the jobs, not one-step
  gating_bench.py      is the conscious pathway really skipped?
  stability_bench.py   400 online updates, with an unguarded control
tests/              114 tests
```

Checkpoints are ~3× the model size: the plasticity controller stores a Fisher
estimate, an anchor and a plasticity value per weight. That is the cost of the
continual-learning machinery, and it is why a 5M-parameter `nano` checkpoint is
~89 MB.

---

## What is deliberately missing

The remaining gaps are **evidence, not architecture**, and they are all
compute-bound rather than design-bound:

- **Scale of the continual-learning test.** Two tasks, 120 steps. Convincing
  evidence needs many task sequences and much longer horizons.
- **Online learning at 100k–10M updates.** Measured here at 400. The trust
  region is shown to slow drift, not to prevent collapse, and the difference
  between those two only shows up over a much longer run.
- **Multimodal and robotics benchmarks.** Vision, audio, navigation and
  manipulation are all wired and none are benchmarked. There is no simulator
  here and no robot.
- **Memory benchmarks beyond retrieval.** Consolidation efficiency and
  forgetting curves over a long lifetime are unmeasured.

All four need GPU-months and, for the robotics ones, hardware. They are not
things a design change fixes.

Two subsystems here are also **built but not validated in the loop they were
built for**: `BodyModel` has no robot to calibrate against, and `SelfCritique`
runs on a 5M-parameter model whose output is too weak for critique to have much
to work with. Both are honest scaffolding, not demonstrated capability.

---

## Limitations

- `nano` is 5M parameters trained on one book for 400 steps. It is a working
  demonstration of the architecture, not a capable language model. Generated
  text is character-level English-ish, not fluent.
- **AMSA loses to SwiGLU on all three benchmarks** — static perplexity, speed,
  and continual learning, the last being the problem it was designed for.
  Driving its adaptive state makes it marginally *worse*. It is kept as the
  default by the author's choice; `ffn="swiglu"` switches.
- **The world model loses to a linear baseline at one-step prediction.** It
  beats an RSSM-family model on both test systems, but a 16k-parameter linear
  layer predicts better than the 119k-parameter dynamics head. The deficit does
  not propagate — the two are tied by three rollout steps and at planning — and
  only one of them can report its own uncertainty. Neither half of that is
  worth much on its own.
- Sensor specs are representative of commodity parts, not measured from
  hardware. No sensor here has ever read a real signal, and the olfactory
  recovery times in particular are plausible rather than calibrated.
- The Reality Cortex reasons over `(subject, relation, object)` triples that
  something must supply. Extracting reliable triples from raw perception is its
  own hard problem and is **not** solved here — `observe()` takes them as given.
  Source reliability priors are also supplied by the operator; only their
  *updates* are learned, from prediction outcomes.
- **Sybil attacks are expensive, not impossible.** Behavioural clustering
  catches identities that vote identically. A patient adversary that varies its
  puppets' patterns, or one that compromises genuinely independent sources,
  still gets through.
- **Grounding needs something observable.** The prediction ledger removes the
  oracle only for beliefs that imply a checkable prediction. A claim about
  something the robot never observes stays anchored to its operator prior
  forever, and that is the correct behaviour rather than a bug — but it means
  trust is grounded for a subset of knowledge, not all of it.
- **`BodyModel` has never met a robot** and `SelfCritique` runs on a model whose
  output is too weak for critique to have much to work with. Both are honest
  scaffolding, not demonstrated capability.
- Online-learning stability is measured over **400** updates, not millions. The
  unguarded control degrades faster but does not diverge at that horizon, so the
  trust region is demonstrated to slow drift, not to prevent collapse.
- The safety controller enforces an envelope and explicit constraints. That is
  the floor, not a safety case. Real deployment needs hazard analysis,
  redundant sensing, a hardware interlock that does not depend on this process
  being alive, and validation against the actual robot.
- The energy argument for spiking is a hardware argument. On a CPU or GPU,
  binary activations do not save wall-clock time — dense matmuls run anyway. The
  sparsity numbers are what neuromorphic hardware would exploit; here they are a
  proxy.
- `full` has never been instantiated.
- Nothing has been run on a robot. `MotorCortex` emits bounded joint targets;
  closing the loop needs a simulator, a safety layer, and real robotics data.
- Continual learning is measured on a two-task language pair. Two tasks is the
  smallest honest test, not a thorough one.

---

Built by **Mohammed Zahid**.
