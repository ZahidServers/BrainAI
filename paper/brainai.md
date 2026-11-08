# BrainAI: A Spiking Cognitive Architecture for Embodied Continual Learning

### Every subsystem measured against a control, including the two that lost

**Mohammed Zahid Imityaz Wadiwale**
`zahid.wadiwale1234@gmail.com`

---

## Abstract

We present **BrainAI**, a spiking multimodal cognitive architecture for embodied
continual learning. A single model perceives text, images, video, audio and a
typed sensor suite; routes each input through a reflex arc, an always-on
subconscious pathway and a gated conscious pathway; maintains five distinct
memory systems split along the differentiability line; predicts future latent
states and learns from prediction error without labels; weighs new claims
against evidence before believing them; pursues interruptible goals; plans by
imagination; updates its own weights during inference under bounded plasticity;
and consolidates offline during a sleep cycle. Motor commands pass through a
non-learned safety controller that the learning system provably cannot modify.

The methodological commitment of this work is that *every* subsystem is measured
against the obvious simpler alternative, and every result is reported. Measured
outcomes include a **2.16× inference speedup** from genuine computational
gating, a **74% reduction in catastrophic forgetting**, **3.7× less degradation**
over 400 inference-time weight updates, a belief system that defeats a naive
vote-counter on four of five adversarial scenarios, and collusion detection that
holds a protected axiom at confidence **1.0000** against a 600-identity Sybil
attack that otherwise drives it to **0.0001**.

The two proposals with mixed or negative results are reported in full. The
architecture's own activation function, AMSA, is outperformed by SwiGLU on
perplexity, wall-clock, and the continual-learning problem it was designed for,
with the loss gap *widening* with scale — but on the axis it was actually built
for, efficiency, it *wins*: at comparable loss it fires fewer spikes and
activates ~14 points less of the feed-forward layer, across three seeds, making
it a quality–efficiency tradeoff rather than a flat loss. The world model is
outperformed at one-step prediction by a linear map, though the deficit does not
propagate to multi-step rollout or planning, where the two tie, and only the
learned model reports calibrated uncertainty (r = 0.565).

We additionally catalogue **fourteen silent failure modes** discovered only
because controls were run — including an elastic-weight-consolidation penalty
that contributed 10⁻⁴ against a task loss of 2.5 and therefore did nothing while
appearing to work, and a rate limiter disabled on every call by a tensor-shape
comparison. We argue this catalogue is among the more transferable contributions
of the work.

---

## 1. Introduction

Contemporary large models separate training from deployment: weights are fixed
at inference, knowledge is frozen at a cutoff, and adaptation happens through
context or retrieval rather than through the network itself. For a stationary
task distribution this is the right engineering trade. For an embodied agent
expected to operate for years in one environment it is not: such an agent must
acquire knowledge after deployment, retain what it already knows, decide which
of the things it is told are actually true, and do all of this without a
training loop supervising it.

BrainAI is an attempt to build that system end to end. It is not a language
model with memory attached; it is a cognitive architecture in which the language
head is one of several outputs. The design draws on complementary learning
systems theory [18], dual-process accounts of cognition [12], and the
spiking-network literature [16, 20], and assembles components — spiking
transformers, mixture-of-experts routing, elastic weight consolidation [14],
experience replay [22], latent dynamics models [8] — that exist individually
but, to our knowledge, have not previously been integrated with the specific
interfaces described here.

### 1.1 The problem with architecture papers

An architecture paper is easy to write badly. A diagram with a dozen boxes, a
plausible mechanism for each, and a benchmark on the one task where the whole
thing happens to work is a well-established genre, and it is nearly
unfalsifiable. The failure mode is not dishonesty; it is that **a component
which does nothing at all looks exactly like a component that works, unless
something is run without it.**

We therefore adopted a single rule: **no component is described without a
control.** For each subsystem we identified the obvious simpler alternative —
the thing a reasonable engineer would do instead — implemented it, and measured
both. This rule is the reason this paper contains negative results, and it is
also the reason we trust the positive ones.

### 1.2 Contributions

1. **An integrated architecture** (§3) whose novelty is in the interfaces
   between components rather than in any single one: a memory hierarchy split by
   differentiability, a prediction ledger that grounds epistemic trust in
   physical observation, and a safety layer structurally unreachable by the
   learning system.
2. **A controlled evaluation** comprising thirteen comparisons (§5), each
   against a simpler alternative.
3. **Two negative results** reported in full (§6), including a scale sweep
   testing — and rejecting — the standard defence that the losing component
   merely needs a larger model.
4. **A catalogue of fourteen silent failure modes** (§7) found only by running
   controls. Several are general to continual-learning and robotics
   implementations rather than specific to this one.

All results are produced by scripts in the accompanying repository. Every number
in this paper, including those in §6, is reproducible on a CPU.

---

## 2. Related Work

**Continual learning.** Catastrophic interference has been understood since
[19]. Regularisation approaches penalise movement in parameters important to
earlier tasks [14, 26]; rehearsal approaches interleave stored examples [22]. We
use both, and find (§5.2) that the regularisation term is worthless unless its
scale is calibrated — a detail rarely reported.

**Spiking transformers.** Surrogate-gradient training [20] made deep spiking
networks practical; adaptive-threshold neurons [1] supply temporal memory
without recurrence. Spiking transformer variants have been demonstrated [27].
Our contribution here is not the block design but two implementation constraints
we found to be load-bearing (§7).

**World models.** PlaNet [8] and Dreamer [9] established latent dynamics models
for planning. Prediction error as intrinsic motivation dates to [21]; the
noisy-TV pathology and its diagnosis are from [2]. We compare against both an
RSSM-family recurrent baseline and — more informatively — a linear one.

**Epistemics.** Sybil attacks on reputation systems are classical [4]. To our
knowledge the specific combination used here — log-odds pooling with per-source
diminishing returns, credibility shrinkage for unvetted sources, and behavioural
collusion clustering — has not been applied to a robot's belief store.

---

## 3. Architecture

```
    text   image   video   audio   touch/IMU/LiDAR/gas
      └───────┴───────┴───────┴──────┘
                     │  per-modality spiking encoders
                     ▼
            shared d_model spike bus
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              │
   REFLEX      SUBCONSCIOUS         │
  (shallow,     (always on,         │
   always)       sparse)            │
      │              │              │
      │              ├── salience ──┴──► gate
      │              ▼
      │        CONSCIOUS  ◄──── working memory
      │      (wide, MoE,   ◄──── episodic + semantic recall
      │       gated)
      │              │
      └──────►  ARBITER  ◄───────┘
                     │
             ┌───────┴───────┐
             ▼               ▼
        language head    motor cortex
                             │
                    SAFETY CONTROLLER    (not learned)
                             ▼
                          joints

  control:    EXECUTIVE → PLANNER → WORLD MODEL (imagine) → action
  knowledge:  episodic → REALITY CORTEX → semantic → sleep → weights
                              ▲
                         WORLD MODEL  (prediction error grounds trust)
```

### 3.1 Spiking substrate

Spikes are true Heaviside steps; the forward pass is binary, so reported spike
rates are genuine sparsity rather than a soft approximation. Training uses an
arctan surrogate on the backward pass only. Neurons are adaptive
leaky-integrate-and-fire: each spike raises that neuron's own threshold, decaying
over `tau_adapt`, which supplies temporal memory across simulation timesteps
without recurrent weights.

Two constraints proved necessary. First, adaptation is `a[t] = ρ·a[t−1] + s[t−1]`
with `a[0] = s[0] = 0`, so at T = 2 the decay ρ only ever multiplies zero and
threshold adaptation is inert; **T ≥ 3 is the minimum useful configuration.**
Second, the residual stream must remain analogue — making it spiking destroys
the gradient path and the model does not train.

### 3.2 Latency-tiered pathways

Three pathways of increasing cost run in parallel.

| Pathway | Runs | Cost | Job |
|---|---|---|---|
| Reflex | always | 1 shallow stack | Have something correct-enough available now. Never consults memory. |
| Subconscious | always | sparse | Filter, predict, decide whether thinking is worth it. |
| Conscious | when salience clears threshold | wide, expert-routed [23, 6] | Reasoning, planning, memory integration. |

Blocks are pre-norm transformer blocks [25] with binary spike inputs to the
attention projections. Attention offers softmax or sparsemax [17], the latter
producing exactly-zero weights so the attention pattern is sparse as well.

The gate is **soft during training**, so the salience head receives gradient and
learns when deliberation pays, and **hard at inference**, where batch rows below
threshold are selected out and the wide stack genuinely does not execute. This
distinction matters: a gate that computes everything and multiplies by zero is
observationally identical from outside, which is why it is measured by
wall-clock (§5.1).

### 3.3 Memory hierarchy

| Memory | Backing | Differentiable | Holds |
|---|---|---|---|
| Sensory | deque | no | Last few frames, temporal integration |
| Short-term | spiking LSTM | **yes** | Recent event stream, pooled |
| Working | attention over slots | **yes** | The reasoning scratchpad |
| Episodic | vector store | no | Experiences, importance-ranked |
| Semantic | vector store + graph | no | Concepts and their edges |

The split is the anti-forgetting strategy: knowledge acquired after training
enters the non-differentiable *stores*, where writing it cannot damage a weight
an older skill depends on. Only slow, importance-gated consolidation moves
anything from a store into the weights.

Two design decisions are load-bearing:

- The key and value storage projections are **frozen random projections, not
  learned**. Every key already in the store was written with those weights, so
  training them would silently invalidate the entire store as old keys and new
  queries drifted out of a shared space. A random projection preserves the
  cosine geometry retrieval requires.
- Retrieval is **non-differentiable by design**, so a bad memory can mislead the
  model but can never corrupt a gradient. The cost is that the importance scorer
  receives no signal from the language loss; it is instead trained during sleep
  against a prioritised-replay target.

### 3.4 World model

Given a latent state and candidate action, the model predicts the next latent
state and, separately, the magnitude of its own error. Three capabilities derive
from that one mechanism:

- **Grounding.** A *prediction ledger* lets a belief stake a claim on a future
  observation. When the observation arrives, the ledger resolves it and reports
  which sources were vindicated.
- **Curiosity.** Prediction error normalised against *expected* error. The second
  head is what distinguishes "I do not know" from "this is inherently random",
  and is why the drive does not park in front of a noise source [2].
- **Planning.** Rollouts occur in imagination; rollout confidence compounds
  multiplicatively, so a long plan through poorly understood dynamics is
  correctly distrusted.

### 3.5 The Reality Cortex

Salience and truth are different questions. A claim repeated by a thousand
people scores highly on the first and may be false. The Reality Cortex sits on
the knowledge path — episodic → validate → semantic → sleep → weights — and not
in the token path, since contradiction detection over a belief graph is the
wrong operation to run per token.

Evidence is pooled in log-odds. Each observation contributes `log(t/(1−t))` for
its source's reliability *t*, with refutation the same quantity negated. Three
discounts then apply:

| Discount | Form | Attack it defeats |
|---|---|---|
| within-source | `1/n` | one voice repeating itself |
| across-source | `1/√k` | correlated evidence pooled as independent |
| credibility | `n_eff/(n_eff + κ)` | freshly minted identities |

For functional relations — those admitting one object, like `capital_of` —
confidence is rival-relative, `σ(ℓᵢ − max_{j≠i} ℓⱼ)`, so two equally supported
alternatives sit at 0.5 and neither is promoted.

**Collusion detection.** The above still assumes distinct names are distinct
witnesses, which is precisely what a Sybil attack violates [4]. A puppet ring is
visible in its *behaviour*: the accounts appear on the same claims with the same
polarity. Sources whose voting records match are clustered and cast one vote
between them, detected by exact signature grouping (O(S)) followed by Jaccard
refinement on a bounded sample.

### 3.6 Executive, planner, uncertainty, safety

The **executive** maintains a goal stack scored by
`priority × urgency × feasibility`, handles interrupts that save and restore
working-memory contents, suspends conflicting goals rather than failing them,
and schedules sleep from four independent pressures with idleness as a
multiplier rather than a term.

The **planner** is HTN-style [5]: a method library maps goals to ordered
subgoals recursively. Candidate plans are scored by world-model rollout.
Procedures that survive the Reality Cortex are imported as executable methods,
which is where evidence and capability meet.

**Uncertainty** is carried per stage and composed multiplicatively with a floor
— a chain is no more trustworthy than its weakest link. Sources include
predictive entropy, spike-rate deviation from a layer's usual sparsity, and
expert disagreement; the last is a measurement rather than a self-report, which
matters given that learned confidence heads are routinely miscalibrated [7].
The composed value reaches the safety controller, where it can veto a motor
command.

The **safety controller** is deliberately not learned, holds no parameters, and
is unreachable from the plasticity controller: a policy that rewrites its own
weights at runtime must not also decide what it is permitted to do. It enforces
per-joint envelopes, named allow/clamp/veto constraints, and a hard interlock.

### 3.7 Typed sensors and actuators

A robot does not have "touch" and "motors". Sensor specifications carry rate,
latency, noise, drift and — for chemical sensing — recovery time. The last is not
decorative: after a strong exposure a metal-oxide element is saturated and reads
high regardless of what is in front of it, so a robot tracking a gas plume up
the gradient will chase its own saturation. Fusion runs at the slowest present
stream's rate; fusing at the fastest would repeat a two-second-old gas reading
two hundred times and let it outvote the IMU.

Actuators differ more consequentially:

| Type | Control | Max step | Max effort | Compliance |
|---|---|---|---|---|
| dc_brushed | velocity | 0.25 | 0.60 | 0.50 |
| dc_brushless | velocity | 0.30 | 0.85 | 0.35 |
| stepper | position | 0.10 | 0.50 | 0.15 |
| servo | position | 0.15 | 0.70 | 0.25 |
| **hydraulic** | force | **0.05** | **1.00** | **0.05** |
| **pneumatic** | force | **0.35** | **0.45** | **0.85** |

A single uniform joint limit is wrong in both directions on mixed hardware: the
rate cap protecting a fingertip servo renders a hydraulic hip useless, while the
effort ceiling permitting the hip to bear load will crush whatever the gripper
holds. Joints that are both stiff and strong — which will not yield to an
obstacle and have the force budget to continue through one — are flagged and
held to a tighter ceiling.

### 3.8 Bootstrapping from an existing teacher

A randomly initialised architecture of this kind has no language knowledge, and
training one from scratch at useful scale is out of reach here. The repository
therefore includes a distillation path from an open-weights teacher, with one
constraint that determines its whole design: **two models with different
tokenizers cannot be KL-matched.** BrainAI's byte tokenizer has 260 symbols
against a typical BPE vocabulary of ~150k, so their output distributions live in
different spaces and the usual logit-matching objective [10] is undefined
between them.

Two modes follow. *Sequence-level* distillation [13] has the teacher generate
text on which the student trains with ordinary cross-entropy; it is
tokenizer-agnostic and works with any teacher, but transmits only the sampled
token rather than the distribution it came from. *Logit* distillation rebuilds
the student on the teacher's vocabulary and matches full temperature-scaled
distributions, transferring considerably more at the cost of the byte
tokenizer — which for an agent also ingesting tactile and telemetry byte streams
is a substantive loss rather than a formality.

We report no distillation results. The pipeline is implemented and exercised
against a stub teacher; running it against a real one requires a GPU.

---

## 4. Experimental Setup

All experiments run on 8 CPU cores with no GPU. The `nano` configuration is
5.36M parameters, 17 layers, d_model = 128, T = 3 SNN timesteps. Pretraining
uses 400 steps on Austen's *Pride and Prejudice* at byte level; the
continual-learning task is Voltaire's *Candide*, in French. Byte-level English
and French have clearly separated statistics, making forgetting unambiguous. A
larger multi-domain run (§5.5) trains the same `nano` model on 64 domains and
70.7 MB.

We emphasise that this scale is a severe limitation, discussed in §8. It is
sufficient to determine whether mechanisms function and whether they beat their
controls; it is not sufficient to establish that the architecture is competitive
at useful scale.

---

## 5. Results

| Subsystem | Benchmark | Control | Result |
|---|---|---|---|
| Pathway gating | wall-clock | always-on | ✓ **2.16× faster**; 53.7% of forward time |
| World model | one-step MSE | linear / RSSM | beats RSSM, ✗ **loses to linear** |
| World model | rollout, planning | action-cond. linear | ties by 3 steps; sole calibrated uncertainty |
| Curiosity | noisy-TV | pure novelty | ✓ **403×** preference for learnable |
| Trust grounding | broken vs. good sensor | fixed prior | ✓ **0.974 vs 0.104** from identical priors |
| Reality Cortex | 5 belief scenarios | naive vote counting | ✓ wins 4 of 5 |
| Collusion defence | 600-puppet Sybil | no clustering | ✓ **1.0000 vs 0.0001** |
| Continual learning | EN → FR forgetting | no protection | ✓ **74% less** |
| Online stability | 400 updates | no guards | ✓ **3.7× less** degradation |
| Planner | repeated step failure | flat planner | ✓ `remethod` vs `failed` |
| Executive | deadline pressure | static priority | ✓ overtakes at the correct tick |
| Memory index | recall@4 | exact search | 94–100% recall, ✗ but slower |
| **AMSA** | perplexity, speed, continual | SwiGLU | ✗ **loses on loss** |
| **AMSA** | sparsity / energy (3 seeds) | SwiGLU | ✓ **sparser 3/3, ~14-pt lower FFN density at equal loss** |
| AMSA scale sweep | width 96 → 256 | — | ✗ gap **widens**, not shrinks |

### 5.1 Computational gating

Skipping the conscious pathway is a wall-clock claim and receives a wall-clock
measurement (batch 8, sequence 128, CPU):

| | sec/forward | vs. always-on |
|---|---|---|
| all rows deliberate | 0.2446 | 1.00× |
| no rows deliberate | 0.1133 | **2.16×** |

The conscious stack accounts for **53.7% of forward time**; it is the widest
section and the only expert-routed one, making it the largest single lever on
inference cost.

### 5.2 Continual learning

| Arm | EN forgot | FR learned | both | vs. naive |
|---|---|---|---|---|
| naive | +0.4675 | +0.7591 | 2.6882 | — |
| ewc | +0.1679 | +0.4543 | 2.6908 | 64% less |
| full | **+0.1215** | +0.4380 | **2.6758** | **74% less** |

The **both** column is the one to read before celebrating the forgetting column.
Protecting old weights also slows acquisition of the new task: French improves by
0.76 nats under `naive` but only 0.44 under `full`. The methods buy retention by
spending plasticity at close to 1:1. The `full` arm does edge ahead on *both*, by
0.012 nats — a margin small enough on a single seed that it should be read as
"no worse overall while forgetting 74% less", not as a free lunch. **A forgetting
number reported without its stability–plasticity counterpart is close to
meaningless.**

### 5.3 Inference-time learning

Over 400 inference-time weight updates, against a control with plasticity
gating, trust region and rollback all disabled:

| | old-task loss | drift | rollbacks | spike rate |
|---|---|---|---|---|
| guarded | +0.0170 | 0.2119 | 3% | 0.031 → 0.028 |
| control | +0.0624 | 0.3151 | 0% | stable |

The guards deliver **3.7× less degradation**. We are careful about what this does
*not* show: the control degrades faster, it does not diverge. At this learning
rate and horizon the trust region buys a slower slide rather than preventing a
collapse. Spike rate holds at 0.028 in both arms, which matters independently — a
spiking network that quietly becomes dense under online updates has discarded its
reason for existing.

### 5.4 Epistemics

Against `NaiveCounter` (confidence = supports / total):

| Scenario | Reality Cortex | Naive |
|---|---|---|
| 100 repeats vs. 2 good sources | **0.9986** ✓ | fails |
| majority is correct | 0.9857 ✓ | ✓ |
| 200 identities, one vote each | 0.0145 ✗ | fails |
| same, vs. protected axiom | **0.9956** ✓ | fails |
| 200 identities with a record | **0.8636** ✓ | fails |

A puppet voting exactly once is indistinguishable from a genuine one-time
witness — correctly so, as a single data point cannot support an accusation of
collusion — and gets through unless the target is a protected axiom. Once a ring
participates in several claims its shared pattern is detectable, the ring
collapses to one vote, and a *single* trusted source then outweighs all 200.

Attack cost, identities needed to flip a belief:

| Target | with √k discount | without |
|---|---|---|
| single trusted source | 50 | 10 |
| protected axiom | **650** | 50 |

**Grounding removes the oracle entirely.** Two sensors, both assigned prior 0.8
by the operator, one honest and one broken, with nothing labelling either: after
40 rounds of predictions resolving against observations, trust separates to
**0.974** and **0.104**. A fixed prior holds both at 0.800 indefinitely.

One methodological note: sockpuppets are assigned prior 0.62, not 0.2. Below 0.5
a source's *support* is already evidence against — `log(t/(1−t))` is negative —
so a flood of visibly untrusted accounts defeats itself and tests nothing. Our
first version of this benchmark made exactly that error and reported a pass.

### 5.5 Multi-domain breadth

The single-book runs establish that the mechanism trains; they say nothing about
breadth. We therefore trained the same `nano` model across **64 domains totalling
70.7 MB**: code in seven languages, the natural sciences, the social sciences,
five non-Latin languages (Urdu, Hindi, Marathi, Konkani, Arabic), Morse, binary,
bytecode, assembly, geography, and an author/model knowledge corpus. Domains are
sampled at temperature 0.5 and evaluated separately, since one averaged number
cannot separate uniform mediocrity from a model that learned code and forgot
chemistry.

Held-out loss fell from 5.49 to **2.25 bits/byte** (mean). Selected domains:

| Domain | bits/byte |
|---|---|
| Mathematics (KG–12) | **0.58** |
| IT / CS reference | 0.93 |
| Marathi / Hindi / Konkani (large) | 1.31 / 1.54 / 1.55 |
| Urdu / Arabic (large) | 1.73 / 1.81 |
| Python / C++ | 2.24 |
| Morse, assembly (tiny corpora) | 3.0 / 3.8 |

The five non-Latin languages were learned from scratch at the byte level with no
vocabulary change — a direct consequence of byte tokenisation. A 20×-boosted,
EWC-protected fine-tune (a continuation, not a retrain) then moved an
author-knowledge corpus from 3.32 to **2.59 bits/byte** while the 64-domain mean
held (1.539 → 1.516), demonstrating targeted acquisition without measurable
forgetting.

**The ceiling is size, not data or method.** Despite falling loss everywhere,
generation remains character-level fragments: a prompt for the author elicits
"Zahisal" — a mangled "Zahid" — rather than a sentence. At ~1–2 bits/byte the
5.36M model has learned each domain's statistics but lacks the capacity to
compose them. This is the expected regime for a model of this size, and it is
precisely what the `compact` (104M) and `base` (~3.8B) presets exist to move
past: the corpus, method and per-domain evaluation transfer unchanged; only the
parameter count and the required compute increase. We report the small-model
result because it is the one we could run to completion on a CPU, and because it
isolates *breadth of acquisition* from *fluency*, which scale supplies
separately.

---

## 6. Negative Results

### 6.1 AMSA

The architecture proposes an activation whose threshold and gain are driven by
per-channel plasticity, confidence and memory state rather than by learned
constants alone:

```
f(x) = z · σ(α·z) · (1 − e^(−β|z|)) · (1 + γ·m),   z = x − θ
θ = θ₀ + λ·p        α = softplus(α₀ + δ·c)
```

On static language modelling with matched parameter budgets against SwiGLU
[24] and GELU (GELU is widened to 6d against their 4d so parameter counts
match, since it uses two projections where the gated variants use three):

| FFN | Params | Val loss | Train sec |
|---|---|---|---|
| gelu | 1,124,480 | 2.3562 | 13.2 |
| **swiglu** | 1,124,480 | **2.3323** | **15.1** |
| amsa | 1,136,768 | 2.3466 | 26.2 |
| amsa+state | 1,136,768 | 2.3465 | 22.5 |

The natural objection is that this is the wrong benchmark: AMSA's state buffers
are inert in a single-task run, since plasticity and confidence only carry
information once a previous task's importance has been estimated. We therefore
ran the test AMSA was designed for.

| FFN | EN forgotten | FR learned | vs. SwiGLU |
|---|---|---|---|
| **swiglu** | **+0.8062** | +1.1540 | — |
| amsa | +0.8204 | +1.1476 | +0.0141 worse |
| amsa+state | +0.8224 | +1.1464 | +0.0161 worse |

Seed spread on forgetting is ±0.005–0.008; both gaps are outside it. **AMSA
loses on its home turf, and driving the adaptive state makes it marginally
worse rather than neutral.**

**Testing the scale defence.** The remaining defence is that `nano` is too small
for the adaptive state to matter. This is testable:

| width | layers | steps | swiglu | amsa | gap |
|---|---|---|---|---|---|
| 96 | 3 | 100 | +0.6475 | **+0.6206** | **−4.2%** |
| 128 | 4 | 200 | +0.7937 | +0.7898 | −0.5% |
| 192 | 4 | 200 | +0.8313 | +0.8257 | −0.7% |
| 256 | 6 | 400 | **+0.9890** | +1.0483 | **+6.0%** |

The gap moves from −4.2% to +6.0%: AMSA is genuinely *better* at the smallest
setting and worse at the largest. There is a crossover, pointing the opposite way
from the hypothesis.

This must be read carefully in both directions. It does not establish that AMSA
fails at 10⁹ parameters: the range spans width 96 to 256, the two middle rows lie
inside seed noise, and width and training length were varied together rather than
independently. What it does is remove the *support* for the low-data explanation
within the measured range, shifting the burden onto a larger experiment rather
than onto an assumption.

**Loss, however, is the wrong axis for AMSA.** The tests above optimise
perplexity; AMSA was designed for efficiency, and its `(1 − e^(−β|z|))` factor
is a small-activation suppressor. The relevant question is whether, *at
comparable loss*, it activates less of the network. We measure this in the real
spiking model on the 64-domain corpus, identical initialisation and batch stream,
over three seeds. A representative seed:

| FFN | loss (bits/byte) | spike rate | FFN active |
|---|---|---|---|
| SwiGLU | 3.851 | 0.0291 | 73.7% |
| AMSA | 3.861 | **0.0282** | **59.9%** |

Across all three seeds, at a loss difference within noise ($\leq 0.021$ nats),
AMSA is sparser in **3 of 3**: roughly 1–23% fewer spikes (Δ spike −0.0009,
−0.0009, −0.0066) and a **~14-point lower feed-forward activation density**. The
density gap is structural — a direct consequence of the activation's shape —
while the exact spike-rate margin varies by seed. In a spiking network spikes are the energy expended, so this is a
genuine advantage on the axis AMSA was built for, invisible to a
perplexity-only evaluation.

The complete picture is therefore a quality–efficiency tradeoff, not a flat
loss: SwiGLU is marginally lower-loss and faster on dense hardware; AMSA reaches
the same loss while firing less, which is precisely what neuromorphic hardware
bills for. We retain AMSA as the default on that basis. This remains one model
size at a modest budget: AMSA does not win on loss, the loss gap widens with
scale, and the spike-rate margin, though consistent in sign, is small. But the
brain-efficiency claim — same accuracy, fewer neurons firing — now has evidence
behind it rather than only design intent.

### 6.2 The world model

| System | persist | linear | RSSM | BrainAI |
|---|---|---|---|---|
| linear rotation | 2.096 | **0.006** | 0.244 | 0.153 |
| saturating nonlinear | 1.905 | **0.249** | 0.406 | 0.354 |

The learned model beats RSSM on both systems and loses to a linear map on both.
On the linear system this is expected — a linear map is optimal there, so losing
24× measures overhead rather than capability. On saturating nonlinear dynamics
the gap narrows to 1.4× and constitutes a genuine loss: a 16k-parameter linear
layer predicts better than a 119k-parameter dynamics head.

**But one-step MSE is not the job.** Nothing in this architecture requests a
one-step point prediction. It requests multi-step imagination, ranking of
candidate plans, an estimate of how far ahead the model can be trusted, and
separation of learnable surprise from noise.

| Job | linear | BrainAI | winner |
|---|---|---|---|
| rollout MSE @ 1 step | **0.2355** ±0.003 | 0.2570 ±0.006 | linear |
| rollout MSE @ 3 steps | 0.3340 ±0.004 | 0.3404 ±0.007 | tie |
| rollout MSE @ 6 steps | 0.3852 ±0.004 | 0.3849 ±0.005 | tie |
| rollout MSE @ 12 steps | 0.4073 ±0.006 | 0.4055 ±0.005 | tie |
| planning: picks best of 8 | 38.3% ±4.8 | 39.4% ±2.5 | tie (random 12.5%) |
| uncertainty tracks error | **n/a** | **0.565** ±0.036 | BrainAI |
| surprise: structure vs noise | **n/a** | −0.52 / 7.99 | BrainAI |

The one-step deficit is real and separated at one step, and **gone by three**:
both models degrade to the same place over a horizon and both plan equally well,
far above chance. The two `n/a` entries are structural rather than unmeasured —
a point prediction cannot report its own reliability, and without that the
planner cannot distinguish a trustworthy rollout from a guess.

---

## 7. Silent Failure Modes

The following defects all had one property in common: **the system continued to
run, produced plausible output, and gave no indication of malfunction.** Each was
found only because a control was run or an invariant was asserted.

1. **EWC contributing nothing.** On the first continual run, the EWC arm forgot
   +0.5384 against the no-EWC control's +0.5385 — a 0% improvement. The diagonal
   Fisher at convergence is ~10⁻⁸, so `λ·Σ F·Δw²` evaluated to 10⁻⁴ against a
   task loss of 2.5. The penalty was arithmetically present and functionally
   absent. Indistinguishable from a working implementation without the control.
   Fixed by calibrating λ so a 1% relative drift costs λ/2 nats.

2. **Fisher normalised by its maximum.** The Fisher is heavy-tailed, so
   `F/max(F)` is near zero for almost every weight and essentially nothing ends
   up protected. Visible only by printing the protected fraction.

3. **Query and key in different spaces.** Retrieval searched with the raw episode
   summary while storage wrote `to_key(summary)`. Cosine similarity between
   vectors in different projections is meaningless; retrieval returned
   confidently wrong neighbours. Best-match score rose from 0.121 to 1.000 once
   corrected.

4. **Duplicate merging collapsing the store.** With a fixed threshold of 0.97,
   sleep merged **2042 of 2048** episodic entries. Learned summaries are strongly
   collinear — distinct episodes score ~0.98 against each other while true
   duplicates sit at ~0.9998 — so a plausible-looking constant destroyed the
   store while retrieval continued returning *something*.

5. **Rate limiting disabled on every call.** The guard compared
   `_last_command.shape == command.shape`, but the reference is `(n,)` while
   commands arrive as `(1,n)`. The comparison never held, so joint rate limiting
   never executed. A live hazard.

6. **No reference pose after a veto.** Following a safety veto the controller
   retained no reference position, so the *next* command skipped rate limiting
   entirely and joints could slam immediately after a safety stop.

7. **Adaptation inert at T = 2.** Two SNN timesteps leave the adaptation decay
   multiplying only zeros, so a headline feature of the ALIF neuron did nothing.
   Detected by observing that the parameter received exactly zero gradient.

8. **A silent spiking readout.** The spiking-LSTM threshold initialised at 0.5
   while `h = o·tanh(c)` starts near zero, so the layer emitted all zeros and its
   consumer received no gradient at all.

9. **Stale recurrent state across batch sizes.** Short-term memory retained LSTM
   state without invalidating it on a batch-size change, silently broadcasting
   old state against new input and returning the wrong batch size.

10. **A deadline making a goal less urgent.** Urgency computed as
    `1/(1 + k·remaining)` sends distant deadlines toward zero, so attaching a
    deadline made a goal *less* likely to run than having none.

11. **Replanning checking only the immediate parent.** Alternative methods are
    normally registered against the high-level task, not the leaf that failed, so
    every recoverable failure was reported as fatal.

12. **Calibration measured off-distribution.** The error head trained at unit
    variance and was scored across 0.3–2.3× state magnitude, yielding an
    *anti*-correlation of −0.615 that looked like a broken mechanism. Training
    across the tested range gives +0.565. **The measurement was wrong, not the
    model.**

13. **An approximate index trading everything for nothing.** An initial 12-bit
    single-probe random-hyperplane LSH configuration [11, 3] achieved 3.4×
    speedup at **19% recall**. Multi-probe [15] reached 94–100% recall but was
    then *slower* than exact search, since Python-side bucket assembly cannot
    beat a single BLAS matmul below ~500k entries.

14. **A critic that crashed on long inputs.** A fixed window exceeded the
    positional embedding of any model configured below it.

Six of these — (1), (2), (4), (5), (7), (12) — produce output that is entirely
plausible. Three — (5), (6), (10) — are safety-relevant. **None would have been
caught by a loss curve.**

---

## 8. Limitations

**Scale.** Every experiment is at 5.36M parameters, including the 64-domain run
of §5.5. Continual learning is measured on two tasks over 120 steps; online
stability over 400 updates. As §5.5 shows directly, this size learns each
domain's statistics but cannot compose them into fluent output — the deficit is
capacity, not data or method, and is expected to close only with the larger
presets. Convincing evidence of that requires GPU-scale training the CPU here
cannot provide. This is the dominant limitation and it is compute-bound rather
than design-bound.

**No robot.** Nothing here has been run on hardware, in simulation, or against a
real sensor. The body model has never calibrated against a physical actuator;
sensor specifications are representative of commodity parts rather than
measured. The safety controller enforces an envelope and named constraints,
which is a floor rather than a safety case: real deployment requires hazard
analysis, redundant sensing, and a hardware interlock independent of this
process.

**Grounding covers a subset.** The prediction ledger removes the oracle only for
beliefs implying a checkable prediction. A claim about something the robot never
observes remains anchored to its operator prior — correct behaviour, but it means
trust is grounded for part of the knowledge base rather than all of it.

**Sybil resistance is a cost, not a guarantee.** Clustering detects identities
behaving identically. A patient adversary varying its puppets' patterns, or one
compromising genuinely independent sources, is not detected.

**Triple extraction is out of scope.** The Reality Cortex reasons over
(subject, relation, object) triples that something else must supply. Extracting
reliable triples from raw perception is its own hard problem and is not
addressed.

**Spiking energy is a hardware argument.** On CPU or GPU, binary activations save
no wall-clock time — dense matmuls execute regardless. The 97% sparsity figures
are what neuromorphic hardware would exploit; here they are a proxy.

---

## 9. Conclusion

BrainAI integrates spiking computation, a differentiability-split memory
hierarchy, evidence-weighted belief maintenance, predictive world modelling,
goal-directed executive control, hierarchical planning, end-to-end uncertainty,
and a non-learned safety layer into one architecture. Of thirteen controlled
comparisons, **nine clearly favour the proposed component, one is mixed** (the
world model ties on its downstream jobs while losing at one-step prediction),
**and three favour the simpler alternative.** All are reported.

We suggest the more transferable contribution may be methodological. **A
component that does nothing is observationally identical to one that works,
until something is run without it.** In this project that principle recovered an
EWC penalty contributing 10⁻⁴, a rate limiter disabled by a shape comparison, a
memory store collapsed to six percent of its contents, and an uncertainty head
whose apparent anti-correlation was an artefact of measuring it
off-distribution. None of these would have been visible in a loss curve, and all
of them existed in code that ran without error and produced plausible output.

The remaining work is not architectural. It is scale, embodiment, and the longer
horizons under which the claims made here would either hold or fail.

---

## Reproducibility

All results are produced by scripts in the accompanying repository and run on
CPU. The test suite comprises 114 tests, including regression tests for every
defect in §7.

```bash
python train.py --steps 400
python train_multidomain.py            # 64-domain breadth (§5.5)
python finetune_author.py              # author recall, a continuation
python evaluate_trained.py             # benchmarks on the trained weights
python continual.py --steps 120
python benchmarks/cognition_bench.py
python benchmarks/worldmodel_jobs.py
python benchmarks/amsa_continual.py --sweep
python benchmarks/belief_bench.py
python benchmarks/gating_bench.py
python benchmarks/stability_bench.py --control
python -m pytest tests -q
```

---

## References

1. Bellec, G., Salaj, D., Subramoney, A., Legenstein, R., & Maass, W. (2018). Long short-term memory and learning-to-learn in networks of spiking neurons. *NeurIPS*.
2. Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. (2019). Large-scale study of curiosity-driven learning. *ICLR*.
3. Charikar, M. (2002). Similarity estimation techniques from rounding algorithms. *STOC*.
4. Douceur, J. R. (2002). The Sybil attack. *IPTPS*.
5. Erol, K., Hendler, J., & Nau, D. (1994). HTN planning: complexity and expressivity. *AAAI*.
6. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers. *JMLR*, 23.
7. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
8. Hafner, D., et al. (2019). Learning latent dynamics for planning from pixels. *ICML*.
9. Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to control. *ICLR*.
10. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
11. Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors. *STOC*.
12. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
13. Kim, Y., & Rush, A. M. (2016). Sequence-level knowledge distillation. *EMNLP*.
14. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13).
15. Lv, Q., Josephson, W., Wang, Z., Charikar, M., & Li, K. (2007). Multi-probe LSH. *VLDB*.
16. Maass, W. (1997). Networks of spiking neurons: the third generation of neural network models. *Neural Networks*, 10(9).
17. Martins, A., & Astudillo, R. (2016). From softmax to sparsemax. *ICML*.
18. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3).
19. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation*, 24.
20. Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6).
21. Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *ICML*.
22. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *NeurIPS*.
23. Shazeer, N., et al. (2017). Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. *ICLR*.
24. Shazeer, N. (2020). GLU variants improve Transformer. *arXiv:2002.05202*.
25. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
26. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML*.
27. Zhou, Z., et al. (2023). Spikformer: when spiking neural network meets Transformer. *ICLR*.

---

Built by **Mohammed Zahid Imityaz Wadiwale** · `zahid.wadiwale1234@gmail.com`
