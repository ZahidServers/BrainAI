# BrainAI - built by Mohammed Zahid.
"""Test suite.

    python -m pytest tests -q

Covers the pieces where a silent failure would be invisible in a loss curve:
spike binarity, adaptation actually adapting, sparsemax producing real zeros,
eviction policy, attribution enforcement, and the online update's rollback.
"""

from __future__ import annotations

import math

import pytest
import torch

from brainai import (
    ALIF,
    AMSA,
    AUTHOR,
    BrainAI,
    BrainConfig,
    ByteTokenizer,
    Identity,
    MoEFFN,
    ReplayBuffer,
    VectorStore,
    available_scales,
    sparsemax,
    spike,
)
from brainai.identity import AttributionError, InvalidName, attribution_line, verify
from brainai.actuators import ActuatorBank
from brainai.activation import GatedAMSA, SwiGLU
from brainai.corpus import (
    DomainSplit,
    MultiDomainCorpus,
    discover_disk_domains,
    evaluate_domains,
)
from brainai.critique import SelfCritique, perplexity_critic
from brainai.prompts import domains as prompt_domains
from brainai.prompts import sample as sample_prompts
from brainai.executive import Executive, GoalStack
from brainai.planner import HierarchicalPlanner, MethodLibrary
from brainai.reality import BELIEVED, OBSOLETE, RealityCortex
from brainai.safety import (
    JointLimits,
    SafetyController,
    effort_ceiling,
    hazardous_joint_ceiling,
    human_in_envelope,
    requires_confidence,
)
from brainai.sensors import SensorSuite
from brainai.sensors import sensor as sensor_spec
from brainai.uncertainty import UncertaintyTracker, entropy_confidence
from brainai.world import CuriosityDrive, WorldModel


@pytest.fixture(scope="module")
def cfg() -> BrainConfig:
    return BrainConfig.for_scale("nano", max_seq_len=64)


@pytest.fixture(scope="module")
def model(cfg: BrainConfig) -> BrainAI:
    torch.manual_seed(0)
    return BrainAI(cfg)


# ---------------------------------------------------------------- identity
def test_author_is_immutable():
    ident = Identity()
    assert ident.author == AUTHOR == "Mohammed Zahid"
    with pytest.raises(AttributeError):
        ident.author = "Someone Else"  # type: ignore[misc]


def test_rename_keeps_attribution():
    ident = Identity().rename("Javri")
    assert ident.display_name == "Javri"
    assert ident.author == AUTHOR
    assert AUTHOR in attribution_line(ident.display_name)
    assert "Javri" in attribution_line(ident.display_name)


@pytest.mark.parametrize("bad", ["", "   ", "x" * 49, "drop\ntable"])
def test_invalid_names_rejected(bad):
    with pytest.raises(InvalidName):
        Identity().rename(bad)


def test_verify_rejects_tampering():
    good = Identity().rename("Javri").to_dict()
    verify(good)  # renamed is fine
    for field in ("author", "codename"):
        bad = dict(good)
        bad[field] = "Someone Else"
        with pytest.raises(AttributionError):
            verify(bad)


# ------------------------------------------------------------------ config
def test_every_scale_builds_a_config():
    for scale in available_scales():
        c = BrainConfig.for_scale(scale)
        assert c.total_layers > 0
        assert c.param_estimate() > 0
        assert "Mohammed Zahid" in c.describe()


def test_full_scale_keeps_the_depth_but_not_the_width():
    """The rescale must preserve the cognitive structure, only shrink widths."""
    full = BrainConfig.for_scale("full")
    wide = BrainConfig.for_scale("full_wide")

    assert full.total_layers == wide.total_layers == 392
    for name in ("conscious", "subconscious", "planner"):
        assert full.section(name).layers == wide.section(name).layers
        assert full.section(name).width < wide.section(name).width

    # An order of magnitude smaller, with more room for stored knowledge.
    assert full.param_estimate() < wide.param_estimate() / 100
    assert full.episodic_capacity > wide.episodic_capacity


def test_config_round_trips(cfg):
    restored = BrainConfig.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()


# -------------------------------------------------------------- activation
def test_amsa_is_zero_at_its_threshold():
    act = AMSA(4)
    with torch.no_grad():
        act.theta0.fill_(0.7)
    out = act(torch.full((1, 4), 0.7))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_amsa_suppresses_small_inputs_more_than_large():
    act = AMSA(1)
    small = float(act(torch.tensor([[0.05]])).abs().detach())
    large = float(act(torch.tensor([[3.0]])).abs().detach())
    assert small < 0.01 < large


def test_amsa_memory_state_modulates_output():
    act = AMSA(4)
    x = torch.randn(2, 4)
    base = act(x)
    act.set_state(memory=torch.ones(4))
    assert not torch.allclose(base, act(x))


def test_amsa_gradients_are_finite():
    act = AMSA(8)
    x = torch.randn(4, 8, requires_grad=True)
    act(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in act.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


# ----------------------------------------------------------------- neurons
def test_spikes_are_binary():
    s = spike(torch.randn(64) * 3)
    assert set(s.unique().tolist()) <= {0.0, 1.0}


def test_surrogate_gradient_flows_through_the_step():
    v = torch.randn(32, requires_grad=True)
    spike(v).sum().backward()
    assert v.grad is not None and float(v.grad.abs().sum()) > 0


def test_alif_threshold_adapts_after_firing():
    """A neuron that just fired must be harder to fire again."""
    n = ALIF(1, v_threshold=0.2, adapt_scale=5.0, learnable_dynamics=False)
    n.reset()
    drive = torch.ones(1, 1) * 5.0
    fired = [float(n(drive)) for _ in range(4)]
    assert fired[0] == 1.0, "should fire on strong input"
    assert 0.0 in fired[1:], "adaptation must suppress at least one later spike"


def test_adapt_decay_is_inert_below_three_timesteps():
    """Documents why every scale preset uses T >= 3.

    a[t] = rho.a[t-1] + s[t-1] with a[0] = s[0] = 0, so rho only ever multiplies
    zero until the third step and receives no gradient before then.
    """
    def adapt_decay_grad(steps: int) -> float:
        n = ALIF(4, v_threshold=0.2, adapt_scale=1.0)
        n.reset()
        out = torch.zeros(2, 4)
        for _ in range(steps):
            out = out + n(torch.ones(2, 4) * 3.0)
        out.sum().backward()
        g = n.adapt_decay_logit.grad
        return 0.0 if g is None else float(g.abs().sum())

    assert adapt_decay_grad(2) == 0.0, "at T=2 the decay only multiplies zeros"
    assert adapt_decay_grad(3) > 0.0, "adaptation must be live by T=3"


# --------------------------------------------------------------- attention
def test_sparsemax_is_a_distribution_with_exact_zeros():
    z = torch.tensor([[5.0, 0.1, -3.0, -4.0]])
    p = sparsemax(z)
    assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-5)
    assert float((p == 0).sum()) >= 2
    assert float(p[0, 0]) > float(p[0, 1])


def test_sparsemax_handles_masked_logits():
    z = torch.tensor([[1.0, 2.0, -1e4]])
    p = sparsemax(z)
    assert float(p[0, 2]) == 0.0
    assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-5)


def test_sparsemax_matches_softmax_on_a_uniform_row():
    z = torch.zeros(1, 6)
    assert torch.allclose(sparsemax(z), torch.full((1, 6), 1 / 6), atol=1e-6)


# ------------------------------------------------------------------ blocks
def test_moe_routes_and_can_grow(cfg):
    moe = MoEFFN(cfg.d_model, cfg)
    before = int(moe.active_experts.sum())
    x = torch.randn(2, 5, cfg.d_model)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(moe.aux_loss)
    assert moe.wake_expert() >= 0
    assert int(moe.active_experts.sum()) == before + 1


def test_moe_reports_zero_headroom_when_all_experts_are_awake(cfg):
    moe = MoEFFN(cfg.d_model, cfg)
    while moe.wake_expert() >= 0:
        pass
    assert int(moe.active_experts.sum()) == cfg.n_experts
    assert moe.wake_expert() == -1


# ------------------------------------------------------------------ memory
def test_vector_store_writes_and_retrieves():
    store = VectorStore(16, 8)
    keys = torch.eye(8)[:4]
    store.write(keys, keys, torch.ones(4))
    assert store.size == 4
    vals, scores, idx = store.search(keys[:1], k=2)
    assert vals.shape == (1, 2, 8)
    assert float(scores[0, 0]) > 0.99


def test_vector_store_pads_when_it_holds_fewer_than_k():
    store = VectorStore(16, 8)
    store.write(torch.randn(1, 8), torch.randn(1, 8), torch.ones(1))
    vals, scores, idx = store.search(torch.randn(2, 8), k=5)
    assert vals.shape == (2, 5, 8)
    assert int((idx == -1).sum()) == 8  # 2 rows x 4 padded slots


def test_vector_store_refuses_to_evict_something_more_important():
    store = VectorStore(2, 4)
    store.write(torch.randn(2, 4), torch.randn(2, 4), torch.tensor([0.9, 0.9]))
    assert store.size == 2
    store.write(torch.randn(1, 4), torch.randn(1, 4), torch.tensor([0.1]))
    assert float(store.importance[store.occupied].min()) == pytest.approx(0.9)


def test_novelty_is_high_for_unseen_and_low_for_stored():
    store = VectorStore(8, 16)
    v = torch.randn(1, 16)
    assert float(store.novelty(v)) == 1.0          # empty store
    store.write(v, v, torch.ones(1))
    assert float(store.novelty(v)) < 0.01
    assert float(store.novelty(-v)) > 0.9


def test_adaptive_merge_survives_collinear_keys():
    """The regression that a fixed threshold hides.

    Real summaries share a large common component, so distinct episodes still
    score ~0.98 against each other while true duplicates sit at ~0.9998.  A
    fixed 0.97 collapses the whole store; the adaptive threshold must not.
    """
    torch.manual_seed(0)
    store = VectorStore(64, 32)
    common = torch.randn(1, 32) * 5.0                 # the shared component
    distinct = torch.randn(6, 32) * 0.3
    entries = common + distinct.repeat_interleave(4, 0)
    entries = entries + 0.001 * torch.randn(24, 32)   # 4 near-copies of each
    store.write(entries, entries, torch.ones(24))

    store.merge_duplicates()                          # adaptive
    assert 4 <= store.size <= 9, f"expected ~6 survivors, got {store.size}"

    fixed = VectorStore(64, 32)
    fixed.write(entries, entries, torch.ones(24))
    fixed.merge_duplicates(threshold=0.97)
    assert fixed.size < store.size, "a fixed 0.97 should over-merge here"


def test_merge_duplicates_frees_slots():
    store = VectorStore(8, 4)
    v = torch.randn(1, 4)
    store.write(v.repeat(3, 1), v.repeat(3, 1), torch.ones(3))
    assert store.size == 3
    assert store.merge_duplicates(0.95) >= 1
    assert store.size < 3


def test_replay_buffer_keeps_tasks_balanced():
    buf = ReplayBuffer(8, 4)
    buf.add(torch.zeros(8, 4, dtype=torch.long), 1.0, task_id=0)
    assert buf.size == 8
    buf.add(torch.ones(4, 4, dtype=torch.long), 2.0, task_id=1)
    counts = buf.task_counts()
    assert counts.get(1, 0) == 4 and counts.get(0, 0) == 4
    assert buf.sample(3).shape == (3, 4)


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_round_trips_unicode():
    tok = ByteTokenizer()
    for text in ("hello", "Voltaire, Candide", "robot arm 90 deg", "naive cafe"):
        assert tok.decode(tok.encode(text)) == text


def test_tokenizer_pads_a_batch():
    tok = ByteTokenizer()
    batch = tok.encode_batch(["a", "much longer string"], 12)
    assert batch.shape == (2, 12)
    assert int((batch[0] == tok.pad_id).sum()) > 0


# ------------------------------------------------------------------- model
def test_forward_shapes_for_each_modality(model, cfg):
    tokens = torch.randint(0, 255, (2, 16))
    out = model(tokens=tokens)
    assert out.logits.shape == (2, 16, cfg.vocab_size)

    img = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    out = model(tokens=tokens, images=img)
    assert out.logits.shape == (2, 16, cfg.vocab_size), "text span must be preserved"

    vid = torch.randn(2, 3, 3, cfg.image_size, cfg.image_size)
    assert model(tokens=tokens, video=vid).logits.shape[1] == 16
    assert model(touch=torch.randn(2, 1, 64)).logits is None


def test_lm_loss_starts_near_uniform_and_backpropagates(model, cfg):
    loss, out = model.lm_loss(torch.randint(0, 255, (2, 17)))
    value = float(loss.detach())
    assert 0.5 * math.log(cfg.vocab_size) < value < 1.5 * math.log(cfg.vocab_size)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    model.zero_grad(set_to_none=True)


def test_motor_output_is_bounded(model, cfg):
    action = model.act(tokens=torch.randint(0, 255, (1, 8)))
    assert action["joint_targets"].shape == (1, cfg.n_actuators)
    assert float(action["joint_targets"].abs().max()) <= 1.0
    assert 0.0 <= float(action["effort"].min()) <= float(action["effort"].max()) <= 1.0


def test_hard_gate_skips_the_conscious_pathway(model):
    tokens = torch.randint(0, 255, (4, 12))
    model.eval()
    model.cfg.conscious_gate_threshold = 0.999
    assert model(tokens=tokens).trace.conscious_fraction == 0.0
    model.cfg.conscious_gate_threshold = 0.0
    assert model(tokens=tokens).trace.conscious_fraction == 1.0
    model.cfg.conscious_gate_threshold = 0.5


def test_spiking_layers_stay_sparse(model):
    model.eval()
    model(tokens=torch.randint(0, 255, (2, 24)))
    rate = model.energy_report()["mean_spike_rate"]
    assert 0.0 < rate < 0.5, f"spike rate {rate} - the network has gone dense"


def test_generate_returns_text(model):
    torch.manual_seed(0)
    assert isinstance(model.generate("hello", max_new_tokens=6), str)


# ------------------------------------------------------- continual learning
def test_ewc_penalty_is_zero_until_consolidation(cfg):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    assert float(m.plasticity.ewc_penalty(m)) == 0.0
    m.lm_loss(torch.randint(0, 255, (2, 17)))[0].backward()
    m.plasticity.accumulate_fisher(m)
    m.plasticity.consolidate(m)
    for p in m.parameters():
        if p.requires_grad:
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.1)
    assert float(m.plasticity.ewc_penalty(m).detach()) > 0.0


def test_consolidation_protects_a_meaningful_slice(cfg):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    for _ in range(3):
        m.lm_loss(torch.randint(0, 255, (2, 17)))[0].backward()
        m.plasticity.accumulate_fisher(m)
        m.zero_grad(set_to_none=True)
    stats = m.plasticity.consolidate(m)
    assert 0.005 < stats["protected_fraction"] < 0.5


def test_online_learning_updates_weights_in_eval_mode(cfg):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.eval()
    tokens = torch.randint(0, 255, (2, 33))
    before = m.lm_head.weight.detach().clone()
    stats = m.learn_online(tokens, verify_update=False)
    assert stats["applied_norm"] > 0
    assert not torch.equal(before, m.lm_head.weight)
    assert not m.training, "learn_online must restore eval mode"


def test_online_update_rolls_back_when_it_makes_things_worse(cfg):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.eval()
    tokens = torch.randint(0, 255, (2, 33))
    before = m.lm_head.weight.detach().clone()
    # An absurd learning rate guarantees the step overshoots.
    loss, _ = m.lm_loss(tokens)
    stats = m.plasticity.online_step(
        m, loss, lr=5.0, max_relative_norm=10.0,
        verify=lambda: float(m.lm_loss(tokens, write_memory=False)[0].detach()),
    )
    if stats["rolled_back"]:
        assert torch.equal(before, m.lm_head.weight)
    else:
        assert stats["loss_after"] <= stats["loss_before"]


def test_sleep_runs_and_reports(cfg):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.replay.add(torch.randint(0, 255, (6, 33)), 1.0, task_id=0)
    report = m.sleep(cycles=2, batch_size=2)
    assert report["replayed_batches"] == 2
    assert "replay_loss_last" in report and math.isfinite(report["replay_loss_last"])
    assert report["consolidations"] >= 1


# ------------------------------------------------------------- persistence
def test_save_load_round_trip_is_exact(cfg, tmp_path):
    torch.manual_seed(0)
    m = BrainAI(cfg).rename("Javri")
    m.eval()
    tokens = torch.randint(0, 255, (2, 16))
    expected = m(tokens=tokens, write_memory=False).logits

    path = m.save(tmp_path / "ckpt.pt")
    restored = BrainAI.load(path)
    restored.eval()

    assert restored.name == "Javri"
    assert restored.identity.author == AUTHOR
    got = restored(tokens=tokens, write_memory=False).logits
    assert torch.allclose(expected, got, atol=1e-5)


def test_memory_survives_a_checkpoint(cfg, tmp_path):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.memory.episodic.write(torch.randn(3, cfg.d_model),
                            torch.randn(3, cfg.d_model), torch.ones(3))
    m.replay.add(torch.randint(0, 255, (4, 33)), 1.0)
    restored = BrainAI.load(m.save(tmp_path / "mem.pt"))
    assert restored.memory.episodic.size == 3
    assert restored.replay.size == 4


# ---------------------------------------------------------- reality cortex
def _cortex() -> RealityCortex:
    rc = RealityCortex()
    rc.register_source("manual", prior=0.92, strength=12)
    rc.register_source("expert_a", prior=0.75, strength=6)
    rc.register_source("expert_b", prior=0.75, strength=6)
    rc.register_source("rumour", prior=0.20, strength=4)
    return rc


def test_repetition_from_one_source_saturates():
    """The echo-chamber guard: 50 repeats must not equal 50 witnesses."""
    rc = _cortex()
    one = rc.submit("x", "prop", "y", "expert_a").claim_id
    after_1 = rc.own_logit(one)
    for _ in range(49):
        rc.submit("x", "prop", "y", "expert_a")
    after_50 = rc.own_logit(one)
    # Harmonic growth: 50 repeats are worth well under 6x one observation.
    assert after_50 < 6.0 * after_1


def test_independent_sources_compound_faster_than_repeats():
    rc = _cortex()
    rc.submit("a", "prop", "v", "expert_a")
    rc.submit("a", "prop", "v", "expert_a")
    repeated = rc.confidence(0)

    rc.submit("b", "prop", "v", "expert_a")
    rc.submit("b", "prop", "v", "expert_b")
    independent = rc.confidence(1)
    assert independent > repeated


def test_axiom_survives_a_flood_of_low_trust_claims():
    """Adaptable is not the same as credulous."""
    rc = _cortex()
    rc.assert_axiom("earth", "shape_of", "oblate spheroid")
    for _ in range(200):
        report = rc.submit("earth", "shape_of", "flat", "rumour")
    assert report.confidence < 0.05
    assert rc.claims[0].status == BELIEVED
    assert rc.confidence(0) > 0.95


def test_functional_relations_make_rivals_exclusive():
    rc = _cortex()
    rc.submit("paris", "capital_of", "france", "manual")
    rc.submit("paris", "capital_of", "france", "expert_a")
    rival = rc.submit("paris", "capital_of", "germany", "rumour")
    assert rc.confidence(0) > 0.9
    assert rival.confidence < 0.1
    # Many-valued relations coexist instead of competing.
    assert not rc.is_functional("has_landmark")
    rc.submit("paris", "has_landmark", "louvre", "manual")
    rc.submit("paris", "has_landmark", "eiffel tower", "manual")
    assert rc.confidence(2) > 0.5 and rc.confidence(3) > 0.5


def test_relation_arity_is_declarable():
    rc = _cortex()
    rc.declare_multivalued("state_of")
    rc.submit("door", "state_of", "locked", "manual")
    rc.submit("door", "state_of", "unlocked", "manual")
    assert rc.confidence(0) > 0.5 and rc.confidence(1) > 0.5   # both stand

    rc.declare_functional("state_of")
    assert rc.confidence(0) == pytest.approx(0.5, abs=0.01)    # now they compete


def test_promotion_requires_independent_corroboration():
    rc = _cortex()
    for _ in range(30):
        rc.submit("bolt", "torque_spec", "40Nm", "manual")   # one source only
    assert rc.confidence(0) > 0.9
    assert rc.promote(memory=None, embed=None) == []

    rc.submit("bolt", "torque_spec", "40Nm", "expert_a")
    promoted = rc.promote(memory=None, embed=None)
    assert len(promoted) == 1 and promoted[0].promoted


def test_beaten_beliefs_are_retired_with_history_not_deleted():
    rc = _cortex()
    rc.submit("door", "state_of", "locked", "rumour")
    rc.submit("door", "state_of", "unlocked", "manual")
    rc.submit("door", "state_of", "unlocked", "expert_a")
    rc.promote(memory=None, embed=None)

    losers = [c for c in rc.claims if c.status == OBSOLETE]
    assert len(losers) == 1
    assert losers[0].object == "locked"
    assert losers[0].superseded_by >= 0          # history kept, not deleted
    assert len(rc.claims) == 2


def test_being_wrong_costs_the_source_its_trust():
    rc = _cortex()
    cid = rc.submit("door", "state_of", "unlocked", "rumour").claim_id
    before_trust = rc.trust("rumour")
    before_conf = rc.confidence(cid)
    for _ in range(5):
        rc.observe_outcome(cid, was_correct=False)
    assert rc.trust("rumour") < before_trust
    assert rc.confidence(cid) < before_conf


def test_correlation_discount_raises_the_price_of_a_sybil_attack():
    """Distinct identities are not distinct witnesses.

    Sockpuppets are given prior 0.62, not 0.2: below 0.5 a source's support is
    already evidence against, so a flood of visibly-untrusted accounts defeats
    itself and tests nothing.  The attack that matters uses identities that each
    look individually plausible.
    """
    def survives(discount: str, n_identities: int) -> bool:
        rc = RealityCortex(correlation_discount=discount)
        rc.assert_axiom("earth", "shape_of", "oblate spheroid")
        for i in range(n_identities):
            rc.register_source(f"sock_{i}", prior=0.62, strength=4)
            rc.submit("earth", "shape_of", "flat", f"sock_{i}")
        return rc.confidence(0) > 0.5

    assert survives("sqrt", 200), "protected axiom must survive 200 sockpuppets"
    assert not survives("none", 200), "without the discount the attack lands"
    # Honest bound: it raises the price, it does not make it impossible.
    assert not survives("sqrt", 2000)


def test_unvetted_sources_are_discounted_until_they_have_a_record():
    rc = RealityCortex()
    rc.register_source("fresh", prior=0.9, strength=2)
    rc.register_source("established", prior=0.9, strength=32)
    assert rc.credibility("fresh") < rc.credibility("established")

    for _ in range(20):
        cid = rc.submit("a", "p", "v", "fresh").claim_id
        rc.observe_outcome(cid, was_correct=True)
    assert rc.credibility("fresh") > 0.8, "a track record should restore weight"


def test_reality_cortex_round_trips():
    rc = _cortex()
    rc.assert_axiom("earth", "shape_of", "oblate spheroid")
    rc.submit("bolt", "torque_spec", "40Nm", "manual")
    rc.submit("bolt", "torque_spec", "40Nm", "expert_a")
    restored = RealityCortex.from_dict(rc.to_dict())
    assert restored.report() == rc.report()
    assert restored.confidence(1) == pytest.approx(rc.confidence(1))
    assert restored.claims[0].protected


def test_beliefs_survive_a_checkpoint(cfg, tmp_path):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.reality.register_source("manual", prior=0.9, strength=10)
    m.observe("bolt", "torque_spec", "40Nm", "manual")
    restored = BrainAI.load(m.save(tmp_path / "beliefs.pt"))
    assert restored.reality.report()["claims"] == 1.0
    assert "manual" in restored.reality.sources


# ------------------------------------------------------------------ safety
def _controller(n: int = 8) -> SafetyController:
    return SafetyController(JointLimits(n), [effort_ceiling(0.85)])


def test_first_command_is_rate_limited_from_home():
    """Without a starting reference the first command can slam the joints."""
    sc = _controller()
    out, _, report = sc.filter(torch.ones(1, 8), torch.full((1, 8), 0.1))
    assert float(out.max()) == pytest.approx(0.15, abs=1e-6)
    assert report.rate_clamped == 8


def test_rate_limit_still_applies_after_a_veto():
    """Regression: a veto used to leave no reference, so the next command slammed."""
    sc = _controller()
    sc.filter(torch.zeros(1, 8), torch.full((1, 8), 0.99))      # vetoed
    out, _, _ = sc.filter(torch.full((1, 8), -1.0), torch.full((1, 8), 0.1))
    assert float(out.min()) >= -0.151


def test_commands_ramp_instead_of_jumping():
    sc = _controller()
    reached = [float(sc.filter(torch.ones(1, 8), torch.full((1, 8), 0.1))[0].max())
               for _ in range(8)]
    assert reached[0] == pytest.approx(0.15, abs=1e-6)
    assert reached[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(b - a <= 0.1501 for a, b in zip(reached, reached[1:]))


def test_position_limits_are_enforced():
    sc = SafetyController(JointLimits(8))
    sc.set_current_position(torch.zeros(8))
    for _ in range(40):
        out, _, _ = sc.filter(torch.full((1, 8), 9.0), torch.full((1, 8), 0.1))
    assert float(out.max()) <= 1.0


def test_estop_holds_position_and_zeroes_effort():
    sc = _controller()
    sc.filter(torch.full((1, 8), 0.1), torch.full((1, 8), 0.1))
    sc.engage_estop("test")
    out, effort, report = sc.filter(torch.ones(1, 8), torch.full((1, 8), 0.5))
    assert report.estop and not report.allowed
    assert float(effort.abs().max()) == 0.0
    assert float(out.max()) <= 0.1 + 1e-6
    sc.release_estop()
    assert not sc.estop_engaged


def test_a_human_in_the_envelope_vetoes_motion():
    sc = SafetyController(JointLimits(8), [human_in_envelope(0.5)])
    _, effort, report = sc.filter(
        torch.ones(1, 8), torch.full((1, 8), 0.2), {"human_distance": 0.2}
    )
    assert report.vetoed_by == "human_in_envelope"
    assert float(effort.max()) == 0.0
    _, _, ok = sc.filter(
        torch.ones(1, 8), torch.full((1, 8), 0.2), {"human_distance": 3.0}
    )
    assert ok.allowed


def test_act_routes_through_safety_and_reports_what_changed(model, cfg):
    model.safety.set_current_position(torch.zeros(cfg.n_actuators))
    action = model.act(tokens=torch.randint(0, 255, (1, 8)),
                       context={"human_distance": 2.0})
    assert "safety" in action and "requested_joint_targets" in action
    assert action["joint_targets"].shape == action["requested_joint_targets"].shape
    assert float(action["joint_targets"].abs().max()) <= 1.0

    model.safety.engage_estop("test")
    stopped = model.act(tokens=torch.randint(0, 255, (1, 8)))
    assert stopped["safety"].estop
    assert float(stopped["effort"].abs().max()) == 0.0
    model.safety.release_estop()


def test_safety_is_not_reachable_by_the_plasticity_controller(model):
    """Online learning must not be able to widen its own operating envelope."""
    tracked = {name for name, _ in model.plasticity._entries}
    assert not any(n.startswith("safety") for n in tracked)
    assert not any(n.startswith("reality") for n in tracked)
    assert not list(getattr(model.safety, "parameters", lambda: [])())


# ------------------------------------------------------------- world model
def test_world_model_learns_dynamics(cfg):
    """A learnable mapping should get easier to predict with training."""
    torch.manual_seed(0)
    world = WorldModel(cfg)
    opt = torch.optim.Adam(world.parameters(), lr=1e-3)
    shift = torch.randn(cfg.d_model) * 0.1

    first = last = 0.0
    for step in range(60):
        state = torch.randn(8, cfg.d_model)
        opt.zero_grad(set_to_none=True)
        loss, parts = world.loss(state, state + shift)
        loss.backward()
        opt.step()
        if step == 0:
            first = parts["dynamics"]
        last = parts["dynamics"]
    assert last < first


def test_rollout_confidence_decays_with_horizon():
    """A long plan through uncertain dynamics must not look as safe as a short one."""
    cfg = BrainConfig.for_scale("nano")
    world = WorldModel(cfg)
    state = torch.randn(1, cfg.d_model)
    _, short = world.imagine(state, [None])
    _, long = world.imagine(state, [None] * 8)
    assert float(long) < float(short)


def test_curiosity_ignores_irreducible_noise():
    """High error that the model *expects* is not interesting."""
    cfg = BrainConfig.for_scale("nano")
    drive = CuriosityDrive(cfg)
    surprise = torch.full((4,), 3.0)
    novelty = torch.ones(4)
    predictable = drive.reward(surprise, novelty, torch.zeros(4))
    noisy = drive.reward(surprise, novelty, torch.full((4,), 5.0))
    assert float(noisy.mean()) < float(predictable.mean())


def test_boredom_rises_without_novelty():
    cfg = BrainConfig.for_scale("nano")
    drive = CuriosityDrive(cfg)
    for _ in range(200):
        drive.reward(torch.full((2,), -5.0), torch.zeros(2))
    assert drive.exploring


def test_prediction_ledger_grounds_trust_without_an_oracle(cfg):
    """The environment supplies the verdict; nothing labels anything."""
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.reality.register_source("sensor", prior=0.8, strength=8)
    claim = m.observe("door", "state_of", "open", "sensor").claim_id
    before = m.reality.trust("sensor")

    state = torch.randn(1, cfg.d_model)
    prediction = m.predict(state, expect_claims=[claim], tolerance=0.1)
    m.observe_reality(-prediction.state[0])            # reality disagrees
    assert m.reality.trust("sensor") < before

    after_wrong = m.reality.trust("sensor")
    for _ in range(3):
        p = m.predict(state, expect_claims=[claim], tolerance=0.5)
        m.observe_reality(p.state[0])                  # reality agrees
    assert m.reality.trust("sensor") > after_wrong


def test_unresolved_predictions_expire_without_punishing_anyone(cfg):
    """Never checking a prediction is not evidence that it was wrong."""
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.reality.register_source("sensor", prior=0.8, strength=8)
    claim = m.observe("x", "state_of", "y", "sensor").claim_id
    before = m.reality.trust("sensor")

    m.predict(torch.randn(1, cfg.d_model), expect_claims=[claim])
    for _ in range(m.ledger.max_age + 1):
        m.ledger.tick()
    assert m.ledger.report()["expired"] >= 1
    assert m.reality.trust("sensor") == pytest.approx(before)


# ---------------------------------------------------------------- executive
def test_a_deadline_never_makes_a_goal_less_urgent():
    """Regression: 1/(1+k*remaining) sent distant deadlines below the baseline."""
    stack = GoalStack()
    far = stack.goals[stack.push("far", priority=0.6, deadline=40)]
    none = stack.goals[stack.push("none", priority=0.6)]
    assert far.urgency(0) >= none.urgency(0)
    assert far.score(0) > none.score(0)
    assert far.urgency(40) == 1.0


def test_interrupt_preempts_and_the_incumbent_resumes():
    stack = GoalStack()
    work = stack.push("work", priority=0.7)
    assert stack.arbitrate() == work

    alarm = stack.interrupt("alarm")
    assert stack.arbitrate() == alarm
    assert stack.goals[work].status == "suspended"

    stack.complete(alarm)
    assert stack.arbitrate() == work
    assert stack.goals[work].resumed_count == 1


def test_conflicting_goals_are_suspended_not_failed():
    stack = GoalStack()
    a = stack.push("charge", priority=0.9)
    b = stack.push("patrol", priority=0.4, conflicts_with=["charge"])
    stack.goals[b].status = "active"
    assert stack.arbitrate() == a
    assert stack.goals[b].status == "suspended"      # "not now", not "never"
    assert not stack.goals[b].finished


def test_failing_a_subgoal_abandons_its_parent():
    stack = GoalStack()
    parent = stack.push("make_tea")
    child = stack.push("boil_water", parent=parent)
    stack.complete(child, achieved=False)
    assert stack.goals[parent].status == "abandoned"


def test_task_switch_saves_and_restores_working_memory(cfg):
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.eval()
    m(tokens=torch.randint(0, 255, (1, 12)))

    first = m.executive.stack.push("first", priority=0.8)
    m.executive.step()
    saved = m.memory.working._content.clone()

    m.executive.stack.interrupt("urgent")
    m.executive.step()
    m.memory.working._content = torch.zeros_like(saved)   # clobber it

    m.executive.stack.complete(1)
    result = m.executive.step()
    assert result["active"] == first and result["context_restored"]
    assert torch.allclose(m.memory.working._content, saved)


def test_sleep_pressure_needs_more_than_one_signal():
    ex = Executive(sleep_pressure_threshold=1.0)
    quiet = ex.sleep_pressure(replay_fill=0.2)
    assert not quiet["should_sleep"]
    loaded = ex.sleep_pressure(replay_fill=1.0, unconsolidated_claims=64, drift=0.5)
    assert loaded["should_sleep"]
    assert ex.sleep_plan(loaded)["cycles"] > ex.sleep_plan(quiet)["cycles"]


# ------------------------------------------------------------------ planner
def test_planner_decomposes_recursively():
    lib = MethodLibrary()
    lib.register("make_tea", ["fill_kettle", "boil", "pour"])
    lib.register("fill_kettle", ["grasp", "move_to_tap"])
    plan = HierarchicalPlanner(lib).decompose("make_tea")
    names = [n.goal for n in plan.nodes]
    assert "grasp" in names and "pour" in names
    assert [plan.nodes[i].goal for i in plan.leaves()][:2] == ["grasp", "move_to_tap"]


def test_planner_escalates_retry_then_remethod_then_failure():
    lib = MethodLibrary()
    lib.register("goal", ["step"], "a")
    lib.register("goal", ["other"], "b")
    planner = HierarchicalPlanner(lib, max_retries=1)
    plan = planner.decompose("goal")
    leaf = plan.next_action()
    assert planner.report_result(plan, leaf, False) == "retry"
    assert planner.report_result(plan, leaf, False) == "remethod"

    single = MethodLibrary()
    single.register("goal", ["step"], "only")
    p2 = HierarchicalPlanner(single, max_retries=0)
    plan2 = p2.decompose("goal")
    leaf2 = plan2.next_action()
    assert p2.report_result(plan2, leaf2, False) == "failed"
    assert plan2.failed


def test_remethod_searches_ancestors_not_just_the_parent():
    """Alternatives sit on the high-level task, not on the leaf that broke.

    Checking only the immediate parent reports a recoverable failure as fatal.
    """
    lib = MethodLibrary()
    lib.register("serve_tea", ["boil_water", "pour"], "kettle")
    lib.register("serve_tea", ["use_dispenser", "pour"], "dispenser")
    lib.register("boil_water", ["fill_kettle", "switch_on"])   # no alternative

    planner = HierarchicalPlanner(lib, max_retries=0)
    plan = planner.decompose("serve_tea")
    leaf = plan.next_action()                                   # fill_kettle
    assert plan.nodes[leaf].goal == "fill_kettle"
    assert planner.report_result(plan, leaf, False) == "remethod"

    replacement = planner.remethod(plan, leaf)
    assert replacement is not None
    assert replacement.nodes[0].method == "dispenser"


def test_believed_procedures_become_executable_plans():
    """Evidence and capability meet: a corroborated procedure becomes a method."""
    rc = RealityCortex()
    for name in ("mech_a", "mech_b"):
        rc.register_source(name, prior=0.85, strength=10)
        rc.submit_procedure("replace_belt", ["loosen", "slip_off", "fit_new"], name)

    lib = MethodLibrary()
    assert lib.learn_from_reality(rc) == 1
    plan = HierarchicalPlanner(lib).decompose("replace_belt")
    assert [plan.nodes[i].goal for i in plan.leaves()] == \
        ["loosen", "slip_off", "fit_new"]


def test_uncorroborated_procedures_are_not_executable():
    rc = RealityCortex()
    rc.register_source("rumour", prior=0.2, strength=4)
    rc.submit_procedure("disable_interlock", ["cut_wire"], "rumour")
    assert MethodLibrary().learn_from_reality(rc) == 0


# -------------------------------------------------------------- uncertainty
def test_entropy_confidence_spans_the_range():
    peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    uniform = torch.zeros(1, 4)
    assert float(entropy_confidence(peaked)) > 0.9
    assert float(entropy_confidence(uniform)) == pytest.approx(0.0, abs=1e-5)


def test_tracker_finds_the_weakest_stage():
    tracker = UncertaintyTracker()
    tracker.record("perception", 0.95)
    tracker.record("language", 0.2)
    tracker.record("motor", 0.9)
    assert tracker.bottleneck() == "language"
    assert 0.2 < tracker.overall() < 0.95


def test_low_confidence_vetoes_a_motor_command(cfg):
    """Uncertainty stops being a diagnostic and becomes a brake."""
    torch.manual_seed(0)
    m = BrainAI(cfg)
    m.safety.add_constraint(requires_confidence(0.99))
    action = m.act(tokens=torch.randint(0, 255, (1, 8)))
    assert action["safety"].vetoed_by == "requires_confidence"
    assert float(action["effort"].abs().max()) == 0.0


def test_forward_reports_end_to_end_confidence(model):
    out = model(tokens=torch.randint(0, 255, (2, 12)))
    confidence = out.trace.extras["confidence"]
    assert 0.0 < confidence <= 1.0
    assert out.trace.extras["bottleneck_stage"] in model.uncertainty.stages


# ------------------------------------------------------------------ critique
def test_debate_reports_disagreement(model):
    result = model.deliberate_with_debate(torch.randint(0, 255, (1, 12)))
    assert result is not None
    assert 0.0 <= result.disagreement <= 1.0
    assert result.confidence == pytest.approx(1.0 - result.disagreement)
    assert len(result.participants) >= 1


def test_self_critique_returns_the_best_round_not_the_last():
    scores = iter([0.2, 0.9, 0.4])
    loop = SelfCritique(
        generate=lambda p: f"draft for {p[:8]}",
        critic=lambda t: (next(scores), ["needs work"]),
        rounds=3,
        accept_score=2.0,          # never accept early
    )
    best = loop.run("prompt")
    assert best.score == 0.9
    assert loop.report()["rounds"] == 3


def test_perplexity_critic_flags_degenerate_output(model):
    critic = perplexity_critic(model)
    empty_score, empty_issues = critic("   ")
    assert empty_score == 0.0 and empty_issues
    _, repeat_issues = critic("the the the the the the the the")
    assert any("repetitive" in i for i in repeat_issues)


# -------------------------------------------------------- memory scaling
def test_lsh_index_matches_exact_search_on_clustered_data():
    torch.manual_seed(0)
    n, dim = 3000, 32
    store = VectorStore(n, dim, index_after=1000, n_bits=8, n_probes=3)
    centres = torch.randn(40, dim) * 3
    keys = centres[torch.randint(0, 40, (n,))] + 0.3 * torch.randn(n, dim)
    store.write(keys, keys, torch.ones(n))
    assert store._indexed

    query = keys[:24] + 0.05 * torch.randn(24, dim)
    _, _, approx = store.search(query, k=4, exact=False)
    _, _, exact = store.search(query, k=4, exact=True)
    recall = sum(
        len(set(approx[b].tolist()) & set(exact[b].tolist())) / 4
        for b in range(24)
    ) / 24
    assert recall > 0.8, f"recall {recall:.2%} too low to be useful"


def test_index_never_returns_an_evicted_slot():
    torch.manual_seed(0)
    store = VectorStore(64, 16, index_after=8)
    keys = torch.randn(64, 16)
    store.write(keys, keys, torch.ones(64))
    store.merge_duplicates(threshold=0.0)          # collapse aggressively
    _, _, idx = store.search(torch.randn(4, 16), k=3, exact=False)
    live = idx[idx >= 0]
    assert bool(store.occupied[live].all())


# ------------------------------------------------------- sensors/actuators
def test_actuator_types_get_different_envelopes():
    """One uniform limit across mixed hardware is wrong in both directions."""
    bank = ActuatorBank.humanoid(32)
    limits = bank.limits()
    kinds = bank.kinds()

    hydraulic = kinds.index("hydraulic")
    pneumatic = kinds.index("pneumatic")
    # Hydraulic: slow but very strong.  Pneumatic: fast but weak.
    assert limits.max_step[hydraulic] < limits.max_step[pneumatic]
    assert limits.max_effort[hydraulic] > limits.max_effort[pneumatic]


def test_stiff_and_strong_joints_are_flagged():
    bank = ActuatorBank.humanoid(32)
    hazardous = bank.hazardous()
    assert hazardous, "hydraulic joints must be flagged"
    kinds = bank.kinds()
    assert all(kinds[i] == "hydraulic" for i in hazardous)
    # A compliant actuator yields against an obstacle; these do not.
    assert not bank.hazardous() == list(range(bank.n_actuators))


def test_steppers_are_quantised_to_reachable_positions():
    bank = ActuatorBank.uniform("stepper", 4)
    spec = bank.specs()[0]
    step = 2.0 * spec.max_travel / spec.discrete_steps
    command = torch.tensor([[0.0, step * 1.4, -step * 2.6, 0.5]])
    snapped = bank.quantise(command)
    remainder = (snapped / step) - torch.round(snapped / step)
    assert torch.allclose(remainder, torch.zeros_like(remainder), atol=1e-5)


def test_continuous_actuators_are_not_quantised():
    bank = ActuatorBank.uniform("servo", 4)
    command = torch.tensor([[0.137, -0.891, 0.5, 0.0]])
    assert torch.allclose(bank.quantise(command), command)


def test_hazardous_ceiling_vetoes_high_effort_on_stiff_joints():
    bank = ActuatorBank.humanoid(16)
    sc = SafetyController(bank.limits(), [hazardous_joint_ceiling(bank, 0.5)])
    n = bank.n_actuators

    safe = torch.full((1, n), 0.2)
    assert sc.filter(torch.zeros(1, n), safe)[2].allowed

    dangerous = torch.full((1, n), 0.2)
    dangerous[0, bank.hazardous()[0]] = 0.95
    assert sc.filter(torch.zeros(1, n), dangerous)[2].vetoed_by == \
        "hazardous_joint_ceiling"


def test_model_derives_safety_limits_from_its_actuators(cfg):
    torch.manual_seed(0)
    mixed = BrainConfig.for_scale("nano", max_seq_len=64,
                                  actuator_profile="humanoid")
    m = BrainAI(mixed)
    assert "hazardous_joint_ceiling" in [c.name for c in m.safety.constraints]
    assert m.safety.limits.max_step.unique().numel() > 1, \
        "mixed hardware must not share one rate cap"


def test_sensor_confidence_reflects_the_hardware():
    """A drifty gas sensor and a laser rangefinder are not equally trustworthy."""
    lidar = sensor_spec("lidar_2d")
    gas = sensor_spec("mos_gas_array")
    assert lidar.confidence() > gas.confidence()

    # A saturated chemical sensor is blind until it recovers.
    assert gas.confidence(saturated=True, age_s=1.0) < 0.1
    assert gas.confidence(saturated=True, age_s=gas.recovery_s + 1) > 0.1


def test_fusion_runs_at_the_slowest_present_rate():
    """Fusing at the fastest rate lets a 2-second-old gas reading outvote the IMU."""
    suite = SensorSuite(64, [sensor_spec(n) for n in
                             ("imu", "lidar_2d", "mos_gas_array")])
    assert suite.fusion_rate(["imu"]) == 200.0
    assert suite.fusion_rate(["imu", "lidar_2d"]) == 10.0
    assert suite.fusion_rate(["imu", "mos_gas_array"]) == 1.0


def test_sensor_suite_fuses_mixed_streams():
    suite = SensorSuite(64, [sensor_spec(n) for n in
                             ("imu", "tactile_skin", "electrochemical_gas")])
    fused, confidence = suite(
        {"imu": torch.randn(2, 9),
         "tactile_skin": torch.randn(2, 4, 64),
         "electrochemical_gas": torch.randn(2, 4)}
    )
    assert fused.shape == (2, 1 + 4 + 1, 64)
    assert confidence["overall"] <= min(
        confidence["imu"], confidence["electrochemical_gas"]
    )


def test_stale_readings_are_detectable():
    imu = sensor_spec("imu")
    assert not imu.stale(age_ms=5.0)
    assert imu.stale(age_ms=200.0)


# ------------------------------------------------------- subject knowledge
def test_rigorous_domains_need_more_corroboration():
    """A maths result is not more true for being popular."""
    rc = RealityCortex()
    for name in ("textbook", "teacher"):
        rc.register_source(name, prior=0.9, strength=10)

    rc.learn_fact("water", "boils_at", "100C", "textbook", domain="physics")
    rc.learn_fact("water", "boils_at", "100C", "teacher", domain="physics")
    assert rc.promote(memory=None, embed=None) == []

    rc.register_source("lab", prior=0.9, strength=10)
    rc.learn_fact("water", "boils_at", "100C", "lab", domain="physics")
    assert len(rc.promote(memory=None, embed=None)) == 1


def test_everyday_claims_use_the_normal_bar():
    rc = RealityCortex()
    for name in ("friend", "sign"):
        rc.register_source(name, prior=0.85, strength=10)
    rc.learn_fact("cafe", "opens_at", "8am", "friend", domain="everyday")
    rc.learn_fact("cafe", "opens_at", "8am", "sign", domain="everyday")
    assert len(rc.promote(memory=None, embed=None)) == 1


def test_one_graph_keeps_cross_domain_links():
    """Per-subject databases would cut exactly the edges worth having."""
    rc = RealityCortex()
    rc.register_source("textbook", prior=0.95, strength=12)
    rc.learn_fact("einstein", "worked_on", "relativity", "textbook",
                  domain="physics")
    rc.learn_fact("einstein", "emigrated_in", "1933", "textbook",
                  domain="history")

    links = rc.cross_domain_links()
    assert "history" in links["physics"]
    assert "physics" in links["history"]


def test_domain_filter_is_a_view_not_a_store():
    rc = RealityCortex()
    rc.register_source("s", prior=0.9, strength=10)
    rc.learn_fact("a", "is", "b", "s", domain="mathematics")
    rc.learn_fact("c", "is", "d", "s", domain="arts")
    assert len(rc.knowledge("mathematics", min_confidence=0.0)) == 1
    assert len(rc.claims) == 2          # one store, filtered


# ---------------------------------------------------------- ffn selection
def test_swiglu_can_replace_amsa_everywhere():
    """The AMSA benchmarks are actionable only if the swap is one line."""
    torch.manual_seed(0)
    swig = BrainAI(BrainConfig.for_scale("nano", max_seq_len=64, ffn="swiglu"))
    assert not any(isinstance(m, GatedAMSA) for m in swig.modules())
    assert any(isinstance(m, SwiGLU) for m in swig.modules())
    loss, _ = swig.lm_loss(torch.randint(0, 255, (2, 17)))
    assert torch.isfinite(loss)

    amsa = BrainAI(BrainConfig.for_scale("nano", max_seq_len=64, ffn="amsa"))
    assert any(isinstance(m, GatedAMSA) for m in amsa.modules())


def test_unknown_ffn_is_rejected():
    with pytest.raises(ValueError, match="unknown ffn"):
        BrainAI(BrainConfig.for_scale("nano", max_seq_len=64, ffn="relu"))


# -------------------------------------------------- multi-domain corpus
def test_prompt_bank_cycles_evenly_across_domains():
    """Coverage is the point; uniform random draw leaves gaps in small domains."""
    picked = sample_prompts(200, seed=0)
    seen = {d for d, _ in picked}
    assert seen == set(prompt_domains())
    counts = {d: sum(1 for x, _ in picked if x == d) for d in seen}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_prompt_bank_rejects_unknown_domains():
    with pytest.raises(KeyError, match="unknown prompt domains"):
        sample_prompts(4, ["not_a_domain"])


def test_temperature_sampling_sits_between_balanced_and_proportional():
    """Code is 50x chemistry; proportional starves it, balanced overfits it."""
    def split(name, windows):
        t = torch.zeros(windows, 8, dtype=torch.long)
        return DomainSplit(name=name, train=t, val=t[:1])

    splits = {"big": split("big", 5000), "small": split("small", 100)}
    ratios = {}
    for temp in (0.0, 0.5, 1.0):
        c = MultiDomainCorpus(splits, temperature=temp)
        w = dict(zip(c.names, c.weights.tolist()))
        ratios[temp] = w["big"] / w["small"]

    assert ratios[0.0] == pytest.approx(1.0, abs=1e-5)   # balanced
    assert ratios[1.0] == pytest.approx(50.0, rel=0.01)  # proportional
    assert 1.0 < ratios[0.5] < ratios[1.0]               # the compromise


def test_corpus_sampling_returns_a_known_domain():
    def split(name, windows):
        t = torch.randint(0, 255, (windows, 16))
        return DomainSplit(name=name, train=t, val=t[:2])

    c = MultiDomainCorpus({"a": split("a", 50), "b": split("b", 50)}, seed=0)
    for _ in range(10):
        name, batch = c.sample(4)
        assert name in ("a", "b")
        assert batch.shape == (4, 16)


def test_per_domain_evaluation_reports_each_domain(cfg):
    """One averaged number hides a model that learned code and forgot chemistry."""
    torch.manual_seed(0)
    m = BrainAI(cfg)

    def split(name):
        t = torch.randint(0, 255, (12, 17))
        return DomainSplit(name=name, train=t, val=t[:4])

    c = MultiDomainCorpus({"x": split("x"), "y": split("y")})
    scores = evaluate_domains(m, c, batch_size=2, batches=1)
    assert set(scores) == {"x", "y", "mean"}
    assert all(v > 0 for v in scores.values())


def test_disk_discovery_finds_txt_and_skips_stubs(tmp_path):
    """'Train on everything in the folder' has to be literal, and it must not
    double-count the one cache that duplicates a generated domain."""
    (tmp_path / "geography.txt").write_text("x" * 5000, encoding="utf-8")
    (tmp_path / "code_klingon.txt").write_text("y" * 5000, encoding="utf-8")
    (tmp_path / "tiny.txt").write_text("z" * 100, encoding="utf-8")   # below floor
    (tmp_path / "domain_code_python.txt").write_text("p" * 5000,
                                                     encoding="utf-8")

    found = discover_disk_domains(tmp_path, min_bytes=2000)
    assert set(found) == {"geography", "code_klingon"}
    assert "tiny" not in found                 # too small
    assert "domain_code_python" not in found   # duplicate of generated domain


def test_load_rejects_a_stripped_checkpoint(cfg, tmp_path):
    torch.manual_seed(0)
    path = BrainAI(cfg).save(tmp_path / "c.pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    del blob["identity"]["author"]
    torch.save(blob, path)
    with pytest.raises(AttributionError):
        BrainAI.load(path)
