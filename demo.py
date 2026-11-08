# BrainAI - built by Mohammed Zahid.
"""Walk through every capability of the architecture.

    python demo.py

Uses ``checkpoints/brainai_pretrained.pt`` if it exists, otherwise a fresh
model (the mechanisms all work untrained; only the text quality suffers).
"""

from __future__ import annotations

from pathlib import Path

import torch

from brainai import AUTHOR, BrainAI, BrainConfig
from brainai.identity import AttributionError, attribution_line

CKPT = Path("checkpoints/brainai_pretrained.pt")


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def load_model() -> BrainAI:
    if CKPT.exists():
        print(f"loading {CKPT}")
        return BrainAI.load(CKPT)
    print("no checkpoint found - using a fresh model")
    return BrainAI(BrainConfig.for_scale("nano"))


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(8)
    model = load_model()
    tok = model.tokenizer

    # ------------------------------------------------------------------
    rule("1. Identity - fixed authorship, free deployment name")
    print(model.identity.banner())
    print()
    for name in ("Javri", "Atlas-7", "BrainAI"):
        model.rename(name)
        print(f"  renamed -> {model.name:<10} {attribution_line(model.name)}")
    print(f"\n  author is immutable: model.identity.author == {AUTHOR!r}")
    try:
        model.identity.author = "someone else"  # type: ignore[misc]
    except AttributeError as exc:
        print(f"  attempting to reassign it raises: AttributeError({exc})")

    # ------------------------------------------------------------------
    rule("2. Scale presets - nano runs here, full is the humanoid spec")
    for scale in ("nano", "small", "base", "full"):
        cfg = BrainConfig.for_scale(scale)
        print(f"  {scale:<6}{cfg.total_layers:>4} layers  "
              f"d_model={cfg.d_model:<6} T={cfg.timesteps}  "
              f"~{cfg.param_estimate() / 1e9:>8.2f}B params (estimate)")
    print("\n  full-scale section breakdown:")
    print("\n".join("  " + l for l in
                    BrainConfig.for_scale("full").describe().splitlines()[4:]))

    # ------------------------------------------------------------------
    rule("3. Multimodal input - text, image, video, touch in one pass")
    cfg = model.cfg
    text = tok.encode_batch(["the robot picked up the red cup"], 32)
    image = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    video = torch.randn(1, 4, 3, cfg.image_size, cfg.image_size)
    touch = torch.randn(1, 1, 64)

    for label, kwargs in [
        ("text only", dict(tokens=text)),
        ("text + image", dict(tokens=text, images=image)),
        ("text + video", dict(tokens=text, video=video)),
        ("all modalities", dict(tokens=text, images=image, video=video, touch=touch)),
    ]:
        out = model(**kwargs)
        fused = model.encoder(**kwargs).shape[1]
        print(f"  {label:<16} fused stream {fused:>4} positions  "
              f"logits {tuple(out.logits.shape)}  {out.trace.summary()}")

    # ------------------------------------------------------------------
    rule("4. Three pathways - reflex, subconscious, conscious")
    model.eval()
    out = model(tokens=text)
    print(f"  {out.trace.summary()}")
    print(f"  reflex weight {out.trace.reflex_weight:.3f} of the final answer")
    print("\n  hard gating at inference actually skips the conscious stack:")
    varied = tok.encode_batch(
        ["hello", "compute the inverse kinematics for the left arm",
         "yes", "why did the grasp fail and what should change next time"], 48
    )
    for threshold in (0.0, 0.5, 0.99):
        model.cfg.conscious_gate_threshold = threshold
        o = model(tokens=varied)
        print(f"    threshold {threshold:<5} -> conscious ran on "
              f"{o.trace.conscious_fraction:.0%} of the batch "
              f"(salience {o.trace.salience:.3f})")
    model.cfg.conscious_gate_threshold = 0.5

    # ------------------------------------------------------------------
    rule("5. Memory hierarchy")
    concepts = ["Paris", "France", "capital city", "the red cup", "grasping"]
    ids = []
    for c in concepts:
        vec = model(tokens=tok.encode_batch([c], 16)).summary
        ids.append(model.memory.learn_concept(vec[0], label=c))
    model.memory.semantic.relate(ids[0], ids[1])      # Paris  - France
    model.memory.semantic.relate(ids[0], ids[2])      # Paris  - capital
    model.memory.semantic.relate(ids[3], ids[4])      # cup    - grasping
    print(f"  stored {len(ids)} concepts; graph edges: "
          f"{ {model.memory.semantic.labels[k]: [model.memory.semantic.labels[n] for n in v] for k, v in model.memory.semantic.edges.items()} }")

    query = model(tokens=tok.encode_batch(["Paris"], 16)).summary
    ctx, info = model.memory.recall(query)
    print(f"\n  recall('Paris') -> context {tuple(ctx.shape)}")
    for k, v in info.items():
        print(f"    {k:<20}{v:.3f}")

    stats = model.memory.stats()
    print(f"\n  episodic {int(stats['episodic']['entries'])} entries, "
          f"semantic {int(stats['semantic']['entries'])} entries")
    print(f"  working memory {model.memory.working.slots} slots, "
          f"sensory buffer {len(model.memory.sensory)}/{cfg.sensory_span} frames")

    # ------------------------------------------------------------------
    rule("5b. Reality Cortex - salience is not truth")
    rc = model.reality
    for name, prior in [("nasa", 0.97), ("journal", 0.95), ("workshop_manual", 0.92),
                        ("mechanic_ali", 0.75), ("mechanic_sara", 0.75),
                        ("passerby", 0.30), ("rumour_feed", 0.20)]:
        rc.register_source(name, prior=prior, strength=12 if prior > 0.9 else 6)
    rc.assert_axiom("earth", "shape_of", "oblate spheroid",
                    text="the earth is an oblate spheroid")

    print("  a) one loud source, repeated 50 times")
    for _ in range(50):
        flat = model.observe("earth", "shape_of", "flat", "rumour_feed",
                             text="the earth is flat")
    print(f"     'earth is flat' after 50 repeats -> confidence {flat.confidence:.4f}")
    print(f"     protected axiom still {rc.claims[0].status}, "
          f"confidence {rc.confidence(0):.4f}")

    print("\n  b) living with mechanics - genuine learning")
    for src in ("mechanic_ali", "mechanic_sara", "workshop_manual"):
        bolt = model.observe("bolt_M10", "torque_spec", "40Nm", src,
                             text="an M10 bolt is torqued to 40 Nm")
    print(f"     '40 Nm' from 3 independent sources -> "
          f"confidence {bolt.confidence:.4f}")

    print("\n  c) a contradiction arrives")
    model.observe("paris", "capital_of", "france", "nasa")
    model.observe("paris", "capital_of", "france", "journal")
    wrong = model.observe("paris", "capital_of", "germany", "passerby")
    print(f"     france  {rc.confidence(wrong.claim_id - 1):.4f}     "
          f"germany {wrong.confidence:.4f}  (rival-relative)")

    print("\n  d) promotion into semantic memory")
    for report in model.validate_and_promote():
        print(f"     {report.summary()}")

    print("\n  e) reality disagrees - the source pays for it")
    door = model.observe("door_north", "state_of", "unlocked", "passerby").claim_id
    print(f"     passerby trust before: {rc.trust('passerby'):.4f}")
    for _ in range(4):
        rc.observe_outcome(door, was_correct=False)
    print(f"     after 4 failed predictions: {rc.trust('passerby'):.4f}")
    print(f"     the germany claim falls with it: {rc.confidence(wrong.claim_id):.4f}")

    print("\n  f) why is this believed?  auditable, not a black box")
    print("\n".join("     " + l for l in rc.explain(bolt.claim_id).splitlines()))
    print(f"\n     {rc.report()}")

    # ------------------------------------------------------------------
    rule("5c. World model - imagine, then learn from being wrong")
    state = model(tokens=tok.encode_batch(["the cup is on the table"], 32)).summary
    pred = model.world(state)
    print(f"  predicted next state {tuple(pred.state.shape)}  "
          f"confidence {float(pred.confidence.detach().mean()):.3f}")
    for horizon in (1, 4, 16):
        _, conf = model.world.imagine(state, [None] * horizon)
        print(f"    imagined {horizon:>2} steps ahead -> confidence {float(conf):.4f}")
    print("  (a long plan through uncertain dynamics is correctly distrusted)")

    actual = state + 0.3 * torch.randn_like(state)
    print(f"\n  curiosity: "
          f"{ {k: round(v, 3) for k, v in model.intrinsic_reward(state, actual).items()} }")

    print("\n  grounding trust with no oracle - the environment decides:")
    model.reality.register_source("wall_sensor", prior=0.8, strength=6)
    claim = model.observe("north_door", "state_of", "open", "wall_sensor").claim_id
    print(f"    wall_sensor trust before      {model.reality.trust('wall_sensor'):.4f}")
    for _ in range(5):
        p = model.predict(state, expect_claims=[claim], tolerance=0.05)
        model.observe_reality(-p.state[0])          # reality disagrees
    print(f"    after 5 failed predictions    {model.reality.trust('wall_sensor'):.4f}")
    print(f"    ledger: {model.ledger.report()}")

    # ------------------------------------------------------------------
    rule("5d. Executive - goals, interrupts, and resuming where you left off")
    ex = model.executive
    ex.stack.push("tidy_workshop", priority=0.55, deadline=30)
    ex.stack.push("recharge", priority=0.35, conflicts_with=["tidy_workshop"])
    print(f"  tick 1: {ex.step()}")
    ex.stack.interrupt("human_entered_cell", priority=1.0)
    print(f"  tick 2: {ex.step()}   <- preempted")
    ex.stack.complete(2)
    print(f"  tick 3: {ex.step()}   <- resumed with its scratchpad intact")
    print(ex.stack.render())
    pressure = model.should_sleep(idle=True)
    print(f"\n  sleep pressure: { {k: round(v, 3) for k, v in pressure.items()} }")
    print(f"  sleep plan:     {ex.sleep_plan(pressure)}")

    # ------------------------------------------------------------------
    rule("5e. Hierarchical planning, checked by imagination")
    lib = model.planner.library
    lib.register("serve_tea", ["boil_water", "pour_cup"], "kettle")
    lib.register("serve_tea", ["use_dispenser", "pour_cup"], "dispenser")
    lib.register("boil_water", ["fill_kettle", "switch_on"])
    plan = model.plan("serve_tea")
    print(f"  chose method {plan.nodes[0].method!r} "
          f"(rollout confidence {plan.confidence:.4f})")
    print(plan.render())

    leaf = plan.next_action()
    verdicts = [model.planner.report_result(plan, leaf, False) for _ in range(3)]
    print(f"  failing {plan.nodes[leaf].goal!r} repeatedly -> {verdicts}")
    alt = model.planner.remethod(plan, leaf)
    print(f"  replanned with method {alt.nodes[0].method!r}" if alt else "  no alternative")

    print("\n  a procedure learned from two mechanics becomes executable:")
    for src in ("mechanic_ali", "mechanic_sara"):
        model.reality.submit_procedure(
            "replace_belt", ["loosen_tensioner", "slip_belt_off", "fit_new_belt"], src
        )
    print(f"    imported {lib.learn_from_reality(model.reality)} method(s)")
    print(f"    plan: {[n.goal for n in model.plan('replace_belt').nodes]}")

    # ------------------------------------------------------------------
    rule("5f. Uncertainty that reaches the actuators")
    out = model(tokens=text)
    print(f"  {model.uncertainty.summary()}")
    debate = model.deliberate_with_debate(text)
    print(f"  internal debate: {debate.summary()}")

    # ------------------------------------------------------------------
    rule("6. Learning at inference time - weights change while running")
    model.eval()
    fact = tok.encode_batch(
        ["the workshop key is kept in the third drawer"] * 2, 64
    )
    before = float(model.lm_loss(fact, write_memory=False)[0].detach())
    snapshot = model.lm_head.weight.detach().clone()

    for i in range(6):
        stats = model.learn_online(fact, reward=1.0, task_id=9)
        if i in (0, 5):
            print(f"  step {i}: loss {stats['loss']:.4f}  "
                  f"applied |dW| {stats['applied_norm']:.5f}  "
                  f"clip {stats['clip']:.3f}  "
                  f"rolled_back {bool(stats['rolled_back'])}")

    after = float(model.lm_loss(fact, write_memory=False)[0].detach())
    drift = float((model.lm_head.weight.detach() - snapshot).norm())
    print(f"\n  loss on the new fact {before:.4f} -> {after:.4f}  "
          f"(improved by {before - after:.4f})")
    print(f"  lm_head weights moved by {drift:.5f} - at inference, in eval mode")
    print(f"  episodic memory now holds {model.memory.episodic.size} entries")

    # ------------------------------------------------------------------
    rule("7. Sleep - replay, merge, re-anchor")
    print(f"  replay buffer: {model.replay.stats()}")
    report = model.sleep(cycles=6, batch_size=4)
    for k in ("replayed_batches", "replay_loss_first", "replay_loss_last",
              "scorer_loss_first", "scorer_loss_last", "episodic_merged",
              "protected_fraction"):
        if k in report:
            print(f"    {k:<22}{report[k]:.4f}")

    # ------------------------------------------------------------------
    rule("8. Growth and energy")
    print(f"  {model.energy_report()}")
    print(f"  plasticity: {model.plasticity.report()}")

    # ------------------------------------------------------------------
    rule("9. Robot control - motor output through the safety controller")
    sc = model.safety
    sc.set_current_position(torch.zeros(cfg.n_actuators))

    action = model.act(images=image, touch=touch, tokens=text,
                       context={"human_distance": 2.5})
    joints = action["joint_targets"][0]
    print(f"  joint targets ({joints.numel()} actuators): "
          f"{[round(float(v), 3) for v in joints[:6]]} ...")
    print(f"  effort:        {[round(float(v), 3) for v in action['effort'][0][:6]]} ...")
    print(f"  reflex only:   {action['reflex_only']}")
    print(f"  safety:        {action['safety'].summary()}")

    print("\n  the network cannot leave its envelope, whatever it asks for:")
    wild = torch.full((1, cfg.n_actuators), 5.0)
    out, eff, rep = sc.filter(wild, torch.full((1, cfg.n_actuators), 0.2))
    print(f"    requested +5.000 -> commanded {float(out.max()):+.3f}   {rep.summary()}")

    print("\n  a human steps inside the work envelope:")
    _, eff2, rep2 = sc.filter(
        torch.ones(1, cfg.n_actuators), torch.full((1, cfg.n_actuators), 0.2),
        {"human_distance": 0.2},
    )
    print(f"    {rep2.summary()}  (effort {float(eff2.max()):.1f})")

    print("\n  emergency stop:")
    sc.engage_estop("operator")
    _, eff3, rep3 = sc.filter(
        torch.ones(1, cfg.n_actuators), torch.full((1, cfg.n_actuators), 0.5)
    )
    print(f"    {rep3.summary()}  (effort {float(eff3.max()):.1f})")
    sc.release_estop()

    print("\n  commands ramp rather than slam:")
    trace = [float(sc.filter(torch.ones(1, cfg.n_actuators),
                             torch.full((1, cfg.n_actuators), 0.2))[0].max())
             for _ in range(8)]
    print(f"    {' -> '.join(f'{v:.2f}' for v in trace)}")
    print(f"\n  {sc.stats()}")

    # ------------------------------------------------------------------
    rule("10. Generation")
    for prompt in ("It is a truth ", "She said "):
        print(f"  {prompt!r} -> {model.generate(prompt, max_new_tokens=48)!r}")

    # ------------------------------------------------------------------
    rule("11. Checkpoints refuse to load without attribution")
    path = Path("checkpoints/demo_rename.pt")
    model.rename("Javri").save(path)
    reloaded = BrainAI.load(path)
    print(f"  saved as 'Javri', reloaded as {reloaded.name!r}, "
          f"author still {reloaded.identity.author!r}")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["identity"]["author"] = "Someone Else"
    torch.save(blob, "checkpoints/demo_tampered.pt")
    try:
        BrainAI.load("checkpoints/demo_tampered.pt")
        print("  ERROR: tampered checkpoint loaded")
    except AttributionError as exc:
        print(f"  tampered checkpoint rejected:\n    {exc}")

    rule("status")
    print(reloaded.status())


if __name__ == "__main__":
    main()
