# BrainAI - built by Mohammed Zahid.
"""BrainAI: the assembled cognitive architecture.

One forward pass runs perception, the reflex arc and the subconscious pathway
unconditionally; memory recall, deliberation and motor output are conditional.
The executive mixes the fast answer with the deliberate one and the language
head reads off the result.

Everything a deployment needs is here: :meth:`learn_online` for inference-time
adaptation, :meth:`sleep` for consolidation, :meth:`act` for actuator commands,
:meth:`rename` for deployment naming, and :meth:`save`/:meth:`load` for
attribution-checked checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import mean_spike_rate
from .config import BrainConfig
from .consolidation import ReplayBuffer, SleepConsolidator
from .encoders import MultimodalEncoder
from .identity import Identity, attribution_line, verify
from .memory import MemoryController
from .pathways import (
    ConsciousPathway,
    ExecutiveController,
    MotorCortex,
    PathwayTrace,
    ReflexPathway,
    SubconsciousPathway,
    run_gated,
)
from .actuators import ActuatorBank
from .critique import InternalDebate, SelfCritique, perplexity_critic
from .executive import Executive
from .planner import HierarchicalPlanner, MethodLibrary
from .plasticity import GrowthController, PlasticityController
from .reality import RealityCortex
from .uncertainty import UncertaintyTracker, entropy_confidence, spike_rate_confidence
from .world import BodyModel, CuriosityDrive, PredictionLedger, WorldModel
from .safety import (
    JointLimits,
    SafetyController,
    SafetyReport,
    default_constraints,
    hazardous_joint_ceiling,
)
from .sensors import SensorSuite
from .sensors import sensor as sensor_spec
from .tokenizer import ByteTokenizer


@dataclass
class BrainOutput:
    """Everything one forward pass produced."""

    logits: Optional[torch.Tensor] = None
    summary: Optional[torch.Tensor] = None
    #: Per-position subconscious states, (B, S, d).  Kept so the world model
    #: can take its temporal signal from inside a single forward pass.
    states: Optional[torch.Tensor] = None
    motor: Dict[str, torch.Tensor] = field(default_factory=dict)
    trace: PathwayTrace = field(default_factory=PathwayTrace)
    aux_loss: Optional[torch.Tensor] = None
    memory_info: Dict[str, float] = field(default_factory=dict)


class BrainAI(nn.Module):
    """Spiking multimodal cognitive architecture, built by Mohammed Zahid.

    The default deployment name is ``BrainAI``; call :meth:`rename` to run it
    under any other name.  Authorship is fixed and is verified on load.
    """

    def __init__(self, cfg: Optional[BrainConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or BrainConfig.for_scale("nano")
        self.tokenizer = ByteTokenizer()

        self.encoder = MultimodalEncoder(self.cfg)
        self.reflex = ReflexPathway(self.cfg)
        self.subconscious = SubconsciousPathway(self.cfg)
        self.conscious = ConsciousPathway(self.cfg)
        # The neural output mixer.  Named `arbiter` because that is all it is;
        # the executive that owns goals lives in `self.executive`.
        self.arbiter = ExecutiveController(self.cfg)
        self.motor = MotorCortex(self.cfg)
        self.memory = MemoryController(self.cfg)

        # Prediction: the only learning signal here that needs no labels.
        self.world = WorldModel(self.cfg)
        self.curiosity = CuriosityDrive(self.cfg)
        self.body = BodyModel(self.cfg)
        self.debate = InternalDebate(self.cfg.d_model)

        self.wm_gate = nn.Parameter(torch.tensor(0.1))
        self.stm_gate = nn.Parameter(torch.tensor(0.1))
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.reflex_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

        self.replay = ReplayBuffer(1024, min(self.cfg.max_seq_len, 128))
        self.growth = GrowthController(self.cfg)

        # Truth, and permission to move.  Neither is a neural network, and
        # neither is reachable by the plasticity controller: a system that
        # rewrites its own weights at runtime must not also be able to rewrite
        # what it is allowed to believe or to do.
        self.reality = RealityCortex()

        # Per-joint envelopes derived from what each actuator actually is.
        # One uniform limit across mixed hardware is wrong in both directions:
        # the rate cap that protects a fingertip servo makes a hydraulic hip
        # useless, and the effort ceiling that lets the hip stand up will crush
        # what the gripper is holding.
        self.actuators = (
            ActuatorBank.humanoid(self.cfg.n_actuators)
            if self.cfg.actuator_profile == "humanoid"
            else ActuatorBank.uniform(
                self.cfg.actuator_profile, self.cfg.n_actuators
            )
        )
        constraints = default_constraints()
        if self.actuators.hazardous():
            constraints.append(hazardous_joint_ceiling(self.actuators))
        self.safety = SafetyController(self.actuators.limits(), constraints)

        self.sensors = (
            SensorSuite(self.cfg.d_model,
                        [sensor_spec(n) for n in self.cfg.sensor_profile])
            if self.cfg.sensor_profile else None
        )
        # Goals, interrupts and sleep scheduling.  Holds no parameters, and
        # like the safety controller is not something gradient descent can
        # rewrite -- a system that edits its own priorities under optimisation
        # pressure is much harder to reason about.
        self.executive = Executive(working_memory=self.memory.working)
        self.planner = HierarchicalPlanner(MethodLibrary())
        self.ledger = PredictionLedger()
        self.uncertainty = UncertaintyTracker()

        # Non-persistent: spike statistics describe this process's activity, so
        # a freshly loaded model must not report a stale 0% firing rate as
        # though it were a measurement.
        self.register_buffer(
            "forward_passes", torch.zeros((), dtype=torch.long), persistent=False
        )

        # Must be last: it enumerates every parameter that exists above it.
        # It holds only buffers, so attaching it adds no parameters of its own.
        self.plasticity = PlasticityController(self, self.cfg)
        self.sleeper = SleepConsolidator(self.cfg)

    # -- identity --------------------------------------------------------
    @property
    def identity(self) -> Identity:
        return self.cfg.identity

    @property
    def name(self) -> str:
        return self.cfg.identity.display_name

    def rename(self, new_name: str) -> "BrainAI":
        """Rename this deployment.  Authorship is unaffected."""
        self.cfg.identity.rename(new_name)
        return self

    def describe(self) -> str:
        return self.cfg.describe()

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        touch: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        include_motor: bool = False,
        write_memory: Optional[bool] = None,
    ) -> BrainOutput:
        """``write_memory`` defaults to ``self.training``: the long-term-memory
        interface only receives gradient once the stores hold something, so
        training has to populate them as it goes."""
        if write_memory is None:
            write_memory = self.training
        self.forward_passes += 1
        x = self.encoder(tokens, images, video, audio, touch)
        self.memory.sensory.push(x.mean(dim=1))

        reflex_out, reflex_conf = self.reflex(x)
        sub_out, salience = self.subconscious(x)

        summary = sub_out.mean(dim=1)                      # (B, d) episode vector

        context = None
        mem_info: Dict[str, float] = {}
        if use_memory:
            context, mem_info = self.memory.recall(summary)

        wm_read = self.memory.working.read(sub_out)
        deliberate, fraction = run_gated(
            self.conscious,
            sub_out,
            salience,
            self.cfg.conscious_gate_threshold,
            self.training,
            memory_context=context,
            working_read=wm_read,
        )

        # Write the deliberation to the scratchpad, then read it back within the
        # same pass - that read-after-write is what trains the write gates.
        self.memory.working.write(deliberate)
        wm_after = self.memory.working.read(deliberate)

        # Short-term memory runs over pooled *events*, not raw tokens: it holds
        # "the last few things that happened", and a per-token recurrence would
        # cost S sequential steps for no extra signal.
        events = F.adaptive_avg_pool1d(
            sub_out.transpose(1, 2), self.cfg.sensory_span
        ).transpose(1, 2)
        stm = self.memory.short_term(events).unsqueeze(1)      # (B, 1, d)

        thought = (
            sub_out + deliberate + self.wm_gate * wm_after + self.stm_gate * stm
        )
        mixed, reflex_weight = self.arbiter(reflex_out, thought, reflex_conf)

        span = self.encoder.text_span(tokens)
        logits = None
        if span:
            # Reflex and deliberation both propose tokens; the executive's
            # mixing weight decides how much of each survives.
            tail = slice(-span, None)
            logits = (
                reflex_weight[:, tail] * self.reflex_head(reflex_out[:, tail])
                + (1.0 - reflex_weight[:, tail]) * self.lm_head(mixed[:, tail])
            )

        # Confidence, stage by stage, so the weakest link is visible rather
        # than being averaged away by whatever came after it.
        rate = mean_spike_rate(self)
        self.uncertainty.reset()
        self.uncertainty.record("perception", spike_rate_confidence(rate))
        self.uncertainty.record("reflex", reflex_conf)
        self.uncertainty.record("salience", salience)
        if logits is not None:
            self.uncertainty.record("language", entropy_confidence(logits))

        trace = PathwayTrace(
            salience=float(salience.detach().mean()),
            reflex_confidence=float(reflex_conf.detach().mean()),
            conscious_ran=fraction > 0.0,
            conscious_fraction=fraction,
            reflex_weight=float(reflex_weight.detach().mean()),
            extras={
                "spike_rate": rate,
                "confidence": self.uncertainty.overall(),
                "bottleneck_stage": self.uncertainty.bottleneck(),
            },
        )

        if write_memory:
            with torch.no_grad():
                surprise = torch.zeros(summary.shape[0], device=summary.device)
                importance = self.memory.score_importance(
                    summary, surprise, salience=salience.reshape(-1)
                )
                self.memory.remember(summary, importance)

        out = BrainOutput(
            logits=logits,
            summary=summary,
            states=sub_out,
            trace=trace,
            aux_loss=self.conscious.aux_loss(),
            memory_info=mem_info,
        )
        if include_motor:
            out.motor = self.motor(mixed)

        self.memory.working.detach()
        return out

    # -- language modelling ----------------------------------------------
    def lm_loss(
        self,
        tokens: torch.Tensor,
        aux_weight: float = 0.01,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, BrainOutput]:
        """Next-token cross-entropy over a (B, S) batch."""
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        out = self(tokens=inputs, **kwargs)
        assert out.logits is not None
        loss = F.cross_entropy(
            out.logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_id,
        )
        if out.aux_loss is not None and aux_weight:
            loss = loss + aux_weight * out.aux_loss.to(loss.device)
        return loss, out

    def loss_fn(self, tokens: torch.Tensor) -> torch.Tensor:
        """Bare loss callable."""
        return self.lm_loss(tokens)[0]

    def world_loss_from(self, out: BrainOutput):
        """Dynamics loss taken from a forward pass that already happened.

        The temporal signal comes from *inside* the sequence -- the pooled
        state of the first half must predict the pooled state of the second --
        so the world model trains on the same forward pass as the language
        head instead of costing a second one.  Consecutive training batches
        would be the obvious alternative and are worthless here: shuffled text
        windows have no temporal relationship to learn.
        """
        if out.states is None or out.states.shape[1] < 2:
            return None, {}
        half = out.states.shape[1] // 2
        before = out.states[:, :half].mean(dim=1)
        after = out.states[:, half:].mean(dim=1)
        return self.world.loss(before, after.detach())

    def sleep_step(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """One replay step: language loss plus the importance scorer's own loss.

        The scorer cannot learn from the language objective -- retrieval is
        non-differentiable on purpose -- so it is trained here against a
        prioritised-replay target: an episode is important to the degree the
        model still gets it wrong.  Both objectives share one forward pass.
        """
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        out = self(tokens=inputs, write_memory=False)
        assert out.logits is not None and out.summary is not None

        per_token = F.cross_entropy(
            out.logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_id,
            reduction="none",
        ).view(targets.shape)
        mask = (targets != self.tokenizer.pad_id).float()
        per_sample = (per_token * mask).sum(1) / mask.sum(1).clamp_min(1.0)

        lm = per_sample.mean()
        if out.aux_loss is not None:
            lm = lm + 0.01 * out.aux_loss.to(lm.device)

        detached = per_sample.detach()
        target = (detached / detached.max().clamp_min(1e-6)).clamp(0.0, 1.0)
        predicted = self.memory.score_importance(
            out.summary,
            surprise=detached.clamp(0, 10) / 10.0,
            salience=out.trace.salience,
        )
        importance_loss = F.mse_loss(predicted, target)

        return lm + 0.1 * importance_loss, {
            "lm": float(lm.detach()),
            "importance": float(importance_loss.detach()),
        }

    # -- inference-time learning -----------------------------------------
    def learn_online(
        self,
        tokens: torch.Tensor,
        reward: float = 0.0,
        task_id: int = 0,
        verify_update: bool = True,
        max_relative_norm: float = 0.01,
    ) -> Dict[str, float]:
        """Observe something, update a little, and remember it.

        Runs even in ``eval`` mode -- this is the continual-learning path, not
        a training step.  Weight movement is bounded by per-parameter plasticity
        and a trust region, and rolled back if it made the loss worse.
        """
        was_training = self.training
        self.train()  # gradients needed; gating stays soft for a stable signal

        # write_memory=False here: the episodic write is done explicitly below,
        # where the reward and the measured surprise are available to score it.
        loss, out = self.lm_loss(tokens, write_memory=False)
        surprise = torch.full(
            (tokens.shape[0],), float(loss.detach()), device=tokens.device
        )

        def recheck() -> float:
            with torch.no_grad():
                return float(self.lm_loss(tokens, write_memory=False)[0])

        stats = self.plasticity.online_step(
            self, loss,
            max_relative_norm=max_relative_norm,
            verify=recheck if verify_update else None,
        )

        with torch.no_grad():
            summary = out.summary
            assert summary is not None
            importance = self.memory.score_importance(
                summary,
                surprise=surprise.clamp(0, 10) / 10.0,
                reward=reward,
                salience=out.trace.salience,
            )
            self.memory.remember(summary, importance)
            self.replay.add(tokens, importance, task_id=task_id)
            novelty = self.memory.episodic.novelty(self.memory.key_of(summary))
            stats.update(self.growth.observe(novelty, self))

        stats["loss"] = float(loss.detach())
        stats["mean_importance"] = float(importance.mean())
        if not was_training:
            self.eval()
        return stats

    def should_sleep(self, idle: bool = False) -> Dict[str, float]:
        """Ask the executive whether it is time to consolidate.

        Four independent pressures -- replay buffer fill, unpromoted claims,
        weight drift and time since the last sleep -- so no single one can
        force a consolidation on its own.  Idleness multiplies rather than
        adds: the right moment is when nothing is being asked of you.
        """
        report = self.reality.report()
        return self.executive.sleep_pressure(
            replay_fill=self.replay.size / max(self.replay.capacity, 1),
            unconsolidated_claims=int(report["hypotheses"]),
            drift=float(self.plasticity.report()["mean_plasticity"] < 0.5),
            idle=idle,
        )

    def sleep(
        self,
        cycles: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: float = 3e-5,
    ) -> Dict[str, float]:
        """Offline consolidation: validate, replay, hygiene, re-anchor.

        Validation runs *first*.  Only claims that survived the Reality Cortex
        reach semantic memory, and only settled ones are ever allowed to
        influence the weights -- so a false statement heard yesterday cannot be
        written into the network overnight.

        With no arguments the executive sizes the cycle from current pressure
        rather than using a fixed number.
        """
        if cycles is None or batch_size is None:
            plan = self.executive.sleep_plan(self.should_sleep())
            cycles = cycles if cycles is not None else plan["cycles"]
            batch_size = batch_size if batch_size is not None else plan["batch_size"]

        promoted = self.validate_and_promote()

        report = self.sleeper.sleep(
            self,
            self.replay,
            self.plasticity,
            self.sleep_step,
            cycles=cycles,
            batch_size=batch_size,
            lr=lr,
            memory_controller=self.memory,
        )

        self.reality.decay()
        self.ledger.tick()
        self.executive.slept()
        report["claims_promoted"] = float(len(promoted))
        for key, value in self.reality.report().items():
            report[f"reality_{key}"] = value
        return report

    # -- generation ------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        self.eval()
        device = next(self.parameters()).device
        ids = self.tokenizer.encode(prompt, add_eos=False)
        seq = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            window = seq[:, -self.cfg.max_seq_len:]
            out = self(tokens=window, use_memory=True)
            assert out.logits is not None
            logits = out.logits[0, -1] / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, logits.numel())
                thresh = logits.topk(k).values[-1]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if int(nxt) == self.tokenizer.eos_id:
                break
            seq = torch.cat([seq, nxt.view(1, 1)], dim=1)

        return self.tokenizer.decode(seq[0, len(ids):].tolist())

    # -- embodiment ------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        images: Optional[torch.Tensor] = None,
        touch: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """One perceive-decide-move cycle, through the safety controller.

        ``joint_targets`` is what will actually be sent;
        ``requested_joint_targets`` is what the network asked for.  When they
        differ the policy tried to leave the envelope, and that is worth
        surfacing rather than silently correcting.
        """
        self.eval()
        out = self(
            tokens=tokens, images=images, touch=touch, include_motor=True, **kwargs
        )
        # Snap to positions each joint can actually reach before the safety
        # check.  A stepper cannot sit between steps, and letting the driver
        # round silently makes commanded and achieved disagree -- which later
        # shows up as unexplained body-model error.
        requested = self.actuators.quantise(out.motor["joint_targets"])
        requested_effort = out.motor["effort"]

        ctx = dict(context or {})
        ctx.setdefault("salience", out.trace.salience)
        # End-to-end confidence, so `requires_confidence` can veto an action
        # the pipeline was never sure about.  This is the point of propagating
        # uncertainty: it stops being a diagnostic and starts being a brake.
        ctx.setdefault("confidence", out.trace.extras.get("confidence", 1.0))
        safe, safe_effort, report = self.safety.filter(
            requested, requested_effort, ctx
        )
        return {
            "joint_targets": safe,
            "effort": safe_effort,
            "requested_joint_targets": requested,
            "requested_effort": requested_effort,
            "safety": report,
            "reflex_only": not out.trace.conscious_ran,
            "confidence": out.trace.extras.get("confidence", 1.0),
            "uncertainty": self.uncertainty.report(),
            "trace": out.trace,
        }

    def perceive(self, readings: Dict[str, torch.Tensor]):
        """Fuse typed sensor readings into the shared bus.

        ``readings`` maps sensor names from :mod:`brainai.sensors` to tensors.
        Streams at different rates are fused at the slowest present rate, and
        each stream's confidence -- from its noise floor, drift and any
        chemical-recovery dead time -- is recorded so downstream stages know
        which inputs to lean on.
        """
        if self.sensors is None:
            raise RuntimeError(
                "no sensor suite configured; build the model with "
                "BrainConfig.for_scale(..., sensor_profile=('imu', 'lidar_2d'))"
            )
        fused, confidence = self.sensors(readings)
        for name, value in confidence.items():
            if name not in ("fusion_rate_hz", "overall"):
                self.uncertainty.record(f"sensor:{name}", value)
        return fused, confidence

    # -- prediction, curiosity, grounding --------------------------------
    @torch.no_grad()
    def predict(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        expect_claims: Sequence[int] = (),
        tolerance: float = 0.25,
    ):
        """Predict the next latent state, optionally staking beliefs on it.

        Claims listed in ``expect_claims`` are registered with the ledger; when
        the real observation arrives via :meth:`observe_reality`, whichever
        sources backed them gain or lose trust accordingly.
        """
        prediction = self.world(state, action)
        if expect_claims:
            self.ledger.expect(
                expect_claims, prediction.state[0], tolerance=tolerance
            )
        return prediction

    @torch.no_grad()
    def observe_reality(self, actual: torch.Tensor) -> Dict[str, float]:
        """Score open predictions against what happened, and update trust.

        This is what removes the oracle from ``observe_outcome``: the verdict
        comes from the environment disagreeing with a prediction, not from an
        operator labelling anything.
        """
        verdicts = self.ledger.resolve(actual)
        touched = self.reality.ground(verdicts)
        return {
            "resolved": float(len(verdicts)),
            "correct": float(sum(1 for _, ok in verdicts if ok)),
            "sources_updated": float(len(touched)),
            **{f"trust:{k}": v for k, v in touched.items()},
        }

    def world_loss(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """Self-supervised dynamics loss.  The next observation is the label."""
        return self.world.loss(state, next_state, action)

    @torch.no_grad()
    def intrinsic_reward(
        self, state: torch.Tensor, actual_next: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Curiosity: how much was there to learn from what just happened?"""
        prediction = self.world(state, action)
        error = self.world.prediction_error(prediction, actual_next)
        surprise = self.world.surprise(error)
        novelty = self.memory.episodic.novelty(self.memory.key_of(state))
        reward = self.curiosity.reward(surprise, novelty, prediction.expected_error)
        return {
            "intrinsic_reward": float(reward.mean()),
            "prediction_error": float(error.mean()),
            "surprise": float(surprise.mean()),
            "novelty": float(novelty.mean()),
            **self.curiosity.report(),
        }

    # -- planning --------------------------------------------------------
    def plan(
        self,
        goal: str,
        state: Optional[torch.Tensor] = None,
        action_for: Optional[Any] = None,
    ):
        """Decompose a goal and pick the variant that imagination likes best."""
        self.planner.library.learn_from_reality(self.reality)
        if state is None:
            state = torch.zeros(1, self.cfg.d_model)
        if action_for is None:
            def action_for(name: str):
                return self.embed_text(name)[: self.cfg.n_actuators].unsqueeze(0)
        return self.planner.best_plan(goal, state, self.world, action_for)

    # -- second thoughts -------------------------------------------------
    def deliberate_with_debate(self, tokens: torch.Tensor):
        """Let the routed experts argue instead of averaging them.

        Disagreement between experts is a measured uncertainty signal rather
        than a self-reported one.
        """
        x = self.encoder(tokens)
        sub_out, _ = self.subconscious(x)
        for block in self.conscious.reason.blocks:
            if block.uses_experts:
                return self.debate.deliberate(
                    block.ffn, self.conscious.reason.in_proj(sub_out)
                )
        return None

    def critique(self, prompt: str, rounds: int = 3, max_new_tokens: int = 48):
        """Generate, criticise, revise; return the best round, not the last."""
        loop = SelfCritique(
            generate=lambda p: self.generate(p, max_new_tokens=max_new_tokens),
            critic=perplexity_critic(self),
            rounds=rounds,
        )
        return loop.run(prompt), loop.report()

    # -- knowledge -------------------------------------------------------
    def observe(
        self,
        subject: str,
        relation: str,
        obj: str,
        source: str,
        supports: bool = True,
        text: str = "",
        note: str = "",
    ):
        """Submit one piece of evidence to the Reality Cortex.

        This does not make the system believe anything.  Belief requires enough
        independent corroboration to clear the promotion thresholds.
        """
        return self.reality.submit(
            subject, relation, obj, source,
            supports=supports, text=text, note=note,
        )

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        """A (d_model,) summary vector for a short string."""
        was_training = self.training
        self.eval()
        toks = self.tokenizer.encode_batch([text], min(64, self.cfg.max_seq_len))
        summary = self(tokens=toks, write_memory=False).summary
        if was_training:
            self.train()
        assert summary is not None
        return summary[0]

    def validate_and_promote(self):
        """Move sufficiently corroborated hypotheses into semantic memory."""
        return self.reality.promote(self.memory, embed=self.embed_text)

    # -- reporting -------------------------------------------------------
    def energy_report(self) -> Dict[str, float]:
        """Sparsity statistics from the most recent forward pass."""
        rate = mean_spike_rate(self)
        report = {
            "mean_spike_rate": rate,
            "sparsity": 1.0 - rate,
            "snn_timesteps": float(self.cfg.timesteps),
            "forward_passes": float(self.forward_passes),
        }
        report.update(self.growth.capacity(self))
        return report

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def status(self) -> str:
        p = self.plasticity.report()
        e = self.energy_report()
        m = self.memory.stats()
        r = self.reality.report()
        s = self.safety.stats()
        g = self.executive.report()
        l = self.ledger.report()
        c = self.curiosity.report()
        spikes = (
            f"{e['mean_spike_rate']:.3f} (sparsity {e['sparsity']:.1%})"
            if self.forward_passes > 0
            else "no forward pass recorded yet"
        )
        return "\n".join([
            attribution_line(self.name),
            f"parameters       {self.parameter_count():,}",
            f"spike rate       {spikes}",
            f"experts active   {int(e['active_experts'])}/{int(e['total_experts'])}",
            f"weights          {int(p['tracked_weights']):,} tracked, "
            f"{p['protected_fraction']:.1%} protected",
            f"consolidations   {int(p['consolidations'])}",
            f"episodic memory  {int(m['episodic']['entries'])} entries",
            f"semantic memory  {int(m['semantic']['entries'])} entries",
            f"replay buffer    {int(self.replay.stats()['entries'])} sequences",
            f"beliefs          {int(r['believed'])} believed, "
            f"{int(r['hypotheses'])} unresolved, {int(r['obsolete'])} retired "
            f"({int(r['sources'])} sources)",
            f"safety           {int(s['commands'])} commands, "
            f"{int(s['interventions'])} interventions, {int(s['vetoes'])} vetoed"
            + ("  [E-STOP]" if self.safety.estop_engaged else ""),
            f"goals            {int(g['goals'])} total, {int(g['active'])} active, "
            f"{int(g['suspended'])} suspended, {int(g['switches'])} switches",
            f"prediction       {int(l['resolved'])} resolved "
            f"({l['accuracy']:.0%} correct), {int(l['open'])} open",
            f"curiosity        boredom {c['boredom']:.2f}"
            + ("  [exploring]" if self.curiosity.exploring else ""),
        ])

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "identity": self.cfg.identity.to_dict(),
                "config": self.cfg.to_dict(),
                "state_dict": self.state_dict(),
                # Beliefs are Python objects, not tensors, so they ride
                # alongside the state dict.  What the robot believes has to
                # survive a restart or every reboot is an amnesia event.
                "reality": self.reality.to_dict(),
                "attribution": attribution_line(self.name),
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: str | Path, map_location: str = "cpu", strict: bool = True
    ) -> "BrainAI":
        """Load a checkpoint, refusing any whose attribution was stripped."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        verify(blob.get("identity", {}))
        cfg = BrainConfig.from_dict(blob["config"])
        model = cls(cfg)
        model.load_state_dict(blob["state_dict"], strict=strict)
        if "reality" in blob:
            model.reality = RealityCortex.from_dict(blob["reality"])
        return model
