# BrainAI - built by Mohammed Zahid.
"""Scale presets and hyper-parameters.

The architecture is described once and instantiated at four scales.  ``nano``
and ``small`` are trainable on a CPU; ``base`` wants a GPU; ``full`` is the
~400-layer humanoid-robot specification and is included so the design is
written down honestly, not because it fits on this machine.  Call
:meth:`BrainConfig.param_estimate` before trying to build one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .identity import Identity

# Canonical sections of the cognitive architecture.
SECTION_ROLES: Dict[str, str] = {
    "vision": "spatial perception encoder",
    "language": "symbolic / linguistic encoder",
    "audio": "acoustic encoder",
    "touch": "tactile and proprioceptive encoder",
    "reflex": "spinal cord - sub-100ms responses, bypasses reasoning",
    "subconscious": "always-on sparse pattern processing and filtering",
    "conscious": "gated deliberate reasoning and planning",
    "working_memory": "actively manipulated content",
    "ltm_interface": "read/write bridge to episodic and semantic stores",
    "planner": "multi-step action sequencing",
    "motor": "motor cortex - actuator command generation",
    "plasticity": "meta-controller deciding what is allowed to change",
    "executive": "arbitration between pathways and modules",
}

# scale -> section -> (layers, width, heads)
_PRESETS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "nano": {
        "vision": (2, 128, 4),
        "language": (2, 128, 4),
        "audio": (1, 64, 2),
        "touch": (1, 64, 2),
        "reflex": (1, 64, 2),
        "subconscious": (2, 128, 4),
        "conscious": (2, 128, 4),
        "working_memory": (1, 128, 4),
        "ltm_interface": (1, 128, 4),
        "planner": (1, 128, 4),
        "motor": (1, 64, 2),
        "plasticity": (1, 64, 2),
        "executive": (1, 64, 2),
    },
    "small": {
        "vision": (4, 256, 8),
        "language": (4, 256, 8),
        "audio": (2, 128, 4),
        "touch": (2, 128, 4),
        "reflex": (2, 128, 4),
        "subconscious": (4, 256, 8),
        "conscious": (6, 256, 8),
        "working_memory": (2, 256, 8),
        "ltm_interface": (2, 256, 8),
        "planner": (2, 256, 8),
        "motor": (2, 128, 4),
        "plasticity": (1, 128, 4),
        "executive": (2, 128, 4),
    },
    # ~100M parameters: the smallest size at which generation starts to look
    # like language rather than domain-flavoured character soup. Still trainable
    # on one GPU; on CPU it is for instantiation and short fine-tunes, not a
    # from-scratch run (see the note in `for_scale`).
    "compact": {
        "vision": (4, 352, 8),
        "language": (4, 352, 8),
        "audio": (2, 352, 8),
        "touch": (2, 352, 8),
        "reflex": (2, 352, 8),
        "subconscious": (4, 352, 8),
        "conscious": (6, 352, 8),
        "working_memory": (2, 352, 8),
        "ltm_interface": (2, 352, 8),
        "planner": (3, 352, 8),
        "motor": (2, 352, 8),
        "plasticity": (1, 352, 8),
        "executive": (2, 352, 8),
    },
    "base": {
        "vision": (8, 768, 12),
        "language": (8, 768, 12),
        "audio": (6, 512, 8),
        "touch": (4, 256, 8),
        "reflex": (2, 256, 8),
        "subconscious": (12, 768, 12),
        "conscious": (16, 1024, 16),
        "working_memory": (4, 1024, 16),
        "ltm_interface": (6, 768, 12),
        "planner": (8, 768, 12),
        "motor": (6, 512, 8),
        "plasticity": (3, 256, 8),
        "executive": (4, 512, 8),
    },
    # The humanoid target.  Narrow and deep, with knowledge in the memory
    # stores rather than the weights.
    #
    # The original specification asked for 96 conscious layers at width 65,536,
    # which the estimator puts at ~82T parameters -- roughly 50x the largest
    # models ever trained, for one robot.  Transformer cost grows with the
    # *square* of width, so that one number dominated everything else.  Keeping
    # the layer counts and cutting the widths preserves the depth and the
    # cognitive structure at ~1/40th the parameters, and this architecture's
    # whole argument is that lifelong knowledge belongs in episodic and
    # semantic memory, not compiled into weights.  `full_wide` below is the
    # original, kept for comparison.
    "full": {
        "vision": (32, 2048, 32),
        "language": (32, 2048, 32),
        "audio": (24, 1024, 16),
        "touch": (16, 512, 8),
        "reflex": (4, 512, 8),
        "subconscious": (64, 2048, 32),
        "conscious": (96, 4096, 32),
        "working_memory": (16, 2048, 32),
        "ltm_interface": (24, 2048, 32),
        "planner": (32, 2048, 32),
        "motor": (24, 1024, 16),
        "plasticity": (12, 512, 8),
        "executive": (16, 1024, 16),
    },
    # The as-originally-specified variant.  Kept so the comparison in the
    # README is reproducible, not because it should be built.
    "full_wide": {
        "vision": (32, 8192, 64),
        "language": (32, 8192, 64),
        "audio": (24, 4096, 32),
        "touch": (16, 2048, 16),
        "reflex": (4, 1024, 8),
        "subconscious": (64, 32768, 64),
        "conscious": (96, 65536, 128),
        "working_memory": (16, 16384, 64),
        "ltm_interface": (24, 8192, 64),
        "planner": (32, 16384, 64),
        "motor": (24, 8192, 32),
        "plasticity": (12, 2048, 16),
        "executive": (16, 4096, 32),
    },
}

_SCALE_GLOBALS: Dict[str, Dict[str, Any]] = {
    # timesteps >= 3 everywhere: the ALIF adaptation term is a[t] = rho.a[t-1]
    # + s[t-1], so with T=2 the decay rho only ever multiplies zeros and
    # threshold adaptation is inert.  T=3 is the smallest useful SNN.
    "nano": dict(d_model=128, timesteps=3, n_experts=4, expert_top_k=2,
                 max_seq_len=256, wm_slots=8, episodic_capacity=2048,
                 semantic_capacity=4096, sensory_span=4),
    "small": dict(d_model=256, timesteps=3, n_experts=8, expert_top_k=2,
                  max_seq_len=512, wm_slots=16, episodic_capacity=16384,
                  semantic_capacity=32768, sensory_span=8),
    "compact": dict(d_model=352, timesteps=3, n_experts=4, expert_top_k=2,
                    max_seq_len=1024, wm_slots=24, episodic_capacity=65536,
                    semantic_capacity=131072, sensory_span=12),
    "base": dict(d_model=768, timesteps=4, n_experts=16, expert_top_k=2,
                 max_seq_len=2048, wm_slots=32, episodic_capacity=262144,
                 semantic_capacity=524288, sensory_span=16),
    # Memory capacity is deliberately *not* reduced alongside the widths: the
    # knowledge has to live somewhere, and a store entry costs d_model floats
    # against a weight matrix's d_model squared.
    "full": dict(d_model=2048, timesteps=8, n_experts=16, expert_top_k=2,
                 max_seq_len=8192, wm_slots=128, episodic_capacity=50_000_000,
                 semantic_capacity=50_000_000, sensory_span=32),
    "full_wide": dict(d_model=4096, timesteps=8, n_experts=16, expert_top_k=2,
                      max_seq_len=8192, wm_slots=128,
                      episodic_capacity=10_000_000,
                      semantic_capacity=10_000_000, sensory_span=32),
}


# How many SpikingStacks each section is actually instantiated as.  Vision is
# built twice - ImageEncoder and VideoEncoder each own one.  working_memory and
# plasticity are not transformer stacks at all, so they cost nothing here.
STACK_COUNT: Dict[str, int] = {
    "vision": 2, "language": 1, "audio": 1, "touch": 1,
    "reflex": 1, "subconscious": 1, "conscious": 1,
    "ltm_interface": 1, "planner": 1, "motor": 1, "executive": 1,
}

#: Sections whose feed-forward is a mixture of experts.
EXPERT_SECTIONS = frozenset({"conscious"})


@dataclass(frozen=True)
class SectionSpec:
    """One section of the architecture."""

    name: str
    layers: int
    width: int
    heads: int

    @property
    def role(self) -> str:
        return SECTION_ROLES.get(self.name, "")

    def param_estimate(self, n_ffn: int = 1) -> int:
        """Parameters for this section.

        Per layer: 4w^2 for attention (q, k, v, o) and 12w^2 for each gated
        feed-forward (up and gate are w x 4w, down is 4w x w).  ``n_ffn`` is the
        number of feed-forward copies -- 1 normally, or the expert count in an
        MoE section.
        """
        per_layer = (4 + 12 * n_ffn) * self.width * self.width
        return STACK_COUNT.get(self.name, 0) * self.layers * per_layer

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "layers": self.layers,
                "width": self.width, "heads": self.heads}


@dataclass
class BrainConfig:
    """Full configuration for a BrainAI instance."""

    scale: str = "nano"
    d_model: int = 128
    vocab_size: int = 260
    max_seq_len: int = 256
    timesteps: int = 2

    sections: Dict[str, SectionSpec] = field(default_factory=dict)

    # Mixture of experts
    n_experts: int = 4
    expert_top_k: int = 2

    # Memory
    sensory_span: int = 4
    stm_slots: int = 32
    wm_slots: int = 8
    episodic_capacity: int = 2048
    semantic_capacity: int = 4096
    retrieval_top_k: int = 4

    # Spiking dynamics
    surrogate_slope: float = 2.0
    v_threshold: float = 1.0
    adapt_scale: float = 0.18

    # Plasticity / continual learning
    ewc_lambda: float = 12.0
    online_lr: float = 1e-3
    plasticity_floor: float = 0.05
    importance_threshold: float = 0.55
    novelty_threshold: float = 0.62

    # Gating
    conscious_gate_threshold: float = 0.5
    reflex_confidence_threshold: float = 0.85

    #: Feed-forward nonlinearity: "amsa" (the architecture's own, kept
    #: deliberately) or "swiglu" (faster, and better on every benchmark here).
    ffn: str = "amsa"
    attention: str = "softmax"  # or "sparsemax"
    dropout: float = 0.0

    # Embodiment
    n_actuators: int = 32
    #: "servo" gives every joint the same well-behaved envelope; "humanoid"
    #: mixes hydraulic legs, brushless arms, pneumatic hands and servo neck,
    #: each with its own limits.  Any key of STANDARD_ACTUATORS also works.
    actuator_profile: str = "servo"
    #: Named sensors from brainai.sensors, or () for the legacy generic touch
    #: vector.  Typed sensors carry rate, latency, noise and recovery time.
    sensor_profile: Tuple[str, ...] = ()
    image_size: int = 64
    patch_size: int = 8
    video_frames: int = 8

    # All five by default: a humanoid needs audio and proprioception, and the
    # audio/touch sections exist in every preset, so leaving them out of the
    # encoder set would silently drop those inputs on the floor.
    modalities: Tuple[str, ...] = ("text", "image", "video", "audio", "touch")
    identity: Identity = field(default_factory=Identity)

    # -- construction ----------------------------------------------------
    @classmethod
    def for_scale(cls, scale: str = "nano", **overrides: Any) -> "BrainConfig":
        if scale not in _PRESETS:
            raise ValueError(
                f"unknown scale {scale!r}; choose from {sorted(_PRESETS)}"
            )
        sections = {
            name: SectionSpec(name, layers, width, heads)
            for name, (layers, width, heads) in _PRESETS[scale].items()
        }
        kwargs: Dict[str, Any] = dict(_SCALE_GLOBALS[scale])
        kwargs.update(scale=scale, sections=sections)
        kwargs.update(overrides)
        return cls(**kwargs)

    def section(self, name: str) -> SectionSpec:
        try:
            return self.sections[name]
        except KeyError:
            raise KeyError(
                f"no section {name!r}; known sections: {sorted(self.sections)}"
            ) from None

    # -- reporting -------------------------------------------------------
    @property
    def total_layers(self) -> int:
        return sum(s.layers for s in self.sections.values())

    def param_estimate(self, active_only: bool = False) -> int:
        """Estimated parameter count.

        ``active_only=True`` counts just the top-k experts that actually run per
        token, which for an MoE model is the number that determines compute.
        The gap between the two is the whole point of routing.
        """
        total = 0
        for name, spec in self.sections.items():
            n_ffn = 1
            if name in EXPERT_SECTIONS:
                n_ffn = self.expert_top_k if active_only else self.n_experts
            total += spec.param_estimate(n_ffn)
            if spec.width != self.d_model:
                # in_proj / out_proj onto the shared bus
                total += STACK_COUNT.get(name, 0) * 2 * spec.width * self.d_model
        # token embedding + language head + reflex head, plus positions
        total += 3 * self.vocab_size * self.d_model
        total += self.max_seq_len * self.d_model
        return total

    def describe(self) -> str:
        lines = [
            self.identity.banner(),
            "",
            f"scale={self.scale}  d_model={self.d_model}  "
            f"snn_timesteps={self.timesteps}  attention={self.attention}",
            f"{'section':<16}{'layers':>7}{'width':>8}{'heads':>7}  role",
            "-" * 88,
        ]
        for name in SECTION_ROLES:
            if name not in self.sections:
                continue
            s = self.sections[name]
            lines.append(
                f"{s.name:<16}{s.layers:>7}{s.width:>8}{s.heads:>7}  {s.role}"
            )
        lines.append("-" * 88)
        total = self.param_estimate()
        active = self.param_estimate(active_only=True)
        lines.append(
            f"{'TOTAL':<16}{self.total_layers:>7} layers   "
            f"~{total / 1e6:,.1f}M parameters "
            f"(~{active / 1e6:,.1f}M active per token, top-{self.expert_top_k} routing)"
        )
        lines.append(
            f"memory: {self.wm_slots} working slots, "
            f"{self.episodic_capacity:,} episodic, "
            f"{self.semantic_capacity:,} semantic"
        )
        lines.append(
            f"experts: {self.n_experts} (top-{self.expert_top_k} routed)"
        )
        return "\n".join(lines)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "timesteps": self.timesteps,
            "sections": {k: v.to_dict() for k, v in self.sections.items()},
            "n_experts": self.n_experts,
            "expert_top_k": self.expert_top_k,
            "sensory_span": self.sensory_span,
            "stm_slots": self.stm_slots,
            "wm_slots": self.wm_slots,
            "episodic_capacity": self.episodic_capacity,
            "semantic_capacity": self.semantic_capacity,
            "retrieval_top_k": self.retrieval_top_k,
            "surrogate_slope": self.surrogate_slope,
            "v_threshold": self.v_threshold,
            "adapt_scale": self.adapt_scale,
            "ewc_lambda": self.ewc_lambda,
            "online_lr": self.online_lr,
            "plasticity_floor": self.plasticity_floor,
            "importance_threshold": self.importance_threshold,
            "novelty_threshold": self.novelty_threshold,
            "conscious_gate_threshold": self.conscious_gate_threshold,
            "reflex_confidence_threshold": self.reflex_confidence_threshold,
            "attention": self.attention,
            "dropout": self.dropout,
            "n_actuators": self.n_actuators,
            "actuator_profile": self.actuator_profile,
            "sensor_profile": list(self.sensor_profile),
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "video_frames": self.video_frames,
            "modalities": list(self.modalities),
            "identity": self.identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainConfig":
        data = dict(data)
        sections = {
            k: SectionSpec(**v) for k, v in data.pop("sections", {}).items()
        }
        identity = Identity.from_dict(data.pop("identity", {}))
        modalities = tuple(data.pop("modalities", ("text",)))
        return cls(sections=sections, identity=identity,
                   modalities=modalities, **data)


def available_scales() -> Tuple[str, ...]:
    return tuple(_PRESETS)
