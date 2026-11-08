# BrainAI - built by Mohammed Zahid.
"""BrainAI - a spiking multimodal cognitive architecture.

Built by Mohammed Zahid.  Default deployment name ``BrainAI``; rename any
instance with ``model.rename("Javri")`` without affecting authorship.

    from brainai import BrainAI, BrainConfig

    model = BrainAI(BrainConfig.for_scale("nano"))
    print(model.describe())
    loss, out = model.lm_loss(tokens)
"""

from .activation import AMSA, GatedAMSA, SwiGLU
from .actuators import STANDARD_ACTUATORS, ActuatorBank, ActuatorSpec, actuator
from .sensors import STANDARD_SENSORS, SensorSpec, SensorSuite, sensor
from .attention import RMSNorm, SpikingAttention, sparsemax
from .blocks import (
    MoEFFN,
    SpikingStack,
    SpikingTransformerBlock,
    mean_spike_rate,
)
from .config import BrainConfig, SectionSpec, available_scales
from .consolidation import ReplayBuffer, SleepConsolidator
from .encoders import (
    AudioEncoder,
    ImageEncoder,
    MultimodalEncoder,
    TextEncoder,
    TouchEncoder,
    VideoEncoder,
)
from .identity import AUTHOR, CODENAME, VERSION, AttributionError, Identity
from .memory import (
    MemoryController,
    SemanticMemory,
    SensoryBuffer,
    ShortTermMemory,
    VectorStore,
    WorkingMemory,
)
from .model import BrainAI, BrainOutput
from .neurons import ALIF, SpikingLSTMCell, spike
from .pathways import (
    ConsciousPathway,
    ExecutiveController,
    MotorCortex,
    PathwayTrace,
    ReflexPathway,
    SubconsciousPathway,
)
from .critique import (
    CritiqueRound,
    DebateResult,
    InternalDebate,
    SelfCritique,
    perplexity_critic,
)
from .executive import Executive, Goal, GoalStack
from .planner import HierarchicalPlanner, MethodLibrary, Plan, PlanNode
from .plasticity import GrowthController, PlasticityController
from .uncertainty import (
    Uncertain,
    UncertaintyTracker,
    entropy_confidence,
    margin_confidence,
    spike_rate_confidence,
)
from .world import (
    BodyModel,
    CuriosityDrive,
    Prediction,
    PredictionLedger,
    WorldModel,
)
from .reality import (
    BELIEVED,
    HYPOTHESIS,
    OBSOLETE,
    Claim,
    ClaimReport,
    Evidence,
    RealityCortex,
    Source,
)
from .safety import (
    Constraint,
    JointLimits,
    SafetyController,
    SafetyReport,
    default_constraints,
    effort_ceiling,
    human_in_envelope,
    requires_confidence,
)
from .tokenizer import ByteTokenizer

__author__ = AUTHOR
__version__ = VERSION

__all__ = [
    "AMSA", "GatedAMSA", "SwiGLU", "ALIF", "SpikingLSTMCell", "spike",
    "SensorSpec", "SensorSuite", "STANDARD_SENSORS", "sensor",
    "ActuatorSpec", "ActuatorBank", "STANDARD_ACTUATORS", "actuator",
    "SpikingAttention", "RMSNorm", "sparsemax",
    "SpikingTransformerBlock", "SpikingStack", "MoEFFN", "mean_spike_rate",
    "BrainConfig", "SectionSpec", "available_scales",
    "TextEncoder", "ImageEncoder", "VideoEncoder", "AudioEncoder",
    "TouchEncoder", "MultimodalEncoder",
    "VectorStore", "SemanticMemory", "SensoryBuffer", "ShortTermMemory",
    "WorkingMemory", "MemoryController",
    "ReflexPathway", "SubconsciousPathway", "ConsciousPathway",
    "ExecutiveController", "MotorCortex", "PathwayTrace",
    "PlasticityController", "GrowthController",
    "ReplayBuffer", "SleepConsolidator",
    "RealityCortex", "Claim", "ClaimReport", "Evidence", "Source",
    "HYPOTHESIS", "BELIEVED", "OBSOLETE",
    "WorldModel", "Prediction", "PredictionLedger", "CuriosityDrive", "BodyModel",
    "Executive", "Goal", "GoalStack",
    "HierarchicalPlanner", "MethodLibrary", "Plan", "PlanNode",
    "UncertaintyTracker", "Uncertain", "entropy_confidence",
    "margin_confidence", "spike_rate_confidence",
    "InternalDebate", "DebateResult", "SelfCritique", "CritiqueRound",
    "perplexity_critic",
    "SafetyController", "JointLimits", "Constraint", "SafetyReport",
    "default_constraints", "human_in_envelope", "effort_ceiling",
    "requires_confidence",
    "ByteTokenizer",
    "BrainAI", "BrainOutput",
    "Identity", "AttributionError", "AUTHOR", "CODENAME", "VERSION",
]
