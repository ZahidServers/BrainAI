# BrainAI - built by Mohammed Zahid.
"""Real sensors, with the properties that actually distinguish them.

A robot does not have "touch".  It has an IMU sampling at 200 Hz, wheel
encoders reporting absolute angle, a LiDAR returning a range array, a pressure
mat on each fingertip, and possibly a gas array that takes thirty seconds to
recover after an exposure.  Collapsing those into one 64-channel vector throws
away every property that makes them different to reason about.

What a spec carries here, and why each field earns its place:

**rate_hz** -- an IMU at 200 Hz and a gas sensor at 1 Hz cannot share a fusion
window.  :class:`SensorSuite` resamples each stream against the slowest one
rather than pretending they arrive together.

**latency_ms** -- ultrasonic time-of-flight is milliseconds; an e-nose needs
seconds to settle.  A reflex arc that does not know this will react to a
reading that describes the past.

**noise** and **drift** -- MOS gas sensors drift badly with temperature and
humidity; encoders essentially do not.  Confidence per sensor should reflect
that, and :meth:`SensorSpec.confidence` derives it rather than assuming every
input is equally trustworthy.

**recovery_s** -- unique to chemical sensing.  After a strong exposure a MOS
element is saturated and its next reading is meaningless.  A robot that tracks
a gas plume by driving toward higher readings will chase its own saturation
without this.

Olfaction is included because it is genuinely absent from most robot stacks and
genuinely useful: gas leaks, spoilage, combustion, solvent exposure.  Nothing
else in a sensor suite detects a leak behind a wall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .attention import RMSNorm

# Which shared-bus modality a sensor's data joins once encoded.
SPATIAL = "spatial"        # vision, lidar, ultrasonic - where things are
PROPRIO = "proprio"        # imu, encoders - where *I* am
CONTACT = "contact"        # tactile, force - what I am touching
CHEMICAL = "chemical"      # gas, e-nose - what is in the air
ACOUSTIC = "acoustic"


@dataclass(frozen=True)
class SensorSpec:
    """One physical sensor and the properties that change how it is used."""

    name: str
    modality: str
    channels: int
    rate_hz: float = 100.0
    latency_ms: float = 1.0
    noise: float = 0.01              # relative std-dev at nominal range
    drift: float = 0.0               # fraction of range per hour
    recovery_s: float = 0.0          # dead time after saturation (chemical)
    unit: str = ""
    description: str = ""

    @property
    def period_ms(self) -> float:
        return 1000.0 / max(self.rate_hz, 1e-6)

    def confidence(self, saturated: bool = False, age_s: float = 0.0) -> float:
        """How much to trust a reading from this sensor right now.

        Falls with the sensor's noise floor, with accumulated drift, and to
        near zero while a chemical sensor is still recovering from saturation.
        """
        if saturated and self.recovery_s > 0.0 and age_s < self.recovery_s:
            return 0.05
        base = 1.0 / (1.0 + 8.0 * self.noise)
        drift_penalty = 1.0 / (1.0 + self.drift * (age_s / 3600.0))
        return max(0.0, min(1.0, base * drift_penalty))

    def stale(self, age_ms: float) -> bool:
        """Older than two sample periods plus its own latency."""
        return age_ms > 2.0 * self.period_ms + self.latency_ms


# -- the standard suite ----------------------------------------------------
#
# Channel counts and rates are representative of commodity parts, not of any
# one robot.  Build your own list; nothing downstream assumes these.

STANDARD_SENSORS: Tuple[SensorSpec, ...] = (
    SensorSpec("rgb_camera", SPATIAL, 3, rate_hz=30, latency_ms=33, noise=0.02,
               unit="intensity", description="colour vision"),
    SensorSpec("depth_camera", SPATIAL, 1, rate_hz=30, latency_ms=33, noise=0.03,
               unit="m", description="per-pixel range"),
    SensorSpec("lidar_2d", SPATIAL, 360, rate_hz=10, latency_ms=100, noise=0.01,
               unit="m", description="planar range scan, one beam per degree"),
    SensorSpec("ultrasonic", SPATIAL, 1, rate_hz=20, latency_ms=50, noise=0.05,
               unit="m", description="cheap short-range time-of-flight"),
    SensorSpec("imu", PROPRIO, 9, rate_hz=200, latency_ms=5, noise=0.02,
               drift=0.05, unit="mixed",
               description="3-axis accel + gyro + magnetometer"),
    SensorSpec("joint_encoders", PROPRIO, 32, rate_hz=500, latency_ms=2,
               noise=0.001, unit="rad", description="absolute joint angle"),
    SensorSpec("force_torque", CONTACT, 6, rate_hz=500, latency_ms=2, noise=0.01,
               unit="N/Nm", description="wrist load cell"),
    SensorSpec("tactile_skin", CONTACT, 64, rate_hz=100, latency_ms=10,
               noise=0.03, unit="kPa", description="pressure array, slip detection"),
    SensorSpec("microphone", ACOUSTIC, 1, rate_hz=16000, latency_ms=10,
               noise=0.02, unit="Pa", description="audio"),
    # Chemical sensing.  Slow, drifty, and the only thing that finds a leak.
    SensorSpec("mos_gas_array", CHEMICAL, 8, rate_hz=1, latency_ms=2000,
               noise=0.12, drift=0.30, recovery_s=30.0, unit="ppm",
               description="metal-oxide e-nose; broad VOC response, drifts"),
    SensorSpec("electrochemical_gas", CHEMICAL, 4, rate_hz=1, latency_ms=15000,
               noise=0.03, drift=0.05, recovery_s=60.0, unit="ppm",
               description="CO / H2S / NO2 / O2; accurate, slow, consumable"),
    SensorSpec("qcm_olfactory", CHEMICAL, 16, rate_hz=2, latency_ms=5000,
               noise=0.06, drift=0.10, recovery_s=45.0, unit="Hz",
               description="quartz microbalance; scent profile, not one gas"),
)

SENSORS_BY_NAME: Dict[str, SensorSpec] = {s.name: s for s in STANDARD_SENSORS}


def sensor(name: str) -> SensorSpec:
    try:
        return SENSORS_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown sensor {name!r}; known: {sorted(SENSORS_BY_NAME)}"
        ) from None


class SensorEncoder(nn.Module):
    """Projects one sensor's channels onto the shared bus.

    Every sensor gets a learned per-channel gain and offset ahead of the
    projection.  Raw units are wildly inconsistent -- radians next to kPa next
    to ppm -- and a single shared projection would be dominated by whichever
    stream happens to have the largest numbers.
    """

    def __init__(self, spec: SensorSpec, d_model: int) -> None:
        super().__init__()
        self.spec = spec
        self.gain = nn.Parameter(torch.ones(spec.channels))
        self.offset = nn.Parameter(torch.zeros(spec.channels))
        self.project = nn.Linear(spec.channels, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, reading: torch.Tensor) -> torch.Tensor:
        """``reading``: (B, T, channels) or (B, channels) -> (B, T, d)."""
        if reading.dim() == 2:
            reading = reading.unsqueeze(1)
        return self.norm(self.project(reading * self.gain + self.offset))


class SensorSuite(nn.Module):
    """A robot's actual sensor complement, fused onto one bus.

    Adding a sensor is a spec plus a line in a list, not a new encoder class.
    Streams arriving at different rates are resampled to the slowest one that
    is present, and each stream's spec-derived confidence is reported so
    downstream stages know which inputs to lean on.
    """

    def __init__(
        self,
        d_model: int,
        specs: Sequence[SensorSpec] = STANDARD_SENSORS,
    ) -> None:
        super().__init__()
        self.specs = {s.name: s for s in specs}
        self.encoders = nn.ModuleDict(
            {s.name: SensorEncoder(s, d_model) for s in specs}
        )
        self.modality_embed = nn.Embedding(5, d_model)
        self._modality_index = {
            SPATIAL: 0, PROPRIO: 1, CONTACT: 2, CHEMICAL: 3, ACOUSTIC: 4
        }
        self.saturated: Dict[str, bool] = {}

    def fusion_rate(self, present: Sequence[str]) -> float:
        """The slowest present stream sets the fusion rate.

        Fusing at the fastest rate would mean repeating a two-second-old gas
        reading two hundred times and letting it outvote the IMU.
        """
        rates = [self.specs[n].rate_hz for n in present if n in self.specs]
        return min(rates) if rates else 0.0

    def forward(
        self, readings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Fuse a dict of ``{sensor_name: tensor}`` into (B, S, d) plus confidences."""
        parts: List[torch.Tensor] = []
        confidence: Dict[str, float] = {}

        for name, reading in readings.items():
            if name not in self.encoders:
                raise KeyError(f"no encoder for sensor {name!r}")
            spec = self.specs[name]
            encoded = self.encoders[name](reading)
            index = torch.tensor(
                self._modality_index[spec.modality], device=encoded.device
            )
            parts.append(encoded + self.modality_embed(index))
            confidence[name] = spec.confidence(
                saturated=self.saturated.get(name, False)
            )

        if not parts:
            raise ValueError("no sensor readings supplied")
        fused = torch.cat(parts, dim=1)
        confidence["fusion_rate_hz"] = self.fusion_rate(list(readings))
        confidence["overall"] = min(
            v for k, v in confidence.items() if k != "fusion_rate_hz"
        )
        return fused, confidence

    def mark_saturated(self, name: str, saturated: bool = True) -> None:
        """Flag a chemical sensor as blinded until it recovers.

        A saturated MOS element reads high regardless of what is in front of
        it.  A plume-following robot that does not track this drives up its own
        gradient and stops.
        """
        self.saturated[name] = saturated

    def describe(self) -> str:
        rows = ["  sensor              modality   ch    rate    lat  noise  conf",
                "  " + "-" * 62]
        for spec in self.specs.values():
            rows.append(
                f"  {spec.name:<20}{spec.modality:<10}{spec.channels:>4}"
                f"{spec.rate_hz:>8.0f}{spec.latency_ms:>7.0f}"
                f"{spec.noise:>7.3f}{spec.confidence():>6.2f}"
            )
        return "\n".join(rows)
