# BrainAI - built by Mohammed Zahid.
"""Per-modality encoders feeding one shared spike representation.

Each modality gets its own front end and its own spiking stack, then projects
onto the common ``d_model`` bus with a modality-type embedding added.  Once on
the bus, a tactile event and a word are the same kind of object as far as the
pathways and the memory hierarchy are concerned -- which is what allows an
episode to be stored as a single linked multimodal memory rather than as four
unrelated embeddings.

Video shares the vision stack and patch embedding with still images and adds
only temporal position, so a frame is not re-learned from scratch.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .attention import RMSNorm
from .blocks import SpikingStack
from .config import BrainConfig

MODALITY_IDS: Dict[str, int] = {
    "text": 0, "image": 1, "video": 2, "audio": 3, "touch": 4
}


class TextEncoder(nn.Module):
    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.max_seq_len, cfg.d_model) * 0.02)
        self.stack = SpikingStack("language", cfg, causal=True)
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        s = tokens.shape[1]
        h = self.embed(tokens) + self.pos[:s]
        return self.norm(self.stack(h))


class ImageEncoder(nn.Module):
    """Patch embedding plus a non-causal vision stack."""

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid = cfg.image_size // cfg.patch_size
        self.n_patches = self.grid * self.grid

        self.patch = nn.Conv2d(
            3, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.pos = nn.Parameter(torch.randn(self.n_patches, cfg.d_model) * 0.02)
        self.stack = SpikingStack("vision", cfg, causal=False)
        self.norm = RMSNorm(cfg.d_model)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, n_patches, d)."""
        return self.patch(images).flatten(2).transpose(1, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.patchify(images) + self.pos
        return self.norm(self.stack(h))


class VideoEncoder(nn.Module):
    """Frames through the shared vision front end, plus temporal position.

    Also emits a motion channel -- the frame-to-frame difference -- because a
    spiking network is naturally a change detector and throwing that away would
    waste the architecture.
    """

    def __init__(self, cfg: BrainConfig, vision: ImageEncoder) -> None:
        super().__init__()
        self.vision = vision
        self.max_frames = cfg.video_frames
        self.time_pos = nn.Parameter(
            torch.randn(cfg.video_frames, cfg.d_model) * 0.02
        )
        self.motion = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.stack = SpikingStack("vision", cfg, causal=False)
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``video``: (B, T, 3, H, W) -> (B, T * n_patches, d)."""
        b, t = video.shape[:2]
        t = min(t, self.max_frames)
        video = video[:, :t]

        flat = video.reshape(b * t, *video.shape[2:])
        patches = self.vision.patchify(flat)                    # (B*T, P, d)
        patches = patches + self.vision.pos
        patches = patches.view(b, t, -1, patches.shape[-1])
        patches = patches + self.time_pos[:t].view(1, t, 1, -1)

        # Motion: what changed since the previous frame.
        delta = torch.zeros_like(patches)
        delta[:, 1:] = patches[:, 1:] - patches[:, :-1]
        h = patches + self.motion(delta)

        h = h.reshape(b, -1, h.shape[-1])
        return self.norm(self.stack(h))


class AudioEncoder(nn.Module):
    """Strided 1-D convolution over a waveform or spectrogram frame."""

    def __init__(self, cfg: BrainConfig, in_channels: int = 1, stride: int = 16) -> None:
        super().__init__()
        self.front = nn.Conv1d(
            in_channels, cfg.d_model, kernel_size=stride * 2, stride=stride
        )
        self.stack = SpikingStack("audio", cfg, causal=True)
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio``: (B, C, L) -> (B, L', d)."""
        h = self.front(audio).transpose(1, 2)
        return self.norm(self.stack(h))


class TouchEncoder(nn.Module):
    """Tactile and proprioceptive vectors: joint angles, force, skin pressure."""

    def __init__(self, cfg: BrainConfig, n_sensors: int = 64) -> None:
        super().__init__()
        self.project = nn.Linear(n_sensors, cfg.d_model)
        self.stack = SpikingStack("touch", cfg, causal=True)
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, touch: torch.Tensor) -> torch.Tensor:
        """``touch``: (B, S, n_sensors) or (B, n_sensors) -> (B, S, d)."""
        if touch.dim() == 2:
            touch = touch.unsqueeze(1)
        return self.norm(self.stack(self.project(touch)))


class MultimodalEncoder(nn.Module):
    """Fuses whichever modalities are present into one token stream."""

    def __init__(self, cfg: BrainConfig, n_touch_sensors: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.modality_embed = nn.Embedding(len(MODALITY_IDS), cfg.d_model)
        self.norm = RMSNorm(cfg.d_model)

        self.text = TextEncoder(cfg)
        self.image = ImageEncoder(cfg) if "image" in cfg.modalities else None
        self.video = (
            VideoEncoder(cfg, self.image)
            if "video" in cfg.modalities and self.image is not None
            else None
        )
        self.audio = AudioEncoder(cfg) if "audio" in cfg.modalities else None
        self.touch = (
            TouchEncoder(cfg, n_touch_sensors) if "touch" in cfg.modalities else None
        )

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        touch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        def tag(h: torch.Tensor, modality: str) -> torch.Tensor:
            idx = torch.tensor(MODALITY_IDS[modality], device=h.device)
            return h + self.modality_embed(idx)

        # Perception first, language last.  Attention downstream is causal, so
        # this ordering is what lets a token attend back over the image it is
        # describing rather than the other way round.
        if images is not None and self.image is not None:
            parts.append(tag(self.image(images), "image"))
        if video is not None and self.video is not None:
            parts.append(tag(self.video(video), "video"))
        if audio is not None and self.audio is not None:
            parts.append(tag(self.audio(audio), "audio"))
        if touch is not None and self.touch is not None:
            parts.append(tag(self.touch(touch), "touch"))
        if tokens is not None:
            parts.append(tag(self.text(tokens), "text"))

        if not parts:
            raise ValueError("at least one modality must be provided")
        return self.norm(torch.cat(parts, dim=1))

    @staticmethod
    def text_span(tokens: Optional[torch.Tensor]) -> int:
        """Length of the trailing text segment of the fused stream.

        The language head scores only those positions, so image patches never
        become prediction targets.
        """
        return 0 if tokens is None else tokens.shape[1]
