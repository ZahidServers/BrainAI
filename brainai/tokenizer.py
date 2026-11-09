# BrainAI - built by Mohammed Zahid.
"""Byte-level tokenizer.

Deliberately the simplest thing that works: 256 byte values plus four control
tokens.  No vocabulary file to ship, no OOV, every language and every binary
sensor payload encodes without special handling -- which matters for a robot
that will see tactile and telemetry streams alongside text.

The cost is longer sequences than a BPE tokenizer.  At ``base`` scale and above,
swap this for a learned BPE vocabulary; nothing else in the model depends on the
tokenizer beyond ``vocab_size``.
"""

from __future__ import annotations

from typing import Iterable, List

import torch

PAD, BOS, EOS, SEP = 256, 257, 258, 259
VOCAB_SIZE = 260

SPECIAL_NAMES = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", SEP: "<sep>"}


class ByteTokenizer:
    """UTF-8 byte tokenizer with padding and batching helpers."""

    pad_id = PAD
    bos_id = BOS
    eos_id = EOS
    sep_id = SEP
    vocab_size = VOCAB_SIZE

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS)
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i >= 256:
                if not skip_special:
                    out.extend(SPECIAL_NAMES.get(i, "<unk>").encode("utf-8"))
                continue
            out.append(i)
        return out.decode("utf-8", errors="replace")

    def encode_batch(
        self, texts: List[str], max_len: int, add_bos: bool = True
    ) -> torch.Tensor:
        """Right-pad a list of strings into a (B, max_len) tensor."""
        rows = []
        for t in texts:
            ids = self.encode(t, add_bos=add_bos)[:max_len]
            ids += [PAD] * (max_len - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)

    def chunks(self, text, seq_len: int, stride: int | None = None,
               dtype: torch.dtype = torch.long) -> torch.Tensor:
        """Slice a long document into (N, seq_len) training windows.

        ``text`` may be ``str`` or raw ``bytes``. Built on numpy views, so a
        500 MB file costs ~its own size rather than the many GB a Python list of
        ints would (that list is what killed Colab sessions on large corpora).
        ``dtype=torch.int16`` stores windows compactly; cast to long per batch.
        """
        import numpy as np
        stride = stride or seq_len
        raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        ids = np.frombuffer(raw, dtype=np.uint8)
        if len(ids) < seq_len:
            padded = np.full(seq_len, PAD, dtype=np.int16)
            padded[: len(ids)] = ids
            return torch.from_numpy(padded[None, :].astype(np.int64)).to(dtype)
        starts = np.arange(0, len(ids) - seq_len + 1, stride)
        view = np.lib.stride_tricks.sliding_window_view(ids, seq_len)[starts]
        return torch.from_numpy(np.ascontiguousarray(view).astype(np.int16)).to(dtype)
