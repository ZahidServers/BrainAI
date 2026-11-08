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

    def chunks(self, text: str, seq_len: int, stride: int | None = None) -> torch.Tensor:
        """Slice a long document into (N, seq_len) training windows."""
        stride = stride or seq_len
        ids = list(text.encode("utf-8"))
        windows = [
            ids[i : i + seq_len]
            for i in range(0, max(1, len(ids) - seq_len + 1), stride)
        ]
        windows = [w for w in windows if len(w) == seq_len]
        if not windows:
            padded = ids + [PAD] * (seq_len - len(ids))
            windows = [padded[:seq_len]]
        return torch.tensor(windows, dtype=torch.long)
