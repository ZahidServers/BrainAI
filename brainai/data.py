# BrainAI - built by Mohammed Zahid.
"""Corpora for training and for the continual-learning experiment.

Two Project Gutenberg texts in different languages.  At byte level English and
French have clearly different statistics, so "did task B damage task A" is a
question with an unambiguous numerical answer -- which is the point of the
continual-learning benchmark.  Both are public domain.

Downloads are cached under ``data/``.  If the network is unavailable a small
built-in corpus is used instead so nothing in the repo hard-fails offline.
"""

from __future__ import annotations

import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

from .tokenizer import ByteTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCES: Dict[str, Tuple[str, str]] = {
    # task name -> (url, human description)
    "english": (
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "Austen, Pride and Prejudice (English)",
    ),
    "french": (
        "https://www.gutenberg.org/cache/epub/4650/pg4650.txt",
        "Voltaire, Candide (French)",
    ),
}

_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END = "*** END OF THE PROJECT GUTENBERG EBOOK"

_FALLBACK = {
    "english": (
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife. However "
        "little known the feelings or views of such a man may be on his first "
        "entering a neighbourhood, this truth is so well fixed in the minds of "
        "the surrounding families, that he is considered as the rightful "
        "property of some one or other of their daughters. "
    ) * 200,
    "french": (
        "Il y avait en Westphalie, dans le chateau de monsieur le baron de "
        "Thunder-ten-tronckh, un jeune garcon a qui la nature avait donne les "
        "moeurs les plus douces. Sa physionomie annoncait son ame. Il avait le "
        "jugement assez droit, avec l'esprit le plus simple. "
    ) * 200,
}


def _strip_gutenberg(text: str) -> str:
    """Drop the licence header and footer, keep the book."""
    start = text.find(_START)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find(_END)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def fetch(task: str, cache_dir: Path | None = None) -> str:
    """Return the raw text for ``task``, downloading and caching on first use."""
    if task not in SOURCES:
        raise KeyError(f"unknown task {task!r}; known: {sorted(SOURCES)}")

    cache_dir = cache_dir or DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{task}.txt"

    if path.exists():
        return path.read_text(encoding="utf-8")

    url, _ = SOURCES[task]
    try:
        ctx = ssl._create_unverified_context()
        raw = urllib.request.urlopen(url, timeout=60, context=ctx).read()
        text = _strip_gutenberg(raw.decode("utf-8", errors="replace"))
    except Exception as exc:  # offline, blocked, mirror down
        print(f"  [data] download failed for {task} ({exc}); using fallback text")
        text = _FALLBACK[task]

    path.write_text(text, encoding="utf-8")
    return text


@dataclass
class Corpus:
    """A tokenised task with a held-out split."""

    name: str
    train: torch.Tensor       # (N, seq_len)
    val: torch.Tensor         # (M, seq_len)
    description: str = ""

    def batch(self, size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.randint(
            0, self.train.shape[0], (size,), generator=generator
        )
        return self.train[idx]

    def val_batches(self, size: int, limit: int = 8):
        for i in range(0, min(self.val.shape[0], size * limit), size):
            yield self.val[i : i + size]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Corpus({self.name}: {self.train.shape[0]} train / "
            f"{self.val.shape[0]} val windows of {self.train.shape[1]})"
        )


def load_corpus(
    task: str,
    seq_len: int = 128,
    val_fraction: float = 0.1,
    max_windows: int = 4000,
    stride: int | None = None,
) -> Corpus:
    """Tokenise a task into fixed-length training windows."""
    text = fetch(task)
    tok = ByteTokenizer()
    windows = tok.chunks(text, seq_len, stride=stride or seq_len // 2)

    # Deterministic shuffle so the split is stable across runs.
    g = torch.Generator().manual_seed(1234)
    windows = windows[torch.randperm(windows.shape[0], generator=g)]
    windows = windows[:max_windows]

    n_val = max(1, int(windows.shape[0] * val_fraction))
    return Corpus(
        name=task,
        train=windows[n_val:],
        val=windows[:n_val],
        description=SOURCES[task][1],
    )


@torch.no_grad()
def evaluate(model, corpus: Corpus, batch_size: int = 8, batches: int = 8) -> float:
    """Mean held-out loss.  Memory writes are disabled so evaluation is read-only."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for batch in corpus.val_batches(batch_size, limit=batches):
        loss, _ = model.lm_loss(batch, aux_weight=0.0, write_memory=False)
        total += float(loss)
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)
