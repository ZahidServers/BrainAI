# BrainAI - built by Mohammed Zahid.
"""Multi-domain corpus: code, science, history, economics, psychology, prose.

The single-book corpus in :mod:`brainai.data` was enough to prove that training
works. It cannot show anything about breadth, because there is only one domain
to be broad across. This module assembles roughly 29 MB spanning eleven
domains, which at byte level is ~29M tokens -- almost exactly one epoch of what
8 CPU cores can process in a working day.

Two sources, both usable without a licence question:

**Python's own standard library**, already on this machine and PSF-licensed. It
is a genuinely good code corpus: 1,020 files of idiomatic, reviewed, documented
Python. Generated tables (``unicodedata``, ``pydoc_data``) and the test suite
are excluded -- the first is not code anyone writes, the second is repetitive.

**Project Gutenberg**, public domain in the US. Every identifier here was
checked against the *actual* title returned, because ID guessing is unreliable:
three of the first ten candidates downloaded a different book than intended,
and one of those (an ID nominally for an astronomy text) returned a novel.

Domains are sampled at **temperature 0.5** rather than in proportion to size.
Code is 15 MB and chemistry is 0.25 MB; proportional sampling would show the
model chemistry roughly once per epoch, while balanced sampling would replay
those same 0.25 MB sixty times and overfit them. ``p_i ∝ n_i^0.5`` is the usual
compromise from multilingual pretraining.
"""

from __future__ import annotations

import glob
import os
import ssl
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .data import _strip_gutenberg
from .tokenizer import ByteTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: domain -> (gutenberg ids, human description). Titles verified by fetching
#: the header and reading the actual Title field, not assumed from the ID.
GUTENBERG_DOMAINS: Dict[str, Tuple[Sequence[int], str]] = {
    "physics":     ((30155, 28233), "Einstein, Relativity; Newton, Principia"),
    "mathematics": ((16713,),       "Dudeney, Amusements in Mathematics"),
    "biology":     ((1228, 15491),  "Darwin, Origin of Species; Hooke, Micrographia"),
    "chemistry":   ((14474,),       "Faraday, The Chemical History of a Candle"),
    "science":     ((20417,),       "Thomson, The Outline of Science"),
    "history":     ((45368,),       "Wells, The Outline of History"),
    "economics":   ((3300,),        "Smith, The Wealth of Nations"),
    "psychology":  ((57628,),       "James, The Principles of Psychology"),
    "philosophy":  ((10615,),       "Locke, An Essay Concerning Human Understanding"),
    "literature":  ((1342,),        "Austen, Pride and Prejudice"),
    "french":      ((4650,),        "Voltaire, Candide"),
}

#: Files that are data tables rather than code, and would teach the model
#: nothing except how to emit long comma-separated literals.
_CODE_EXCLUDE = ("test", "idlelib", "unicodedata", "pydoc_data", "encodings")
_CODE_MAX_BYTES = 200_000


def _gutenberg(book_id: int) -> str:
    ctx = ssl._create_unverified_context()
    for url in (
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ):
        try:
            raw = urllib.request.urlopen(url, timeout=90, context=ctx).read()
            return _strip_gutenberg(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
    raise RuntimeError(f"could not fetch Gutenberg id {book_id}")


def python_stdlib(max_bytes: int = 16_000_000) -> str:
    """Real Python source from the local interpreter's standard library.

    Files are separated by a path comment so the model sees module boundaries
    rather than one undifferentiated stream.
    """
    lib = Path(sys.executable).parent / "Lib"
    if not lib.exists():
        lib = Path(os.__file__).parent

    chunks: List[str] = []
    total = 0
    for path in sorted(glob.glob(str(lib / "**" / "*.py"), recursive=True)):
        low = path.lower()
        if any(bad in low for bad in _CODE_EXCLUDE):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > _CODE_MAX_BYTES:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = os.path.relpath(path, lib).replace("\\", "/")
        chunks.append(f"# ==== {rel} ====\n{text}\n")
        total += len(text)
        if total >= max_bytes:
            break
    return "\n".join(chunks)


def fetch_domain(domain: str, cache_dir: Optional[Path] = None) -> str:
    """Raw text for one domain, cached on disk after the first call."""
    cache_dir = cache_dir or DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"domain_{domain}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")

    if domain == "code_python":
        text = python_stdlib()
    elif domain in GUTENBERG_DOMAINS:
        ids, _ = GUTENBERG_DOMAINS[domain]
        text = "\n\n".join(_gutenberg(i) for i in ids)
    else:
        raise KeyError(f"unknown domain {domain!r}; known: {sorted(all_domains())}")

    path.write_text(text, encoding="utf-8")
    return text


def all_domains() -> List[str]:
    return ["code_python"] + list(GUTENBERG_DOMAINS)


#: The one cache file that genuinely duplicates a generated domain: the cached
#: Python stdlib. ``code_python`` is always regenerated from the live
#: interpreter, so its cache must be skipped or the domain is counted twice.
#: Every other ``domain_*.txt`` holds unique Gutenberg text and is kept.
_DISCOVERY_SKIP_EXACT = ("domain_code_python",)


def discover_disk_domains(
    data_dir: Optional[Path] = None,
    min_bytes: int = 2_000,
) -> Dict[str, Path]:
    """Every ``*.txt`` in ``data/`` that is a corpus in its own right.

    This is what makes "train on everything in the data folder" literal: drop a
    ``.txt`` in and it becomes a domain, no code change. Files below
    ``min_bytes`` (the tiny grade-level stubs) and the ``domain_*`` cache files
    are skipped so the domain list reflects real, distinct corpora.

    A domain name is the filename stem, lower-cased. Where a hand-curated domain
    and a disk file would collide, the curated one wins and the disk file is
    left for the caller to rename if they want both.
    """
    data_dir = data_dir or DATA_DIR
    found: Dict[str, Path] = {}
    if not data_dir.exists():
        return found
    for path in sorted(data_dir.glob("*.txt")):
        name = path.stem
        if name in _DISCOVERY_SKIP_EXACT:
            continue
        try:
            if path.stat().st_size < min_bytes:
                continue
        except OSError:
            continue
        found[name.lower()] = path
    return found


@dataclass
class DomainSplit:
    """One domain's tokenised windows, with a held-out slice."""

    name: str
    train: torch.Tensor
    val: torch.Tensor
    description: str = ""

    @property
    def train_tokens(self) -> int:
        return int(self.train.numel())

    def batch(self, size: int, generator: Optional[torch.Generator] = None):
        idx = torch.randint(0, self.train.shape[0], (size,), generator=generator)
        return self.train[idx].long()      # windows are stored int16 to save RAM

    def val_batches(self, size: int, limit: int = 6):
        for i in range(0, min(self.val.shape[0], size * limit), size):
            yield self.val[i : i + size].long()


def read_sampled_bytes(path: Path, cap_bytes: int, block: int = 1 << 20,
                       seed: int = 1234) -> bytes:
    """At most ``cap_bytes`` of a file, taken as random 1 MB blocks spread over it.

    Reading a whole 500 MB file just to keep ~30 MB of windows is what ran
    Colab out of RAM. Small files are returned whole; big ones are sampled
    across their full length so the sample is not just the file's beginning.
    """
    size = path.stat().st_size
    if size <= cap_bytes:
        return path.read_bytes()
    import random
    rng = random.Random(seed)
    n_blocks = max(1, cap_bytes // block)
    starts = sorted(rng.sample(range(0, max(1, (size - block) // block)), n_blocks))         if (size - block) // block >= n_blocks else [0]
    parts = []
    with open(path, "rb") as f:
        for st in starts:
            f.seek(st * block)
            parts.append(f.read(block))
    return b"".join(parts)


class MultiDomainCorpus:
    """Eleven domains, sampled at temperature 0.5.

    ``sample()`` draws a domain then a batch from it. Per-domain held-out sets
    are kept separate so breadth can be measured rather than assumed -- a single
    averaged loss hides a model that has learned code and forgotten chemistry.
    """

    def __init__(
        self,
        splits: Dict[str, DomainSplit],
        temperature: float = 0.5,
        seed: int = 0,
        boost: Optional[Dict[str, float]] = None,
    ) -> None:
        self.splits = splits
        self.temperature = temperature
        self._gen = torch.Generator().manual_seed(seed)
        self.boost = dict(boost or {})

        sizes = torch.tensor(
            [float(s.train_tokens) for s in splits.values()]
        ).clamp_min(1.0)
        weights = sizes.pow(temperature)
        # Per-domain multipliers, for deliberately over-representing a small but
        # important corpus (the author corpus is ~0.06 MB against a 16 MB maths
        # file, so at temperature 0.5 it is still only ~0.5% of samples; a x15
        # boost is what makes its facts actually stick).
        self.names = list(splits)
        for i, name in enumerate(self.names):
            weights[i] *= float(self.boost.get(name, 1.0))
        self.weights = weights / weights.sum()

    def sample(self, batch_size: int) -> Tuple[str, torch.Tensor]:
        i = int(torch.multinomial(self.weights, 1, generator=self._gen))
        name = self.names[i]
        return name, self.splits[name].batch(batch_size, generator=self._gen)

    @property
    def total_tokens(self) -> int:
        return sum(s.train_tokens for s in self.splits.values())

    def describe(self) -> str:
        rows = [f"  {'domain':<14}{'MB':>7}{'windows':>9}{'sample %':>10}  source",
                "  " + "-" * 88]
        for name, weight in zip(self.names, self.weights.tolist()):
            s = self.splits[name]
            rows.append(
                f"  {name:<14}{s.train_tokens / 1e6:>7.2f}{s.train.shape[0]:>9,}"
                f"{weight * 100:>9.1f}%  {s.description[:44]}"
            )
        rows.append("  " + "-" * 88)
        rows.append(f"  {'TOTAL':<14}{self.total_tokens / 1e6:>7.2f} MB across "
                    f"{len(self.names)} domains")
        return "\n".join(rows)


def load_multidomain(
    domains: Optional[Sequence[str]] = None,
    seq_len: int = 128,
    val_fraction: float = 0.02,
    max_windows_per_domain: int = 130_000,
    temperature: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
    discover: bool = True,
    boost: Optional[Dict[str, float]] = None,
) -> MultiDomainCorpus:
    """Build the multi-domain corpus, optionally from every file in ``data/``.

    ``discover=True`` (the default) scans ``data/`` and treats every substantial
    ``*.txt`` as its own domain, so anything dropped in the folder is trained on
    without a code change. ``code_python`` is always included because it is
    generated from the local interpreter rather than a file. Passing an explicit
    ``domains`` list overrides discovery.
    """
    tok = ByteTokenizer()
    splits: Dict[str, DomainSplit] = {}

    disk = discover_disk_domains() if (discover and domains is None) else {}
    if domains is not None:
        names = list(domains)
    elif disk:
        names = ["code_python"] + sorted(disk)
    else:
        names = all_domains()

    for name in names:
        if verbose:
            print(f"  [corpus] {name} ...", end=" ", flush=True)
        try:
            if name in disk:
                # Bounded read: never hold more than the windows we will keep.
                text = read_sampled_bytes(
                    disk[name], cap_bytes=max_windows_per_domain * seq_len * 2
                ).decode("utf-8", errors="replace")
            else:
                text = fetch_domain(name)
        except Exception as exc:
            if verbose:
                print(f"skipped ({type(exc).__name__})")
            continue

        windows = tok.chunks(text, seq_len, stride=seq_len, dtype=torch.int16)
        del text
        g = torch.Generator().manual_seed(1234)
        windows = windows[torch.randperm(windows.shape[0], generator=g)]
        windows = windows[:max_windows_per_domain]

        n_val = max(1, int(windows.shape[0] * val_fraction))
        if name == "code_python":
            description = "Python standard library (PSF)"
        elif name in GUTENBERG_DOMAINS:
            description = GUTENBERG_DOMAINS[name][1]
        elif name in disk:
            description = f"data/{disk[name].name}"
        else:
            description = name
        splits[name] = DomainSplit(
            name=name, train=windows[n_val:], val=windows[:n_val],
            description=description,
        )
        if verbose:
            print(f"{windows.shape[0]:,} windows")

    if not splits:
        raise RuntimeError("no domains could be loaded")
    return MultiDomainCorpus(splits, temperature=temperature, seed=seed,
                             boost=boost)


@torch.no_grad()
def evaluate_domains(
    model, corpus: MultiDomainCorpus, batch_size: int = 8, batches: int = 4
) -> Dict[str, float]:
    """Held-out loss per domain.

    Reported separately on purpose: one averaged number cannot distinguish a
    model that is mediocre everywhere from one that learned code and forgot
    everything else.
    """
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}
    for name, split in corpus.splits.items():
        losses = []
        for batch in split.val_batches(batch_size, limit=batches):
            batch = batch.to(next(model.parameters()).device)
            loss, _ = model.lm_loss(batch, aux_weight=0.0, write_memory=False)
            losses.append(float(loss))
        out[name] = sum(losses) / max(len(losses), 1)
    out["mean"] = sum(v for k, v in out.items() if k != "mean") / max(len(out), 1)
    if was_training:
        model.train()
    return out
