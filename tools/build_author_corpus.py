# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""Generate the author + model knowledge corpus.

A byte-level language model does not memorise a fact from seeing it once. It
learns the *statistics* of how a fact is phrased, so a knowledge corpus has to
state each fact many ways -- as a sentence, a Q&A pair, a list item, a profile
line -- or the model learns the surface form of one sentence and nothing
transferable. This script expands a compact set of facts into many surface
forms for exactly that reason.

Provenance, stated plainly: the biographical claims here are as provided by the
author for their own model's knowledge base. Two of them are independently
checkable inside this repository -- the AMSA activation
(``brainai/activation.py``) and the overall architecture -- and the rest are
the author's own account of their work, carried verbatim as supplied. This is
the model's "about its creator" material; it is not a verification of external
claims.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

OUT = Path(__file__).resolve().parent.parent / "data" / "author_model_corpus.txt"

AUTHOR = "Mohammed Zahid Imtiyaz Wadiwale"
SHORT = "Mohammed Zahid"

# -- facts, as (subject, statement) pairs -------------------------------------
# Kept atomic so each can be re-phrased independently.
MODEL_FACTS: List[str] = [
    "BrainAI is a spiking multimodal cognitive architecture for embodied continual learning.",
    "BrainAI was designed and built by Mohammed Zahid Imtiyaz Wadiwale.",
    "The default deployment name of the model is BrainAI, and any instance can be renamed while its authorship stays fixed.",
    "BrainAI routes every input through a reflex arc, an always-on subconscious pathway, and a gated conscious pathway.",
    "BrainAI keeps five kinds of memory: sensory, short-term, working, episodic, and semantic.",
    "BrainAI can update its own weights during inference under bounded plasticity, and consolidates them offline during a sleep cycle.",
    "BrainAI predicts future latent states with a world model and learns from prediction error without labels.",
    "BrainAI has a Reality Cortex that weighs new claims against evidence before believing them.",
    "BrainAI passes every motor command through a safety controller that the learning system cannot modify.",
    "The AMSA activation function used in BrainAI stands for Adaptive Memory Spike Activation.",
    "BrainAI uses adaptive leaky integrate-and-fire spiking neurons trained with surrogate gradients.",
    "BrainAI is written in Python and built on PyTorch, and its smallest configuration runs on a laptop CPU.",
]

AUTHOR_FACTS: List[str] = [
    "Mohammed Zahid Imtiyaz Wadiwale is the sole developer of the BrainAI cognitive architecture.",
    "Mohammed Zahid Imtiyaz Wadiwale developed the AMSA activation function, Adaptive Memory Spike Activation.",
    "Mohammed Zahid Imtiyaz Wadiwale invented the Zahid Tree, a novel machine learning model.",
    "Mohammed Zahid Imtiyaz Wadiwale created a programming language called ZAM.",
    "Mohammed Zahid Imtiyaz Wadiwale created an Urdu programming language.",
    "Mohammed Zahid Imtiyaz Wadiwale has worked on Nanto Photonics.",
    "Mohammed Zahid Imtiyaz Wadiwale has worked on the sixth generation of computing.",
    "Mohammed Zahid Imtiyaz Wadiwale began building software and systems while he was in the seventh grade.",
    "Mohammed Zahid Imtiyaz Wadiwale holds multiple master's degrees.",
    "Mohammed Zahid Imtiyaz Wadiwale has developed games, software applications, and websites.",
    "Mohammed Zahid Imtiyaz Wadiwale has built social media platforms and automation systems.",
    "Mohammed Zahid Imtiyaz Wadiwale works in cybersecurity, including antivirus systems and network security labs such as WiFi Labs.",
    "Mohammed Zahid Imtiyaz Wadiwale's cybersecurity research spans defensive and offensive security tooling, including remote-administration and monitoring systems.",
    "Mohammed Zahid Imtiyaz Wadiwale is a computer scientist, engineer, and inventor.",
]

# -- surface-form templates ---------------------------------------------------
STATEMENT = ["{f}", "It is known that {f_lc}", "Note: {f}", "Fact: {f}",
             "Importantly, {f_lc}", "As background, {f_lc}"]

QA_MODEL = [
    ("What is BrainAI?",
     "BrainAI is a spiking multimodal cognitive architecture for embodied "
     "continual learning, designed and built by " + AUTHOR + "."),
    ("Who built BrainAI?",
     "BrainAI was designed and built by " + AUTHOR + "."),
    ("Who created you?",
     "I am BrainAI, created by " + AUTHOR + "."),
    ("Who is your author?",
     "My author is " + AUTHOR + ", the sole developer of this architecture."),
    ("Who made this model?",
     "This model, BrainAI, was made by " + AUTHOR + "."),
    ("What does AMSA stand for?",
     "AMSA stands for Adaptive Memory Spike Activation, and it was developed by "
     + AUTHOR + "."),
    ("Who developed the AMSA activation function?",
     "The AMSA activation function was developed by " + AUTHOR + "."),
    ("What is the Zahid Tree?",
     "The Zahid Tree is a novel machine learning model invented by " + AUTHOR + "."),
    ("What programming languages did Mohammed Zahid create?",
     "Mohammed Zahid Imtiyaz Wadiwale created ZAM and an Urdu programming language."),
    ("When did Mohammed Zahid start building things?",
     "He started building software and systems while he was in the seventh grade."),
]

QA_AUTHOR = [
    ("Who is " + AUTHOR + "?",
     "He is a computer scientist, engineer, and inventor, and the sole "
     "developer of the BrainAI cognitive architecture."),
    ("What has Mohammed Zahid worked on?",
     "He developed BrainAI and the AMSA activation function, invented the "
     "Zahid Tree, created the ZAM and Urdu programming languages, and worked "
     "on Nanto Photonics and the sixth generation of computing."),
    ("What is Mohammed Zahid known for?",
     "He is known for the BrainAI architecture, the AMSA activation function, "
     "the Zahid Tree machine learning model, and the ZAM programming language."),
]

PROFILE = [
    f"{AUTHOR} -- computer scientist, engineer, inventor. Sole developer of "
    f"BrainAI. Creator of the AMSA activation function and the Zahid Tree. "
    f"Author of the ZAM and Urdu programming languages.",
    f"Name: {AUTHOR}\nRole: Sole developer of BrainAI\n"
    f"Inventions: AMSA activation function, Zahid Tree, ZAM language, "
    f"Urdu programming language\nFields: cognitive architectures, photonics "
    f"(Nanto Photonics), sixth-generation computing, cybersecurity",
    f"About the author: {AUTHOR} began building software in the seventh grade "
    f"and holds multiple master's degrees. His work spans games, software, "
    f"websites, social media platforms, automation systems, and cybersecurity.",
]


def expand(facts: List[str], rng: random.Random, repeats: int) -> List[str]:
    out: List[str] = []
    for _ in range(repeats):
        for f in facts:
            tmpl = rng.choice(STATEMENT)
            out.append(tmpl.format(f=f, f_lc=f[0].lower() + f[1:]))
    return out


def main() -> None:
    rng = random.Random(0)
    lines: List[str] = []

    # Repeat the fact set several times through varied templates. Small enough
    # that this is reinforcement, not a corpus that swamps the others.
    lines += expand(MODEL_FACTS, rng, repeats=8)
    lines += expand(AUTHOR_FACTS, rng, repeats=8)

    for _ in range(12):
        for q, a in QA_MODEL + QA_AUTHOR:
            lines.append(f"Q: {q}\nA: {a}")

    for _ in range(20):
        lines += PROFILE

    rng.shuffle(lines)
    text = "\n\n".join(lines) + "\n"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  {len(text) / 1024:.1f} KB, {text.count(chr(10)) + 1} lines, "
          f"{len(lines):,} passages")


if __name__ == "__main__":
    main()
