# Paper

**BrainAI: A Spiking Cognitive Architecture for Embodied Continual Learning**
Mohammed Zahid Imityaz Wadiwale · `zahid.wadiwale1234@gmail.com`

| File | Use |
|---|---|
| `brainai.tex` | Submission source. Two-column `article`, self-contained bibliography, no external `.bib`. |
| `brainai.md` | Same paper, readable without a TeX install. |

## Building the PDF

No LaTeX toolchain is installed on this machine, so the PDF is not committed.

**Overleaf** — upload `brainai.tex` alone and compile. It needs no other files.

**Locally**, once a distribution is installed:

```bash
pdflatex brainai.tex && pdflatex brainai.tex   # twice, to resolve references
```

The second pass is what fills in the `\ref` cross-references; a single pass
leaves them as `??`.

Verified before writing: 27 references, all cited and all defined; every
environment balanced; braces and math delimiters balanced.

## Provenance of the numbers

Every figure in the paper comes from a script in the parent repository, run on
CPU. Nothing is estimated or carried over from the literature.

| Paper section | Produced by |
|---|---|
| §5.1 gating | `benchmarks/gating_bench.py` |
| §5.2 continual learning | `continual.py` |
| §5.3 online stability | `benchmarks/stability_bench.py --control` |
| §5.4 epistemics | `benchmarks/belief_bench.py` |
| §6.1 AMSA | `benchmarks/activation_bench.py`, `benchmarks/amsa_continual.py --sweep` |
| §6.2 world model | `benchmarks/cognition_bench.py`, `benchmarks/worldmodel_jobs.py` |
| §7 regressions | `tests/` (108 tests) |

## A note on the negative results

Sections 6 and 7 are not padding. The architecture's own activation loses to
SwiGLU on all three axes including the one it was designed for, and the world
model loses to a linear map at one-step prediction. Both stay in the paper
because the alternative — reporting only the comparisons that went well — is the
failure mode §1.1 is about.
