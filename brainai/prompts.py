# BrainAI - built by Mohammed Zahid.
"""Domain prompt bank for teacher-generated distillation data.

Sequence-level distillation is only as broad as the prompts driving it. A
teacher asked six robot questions produces a student that knows about robots,
whatever the teacher itself was capable of. This bank spans the capability
areas the architecture is aimed at, so the generated corpus inherits breadth
rather than the prompt author's imagination.

Prompts are open-ended continuations rather than questions. The student is a
byte-level language model, not an instruction-following assistant; what
transfers is the *shape* of technical prose and code in each domain, and a
half-finished sentence elicits that better than a question does.

Health and legal prompts are framed as **educational explanation**, and the
generated text is training data for a small language model rather than advice
to anyone. Nothing here should end up in front of a patient or a litigant.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

#: domain -> continuation stems. Kept deliberately partial.
DOMAIN_PROMPTS: Dict[str, List[str]] = {
    "code_python": [
        "def binary_search(items, target):",
        "class LRUCache:\n    def __init__(self, capacity):",
        "# Read a CSV file and group the rows by column\nimport csv\n",
        "async def fetch_all(urls):",
        "The difference between a list and a generator in Python is",
        "@dataclass\nclass Config:",
        "def merge_sort(a):\n    if len(a) <= 1:",
        "try:\n    result = compute()\nexcept ValueError as exc:",
    ],
    "code_web": [
        "function debounce(fn, delay) {",
        "const useFetch = (url) => {\n  const [data, setData] = useState(null);",
        "SELECT customers.name, COUNT(orders.id) FROM customers",
        "<?php\nfunction sanitize_input($data) {",
        "To centre a div both horizontally and vertically with flexbox,",
        "app.get('/api/users/:id', async (req, res) => {",
    ],
    "code_systems": [
        "func worker(jobs <-chan int, results chan<- int) {",
        "impl Iterator for Counter {\n    type Item = u32;",
        "template <typename T>\nclass Stack {",
        "public class ThreadSafeCounter {\n    private final AtomicInteger",
        "A segmentation fault usually means the program",
        "The difference between a stack allocation and a heap allocation is",
    ],
    "algorithms": [
        "The time complexity of quicksort in the average case is",
        "A hash table resolves collisions by",
        "Dijkstra's algorithm finds the shortest path by",
        "The key insight behind dynamic programming is that",
        "A red-black tree maintains balance by ensuring",
    ],
    "devops": [
        "FROM python:3.12-slim\nWORKDIR /app",
        "To debug a pod stuck in CrashLoopBackOff, first",
        "The difference between horizontal and vertical scaling is",
        "A CI pipeline should fail fast because",
        "SQL injection is prevented by",
        "TLS certificate validation works by",
    ],
    "physics": [
        "The second law of thermodynamics states that",
        "In special relativity, simultaneity is relative because",
        "Conservation of angular momentum explains why a spinning skater",
        "The photoelectric effect demonstrated that light",
        "An electric field is defined as the force per unit charge, so",
    ],
    "mathematics": [
        "To prove that the square root of two is irrational, assume",
        "The derivative of a function measures",
        "A matrix is invertible if and only if",
        "Bayes' theorem relates the conditional probability",
        "The fundamental theorem of calculus connects",
        "A group is a set with an operation satisfying",
    ],
    "chemistry": [
        "An exothermic reaction releases energy because",
        "The periodic table is arranged by atomic number because",
        "A buffer solution resists changes in pH by",
        "Covalent bonding differs from ionic bonding in that",
        "Le Chatelier's principle predicts that when a system at equilibrium",
    ],
    "biology": [
        "DNA replication is semi-conservative, which means",
        "The mitochondrion produces ATP through",
        "Natural selection acts on variation within a population by",
        "An enzyme lowers the activation energy of a reaction by",
        "Homeostasis in the human body is maintained through",
    ],
    "astronomy": [
        "A star reaches the main sequence when",
        "Redshift indicates that distant galaxies are",
        "The habitable zone of a star is the region where",
        "A black hole's event horizon is the boundary at which",
    ],
    "engineering": [
        "A cantilever beam under a point load experiences maximum bending moment",
        "Ohm's law relates voltage, current and resistance so that",
        "The factor of safety in structural design accounts for",
        "A PID controller adjusts its output based on",
        "Reinforced concrete works because steel and concrete",
        "Heat transfer occurs by conduction, convection and radiation, where",
    ],
    "data_ai": [
        "Overfitting occurs when a model",
        "Gradient descent updates parameters in the direction of",
        "The bias-variance tradeoff describes",
        "A convolutional layer differs from a fully connected layer because",
        "Cross-validation estimates generalisation error by",
        "The p-value in a hypothesis test represents",
        "To clean a dataset with missing values, the first question is",
    ],
    "business": [
        "A balance sheet must balance because assets equal",
        "Discounted cash flow values a business by",
        "Price elasticity of demand measures",
        "A SWOT analysis examines strengths, weaknesses,",
        "The difference between gross margin and net margin is",
        "A project's critical path is the sequence of tasks that",
    ],
    "education": [
        "To explain fractions to a beginner, start with",
        "A good thesis statement should",
        "When revising for an exam, spaced repetition works because",
        "To structure a literature review, begin by",
    ],
    "writing": [
        "The difference between active and passive voice is",
        "Good technical documentation begins with",
        "To summarise a long document, first identify",
        "A subordinate clause cannot stand alone because",
        "Effective copy leads with the benefit rather than",
    ],
    "creative": [
        "The lighthouse had been dark for three years when",
        "A good game mechanic creates tension by",
        "She opened the letter and found",
        "Three product ideas for reducing food waste at home:",
    ],
    "health": [
        "For general education: carbohydrates provide energy because",
        "For general education: skeletal muscle adapts to resistance training by",
        "For general education: the cardiovascular system responds to exercise by",
        "For general education: dietary fibre supports digestion because",
    ],
    "law": [
        "As a general explanation of legal concepts, a contract requires offer,",
        "As a general explanation, the difference between civil and criminal law is",
        "As a general explanation, an indemnity clause allocates risk by",
        "As a general explanation, data protection regulation typically requires",
    ],
    "everyday": [
        "Subject: Following up on our conversation\n\nHi Sarah,",
        "A strong CV summary states your role, years of experience, and",
        "When asked to describe a weakness in an interview, a useful approach is",
        "To plan a week of work, start by listing",
    ],
    "robotics": [
        "The robot picked up the",
        "To grasp a fragile object safely, the controller should",
        "A reflex is faster than deliberation because",
        "Inverse kinematics solves for joint angles given",
        "Sensor fusion combines an IMU and a LiDAR by",
    ],
}


def domains() -> List[str]:
    return list(DOMAIN_PROMPTS)


def sample(
    n: int,
    domains_wanted: Sequence[str] | None = None,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """``n`` (domain, prompt) pairs, cycled evenly across domains.

    Even cycling rather than uniform random sampling: with 20 domains and a few
    thousand samples, random draw leaves visible gaps in the smaller ones, and
    the whole point of the bank is coverage.
    """
    names = list(domains_wanted) if domains_wanted else domains()
    unknown = [d for d in names if d not in DOMAIN_PROMPTS]
    if unknown:
        raise KeyError(f"unknown prompt domains: {unknown}")

    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    i = 0
    while len(out) < n:
        name = names[i % len(names)]
        out.append((name, rng.choice(DOMAIN_PROMPTS[name])))
        i += 1
    return out


def report() -> str:
    rows = [f"  {len(DOMAIN_PROMPTS)} domains, "
            f"{sum(len(v) for v in DOMAIN_PROMPTS.values())} prompts", ""]
    for name, prompts in DOMAIN_PROMPTS.items():
        rows.append(f"  {name:<14}{len(prompts):>3}  e.g. {prompts[0][:48]!r}")
    return "\n".join(rows)
