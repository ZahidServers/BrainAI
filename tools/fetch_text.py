# BrainAI - built by Mohammed Zahid.
"""Stream large open text corpora into data/*.txt shards (each becomes a domain).

    python tools/fetch_text.py --out /content/drive/MyDrive/brainai/data --mb 500
    python tools/fetch_text.py --only fineweb_edu wikipedia --mb 200

Streams (no full download), stops at --mb per source, so it is safe on Colab.
All sources are openly licensed / permissive; check each card before shipping weights.
"""
import argparse
from pathlib import Path

# name -> (hf path, config, split, text field)
SOURCES = {
    "fineweb_edu":   ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),   # reading/writing prose
    "wikipedia":     ("wikimedia/wikipedia", "20231101.en", "train", "text"),          # knowledge
    "tinystories":   ("roneneldan/TinyStories", None, "train", "text"),               # simple reading/writing
    "code":          ("codeparrot/codeparrot-clean", None, "train", "content"),
    "math":          ("open-web-math/open-web-math", None, "train", "text"),
    "wiki_hindi":    ("wikimedia/wikipedia", "20231101.hi", "train", "text"),
    "wiki_french":   ("wikimedia/wikipedia", "20231101.fr", "train", "text"),
}


def fetch(name, out, mb):
    from datasets import load_dataset
    path, cfg, split, field = SOURCES[name]
    dest = out / f"corpus_{name}.txt"
    if dest.exists() and dest.stat().st_size >= mb * 1e6 * 0.95:
        print(f"[skip] {name} already {dest.stat().st_size/1e6:.0f} MB"); return
    ds = load_dataset(path, cfg, split=split, streaming=True) if cfg else \
         load_dataset(path, split=split, streaming=True)
    written, cap = 0, int(mb * 1e6)
    with open(dest, "w", encoding="utf-8") as f:
        for row in ds:
            t = (row.get(field) or "").strip()
            if len(t) < 200:
                continue
            f.write(t + "\n\n"); written += len(t.encode("utf-8"))
            if written >= cap:
                break
    print(f"[done] {name}: {written/1e6:.0f} MB -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--mb", type=int, default=300, help="cap per source")
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for n in a.only:
        try: fetch(n, out, a.mb)
        except Exception as e: print(f"[fail] {n}: {e}")
