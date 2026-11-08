# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""Download larger corpora for Urdu, Hindi, Marathi, Konkani and Arabic.

The hand-added language files were 60-90 KB each -- too small to sit alongside
the ~0.7 MB English and French domains. This builds script-correct corpora of
roughly comparable size from free, programmatically-fetchable sources.

Two source types, combined on purpose so no language is scripture-only:

  Tanzil (tanzil.net)  Quran text and translations in Arabic, Urdu and Hindi.
      Clean, consistent, freely redistributable, and a standard NLP source for
      exactly these scripts. It supplies volume and correct orthography.
  Wikipedia (MediaWiki API)  random encyclopedic articles as plain-text
      extracts, for secular subject matter and for Marathi and Konkani, which
      Tanzil does not cover. Konkani uses the ``gom`` wiki, which is small, so
      that corpus is whatever the wiki actually holds rather than a target.

Byte-level tokenisation handles these scripts directly -- each Devanagari or
Arabic character is 2-3 UTF-8 bytes and needs no special vocabulary -- which is
why a multilingual corpus costs nothing here beyond the download.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

ssl._create_default_https_context = ssl._create_unverified_context
DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "BrainAI-corpus/1.0 (research; contact zahid.wadiwale1234@gmail.com)"}

TANZIL = {
    "arabic": ["https://tanzil.net/pub/download/index.php?"
               "quranType=uthmani&outType=txt&agree=true"],
    "urdu": ["https://tanzil.net/trans/ur.jalandhry",
             "https://tanzil.net/trans/ur.ahmedali"],
    "hindi": ["https://tanzil.net/trans/hi.farooq"],
}

GUTENBERG = {
    "hindi": [48512],  # a Devanagari text, verified ~0.64 MB
}

# language -> wiki subdomain. gom = Goan Konkani.
WIKI = {
    "marathi": "mr",
    "konkani": "gom",
    "urdu": "ur",
    "hindi": "hi",
    "arabic": "ar",
}


def _get(url: str, timeout: int = 60) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout
    ).read()


def _script_fraction(text: str) -> float:
    """Fraction of a sample that is Arabic or Devanagari -- a sanity check."""
    sample = text[:20000]
    if not sample:
        return 0.0
    hits = sum(1 for c in sample
               if ("؀" <= c <= "ۿ") or ("ऀ" <= c <= "ॿ")
               or ("ঀ" <= c <= "৿"))
    return hits / len(sample)


def harvest_wikipedia(lang_code: str, target_bytes: int,
                      max_calls: int = 400) -> str:
    """Random plain-text articles from one Wikipedia edition."""
    api = (f"https://{lang_code}.wikipedia.org/w/api.php?action=query"
           f"&generator=random&grnnamespace=0&grnlimit=10"
           f"&prop=extracts&explaintext=1&exlimit=10&format=json")
    seen: set = set()
    parts: List[str] = []
    total = 0
    for call in range(max_calls):
        try:
            data = json.loads(_get(api, timeout=40).decode("utf-8", "replace"))
        except Exception:
            time.sleep(1.0)
            continue
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            extract = page.get("extract", "")
            if title in seen or len(extract) < 200:
                continue
            seen.add(title)
            parts.append(f"= {title} =\n{extract}")
            total += len(extract)
        if total >= target_bytes:
            break
        time.sleep(0.3)  # be polite to the API
    return "\n\n".join(parts)


def _strip_gutenberg(raw: str) -> str:
    a = raw.find("*** START")
    b = raw.find("*** END")
    if a != -1:
        a = raw.find("\n", a) + 1
    return raw[a if a > 0 else 0 : b if b != -1 else len(raw)].strip()


def build_language(lang: str, target_mb: float = 1.2) -> Optional[Path]:
    target = int(target_mb * 1_000_000)
    parts: List[str] = []

    for url in TANZIL.get(lang, []):
        try:
            parts.append(_get(url).decode("utf-8", "replace"))
            print(f"    tanzil ok  ({len(parts[-1])/1e6:.2f}MB)")
        except Exception as exc:
            print(f"    tanzil failed ({type(exc).__name__})")

    for book in GUTENBERG.get(lang, []):
        try:
            raw = _get(f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt")
            parts.append(_strip_gutenberg(raw.decode("utf-8", "replace")))
            print(f"    gutenberg {book} ok")
        except Exception as exc:
            print(f"    gutenberg {book} failed ({type(exc).__name__})")

    have = sum(len(p) for p in parts)
    if lang in WIKI and have < target:
        print(f"    wikipedia {WIKI[lang]} (need {(target-have)/1e6:.2f}MB) ...",
              flush=True)
        wiki = harvest_wikipedia(WIKI[lang], target - have)
        if wiki:
            parts.append(wiki)
            print(f"    wikipedia ok ({len(wiki)/1e6:.2f}MB)")

    text = "\n\n".join(p for p in parts if p).strip()
    if not text:
        print(f"  {lang}: NOTHING fetched")
        return None

    frac = _script_fraction(text)
    out = DATA / f"language_{lang}_large.txt"
    out.write_text(text, encoding="utf-8")
    print(f"  {lang:<10}{len(text)/1e6:>6.2f}MB  script={frac:.0%}  -> {out.name}")
    return out


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print("Downloading larger corpora for non-Latin languages:\n")
    for lang in ("arabic", "urdu", "hindi", "marathi", "konkani"):
        print(f"  {lang} ...")
        build_language(lang)
    print("\nSources: Tanzil (Quran text/translations), Project Gutenberg, "
          "Wikipedia (CC BY-SA). Retain attribution if redistributing.")


if __name__ == "__main__":
    main()
