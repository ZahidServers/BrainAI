# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""Assemble programming-language corpora from real, permissively-licensed code.

Two sources, both chosen so the text is genuine code rather than something that
merely looks like it:

  local   Apache httpd C/C++ headers already on this machine (Apache-2.0), and
          the Python standard library (PSF).
  remote  a small curated set of canonical files from permissively-licensed
          GitHub repositories -- Guava (Apache-2.0), .NET runtime (MIT),
          nlohmann/json (MIT), Go stdlib (BSD), Rust stdlib (MIT/Apache).

Curated rather than scraped: a handful of canonical, well-written files teaches
a small model more about a language's idioms than a random sweep would, and it
keeps the licensing auditable. Each file is prefixed with a path comment so the
model sees boundaries between files.

Nothing here is fabricated. If a download fails the language is simply smaller,
and the failure is reported rather than papered over with synthetic filler.
"""

from __future__ import annotations

import glob
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

ssl._create_default_https_context = ssl._create_unverified_context
DATA = Path(__file__).resolve().parent.parent / "data"

# language -> list of raw URLs (canonical, permissively licensed)
REMOTE: Dict[str, List[str]] = {
    "code_java": [
        "https://raw.githubusercontent.com/google/guava/master/guava/src/com/google/common/base/Strings.java",
        "https://raw.githubusercontent.com/google/guava/master/guava/src/com/google/common/base/Preconditions.java",
        "https://raw.githubusercontent.com/google/guava/master/guava/src/com/google/common/collect/Lists.java",
        "https://raw.githubusercontent.com/google/guava/master/guava/src/com/google/common/io/ByteStreams.java",
        "https://raw.githubusercontent.com/google/guava/master/guava/src/com/google/common/base/Optional.java",
    ],
    "code_csharp": [
        "https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Private.CoreLib/src/System/String.cs",
        "https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Private.CoreLib/src/System/Collections/Generic/List.cs",
        "https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Linq/src/System/Linq/Where.cs",
        "https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Private.CoreLib/src/System/Math.cs",
    ],
    "code_cpp": [
        "https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp",
    ],
    "code_go": [
        "https://raw.githubusercontent.com/golang/go/master/src/sort/sort.go",
        "https://raw.githubusercontent.com/golang/go/master/src/strings/strings.go",
        "https://raw.githubusercontent.com/golang/go/master/src/sync/mutex.go",
        "https://raw.githubusercontent.com/golang/go/master/src/net/http/server.go",
    ],
    "code_rust": [
        "https://raw.githubusercontent.com/rust-lang/rust/master/library/core/src/option.rs",
        "https://raw.githubusercontent.com/rust-lang/rust/master/library/core/src/result.rs",
        "https://raw.githubusercontent.com/rust-lang/rust/master/library/alloc/src/vec/mod.rs",
    ],
}


def fetch(url: str) -> str:
    return urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")


def build_remote() -> None:
    for lang, urls in REMOTE.items():
        parts: List[str] = []
        ok = 0
        for url in urls:
            name = url.rsplit("/", 1)[-1]
            try:
                parts.append(f"// ==== {name} ====\n{fetch(url)}\n")
                ok += 1
            except Exception as exc:
                print(f"  {lang}: failed {name} ({type(exc).__name__})")
        if parts:
            out = DATA / f"{lang}.txt"
            out.write_text("\n".join(parts), encoding="utf-8")
            size = out.stat().st_size / 1024
            print(f"  {lang:<14}{ok}/{len(urls)} files  {size:>7.0f} KB")


def build_local_c(max_bytes: int = 4_000_000) -> None:
    """Apache httpd C/C++ headers -- real systems C, already on disk."""
    roots = glob.glob(r"C:\wamp64\bin\apache\**\include", recursive=True)
    files: List[str] = []
    for root in roots:
        files += glob.glob(os.path.join(root, "*.h"))
    files = sorted(set(files))

    parts: List[str] = []
    total = 0
    for path in files:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts.append(f"/* ==== {os.path.basename(path)} ==== */\n{text}\n")
        total += len(text)
        if total >= max_bytes:
            break
    if parts:
        out = DATA / "code_c.txt"
        out.write_text("\n".join(parts), encoding="utf-8")
        print(f"  {'code_c':<14}{len(parts)} files  {out.stat().st_size/1024:>7.0f} KB "
              f"(Apache httpd headers)")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print("Collecting real code corpora (permissively licensed):")
    build_local_c()
    build_remote()
    print("\nNote: licences are Apache-2.0 / MIT / BSD / PSF. If you "
          "redistribute a model trained on these, retain their attribution.")


if __name__ == "__main__":
    main()
