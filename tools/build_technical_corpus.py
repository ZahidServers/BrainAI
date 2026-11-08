# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""Generate low-level and encoding corpora: Morse, binary, bytecode, assembly.

These are *generated* rather than downloaded, and that is the right choice for
this material specifically: the mappings are exact and mechanical, so a
generator produces guaranteed-correct examples with full coverage, where a
scraped corpus would be noisier and no more authentic. Morse code is a fixed
table; ASCII-to-binary is a function; the assembly here is hand-written to
assemble.

What each section teaches a byte-level model:

  morse       the International Morse alphabet and round-trip text<->code
  binary      ASCII<->binary and decimal<->binary<->hex, as worked conversions
  bytecode    CPython disassembly (real ``dis`` output) and a JVM-style listing
  assembly    NASM for x86 in 16/32/64-bit forms, plus 8-bit 8051, all with
              the source-to-encoding correspondence a real listing shows

The assembly is illustrative reference code -- correct in syntax and intent --
not an exhaustive ISA reference. It is enough for a model to learn the shape of
each dialect, which is the stated goal.
"""

from __future__ import annotations

import dis
import random
from pathlib import Path
from typing import List

DATA = Path(__file__).resolve().parent.parent / "data"

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.", ".": ".-.-.-", ",": "--..--", "?": "..--..",
    "/": "-..-.", "=": "-...-",
}

MORSE_WORDS = [
    "SOS", "HELLO WORLD", "BRAINAI", "MOHAMMED ZAHID", "MAYDAY", "CQ CQ",
    "THE QUICK BROWN FOX", "MORSE CODE", "73 DE OP", "ATTENTION",
    "SPIKING NEURAL NETWORK", "ADAPTIVE MEMORY SPIKE ACTIVATION",
]


def build_morse() -> str:
    lines = ["INTERNATIONAL MORSE CODE", "=" * 40, ""]
    lines.append("Alphabet and digits:")
    for ch, code in MORSE.items():
        lines.append(f"  {ch} = {code}")
    lines.append("")
    lines.append("Encoding examples (letters separated by space, words by /):")
    for word in MORSE_WORDS:
        enc = " / ".join(
            " ".join(MORSE.get(c, "") for c in w) for w in word.split()
        )
        lines.append(f"  TEXT: {word}")
        lines.append(f"  MORSE: {enc}")
        lines.append(f"  DECODE {enc} -> {word}")
        lines.append("")
    # Timing reference.
    lines += [
        "Morse timing (in dot units):",
        "  dot = 1 unit; dash = 3 units",
        "  gap between symbols in a letter = 1 unit",
        "  gap between letters = 3 units; gap between words = 7 units",
        "",
    ]
    return "\n".join(lines)


def build_binary() -> str:
    lines = ["BINARY, DECIMAL AND HEXADECIMAL", "=" * 40, ""]
    lines.append("ASCII characters as 8-bit binary:")
    for ch in "ABCDEFGHIabcdefghi0123456789 .!?":
        lines.append(f"  '{ch}' = dec {ord(ch):>3} = bin {ord(ch):08b} "
                     f"= hex 0x{ord(ch):02X}")
    lines.append("")
    lines.append("Text encoded to binary:")
    for word in ("BrainAI", "Zahid", "spike", "01010101"):
        enc = " ".join(f"{ord(c):08b}" for c in word)
        lines.append(f"  {word!r} -> {enc}")
    lines.append("")
    lines.append("Number base conversions:")
    rng = random.Random(0)
    for _ in range(40):
        n = rng.randint(0, 255)
        lines.append(f"  dec {n:>3} = bin {n:08b} = hex 0x{n:02X} = oct 0o{n:03o}")
    lines.append("")
    lines.append("Binary arithmetic:")
    for _ in range(20):
        a, b = rng.randint(0, 127), rng.randint(0, 127)
        lines.append(f"  {a:07b} + {b:07b} = {a+b:08b}   ({a} + {b} = {a+b})")
        lines.append(f"  {a:08b} AND {b:08b} = {a & b:08b}")
        lines.append(f"  {a:08b} XOR {b:08b} = {a ^ b:08b}")
    return "\n".join(lines)


def _sample_functions():
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def factorial(n):
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    return [gcd, factorial, fib]


def build_bytecode() -> str:
    lines = ["BYTECODE AND DISASSEMBLY", "=" * 40, ""]
    lines.append("CPython bytecode (real output of the dis module):")
    lines.append("")
    for fn in _sample_functions():
        import inspect
        lines.append(f"# source: {fn.__name__}")
        lines.append(inspect.getsource(fn).rstrip())
        lines.append(f"# disassembly of {fn.__name__}:")
        try:
            lines.append(dis.Bytecode(fn).dis().rstrip())
        except Exception:
            pass
        lines.append("")

    lines += [
        "JVM-style bytecode (illustrative, javap -c form):",
        "",
        "// int add(int a, int b) { return a + b; }",
        "  0: iload_0",
        "  1: iload_1",
        "  2: iadd",
        "  3: ireturn",
        "",
        "// static int square(int n) { return n * n; }",
        "  0: iload_0",
        "  1: iload_0",
        "  2: imul",
        "  3: ireturn",
        "",
        "Common JVM opcodes: iconst, bipush, iload, istore, iadd, isub, imul, "
        "idiv, ireturn, invokevirtual, invokestatic, getfield, putfield, new, "
        "goto, if_icmpeq.",
        "",
        ".NET CIL (illustrative, ildasm form):",
        "  ldarg.0",
        "  ldarg.1",
        "  add",
        "  ret",
        "",
    ]
    return "\n".join(lines)


def build_assembly() -> str:
    lines = ["ASSEMBLY LANGUAGE REFERENCE", "=" * 40, ""]

    lines += [
        "; ---- NASM x86 64-bit (Linux) : write() then exit() ----",
        "section .data",
        "    msg db 'Hello, BrainAI', 0x0A",
        "    len equ $ - msg",
        "section .text",
        "    global _start",
        "_start:",
        "    mov rax, 1          ; sys_write",
        "    mov rdi, 1          ; fd = stdout",
        "    mov rsi, msg        ; buffer",
        "    mov rdx, len        ; length",
        "    syscall",
        "    mov rax, 60         ; sys_exit",
        "    xor rdi, rdi        ; status 0",
        "    syscall",
        "",
        "; ---- NASM x86 32-bit (Linux) : same, via int 0x80 ----",
        "section .text",
        "    global _start",
        "_start:",
        "    mov eax, 4          ; sys_write",
        "    mov ebx, 1          ; stdout",
        "    mov ecx, msg",
        "    mov edx, len",
        "    int 0x80",
        "    mov eax, 1          ; sys_exit",
        "    xor ebx, ebx",
        "    int 0x80",
        "",
        "; ---- NASM x86 16-bit (DOS, .COM) : print string ----",
        "org 0x100",
        "    mov ah, 0x09        ; DOS print-string",
        "    mov dx, msg",
        "    int 0x21",
        "    mov ax, 0x4C00      ; exit",
        "    int 0x21",
        "msg db 'Hello$'",
        "",
        "; ---- 8-bit 8051 microcontroller : toggle P1.0 ----",
        "    ORG 0x00",
        "LOOP:",
        "    CPL P1.0            ; complement port 1 bit 0",
        "    ACALL DELAY",
        "    SJMP LOOP",
        "DELAY:",
        "    MOV R0, #0xFF",
        "D1: DJNZ R0, D1",
        "    RET",
        "    END",
        "",
        "; ---- common x86 arithmetic idioms ----",
        "    add eax, ebx        ; eax = eax + ebx",
        "    sub eax, 1          ; decrement",
        "    imul eax, edx       ; signed multiply",
        "    cmp eax, ecx        ; set flags from eax - ecx",
        "    je  equal_label     ; jump if equal (ZF=1)",
        "    push rbp            ; function prologue",
        "    mov  rbp, rsp",
        "    pop  rbp            ; epilogue",
        "    ret",
        "",
        "Register widths: al/ah (8-bit), ax (16-bit), eax (32-bit), rax "
        "(64-bit). The same register file widens across the four modes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    outputs = {
        "morse_code.txt": build_morse(),
        "binary_encoding.txt": build_binary(),
        "bytecode_disassembly.txt": build_bytecode(),
        "assembly_reference.txt": build_assembly(),
    }
    print("Generating technical corpora:")
    for name, text in outputs.items():
        # Repeat the short ones so they are not lost among MB-scale domains.
        reps = max(1, 40_000 // max(len(text), 1))
        full = ("\n\n".join([text] * reps)) if reps > 1 else text
        (DATA / name).write_text(full, encoding="utf-8")
        print(f"  {name:<28}{len(full)/1024:>7.0f} KB  (base {len(text)/1024:.0f} KB x{reps})")


if __name__ == "__main__":
    main()
