# NML-G — General Purpose Extension Specification

## Overview

NML-G extends the NML instruction set with general-purpose computing capabilities while preserving NML's core properties: tiny runtime, AI-generatability, auditability, and portability.

NML-G adds **5 new instructions** organized around two principles:

1. **SYS** — a single multiplexed syscall instruction for all host I/O
2. **Missing math primitives** — MOD, type conversion, bitwise NOT

## Design Philosophy

Rather than adding dozens of specialized instructions, NML-G follows the RISC-V `ecall` pattern: a single `SYS` instruction dispatches to host-provided operations via a numeric code. This keeps the opcode space compact (1 new opcode covers all I/O) while being infinitely extensible.

## New Instructions (5)

### SYS — System Call (I/O, time, random, exit)

```
SYS  Rd #code       ; Rd = syscall(code, context from other registers)
```

| Code | Name | Behavior | Input | Output |
|------|------|----------|-------|--------|
| 0 | `PRINT_NUM` | Print Rd[0] as number + newline to stdout | Rd = value | — |
| 1 | `PRINT_CHAR` | Print Rd[0] as ASCII character (no newline) | Rd = char code | — |
| 2 | `READ_NUM` | Read number from stdin into Rd | — | Rd = parsed number |
| 3 | `READ_CHAR` | Read single character from stdin into Rd | — | Rd = char code |
| 4 | `TIME` | Current wall-clock time (seconds since epoch) | — | Rd = timestamp |
| 5 | `RAND` | Pseudo-random float in [0, 1) | — | Rd = random |
| 6 | `EXIT` | Terminate with exit code Rd[0] | Rd = code | (no return) |
| 7 | `PRINT_MEM` | Print all elements of @label as CSV | R1 = @label slot index | — |
| 8 | `ARGC` | Argument count (future) | — | Rd = count |
| 9 | `ARGV` | Get argument N (future) | R1 = index | Rd = value |

**Tri-syntax:**

| Classic | Symbolic | Verbose |
|---------|----------|---------|
| `SYS` | `⚙` (U+2699) | `SYSTEM` |

**Example — Hello World (print character codes):**
```
LEAF  R0 #72        ; 'H'
SYS   R0 #1         ; print char
LEAF  R0 #105       ; 'i'
SYS   R0 #1
LEAF  R0 #10        ; newline
SYS   R0 #1
HALT
```

**Example — Print a computed result:**
```
LEAF  R0 #42.0
LEAF  R1 #2.0
EMUL  R0 R0 R1      ; R0 = 84.0
SYS   R0 #0          ; prints "84.000000"
HALT
```

### MOD — Integer Modulo

```
MOD  Rd Rs1 Rs2     ; Rd = (int)Rs1[0] % (int)Rs2[0]
```

Essential for: hash functions, FizzBuzz, cycle detection, round-robin, divisibility tests.

**Tri-syntax:**

| Classic | Symbolic | Verbose |
|---------|----------|---------|
| `MOD` | `%` | `MODULO` |

**Example — Is N even?**
```
LEAF  R0 #17.0
LEAF  R1 #2.0
MOD   R2 R0 R1      ; R2 = 1 (odd)
CMPI  RE R2 #0.5    ; R2 < 0.5? → flag = 0 (no, it's 1)
JMPF  #1            ; flag is false → skip
ST    R0 @even_result
HALT
```

### ITOF — Integer to Float

```
ITOF  Rd Rs         ; Rd = (float)Rs[0] where Rs is i32
```

Converts i32 tensor values to f32. Needed when mixing integer computation with floating-point ML pipelines.

**Tri-syntax:**

| Classic | Symbolic | Verbose |
|---------|----------|---------|
| `ITOF` | `⊶` (U+22B6) | `INT_TO_FLOAT` |

### FTOI — Float to Integer

```
FTOI  Rd Rs         ; Rd = (int)Rs[0] (truncates toward zero)
```

Converts f32/f64 tensor values to i32. Needed for indexing, character codes, loop counters.

**Tri-syntax:**

| Classic | Symbolic | Verbose |
|---------|----------|---------|
| `FTOI` | `⊷` (U+22B7) | `FLOAT_TO_INT` |

### BNOT — Bitwise NOT (Integer)

```
BNOT  Rd Rs         ; Rd = ~(int)Rs[0] (bitwise complement)
```

Combined with the existing EMUL (multiply) and MADD (add), this enables all bitwise logic:
- AND: `EMUL Rd Rs1 Rs2` (when values are 0/1 masks from CMPR)
- OR: `MADD` + `CLMP` (add masks, clamp to 0..1)
- XOR: combine BNOT, EMUL, MADD

**Tri-syntax:**

| Classic | Symbolic | Verbose |
|---------|----------|---------|
| `BNOT` | `¬` (U+00AC) | `BITWISE_NOT` |

## Configurable Resource Limits

NML-G makes all hard limits configurable at compile time:

```c
// Override at compile: gcc -DNML_MAX_INSTRUCTIONS=65536 ...
#ifndef NML_MAX_INSTRUCTIONS
  #define NML_MAX_INSTRUCTIONS 8192
#endif
```

| Resource | Default | NML-G Recommended | Flag |
|----------|---------|-------------------|------|
| Max instructions | 8,192 | 65,536 | `-DNML_MAX_INSTRUCTIONS=65536` |
| Max tensor elements | 65,536 | 262,144 | `-DNML_MAX_TENSOR_SIZE=262144` |
| Max memory slots | 64 | 256 | `-DNML_MAX_MEMORY_SLOTS=256` |
| Max loop depth | 8 | 32 | `-DNML_MAX_LOOP_DEPTH=32` |
| Max call depth | 32 | 128 | `-DNML_MAX_CALL_DEPTH=128` |
| Max cycles | 1,000,000 | 100,000,000 | `--max-cycles 100000000` |

Build with all general-purpose limits:
```bash
gcc -O2 -o nml-gp nml.c -lm \
    -DNML_MAX_INSTRUCTIONS=65536 \
    -DNML_MAX_TENSOR_SIZE=262144 \
    -DNML_MAX_MEMORY_SLOTS=256 \
    -DNML_MAX_LOOP_DEPTH=32 \
    -DNML_MAX_CALL_DEPTH=128
```

## What NML-G Enables

### Console Programs
```
; Read two numbers, print their sum
SYS   R0 #2         ; read number → R0
SYS   R1 #2         ; read number → R1
TACC  R0 R0 R1      ; R0 = R0 + R1
SYS   R0 #0         ; print result
HALT
```

### Algorithms (FizzBuzz, Sorting, etc.)
```
; FizzBuzz 1-100
LEAF  RD #1.0                 ; counter = 1
LEAF  R3 #3.0
LEAF  R5 #5.0
LEAF  R9 #15.0
; -- loop body --
MOD   R0 RD R9               ; R0 = counter % 15
CMPI  RE R0 #0.5
JMPF  #3                     ; if R0 < 0.5 (== 0) → FizzBuzz
SYS   RD #0                  ;   ... handle other cases ...
JUMP  #N                     ;   skip to increment
; ... (fizz, buzz, number branches) ...
HALT
```

### Interactive Programs
Programs can now read input and produce output during execution, not just at the end via register dumps.

### Timing and Benchmarking
```
SYS   R0 #4         ; start time
; ... computation ...
SYS   R1 #4         ; end time
MSUB  R2 R1 R0      ; elapsed
SYS   R2 #0         ; print elapsed seconds
HALT
```

## Opcode Budget

| Extension | Opcodes | Running Total |
|-----------|---------|---------------|
| Core | 35 | 35 |
| NML-V | 4 | 39 |
| NML-T | 4 | 43 |
| NML-R | 4 | 47 |
| NML-S | 2 | 49 |
| NML-M2M | 13 | 62 |
| **NML-G** | **5** | **67** |

This exceeds the current 6-bit (64) opcode encoding. Options:
1. Expand to 7-bit opcodes (128 max) — recommended
2. Replace SYNC (currently a no-op) to reclaim 1 slot
3. Use SYS multiplexing more aggressively (MOD/ITOF/FTOI as SYS subcodes)

Since NML assembles from text at load time, expanding the encoding is a one-line change with no backward compatibility impact on `.nml` source files.

## Future NML-G Additions (v2)

Reserved SYS codes 10-31 for future use:

| Code | Name | Description |
|------|------|-------------|
| 10 | `FILE_OPEN` | Open file, return handle |
| 11 | `FILE_READ` | Read from file handle |
| 12 | `FILE_WRITE` | Write to file handle |
| 13 | `FILE_CLOSE` | Close file handle |
| 14 | `STR_LEN` | String length |
| 15 | `STR_CMP` | String comparison |
| 16 | `ENV_GET` | Read environment variable |
| 17 | `SLEEP` | Sleep for N milliseconds |

## Version

- NML-G v0.1 — Initial specification (5 instructions, configurable limits)
- Requires NML runtime v0.6+
