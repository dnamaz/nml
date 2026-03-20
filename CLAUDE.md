# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is This

NML (Neural Machine Language) is a minimal, deterministic machine language for AI workloads. It compiles to a single ~83 KB binary (zero dependencies beyond libc and libm) and provides 85 opcodes across 8 extensions covering inference, training, vision, transformers, signal processing, and M2M communication.

Key properties: 32 tensor registers (not scalar), tri-syntax (classic / symbolic Unicode / verbose), zero-ambiguity grammar designed for LLM generation.

## Build Commands

```bash
make nml                  # Standard build (85 instructions, portable)
make nml-fast             # BLAS acceleration (Apple Accelerate on macOS, OpenBLAS on Linux)
make nml-metal            # Metal GPU + BLAS (macOS only, requires clang)
make nml-crypto           # Ed25519 + HMAC-SHA256 signing support
make nmld                 # Pre-fork worker pool daemon
make nml-wasm             # WebAssembly (requires emcc)
make release              # Build + strip
```

Direct gcc compilation (minimal):
```bash
gcc -O2 -o nml runtime/nml.c -lm
gcc -O2 -o nml runtime/nml.c -lm -DNML_NO_DEFAULT_EXTENSIONS  # core 35 opcodes only
```

## Running Programs

```bash
./nml programs/hello_world.nml
./nml programs/anomaly_detector.nml programs/anomaly_weights.nml.data
./nml <program.nml> [data.nml.data] [--trace] [--max-cycles N] [--fragment NAME] [--describe]

# Crypto build: sign, patch, verify+execute
./nml-crypto --sign <program.nml> --key <hex> [--agent <name>]
./nml-crypto <program.nml> --patch <patch.ptch>
./nml-crypto <signed.nml> [data.nml.data]
```

## Tests

```bash
make test                 # All core tests (anomaly, extensions, symbolic, verbose, features, gp)
make test-anomaly         # 3-layer NN inference
make test-extensions      # Vision/transformer/reduction/signal
make test-symbolic        # Unicode symbolic syntax
make test-verbose         # Verbose syntax
make test-features        # SDIV, EDIV, CMP, CALL/RET, backward jumps
make test-gp              # Hello world, fibonacci, fizzbuzz, primes
```

Run a single test program directly:
```bash
./nml tests/test_features.nml 2>&1 | grep -E "(HALTED|sdiv|cmpi)"

# Per-family opcode coverage tests (01_arithmetic through 16_backward)
./nml tests/opcode_coverage/01_arithmetic.nml
./nml tests/opcode_coverage/11_transformer.nml
./nml tests/opcode_coverage/15_training.nml
```

## Architecture

### Runtime (`runtime/`)

- **`nml.c`** — Single-file C99 runtime (3,647 lines). All 85 opcodes, register file, memory model, instruction decoder, tri-syntax parser. Optional BLAS via `-DNML_USE_ACCELERATE` / `-DNML_USE_OPENBLAS`. Optional Metal via `-DNML_USE_METAL`.
- **`nml_crypto.h`** — Standalone HMAC-SHA256 + Ed25519 (no external deps).
- **`tweetnacl.c/.h`** — Compact NaCl for Ed25519 ops.
- **`nmld.c`** — Daemon wrapping the runtime with pre-fork workers and binary cache.

### Instruction Set Extensions

| Extension | Opcodes | What It Adds |
|-----------|---------|--------------|
| Core | 35 | Arithmetic, activation, memory, control flow, subroutines, tree |
| NML-V | 4 | Vision: CONV, POOL, UPSC, PADZ |
| NML-T | 4 | Transformer: ATTN, NORM, EMBD, GELU |
| NML-R | 4 | Reduction: RDUC, WHER, CLMP, CMPR |
| NML-S | 2 | Signal: FFT, FILT |
| NML-M2M | 12 | Machine-to-machine: META, FRAG, SIGN, VRFY, VOTE, PTCH, … |
| NML-TR | 19 | Training: BKWD, WUPD, LOSS, TNET, TNDEEP, backward passes, config-driven training |
| NML-G | 5 | General-purpose: SYS, MOD, ITOF, FTOI, BNOT |

### Data Files (`.nml.data`)

Named tensor definitions loaded at runtime:
```
@tensor_name shape=rows,cols dtype=f32 data=val1,val2,...
@weights shape=4,3 dtype=f32 data=0.1,0.2,...
```
In programs: `LD R0 @weights` / `ST RA @result`

### Transpilers (`transpilers/`)

Python tooling for grammar validation, LLM constrained decoding, training data generation, and code transpilation:
- `nml_grammar.py` — Full formal grammar validator (all syntax variants)
- `nml_lark_grammar.py` — Lark CFG for Outlines constrained decoding
- `nml_executor.py` — Program retriever + executor + tracer
- `nml_semantic.py` — Semantic type checking
- `nml_to_mojo.py` — NML → Mojo transpiler
- Various `*_gen.py` files — Training data generators (DPO, self-train, etc.)

### HTTP Server & MCP (`serve/`)

```bash
# Full stack (server port 8082 + RAG gateway port 8083 + UI)
bash serve/start_agents.sh
make agent-start

# Headless
make agent-start-headless

# Status
make agent-status         # curl localhost:8082/health + localhost:8083/health

# Standalone
python3 serve/nml_server.py --http --port 8082
python3 serve/nml_server.py --transport stdio  # MCP mode for Cursor/Claude
```

MCP tools exposed: `nml_spec`, `nml_validate`, `nml_execute`, `nml_format`.

### LSP Server (`lsp/`)

VS Code / editor integration. Launch with:
```bash
python3 lsp/nml_lsp/server.py
```
Provides hover docs, go-to-definition, completion, diagnostics, and semantic tokens. Opcode metadata lives in `lsp/nml_lsp/opcode_db.py`.

### Terminal / UI (`terminal/`)

- `nml_terminal.jsx` — React WebView NML emulator (runs WASM build)
- `nml_chat.jsx` — LLM chat interface for code generation

### Domain Layer (`domain/`)

Tax calculation domain: transpilers generate NML programs from real tax data, fine-tune Mistral/Qwen2.5-Coder models via MLX LoRA, and serve a RAG gateway.

```bash
make domain-test-tax
make domain-transpile-library       # Classic syntax
make domain-transpile-library-symbolic
make domain-finetune                # MLX LoRA fine-tune
make domain-finetune-merge          # Merge LoRA adapters
make domain-rag-server
```

## Syntax Variants

Every instruction can be written three ways:
- **Classic:** `MMUL R0 R1 R2`
- **Symbolic:** `⊗ ι κ λ` (Unicode, token-efficient)
- **Verbose:** `matrix.multiply dest=R0 left=R1 right=R2`

The runtime accepts all three interchangeably. Registers have classic (`R0`–`R9`, `RA`–`RV`) and Greek aliases (`ι`, `κ`, `λ`, …).

## Runtime Limits

- 32 tensor registers, 64 named memory slots, 32-level call stack, 8-level loop nesting
- Default cycle limit: 1,000,000 (override: `--max-cycles N`)
- Error codes: -1 to -15 (see `docs/NML_SPEC.md`)
