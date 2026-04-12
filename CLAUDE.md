# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is This

NML (Neural Machine Language) is a minimal, deterministic machine language for AI workloads. It compiles to a single ~83 KB binary (zero dependencies beyond libc and libm) and provides 89 opcodes across 8 extensions covering inference, training, vision, transformers, signal processing, and M2M communication.

Key properties: 32 tensor registers (not scalar), tri-syntax (classic / symbolic Unicode / verbose), zero-ambiguity grammar designed for LLM generation.

## Build Commands

```bash
make nml                  # Standard build (89 opcodes, portable)
make nml-fast             # BLAS acceleration (Apple Accelerate on macOS, OpenBLAS on Linux)
make nml-cuda             # NVIDIA CUDA + cuBLAS GPU backend
make nml-sycl             # Intel SYCL GPU backend (requires icpx / Intel oneAPI)
make nml-metal            # Metal GPU + BLAS (macOS only, requires clang)
make nml-crypto           # Ed25519 + HMAC-SHA256 signing support
make nmld                 # Pre-fork worker pool daemon
make nml-wasm             # WebAssembly (requires emcc)
make release              # Build + strip

# Phase 5 — shared library for embedding
make libnml.so            # Linux shared library (-DNML_BUILD_LIB)
make libnml.dll           # Windows DLL
make libnml.a             # Static library
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

# Phase-level integration tests (each validates the phase + prior regressions)
make test-phase1          # Broadcasting, 4D vision (10b_vision_4d), cross-entropy loss
make test-phase2          # GPU CONV CPU-path validation, kernel fusion
make test-phase3          # N-layer TNET, BN, DROP, WDECAY
make test-phase4          # ONNX import (requires: pip install onnx numpy)
make test-phase5          # Python binding import + subprocess mode + test_python_binding.py
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

- **`nml.c`** — Single-file C99 runtime (~5,000 lines). All 89 opcodes, register file, memory model, instruction decoder, tri-syntax parser. Optional BLAS via `-DNML_USE_ACCELERATE` / `-DNML_USE_OPENBLAS`. Optional Metal via `-DNML_USE_METAL`. Build as library with `-DNML_BUILD_LIB`.
- **`nml_api.h`** — Public C library API (Phase 5). 10 functions: `nml_vm_create`, `nml_vm_destroy`, `nml_vm_load`, `nml_vm_load_mem`, `nml_vm_set_input`, `nml_vm_run`, `nml_vm_get_output`, `nml_vm_get_register`, `nml_vm_last_error`, `nml_version`. Opaque `nml_vm_t*` handle; no global state.
- **`nml_crypto.h`** — Standalone HMAC-SHA256 + Ed25519 (no external deps).
- **`tweetnacl.c/.h`** — Compact NaCl for Ed25519 ops.
- **`nmld.c`** — Daemon wrapping the runtime with pre-fork workers and binary cache.
- **`nml_backend_cuda.cu`** — NVIDIA CUDA kernels (Phase 2): CONV (im2col + cuBLAS batched GEMM), ATTN (chained GEMM + softmax), GELU, SOFTMAX, LAYERNORM. Elementwise fusion: CONV/MMUL immediately followed by RELU/SIGM/TANH/GELU on the same register executes in-place.
- **`nml_backend_sycl.cpp`** — Intel SYCL kernels (Phase 2): same op set as CUDA backend.

### Instruction Set Extensions

| Extension | Opcodes | What It Adds |
|-----------|---------|--------------|
| Core | 35 | Arithmetic, activation, memory, control flow, subroutines, tree |
| NML-V | 4 | Vision: CONV, POOL, UPSC, PADZ — accept 4D NCHW tensors (Phase 1) |
| NML-T | 4 | Transformer: ATTN, NORM, EMBD, GELU |
| NML-R | 4 | Reduction: RDUC, WHER, CLMP, CMPR |
| NML-S | 2 | Signal: FFT, FILT |
| NML-M2M | 12 | Machine-to-machine: META, FRAG, SIGN, VRFY, VOTE, PTCH, … |
| NML-TR | 21 | Training: BKWD, WUPD, LOSS, TNET (N-layer, Phase 3), TNDEEP, backward passes, config-driven training, BN (Phase 3), DROP (Phase 3) |
| NML-G | 5 | General-purpose: SYS, MOD, ITOF, FTOI, BNOT |

**Total: 89 opcodes** (was 85 at v0.9.0; BN and DROP added in Phase 3).

#### Phase 1 — Broadcasting and 4D Vision

- **Broadcasting (MADD, MSUB, EMUL, EDIV):** NumPy-style right-align rules. Dimensions broadcast when one side is 1. Example: `[4,3]` MADD `[3]` → right-aligns → broadcasts to `[4,3]`.
- **4D vision ops:** CONV/POOL/UPSC/PADZ now accept `[N,C,H,W]` input tensors and `[C_out,C_in,KH,KW]` kernel tensors (NCHW layout). Prior 2D behaviour is preserved.
- **Cross-entropy loss:** `LOSS #2` implements cross-entropy (was previously MSE only at `#0`).
- **GPU thresholds via env vars:** Override at runtime without recompiling:
  - `NML_CUDA_MMUL_THRESHOLD`, `NML_CUDA_EW_THRESHOLD`
  - `NML_SYCL_MMUL_THRESHOLD`, `NML_SYCL_EW_THRESHOLD`

#### Phase 3 — New Opcodes

- **`BN` (batch normalization):** symbolic `⊞`, verbose `BATCH_NORM`. Normalizes a batch tensor: `BN Rdst Rsrc Rgamma Rbeta`.
- **`DROP` (inverted dropout):** symbolic `≋`, verbose `DROPOUT`. `DROP Rdst Rsrc #rate` — applies inverted dropout with the given rate; at inference time pass `#0.0` to disable.
- **N-layer TNET:** `TNET R_config #epochs` trains arbitrary-depth MLPs (1–8 layers). Config tensor shape `[L,3]` where each row is `[input_size, hidden_size, output_size]`.
- **WDECAY:** L2 weight decay for AdamW optimizer is now fully implemented.

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
- `nml_from_onnx.py` — ONNX → NML program + `.nml.data` weight file (Phase 4). Handles 28 ONNX op types. Register allocation: R0 = primary input, R1–R28 = intermediate activations, initializers → named memory slots. Usage: `python3 transpilers/nml_from_onnx.py model.onnx [--dry-run]`
- `nml_to_hailo.py` — NML → ONNX → HEF compiler for Hailo NPU. Updated in Phase 4 to support CONV4D (was aborting on 4D CONV before).
- Various `*_gen.py` files — Training data generators (DPO, self-train, etc.)

### Python Binding (`python/`)

- `nml.py` — Python binding (Phase 5). Two modes:
  - **Library mode** (`VM` class via ctypes): `nml_vm_create/load/set_input/run/get_output/destroy`. Requires `libnml.so` / `libnml.dll`.
  - **Subprocess fallback** (`run_program()`, `infer()`): always works; uses the CLI binary.
  - numpy optional (falls back to plain Python lists).

### Test Files (`tests/`)

Key test files added across phases:
- `tests/opcode_coverage/17_broadcast.nml` — broadcasting (Phase 1)
- `tests/opcode_coverage/10b_vision_4d.nml` — 4D NCHW vision (Phase 1)
- `tests/opcode_coverage/15b_loss_ce.nml` — cross-entropy loss (Phase 1)
- `tests/opcode_coverage/18_gpu_conv.nml` — GPU CONV CPU-path (Phase 2)
- `tests/opcode_coverage/19_fusion.nml` — kernel fusion (Phase 2)
- `tests/opcode_coverage/20_nlayer_tnet.nml` — N-layer TNET (Phase 3)
- `tests/opcode_coverage/21_batchnorm.nml` — BN (Phase 3)
- `tests/opcode_coverage/22_dropout.nml` — DROP (Phase 3)
- `tests/opcode_coverage/23_wdecay.nml` — WDECAY (Phase 3)
- `tests/test_onnx_import.py` — end-to-end ONNX import (Phase 4)
- `tests/test_python_binding.py` — Python binding (Phase 5)

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

- 32 tensor registers, 256 named memory slots, 32-level call stack, 8-level loop nesting
- Default cycle limit: 1,000,000 (override: `--max-cycles N`)
- Error codes: -1 to -15 (see `docs/NML_SPEC.md`)

## Windows Build & Environment Variables

### Building on Windows

Requires [MSYS2](https://www.msys2.org/) (MinGW-w64 gcc) or any gcc-compatible toolchain. Run make targets from a Git Bash / MSYS2 shell:

```bash
make nml              # produces nml.exe
make libnml.dll       # Windows DLL for Python binding
make nml-cuda         # CUDA build — requires nvcc on PATH
make nml-sycl         # SYCL build — requires icpx (Intel oneAPI)
make release          # build + strip nml.exe
```

Intermediate `.o` files go into `build/` on Windows (not `/tmp/`). Override with:
```bash
make nml-cuda OBJDIR=my_build_dir
```

### Environment Variables — Runtime GPU Tuning

These control GPU dispatch without recompiling. Set before running `nml-cuda` or `nml-sycl`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NML_CUDA_MMUL_THRESHOLD` | 4096 | Min elements before routing MMUL to CUDA |
| `NML_CUDA_EW_THRESHOLD` | 16384 | Min elements before routing elementwise ops to CUDA |
| `NML_SYCL_MMUL_THRESHOLD` | 4096 | Min elements before routing MMUL to SYCL |
| `NML_SYCL_EW_THRESHOLD` | 16384 | Min elements before routing elementwise ops to SYCL |
| `CUDA_VISIBLE_DEVICES` | (all) | Select NVIDIA GPU(s), e.g. `0` or `0,1` |
| `ONEAPI_DEVICE_SELECTOR` | (auto) | Select SYCL device, e.g. `opencl:gpu` (Intel oneAPI) |
| `SYCL_DEVICE_FILTER` | (auto) | Alternative SYCL selector for AdaptiveCpp |

**PowerShell:**
```powershell
$env:NML_CUDA_MMUL_THRESHOLD = "8192"
$env:CUDA_VISIBLE_DEVICES    = "0"
$env:ONEAPI_DEVICE_SELECTOR  = "opencl:gpu"
.\nml-cuda.exe programs\anomaly_detector.nml programs\anomaly_weights.nml.data
```

**cmd.exe:**
```cmd
set NML_CUDA_MMUL_THRESHOLD=8192
set CUDA_VISIBLE_DEVICES=0
nml-cuda.exe programs\anomaly_detector.nml programs\anomaly_weights.nml.data
```

**Git Bash / MSYS2:**
```bash
NML_CUDA_MMUL_THRESHOLD=8192 CUDA_VISIBLE_DEVICES=0 ./nml-cuda.exe programs/anomaly_detector.nml
```

### Environment Variables — Python Training Tools

These affect `nml_corpus_builder.py`, `nml_verify_gen.py`, `nml_dpo_gen.py`, and `nml_selftrain_pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NML_RUNTIME` | `nml.exe` (Windows) / `nml` (Linux/Mac) | Path to the NML runtime binary used for execution-validation during corpus building and self-training |
| `PYTHONUTF8` | (unset) | Set to `1` to force UTF-8 output on Windows — already applied inside `run_all_generators.py` |
| `PYTHONIOENCODING` | (unset) | Set to `utf-8` as an alternative to `PYTHONUTF8` |

**Set `NML_RUNTIME` in PowerShell:**
```powershell
$env:NML_RUNTIME = "C:\Users\you\nml\nml.exe"
python transpilers\nml_corpus_builder.py
python transpilers\nml_verify_gen.py --model path\to\model
```

**Set `NML_RUNTIME` in cmd.exe:**
```cmd
set NML_RUNTIME=C:\Users\you\nml\nml.exe
python transpilers\nml_corpus_builder.py
```

**Persistent (PowerShell profile or System Properties):**
```powershell
[System.Environment]::SetEnvironmentVariable("NML_RUNTIME","C:\Users\you\nml\nml.exe","User")
```

### Fine-Tuning on Windows

The `domain-finetune` / `domain-finetune-merge` Make targets and `nml_selftrain_pipeline.py` use **MLX** (Apple Silicon only). On Windows, use Hugging Face `transformers` + `peft` (LoRA) instead:

```bash
pip install transformers peft datasets accelerate bitsandbytes
```

The JSONL files produced by `run_all_generators.py` → `nml_corpus_builder.py` (`train.jsonl`, `valid.jsonl`) are directly compatible with the HuggingFace `SFTTrainer` / `DPOTrainer` workflow. Pass `train.jsonl` to your trainer in place of the MLX pipeline.

## New Files (Phases 1–5)

| File | Phase | Description |
|------|-------|-------------|
| `runtime/nml_api.h` | 5 | Public C library API |
| `runtime/nml_backend_cuda.cu` | 2 | NVIDIA CUDA GPU kernels |
| `python/nml.py` | 5 | Python ctypes binding + subprocess fallback |
| `transpilers/nml_from_onnx.py` | 4 | ONNX → NML transpiler |
| `tests/test_onnx_import.py` | 4 | End-to-end ONNX import test |
| `tests/test_python_binding.py` | 5 | Python binding test suite |
| `tests/opcode_coverage/17_broadcast.nml` | 1 | Broadcasting test |
| `tests/opcode_coverage/10b_vision_4d.nml` | 1 | 4D NCHW vision test |
| `tests/opcode_coverage/18_gpu_conv.nml` | 2 | GPU CONV CPU-path validation |
| `tests/opcode_coverage/19_fusion.nml` | 2 | Kernel fusion test |
| `tests/opcode_coverage/20_nlayer_tnet.nml` | 3 | N-layer TNET test |
| `tests/opcode_coverage/21_batchnorm.nml` | 3 | BN opcode test |
| `tests/opcode_coverage/22_dropout.nml` | 3 | DROP opcode test |
| `tests/opcode_coverage/23_wdecay.nml` | 3 | WDECAY test |
