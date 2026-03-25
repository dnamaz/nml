# NML Runtime Backends

NML supports multiple compute backends selected at compile time.  All backends
share the same NML instruction set, register file, and memory model.  Only the
low-level tensor compute path changes.

---

## Dispatch Chain

Every MMUL (and, where applicable, activation) follows this priority order at
runtime.  The first backend that accepts the operation wins; the rest are
skipped.

```
NML_USE_SYCL  ──► SYCL GPU (Intel / AMD / NVIDIA via AdaptiveCpp)
                   │  returns -1 below threshold or on error
                   ▼
NML_USE_METAL ──► Metal GPU (macOS / iOS only)
                   │  returns -1 below threshold or on error
                   ▼
NML_HAS_BLAS  ──► CPU BLAS (Accelerate / OpenBLAS / oneMKL / AOCL)
                   │  F32 + F64 only; I32 falls through
                   ▼
              ──► CPU scalar loop (always available, all dtypes)
```

For Hailo NPU, dispatch happens at the **whole-program level** before the
interpreter starts — see [Hailo NPU](#8-hailo-npu-raspberry-pi-ai-hat).

---

## 1. Portable CPU (default)

| | |
|---|---|
| **Define** | *(none)* |
| **Build** | `make nml` |
| **Hardware** | Any CPU with C99 support |
| **Ops accelerated** | None — pure C scalar loops |
| **Dependencies** | libc, libm only |

The default build.  83 KB binary, zero external dependencies.  Suitable for
MCUs, embedded targets, and any platform where BLAS is unavailable.

```bash
gcc -O2 -o nml runtime/nml.c -lm
```

---

## 2. OpenBLAS (CPU)

| | |
|---|---|
| **Define** | `NML_USE_OPENBLAS` |
| **Build** | `make nml-fast` (Linux / Windows) |
| **Hardware** | x86-64, ARM64, POWER — any platform OpenBLAS supports |
| **Ops accelerated** | MMUL (F32, F64) |
| **Dependencies** | libopenblas |

OpenBLAS is auto-detected by `make nml-fast` on non-macOS platforms.  Uses
AVX2 / AVX-512 / NEON automatically based on what the CPU reports.

```bash
# Linux
sudo apt install libopenblas-dev
make nml-fast

# Cross-compile for ARM64 (e.g. Raspberry Pi)
aarch64-linux-gnu-gcc -O3 -march=armv8.2-a -DNML_USE_OPENBLAS \
    runtime/nml.c -lm -lopenblas -o nml-fast
```

**Raspberry Pi specific** — use the dedicated RPi target for Cortex-A76 tuning:

```bash
make nml-rpi          # -march=armv8.2-a -mcpu=cortex-a76
```

---

## 3. Apple Accelerate (CPU + AMX)

| | |
|---|---|
| **Define** | `NML_USE_ACCELERATE` |
| **Build** | `make nml-fast` (macOS) |
| **Hardware** | Apple Silicon (M-series), Intel Mac |
| **Ops accelerated** | MMUL (F32, F64) |
| **Dependencies** | Accelerate.framework (ships with Xcode) |

`make nml-fast` auto-selects Accelerate on macOS.  On Apple Silicon the
framework calls into the AMX (Apple Matrix Extensions) coprocessor for large
matrix multiplies.

```bash
make nml-fast
# Equivalent:
clang -O3 -march=native -DNML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK \
    runtime/nml.c -lm -framework Accelerate -o nml-fast
```

Also used alongside Metal in `make nml-metal` — Accelerate handles MMUL below
the Metal threshold and F64 workloads Metal cannot process.

---

## 4. Intel oneMKL CPU

| | |
|---|---|
| **Define** | `NML_USE_ONEMKL_CPU` |
| **Build** | `make nml-mkl-cpu` / `make nml-mkl-cpu-parallel` |
| **Hardware** | Intel Core, Xeon (any generation with AVX2+) |
| **Ops accelerated** | MMUL (F32, F64) |
| **Dependencies** | Intel oneAPI Base Toolkit (`icx`, `-qmkl`) |
| **Header** | `<mkl_cblas.h>` |

oneMKL is Intel's vendor-tuned BLAS.  It uses the same `cblas_sgemm` /
`cblas_dgemm` dispatch path as OpenBLAS — the only difference is the include
and link flags.  No GPU or SYCL runtime is required.

```bash
# Source the oneAPI environment first
source /opt/intel/oneapi/setvars.sh

make nml-mkl-cpu             # single-threaded MKL
make nml-mkl-cpu-parallel    # multi-threaded MKL (uses all cores)
```

**When to use over OpenBLAS:** Intel CPUs on Intel-optimised clusters or
production servers where you want vendor support and peak AMX-512/VNNI
throughput.

---

## 5. AMD AOCL (CPU)

| | |
|---|---|
| **Define** | `NML_USE_AOCL` |
| **Build** | `make nml-aocl` |
| **Hardware** | AMD EPYC, Ryzen (Zen2 and later recommended) |
| **Ops accelerated** | MMUL (F32, F64) |
| **Dependencies** | AOCL — `libopenblas` / `libblis-mt` |
| **Header** | `<cblas.h>` (AOCL installs a cblas-compatible header) |

AMD Optimizing CPU Libraries (AOCL) is AMD's equivalent of Intel oneMKL — a
vendor-tuned BLAS built on BLIS, optimised for Zen micro-architecture cache
hierarchy and AVX-512 FP units.

```bash
# Install AOCL: https://developer.amd.com/amd-aocl/
# Set AOCL_ROOT to your install prefix (default /opt/aocl)
make nml-aocl AOCL_ROOT=/opt/aocl

# Or directly:
gcc -O3 -march=native -DNML_USE_AOCL \
    -I/opt/aocl/include -L/opt/aocl/lib \
    runtime/nml.c -lm -lblis-mt -o nml-aocl
```

**When to use over OpenBLAS:** AMD EPYC servers in data centres or Ryzen
workstations where you want Zen-specific cache blocking and AVX-512 paths.

---

## 6. Apple Metal (GPU)

| | |
|---|---|
| **Define** | `NML_USE_METAL` |
| **Build** | `make nml-metal` (macOS only) |
| **Hardware** | Apple Silicon GPU (M1/M2/M3/M4), AMD Radeon (older Intel Macs) |
| **Ops accelerated** | MMUL (F32), via MPS `MPSMatrixMultiplication` |
| **Dispatch threshold** | `NML_METAL_MMUL_THRESHOLD` = 1,048,576 (m×n ≥ 1024×1024) |
| **Dependencies** | Metal.framework, MetalPerformanceShaders.framework, Accelerate.framework |
| **Implementation** | `runtime/nml_backend_metal.m` (Objective-C) |

Metal GPU is only profitable above the 1M-element threshold due to MPS
framework launch cost. General builds are memory-bounded (default 16M elements),
so Metal fires for large matmuls. Embedded/RPi targets use
`-DNML_MAX_TENSOR_SIZE=65536` (256×256), where Metal never fires — use BLAS there.

```bash
make nml-metal
```

Falls back to Accelerate (CPU BLAS) for sizes below threshold and for F64.

---

## 7. Intel SYCL GPU

| | |
|---|---|
| **Define** | `NML_USE_SYCL` |
| **Build** | `make nml-sycl` |
| **Hardware** | Intel Xe GPU, AMD GPU (via AdaptiveCpp / HIP), NVIDIA GPU (via AdaptiveCpp / CUDA) |
| **Ops accelerated** | MMUL (F32), RELU / SIGMOID / TANH (F32) |
| **Dispatch threshold** | `NML_SYCL_MMUL_THRESHOLD` = 4,096 (m×n ≥ 64×64) |
| **EW threshold** | `NML_SYCL_EW_THRESHOLD` = 4,096 elements |
| **Dependencies** | Intel oneAPI DPC++ (`icpx`), or AdaptiveCpp (`clang++ -fsycl`) |
| **Implementation** | `runtime/nml_backend_sycl.cpp` |

### GEMV vs GEMM dispatch

When m = 1 (single-program inference), the backend uses a **work-group
parallel reduction** over K to produce N output elements.  This launches
N × 32 work-items vs. the naive N, giving ~32× better GPU occupancy on small
matrices.

When m > 1 (batched inference from `nmld`), a standard 2-D `parallel_for`
GEMM is used.

### oneMKL upgrade path

Add `NML_USE_ONEMKL` to replace the custom kernels with Intel-tuned BLAS:

```cmake
cmake .. -DNML_SYCL=ON -DNML_SYCL_MKL=ON
# or:
make nml-sycl   # then link with -DNML_USE_ONEMKL -lmkl_sycl ...
```

| Kernel | Without oneMKL | With oneMKL |
|--------|---------------|-------------|
| GEMV (m=1) | Work-group reduction | `mkl::blas::gemv` |
| GEMM (m>1) | Naive 2-D parallel_for | `mkl::blas::gemm` |
| Activations | Custom 1-D parallel_for | Custom (unchanged) |

### Build

```bash
# Source the oneAPI environment
source /opt/intel/oneapi/setvars.sh

make nml-sycl              # portable custom kernels
# or via CMake:
cmake .. -DNML_SYCL=ON && make nml-sycl
cmake .. -DNML_SYCL=ON -DNML_SYCL_MKL=ON && make nml-sycl
```

### Device selection

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./nml-sycl program.nml   # Intel GPU
ONEAPI_DEVICE_SELECTOR=opencl:gpu     ./nml-sycl program.nml   # OpenCL
SYCL_DEVICE_FILTER=gpu                ./nml-sycl program.nml   # AdaptiveCpp
```

### Batch break-even (nmld)

Dispatching to SYCL is only profitable when enough programs are batched to
amortise the kernel-launch overhead:

| Hardware | Break-even batch | 2× CPU throughput |
|---|---|---|
| Intel Iris Xe (integrated, USM shared) | ~16 programs | ~50 programs |
| Intel Arc A770 (discrete, PCIe) | ~54 programs | ~150 programs |

Use `nmld-sycl` with a batch accumulator (flush at N=64 or 5 ms timeout).

---

## 8. Hailo NPU (Raspberry Pi AI HAT+)

| | |
|---|---|
| **Define** | `NML_USE_HAILO` |
| **Build** | `make nml-rpi-hailo` |
| **Hardware** | Hailo-8 (AI HAT+, 26 TOPS), Hailo-8L (AI Kit, 13 TOPS), Hailo-10H, Hailo-15H |
| **Dispatch level** | Whole-program (not per-op) |
| **Dependencies** | HailoRT ≥ 4.17 (`sudo apt install hailo-all`) |
| **Implementation** | `runtime/nml_backend_hailo.cpp`, `runtime/nml_backend_hailo.h` |

### How it differs from all other backends

Every other backend accelerates individual tensor operations (MMUL, RELU, …)
while the NML interpreter continues to run the program.  The Hailo backend
**replaces the NML interpreter entirely** for programs that have a pre-compiled
HEF (Hailo Executable Format) file.

```
Other backends:   NML interpreter → per-op dispatch → GPU/CPU
Hailo backend:    HEF detected → skip interpreter → Hailo NPU runs whole program
```

Weights are **baked into the HEF** at compile time.  Only runtime inputs
(sensor readings, etc.) are passed at inference time.

### I/O name matching

HEF input/output stream names are matched to NML memory slot labels by exact
string comparison.  `transpilers/nml_to_hailo.py` preserves the `.nml.data`
slot names when generating the HEF, so matching is automatic.

### Deployment modes

#### Mode A — HEF sidecar file (default)

```
programs/
  anomaly_detector.nml
  anomaly_detector.hailo8l.hef   ← produced offline, deployed alongside
  anomaly_weights.nml.data
```

HEF lookup priority at runtime (first found wins):

1. `$HAILO_ARCH` env var → `<stem>.<env_arch>.hef`
2. Auto-detected device arch → `<stem>.<device_arch>.hef`
3. Generic fallback → `<stem>.hef`

Arch is queried once from the device via `nml_hailo_arch()` and cached.

```bash
./nml-rpi-hailo programs/anomaly_detector.nml programs/anomaly_weights.nml.data
# [NML] Hailo HEF: programs/anomaly_detector.hailo8l.hef (arch: hailo8l) — dispatching to NPU
```

#### Mode B — Embedded HEF (single self-contained binary)

The HEF is compiled into the binary as a C byte-array.  No file I/O at
runtime — ship one binary, nothing else.

```bash
make nml-rpi-hailo-embed \
     PROGRAM=programs/anomaly_detector.nml \
     DATA=programs/anomaly_weights.nml.data \
     ARCH=hailo8l
# Built: programs/anomaly_detector-standalone

./programs/anomaly_detector-standalone \
     programs/anomaly_detector.nml \
     programs/anomaly_weights.nml.data
# [NML] Hailo embedded HEF (45312 bytes) — dispatching to NPU
```

The three-step build pipeline runs automatically via Make dependencies:

```
.nml + .nml.data
    │  python3 transpilers/nml_to_hailo.py --arch <ARCH>
    ▼
program.<arch>.hef
    │  python3 -c "hex(b) for b in data..."
    ▼
runtime/nml_hef_resource.h
    │  g++ -DNML_USE_HAILO -DNML_EMBEDDED_HEF
    ▼
program-standalone
```

### HEF compilation (offline, dev machine only)

Python and the Hailo Dataflow Compiler are required **only on the dev machine**,
never on the Raspberry Pi.

```bash
# Install on dev machine (x86 Linux, Ubuntu 20.04/22.04)
pip install hailo-sdk-client hailo-sdk-common onnx numpy

# Compile for a specific chip
python3 transpilers/nml_to_hailo.py program.nml weights.nml.data --arch hailo8l

# Or use Docker (any OS)
docker run --rm -v $(pwd):/nml hailo/hailo_sdk:latest \
    python3 /nml/transpilers/nml_to_hailo.py \
        /nml/programs/anomaly_detector.nml \
        /nml/programs/anomaly_weights.nml.data --arch hailo8l

# Pre-compile a full library for all chip variants
for arch in hailo8 hailo8l hailo10h; do
    for prog in programs/*.nml; do
        data="${prog%.nml}.nml.data"
        [ -f "$data" ] && python3 transpilers/nml_to_hailo.py "$prog" "$data" --arch "$arch"
    done
done
```

### Supported NML ops for HEF compilation

`transpilers/nml_to_hailo.py` maps these NML ops to ONNX:

| NML op | ONNX node |
|--------|-----------|
| MMUL | MatMul |
| MADD | Add |
| MSUB | Sub |
| EMUL | Mul |
| RELU | Relu |
| SIGM | Sigmoid |
| TANH | Tanh |
| GELU | Gelu |
| SOFT | Softmax |
| NORM | LayerNormalization |

Programs using unsupported ops (CONV, ATTN, LOOP, CALL/RET, training ops, …)
abort compilation with a clear error — use the CPU/SYCL path for those.

### Chip variants

| `--arch` | Chip | TOPS | Product |
|---|---|---|---|
| `hailo8l` | Hailo-8L | 13 | Raspberry Pi AI Kit |
| `hailo8` | Hailo-8 | 26 | Raspberry Pi AI HAT+ |
| `hailo10h` | Hailo-10H | ~40 | AI HAT+ 2 (if applicable) |
| `hailo15h` | Hailo-15H | ~45+ | Hailo-15H platforms |

HEF files are chip-specific but not chip-instance-specific — a HEF compiled
for `hailo8l` runs on any Hailo-8L device.

### Environment variables

| Variable | Effect |
|---|---|
| `HAILO_ARCH` | Pin chip variant for HEF lookup and `nml_to_hailo.py` default arch |
| `HAILO_DEVICE_ID` | Pin to a specific PCIe device (BDF string, e.g. `0000:01:00.0`) |

---

## Backend Comparison

| Backend | Hardware | Ops | Build define | Build target |
|---------|----------|-----|-------------|--------------|
| Portable | Any CPU | — | *(none)* | `make nml` |
| OpenBLAS | Any x86/ARM | MMUL | `NML_USE_OPENBLAS` | `make nml-fast` |
| Accelerate | Apple Silicon/Mac | MMUL | `NML_USE_ACCELERATE` | `make nml-fast` |
| oneMKL CPU | Intel CPU | MMUL | `NML_USE_ONEMKL_CPU` | `make nml-mkl-cpu` |
| AOCL | AMD Zen CPU | MMUL | `NML_USE_AOCL` | `make nml-aocl` |
| Metal | Apple GPU | MMUL | `NML_USE_METAL` | `make nml-metal` |
| SYCL | Intel/AMD/NVIDIA GPU | MMUL, RELU, SIGMOID, TANH | `NML_USE_SYCL` | `make nml-sycl` |
| SYCL + oneMKL | Intel GPU | MMUL (tuned), RELU, SIGMOID, TANH | `NML_USE_SYCL` + `NML_USE_ONEMKL` | `make nml-sycl` |
| Hailo NPU | Hailo-8/8L/10H/15H | Whole program | `NML_USE_HAILO` | `make nml-rpi-hailo` |
| Hailo embedded | Hailo-8/8L/10H/15H | Whole program | `NML_USE_HAILO` + `NML_EMBEDDED_HEF` | `make nml-rpi-hailo-embed` |

### Key thresholds

| Backend | Threshold define | Default | Meaning |
|---------|----------------|---------|---------|
| SYCL MMUL | `NML_SYCL_MMUL_THRESHOLD` | 4,096 | min m×n for GPU dispatch |
| SYCL activation | `NML_SYCL_EW_THRESHOLD` | 4,096 | min elements for GPU dispatch |
| Metal MMUL | `NML_METAL_MMUL_THRESHOLD` | 1,048,576 | min m×n for MPS dispatch |

Override at compile time:
```bash
gcc -DNML_SYCL_MMUL_THRESHOLD=1024 ...   # lower threshold for small matrices
```

---

## Adding a New Backend

1. Create `runtime/nml_backend_<name>.h` — C interface with `extern "C"` guards
2. Create `runtime/nml_backend_<name>.cpp` (or `.c`) — implementation
3. Add `#ifdef NML_USE_<NAME>` include + extern declarations in `runtime/nml.c`
   near the other backend blocks (lines ~63–85)
4. Add dispatch in `tensor_matmul()` following the existing pattern (line ~320)
5. Add a `make nml-<name>` target in `Makefile`
6. Add a `cmake .. -DNML_<NAME>=ON` option in `CMakeLists.txt`
7. Document here

The `-1` return convention is the contract: return `0` on success, `-1` to
fall through to the next backend in the chain.
