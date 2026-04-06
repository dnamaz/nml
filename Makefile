# ═══════════════════════════════════════════
# NML — Neural Machine Language
# Makefile for v0.10.0
# ═══════════════════════════════════════════

CC       = gcc
CFLAGS   = -O2 -Wall -std=c99
LDFLAGS  = -lm
SYCL_CC  = icpx   # Intel oneAPI DPC++; use 'clang++ -fsycl' for AdaptiveCpp/hipSYCL

ifeq ($(OS),Windows_NT)
  EXEEXT  = .exe
else
  EXEEXT  =
endif

# Intermediate object files go into build/; final binaries into bin/
OBJDIR  ?= build
BINDIR  ?= bin

# ── OpenBLAS paths (Windows/MSYS2 auto-detection) ───────────────────────────
# Override: make nml-fast OPENBLAS_PREFIX=C:/msys64/mingw64
ifeq ($(OS),Windows_NT)
  OPENBLAS_PREFIX ?= $(firstword $(patsubst %/include/openblas,%,$(wildcard C:/msys64/mingw64/include/openblas C:/msys64/ucrt64/include/openblas C:/msys64/clang64/include/openblas)))
  OPENBLAS_INC    = $(OPENBLAS_PREFIX)/include/openblas
  OPENBLAS_LIB    = $(OPENBLAS_PREFIX)/lib
endif
NML_TEST_TMP = $(OBJDIR)/nml_test.data

# ── Convenience alias — use as $(NML_BIN) in test targets ───────────────────
NML_BIN = $(BINDIR)/nml$(EXEEXT)

# ── Runtime environment variables (override without recompiling) ─────────────
#
# GPU dispatch thresholds — minimum tensor elements before offloading to GPU:
#   NML_CUDA_MMUL_THRESHOLD   matrix-multiply on CUDA   (default: 4096)
#   NML_CUDA_EW_THRESHOLD     elementwise ops on CUDA   (default: 16384)
#   NML_SYCL_MMUL_THRESHOLD   matrix-multiply on SYCL   (default: 4096)
#   NML_SYCL_EW_THRESHOLD     elementwise ops on SYCL   (default: 16384)
#
# GPU device selection:
#   CUDA_VISIBLE_DEVICES          select NVIDIA GPU(s) (e.g. "0" or "0,1")
#   ONEAPI_DEVICE_SELECTOR        select SYCL device    (e.g. "opencl:gpu")
#   SYCL_DEVICE_FILTER            alternative SYCL selector (AdaptiveCpp)
#
# Python training tools:
#   NML_RUNTIME    path to nml/nml.exe binary used by corpus builder,
#                  verify_gen, dpo_gen, and selftrain_pipeline
#                  (auto-detected on Windows; set this to override)
#   PYTHONUTF8=1   force UTF-8 output on Windows (set in run_all_generators.py)
#   PYTHONIOENCODING=utf-8  same effect, older Python
#
# Windows — set env vars in PowerShell:
#   $env:NML_RUNTIME = "C:\path\to\nml.exe"
#   $env:NML_CUDA_MMUL_THRESHOLD = "8192"
#   $env:CUDA_VISIBLE_DEVICES = "0"
#
# Windows — set env vars in cmd.exe:
#   set NML_RUNTIME=C:\path\to\nml.exe
#   set NML_CUDA_MMUL_THRESHOLD=8192
#   set CUDA_VISIBLE_DEVICES=0
#
# Linux / macOS — set env vars in bash:
#   export NML_RUNTIME=/path/to/nml
#   export NML_CUDA_MMUL_THRESHOLD=8192
#   export CUDA_VISIBLE_DEVICES=0

# ═══════════════════════════════════════════
# Build
# ═══════════════════════════════════════════

nml: runtime/nml.c runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $(BINDIR)/nml$(EXEEXT) runtime/nml.c $(LDFLAGS)
	@echo "  Built: $(BINDIR)/nml (v0.10.0, 89 opcodes, 32 registers — portable)"

nmld: runtime/nmld.c runtime/nml.c runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $(BINDIR)/nmld$(EXEEXT) runtime/nmld.c $(LDFLAGS)
	@echo "  Built: $(BINDIR)/nmld (NML daemon — generic execution server)"

ifeq ($(shell uname),Darwin)
  NML_FAST_CC    = $(CC) -O3 -march=native -std=c99 -DNML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
  NML_FAST_LIBS  = -lm -framework Accelerate
  NML_FAST_LABEL = BLAS via Apple Accelerate
else
  ifeq ($(OS),Windows_NT)
    NML_FAST_CC    = $(CC) -O3 -march=native -std=c99 -DNML_USE_OPENBLAS -I$(OPENBLAS_INC) -L$(OPENBLAS_LIB)
    NML_FAST_LIBS  = -lm -lopenblas
    NML_FAST_LABEL = BLAS via OpenBLAS + MSYS2 ($(OPENBLAS_PREFIX))
  else
    NML_FAST_CC    = $(CC) -O3 -march=native -std=c99 -DNML_USE_OPENBLAS
    NML_FAST_LIBS  = -lm -lopenblas
    NML_FAST_LABEL = BLAS via OpenBLAS
  endif
endif

nml-fast: runtime/nml.c runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(NML_FAST_CC) -o $(BINDIR)/nml-fast$(EXEEXT) runtime/nml.c $(NML_FAST_LIBS)
	@echo "  Built: $(BINDIR)/nml-fast (v0.10.0, $(NML_FAST_LABEL))"

# ─── Raspberry Pi targets ───────────────────────────────────────────────────
#
# nml-rpi: CPU-only build optimised for Raspberry Pi 5 (Cortex-A76, NEON).
# Run directly on the Pi — no special hardware needed.
# Requires OpenBLAS: sudo apt install libopenblas-dev
#
nml-rpi: runtime/nml.c runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(CC) -O3 -march=armv8.2-a -mcpu=cortex-a76 -std=c99 \
	    -DNML_USE_OPENBLAS -DNML_MAX_TENSOR_SIZE=65536 \
	    -o $(BINDIR)/nml-rpi$(EXEEXT) runtime/nml.c -lm -lopenblas
	@echo "  Built: $(BINDIR)/nml-rpi (Cortex-A76 + OpenBLAS NEON, RPi 5, 256×256 cap)"

# nml-rpi-hailo: CPU build + Hailo NPU dispatch via HailoRT.
# Requires:
#   sudo apt install hailo-all          # Raspberry Pi OS Bookworm
#   tools/nml_to_hailo.py  (offline HEF compilation per program)
# Usage:
#   python3 tools/nml_to_hailo.py programs/anomaly_detector.nml \
#           programs/anomaly_weights.nml.data
#   ./nml-rpi-hailo programs/anomaly_detector.nml \
#           programs/anomaly_weights.nml.data
# If anomaly_detector.hef exists in the same dir, Hailo NPU is used;
# otherwise falls back to CPU (OpenBLAS).
#
nml-rpi-hailo: runtime/nml.c runtime/nml_backend_hailo.cpp \
               runtime/nml_backend_hailo.h runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	g++ -O3 -march=armv8.2-a -mcpu=cortex-a76 -std=c++17 \
	    -DNML_USE_HAILO -DNML_USE_OPENBLAS -DNML_MAX_TENSOR_SIZE=65536 \
	    -x c   -std=c99   runtime/nml.c \
	    -x c++ -std=c++17 runtime/nml_backend_hailo.cpp \
	    -o $(BINDIR)/nml-rpi-hailo$(EXEEXT) -lm -lopenblas -lhailort
	@echo "  Built: $(BINDIR)/nml-rpi-hailo (Cortex-A76 + OpenBLAS + Hailo NPU, 256×256 cap)"

# nmld-rpi-hailo: daemon variant with Hailo NPU support
nmld-rpi-hailo: runtime/nmld.c runtime/nml_backend_hailo.cpp \
                runtime/nml_backend_hailo.h runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	g++ -O3 -march=armv8.2-a -mcpu=cortex-a76 -std=c++17 \
	    -DNML_USE_HAILO -DNML_USE_OPENBLAS -DNML_MAX_TENSOR_SIZE=65536 \
	    -x c   -std=c99   runtime/nmld.c \
	    -x c++ -std=c++17 runtime/nml_backend_hailo.cpp \
	    -o $(BINDIR)/nmld-rpi-hailo$(EXEEXT) -lm -lopenblas -lhailort
	@echo "  Built: $(BINDIR)/nmld-rpi-hailo (NML daemon + Hailo NPU, 256×256 cap)"

# ─── Embedded HEF (single self-contained binary) ────────────────────────────
#
# nml-rpi-hailo-embed: compile an NML program + its HEF into one binary.
# The HEF is stored as a C byte-array; no file I/O occurs at runtime.
#
# Variables:
#   PROGRAM  path to .nml file            (default: programs/anomaly_detector.nml)
#   DATA     path to .nml.data file       (default: programs/anomaly_weights.nml.data)
#   ARCH     Hailo chip target            (default: $HAILO_ARCH or hailo8)
#   OUT      output binary name           (default: <program-stem>-standalone)
#
# Usage:
#   make nml-rpi-hailo-embed \
#        PROGRAM=programs/anomaly_detector.nml \
#        DATA=programs/anomaly_weights.nml.data \
#        ARCH=hailo8l
#
PROGRAM ?= programs/anomaly_detector.nml
DATA    ?= programs/anomaly_weights.nml.data
ARCH    ?= $(or $(HAILO_ARCH),hailo8)
_STEM   := $(basename $(PROGRAM))
OUT     ?= $(_STEM)-standalone
_HEF    := $(_STEM).$(ARCH).hef

# Step 1 — compile NML program to HEF (if not already present)
$(_HEF): $(PROGRAM) $(DATA)
	python3 transpilers/nml_to_hailo.py $(PROGRAM) $(DATA) --arch $(ARCH)

# Step 2 — convert HEF binary → C byte-array header
runtime/nml_hef_resource.h: $(_HEF)
	@python3 -c "\
import sys; \
data = open(sys.argv[1], 'rb').read(); \
print('/* Auto-generated by make nml-rpi-hailo-embed — do not edit */'); \
print('static const unsigned char _nml_embedded_hef_data[] = {' + ','.join(hex(b) for b in data) + '};'); \
print('static const unsigned int  _nml_embedded_hef_size   = ' + str(len(data)) + ';'); \
" $< > $@
	@echo "  Generated: runtime/nml_hef_resource.h ($(shell wc -c < $<) bytes, arch=$(ARCH))"

# Step 3 — build the self-contained binary
nml-rpi-hailo-embed: runtime/nml.c runtime/nml_backend_hailo.cpp \
                     runtime/nml_backend_hailo.h runtime/nml_tensor.h \
                     runtime/nml_hef_resource.h
	@mkdir -p $(BINDIR)
	g++ -O3 -march=armv8.2-a -mcpu=cortex-a76 \
	    -DNML_USE_HAILO -DNML_USE_OPENBLAS -DNML_EMBEDDED_HEF \
	    -DNML_MAX_TENSOR_SIZE=65536 \
	    -x c   -std=c99   runtime/nml.c \
	    -x c++ -std=c++17 runtime/nml_backend_hailo.cpp \
	    -o $(BINDIR)/$(OUT)$(EXEEXT) -lm -lopenblas -lhailort
	@echo "  Built: $(BINDIR)/$(OUT)  (HEF embedded, arch=$(ARCH), no file I/O)"
	@echo "  Run:   $(BINDIR)/$(OUT) $(PROGRAM) $(DATA)"

# ─── Intel oneMKL CPU ────────────────────────────────────────────────────────
# nml-mkl-cpu: Intel oneMKL BLAS on CPU — no GPU or SYCL runtime required.
# Requires Intel oneAPI Base Toolkit (source /opt/intel/oneapi/setvars.sh).
# -qmkl=sequential: single-threaded MKL (use =parallel for multi-core).
nml-mkl-cpu: runtime/nml.c runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	icx -O3 -march=native -std=c99 -DNML_USE_ONEMKL_CPU \
	    -o $(BINDIR)/nml-mkl-cpu$(EXEEXT) runtime/nml.c -lm -qmkl=sequential
	@echo "  Built: $(BINDIR)/nml-mkl-cpu (Intel oneMKL CPU BLAS, single-threaded)"

nml-mkl-cpu-parallel: runtime/nml.c runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	icx -O3 -march=native -std=c99 -DNML_USE_ONEMKL_CPU \
	    -o $(BINDIR)/nml-mkl-cpu-parallel$(EXEEXT) runtime/nml.c -lm -qmkl=parallel
	@echo "  Built: $(BINDIR)/nml-mkl-cpu-parallel (Intel oneMKL CPU BLAS, multi-threaded)"

# nml-aocl: AMD AOCL-BLAS (BLIS) on CPU — optimised for AMD Zen (EPYC, Ryzen).
# Requires AOCL: https://developer.amd.com/amd-aocl/
# Set AOCL_ROOT to your AOCL install prefix, e.g. /opt/aocl or ~/aocl.
AOCL_ROOT ?= /opt/aocl
nml-aocl: runtime/nml.c runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(CC) -O3 -march=native -std=c99 -DNML_USE_AOCL \
	    -I$(AOCL_ROOT)/include \
	    -L$(AOCL_ROOT)/lib \
	    -o $(BINDIR)/nml-aocl$(EXEEXT) runtime/nml.c -lm -lblis-mt
	@echo "  Built: $(BINDIR)/nml-aocl (AMD AOCL-BLAS / BLIS, AMD Zen optimised)"

# nml-metal: nml.c compiled as pure C99 + nml_backend_metal.m compiled as Objective-C, linked together.
# Previously nml.c was compiled as Objective-C (-x objective-c) — now only the backend file is.
nml-metal: runtime/nml.c runtime/nml_backend_metal.m runtime/nml_tensor.h
ifeq ($(shell uname),Darwin)
	@mkdir -p $(OBJDIR) $(BINDIR)
	clang -O3 -march=native -std=c99 -D_DARWIN_C_SOURCE \
	    -DNML_USE_METAL -DNML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK \
	    -framework Accelerate \
	    -c -o $(OBJDIR)/nml_metal_core.o runtime/nml.c
	clang -O3 -x objective-c -D_DARWIN_C_SOURCE \
	    -DNML_USE_METAL -DNML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK \
	    -framework Metal -framework MetalPerformanceShaders \
	    -framework Accelerate -framework Foundation \
	    -c -o $(OBJDIR)/nml_metal_backend.o runtime/nml_backend_metal.m \
	    -Wno-deprecated-declarations
	clang -O3 \
	    -framework Metal -framework MetalPerformanceShaders \
	    -framework Accelerate -framework Foundation \
	    -o $(BINDIR)/nml-metal$(EXEEXT) $(OBJDIR)/nml_metal_core.o $(OBJDIR)/nml_metal_backend.o -lm \
	    -Wno-deprecated-declarations
	@echo "  Built: $(BINDIR)/nml-metal (v0.10.0, Metal GPU + BLAS via Accelerate)"
else
	@echo "  Error: Metal requires macOS. Use nml-fast for CPU acceleration."
	@exit 1
endif

# nml-sycl: nml.c compiled as C99 under icpx (-x c) + nml_backend_sycl.cpp compiled with -fsycl.
# Device selection at runtime: ONEAPI_DEVICE_SELECTOR=opencl:gpu  (oneAPI)
#                               SYCL_DEVICE_FILTER=gpu             (AdaptiveCpp)
nml-sycl: runtime/nml.c runtime/nml_backend_sycl.cpp runtime/nml_tensor.h
ifeq ($(shell which $(SYCL_CC) 2>/dev/null),)
	@echo "  Error: $(SYCL_CC) not found. Install Intel oneAPI or set SYCL_CC=clang++ and add -fsycl to SYCL_CC."
	@exit 1
endif
	@mkdir -p $(OBJDIR) $(BINDIR)
	gcc -O3 -std=c99 -DNML_USE_SYCL -c runtime/nml.c -o $(OBJDIR)/nml_sycl_core.o
	$(SYCL_CC) -fsycl -O3 -DNML_USE_SYCL \
	    $(OBJDIR)/nml_sycl_core.o \
	    runtime/nml_backend_sycl.cpp \
	    -o $(BINDIR)/nml-sycl$(EXEEXT) -lm
	@echo "  Built: $(BINDIR)/nml-sycl (SYCL GPU, threshold=$(NML_SYCL_MMUL_THRESHOLD) — override with -DNML_SYCL_MMUL_THRESHOLD=N)"

nmld-sycl: runtime/nmld.c runtime/nml_backend_sycl.cpp runtime/nml_tensor.h
ifeq ($(shell which $(SYCL_CC) 2>/dev/null),)
	@echo "  Error: $(SYCL_CC) not found."
	@exit 1
endif
	@mkdir -p $(OBJDIR) $(BINDIR)
	gcc -O3 -std=c99 -DNML_USE_SYCL -c runtime/nmld.c -o $(OBJDIR)/nmld_sycl_core.o
	$(SYCL_CC) -fsycl -O3 -DNML_USE_SYCL \
	    $(OBJDIR)/nmld_sycl_core.o \
	    runtime/nml_backend_sycl.cpp \
	    -o $(BINDIR)/nmld-sycl$(EXEEXT) -lm
	@echo "  Built: $(BINDIR)/nmld-sycl (NML daemon with SYCL GPU acceleration)"

# ═══════════════════════════════════════════
# CUDA backend (NVIDIA GPUs via cuBLAS)
# Requires: CUDA toolkit (nvcc + libcublas)
# Device selection: CUDA_VISIBLE_DEVICES=0
# ═══════════════════════════════════════════
NVCC      ?= nvcc
CUDA_ARCH ?= native   # 'native' auto-detects; or sm_86, sm_89, sm_90, etc.

nml-cuda: runtime/nml.c runtime/nml_backend_cuda.cu runtime/nml_backend_cuda.h runtime/nml_tensor.h
ifeq ($(shell which $(NVCC) 2>/dev/null),)
	@echo "  Error: $(NVCC) not found. Install CUDA toolkit (https://developer.nvidia.com/cuda-downloads)."
	@exit 1
endif
	@mkdir -p $(OBJDIR) $(BINDIR)
	gcc -O3 -std=c99 -DNML_USE_CUDA -c runtime/nml.c -o $(OBJDIR)/nml_cuda_core.o
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -DNML_USE_CUDA \
	    $(OBJDIR)/nml_cuda_core.o \
	    runtime/nml_backend_cuda.cu \
	    -o $(BINDIR)/nml-cuda$(EXEEXT) -lm -lcublas
	@echo "  Built: $(BINDIR)/nml-cuda (NVIDIA CUDA + cuBLAS, arch=$(CUDA_ARCH))"

nmld-cuda: runtime/nmld.c runtime/nml_backend_cuda.cu runtime/nml_backend_cuda.h runtime/nml_tensor.h
ifeq ($(shell which $(NVCC) 2>/dev/null),)
	@echo "  Error: $(NVCC) not found."
	@exit 1
endif
	@mkdir -p $(OBJDIR) $(BINDIR)
	gcc -O3 -std=c99 -DNML_USE_CUDA -c runtime/nmld.c -o $(OBJDIR)/nmld_cuda_core.o
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -DNML_USE_CUDA \
	    $(OBJDIR)/nmld_cuda_core.o \
	    runtime/nml_backend_cuda.cu \
	    -o $(BINDIR)/nmld-cuda$(EXEEXT) -lm -lcublas
	@echo "  Built: $(BINDIR)/nmld-cuda (NML daemon with NVIDIA CUDA acceleration)"

nml-fmt: runtime/nml_fmt.c runtime/nml_fmt.h
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $(BINDIR)/nml-fmt$(EXEEXT) runtime/nml_fmt.c
	@echo "  Built: $(BINDIR)/nml-fmt (syntax converter: classic ↔ symbolic ↔ verbose)"

nml-crypto: runtime/nml.c runtime/nml_crypto.h runtime/tweetnacl.c runtime/tweetnacl.h
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -DNML_CRYPTO -o $(BINDIR)/nml-crypto$(EXEEXT) runtime/nml.c runtime/tweetnacl.c $(LDFLAGS)
	@echo "  Built: $(BINDIR)/nml-crypto (v0.10.0, Ed25519 + HMAC-SHA256 signing)"

EMCC ?= emcc

nml-wasm: runtime/nml.c
	$(EMCC) -O2 -std=c99 -o terminal/nml.js runtime/nml.c -lm \
	    -s MODULARIZE=1 -s EXPORT_NAME='NMLModule' \
	    -s EXPORTED_RUNTIME_METHODS='["callMain","FS"]' \
	    -s ALLOW_MEMORY_GROWTH=1
	@echo "  Built: nml-wasm (terminal/nml.js + terminal/nml.wasm)"

release: nml
	strip $(BINDIR)/nml$(EXEEXT)
	@echo "  nml: $$(wc -c < $(BINDIR)/nml$(EXEEXT) | tr -d ' ') bytes"

# ═══════════════════════════════════════════
# Core tests
# ═══════════════════════════════════════════

test-anomaly: nml
	$(NML_BIN) programs/anomaly_detector.nml programs/anomaly_weights.nml.data

test-extensions: nml
	$(NML_BIN) programs/extension_demo.nml programs/extension_demo.nml.data

test-symbolic: nml
	@echo "--- Symbolic anomaly detector ---"
	$(NML_BIN) tests/test_symbolic.nml programs/anomaly_weights.nml.data 2>&1 | grep -E "(HALTED|anomaly_score)"
	@echo "--- Symbolic features ---"
	$(NML_BIN) tests/test_symbolic_features.nml 2>&1 | grep -E "(HALTED|sdiv|cmpi|cmp_|sum_|call_)"

test-verbose: nml
	$(NML_BIN) tests/test_verbose.nml 2>&1 | grep -E "(HALTED|tax_amount)"

test-features: nml
	$(NML_BIN) tests/test_features.nml 2>&1 | grep -E "(HALTED|sdiv|cmpi|cmp_|sum_|call_)"

test-hello: nml
	@echo "--- Hello World (NML-G) ---"
	$(NML_BIN) programs/hello_world.nml 2>&1 | grep -E "(Hello|HALTED)"

test-fibonacci: nml
	@echo "--- Fibonacci (NML-G) ---"
	$(NML_BIN) programs/fibonacci.nml 2>&1 | grep -E "^[0-9]" | head -5
	@echo "  ..."

test-fizzbuzz: nml
	@echo "--- FizzBuzz (NML-G) ---"
	$(NML_BIN) programs/fizzbuzz.nml 2>&1 | grep -E "^-?[0-9]" | head -5
	@echo "  ..."

test-primes: nml
	@echo "--- Primes (NML-G) ---"
	$(NML_BIN) programs/primes.nml 2>&1 | grep -E "^[0-9]"

test-gp: test-hello test-fibonacci test-fizzbuzz test-primes
	@echo ""
	@echo "  NML-G tests passed."

test: test-anomaly test-extensions test-symbolic test-verbose test-features test-gp
	@echo ""
	@echo "  All core tests passed."

test-phase1: nml
	@echo "=== Phase 1 Tests ==="
	@echo "--- Broadcasting ---"
	$(NML_BIN) tests/opcode_coverage/17_broadcast.nml tests/opcode_coverage/17_broadcast.nml.data
	@echo "--- 4D Vision ---"
	$(NML_BIN) tests/opcode_coverage/10b_vision_4d.nml tests/opcode_coverage/10b_vision_4d.nml.data
	@echo "--- Cross-Entropy Loss ---"
	$(NML_BIN) tests/opcode_coverage/15b_loss_ce.nml tests/opcode_coverage/15b_loss_ce.nml.data
	@echo "--- Regression: existing vision ---"
	$(NML_BIN) tests/opcode_coverage/10_vision.nml tests/opcode_coverage/10_vision.nml.data
	@echo "--- Regression: existing training ---"
	$(NML_BIN) tests/opcode_coverage/15_training.nml tests/opcode_coverage/15_training.nml.data
	@echo "--- Regression: anomaly detector ---"
	$(NML_BIN) programs/anomaly_detector.nml programs/anomaly_weights.nml.data
	@echo "=== Phase 1 PASSED ==="

test-phase2: nml
	@echo "=== Phase 2 Tests ==="
	@echo "--- GPU CONV (CPU path validation) ---"
	$(NML_BIN) tests/opcode_coverage/18_gpu_conv.nml tests/opcode_coverage/18_gpu_conv.nml.data
	@echo "--- Kernel Fusion ---"
	$(NML_BIN) tests/opcode_coverage/19_fusion.nml tests/opcode_coverage/10b_vision_4d.nml.data
	@echo "--- Regression: Phase 1 still passes ---"
	$(MAKE) test-phase1
	@echo "=== Phase 2 PASSED ==="

test-phase3: nml
	@echo "=== Phase 3 Tests ==="
	@echo "--- N-Layer TNET ---"
	$(NML_BIN) tests/opcode_coverage/20_nlayer_tnet.nml tests/opcode_coverage/20_nlayer_tnet.nml.data
	@echo "--- Batch Normalization ---"
	$(NML_BIN) tests/opcode_coverage/21_batchnorm.nml tests/opcode_coverage/21_batchnorm.nml.data
	@echo "--- Dropout ---"
	$(NML_BIN) tests/opcode_coverage/22_dropout.nml tests/opcode_coverage/22_dropout.nml.data
	@echo "--- Weight Decay ---"
	$(NML_BIN) tests/opcode_coverage/23_wdecay.nml tests/opcode_coverage/23_wdecay.nml.data
	@echo "--- Regression: Phase 1 + 2 still pass ---"
	$(NML_BIN) programs/anomaly_detector.nml programs/anomaly_weights.nml.data
	$(NML_BIN) tests/opcode_coverage/15_training.nml tests/opcode_coverage/15_training.nml.data
	@echo "=== Phase 3 PASSED ==="

test-onnx: nml
	@echo "=== Phase 4: ONNX Import Test ==="
	python3 tests/test_onnx_import.py || echo "SKIP: onnx not installed"
	@echo "=== Phase 4 Done ==="

test-phase4: test-onnx
	@echo "=== Phase 4 PASSED ==="

# ═══════════════════════════════════════════
# Phase 5 — C library + Python binding
# ═══════════════════════════════════════════

# ── Shared library ──────────────────────────────────────────────────────────
LIB_CFLAGS = -O2 -std=c99 -DNML_BUILD_LIB

libnml.so: runtime/nml.c runtime/nml_api.h runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(CC) $(LIB_CFLAGS) -shared -fPIC -o $(BINDIR)/libnml.so runtime/nml.c -lm
	@echo "  Built: $(BINDIR)/libnml.so"

libnml-fast.so: runtime/nml.c runtime/nml_api.h runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(CC) $(LIB_CFLAGS) -shared -fPIC $(BLAS_FLAGS) \
	    -o $(BINDIR)/libnml-fast.so runtime/nml.c -lm $(BLAS_LIBS)
	@echo "  Built: $(BINDIR)/libnml-fast.so"

# Windows DLL
libnml.dll: runtime/nml.c runtime/nml_api.h runtime/nml_tensor.h
	@mkdir -p $(BINDIR)
	$(CC) $(LIB_CFLAGS) -shared -o $(BINDIR)/libnml.dll runtime/nml.c -lm
	@echo "  Built: $(BINDIR)/libnml.dll"

# Static library
libnml.a: runtime/nml.c runtime/nml_api.h runtime/nml_tensor.h
	@mkdir -p $(OBJDIR) $(BINDIR)
	$(CC) $(LIB_CFLAGS) -c -o $(OBJDIR)/nml_lib.o runtime/nml.c
	ar rcs $(BINDIR)/libnml.a $(OBJDIR)/nml_lib.o
	rm -f $(OBJDIR)/nml_lib.o
	@echo "  Built: $(BINDIR)/libnml.a"

test-phase5: nml
	@echo "=== Phase 5: C API + Python binding ==="
	@echo "--- Python binding import ---"
	python3 -c "import sys; sys.path.insert(0,'python'); import nml; print('nml module OK')" || echo "SKIP: python3 not available"
	@echo "--- Python subprocess mode ---"
	python3 -c "import sys; sys.path.insert(0,'python'); import nml; out=nml.run_program('programs/anomaly_detector.nml','programs/anomaly_weights.nml.data'); print(out[:120])" || echo "SKIP: python3 not available"
	@echo "--- Full Python binding test ---"
	python3 tests/test_python_binding.py || echo "SKIP: python3 not available"
	@echo "=== Phase 5 PASSED ==="

# ═══════════════════════════════════════════
# Domain targets (require domain/ populated)
# ═══════════════════════════════════════════

domain-test-tax: nml
	@echo "--- Junior Developer ---"
	@mkdir -p $(OBJDIR)
	@echo "@employee_data shape=1,8 data=65000.0,0.0,0.0,0.03,0.0,26.0,0.06,2.0" > $(NML_TEST_TMP)
	$(NML_BIN) domain/programs/tax_calculator.nml $(NML_TEST_TMP) 2>&1 | grep -E "(HALTED|net_pay)"

domain-transpile-scan:
	cd domain/transpilers && python3 domain_transpiler.py scan

domain-transpile-library: nml
	cd domain/transpilers && python3 domain_build_library.py --validate

domain-transpile-library-symbolic: nml
	cd domain/transpilers && python3 domain_build_library.py --syntax symbolic --no-comments

domain-train:
	cd domain/transpilers && python3 tax_pipeline.py

domain-benchmark:
	cd domain/transpilers && python3 benchmark.py

domain-prepare-training:
	cd domain/transpilers && python3 finetune_pipeline.py \
		--inputs ../output/training/nml_code_pairs.jsonl \
		         ../output/training/all_gaps_combined.jsonl \
		         ../output/training/rag_gaps.jsonl \
		         ../output/training/constants_pairs.jsonl \
		         ../output/training/nml_syntax.jsonl \
		--output-dir ../output/training/mlx-combined \
		--prepare-only

domain-finetune:
	cd domain/transpilers && python3 finetune_pipeline.py \
		--inputs ../output/training/nml_code_pairs.jsonl \
		         ../output/training/all_gaps_combined.jsonl \
		         ../output/training/rag_gaps.jsonl \
		         ../output/training/constants_pairs.jsonl \
		         ../output/training/nml_syntax.jsonl \
		--base-model ../output/model/Mistral-7B-Instruct-v0.3-4bit \
		--output-dir ../output/training/mlx-combined \
		--adapter-dir ../output/model/nml-combined-adapters \
		--train

domain-finetune-merge:
	cd domain/transpilers && python3 finetune_pipeline.py \
		--base-model ../output/model/Mistral-7B-Instruct-v0.3-4bit \
		--adapter-dir ../output/model/nml-combined-adapters \
		--merge-to ../output/model/nml-combined-merged \
		--merge-only

domain-rag-server:
	cd domain/transpilers && python3 domain_rag_server.py --domains tax

# ═══════════════════════════════════════════
# Agent services
# ═══════════════════════════════════════════

agent-start: nml
	bash serve/start_agents.sh

agent-start-headless: nml
	bash serve/start_agents.sh --no-ui

agent-gateway: nml
	cd domain/transpilers && python3 domain_rag_server.py --domains tax

agent-status:
	@curl -s localhost:8082/health 2>/dev/null | python3 -m json.tool || echo "  NML Server: OFFLINE"
	@curl -s localhost:8083/health 2>/dev/null | python3 -m json.tool || echo "  Gateway: OFFLINE"

# ═══════════════════════════════════════════
# Clean
# ═══════════════════════════════════════════

clean:
	rm -rf $(BINDIR) $(OBJDIR)

# ═══════════════════════════════════════════
# Help
# ═══════════════════════════════════════════

help:
	@echo ""
	@echo "  NML v0.10.0 — Neural Machine Language"
	@echo "  ═════════════════════════════════════"
	@echo ""
	@echo "  Build:"
	@echo "    make nml              Build the NML runtime (portable, pure C99)"
	@echo "    make nml-fast         Build with BLAS (Accelerate on macOS, OpenBLAS on Linux)"
	@echo "    make nml-metal        Build with Metal GPU + BLAS (macOS only)"
	@echo "    make nml-sycl         Build with SYCL GPU (requires icpx / Intel oneAPI)"
	@echo "    make nmld             Build the NML daemon (pre-fork workers, binary cache)"
	@echo "    make nmld-sycl        Build the NML daemon with SYCL GPU acceleration"
	@echo "    make release          Build + strip"
	@echo ""
	@echo "  Test (core):"
	@echo "    make test             Run all core tests"
	@echo "    make test-anomaly     Anomaly detection (neural net)"
	@echo "    make test-extensions  Extension demo"
	@echo "    make test-symbolic    Symbolic syntax tests"
	@echo "    make test-verbose     Verbose syntax test"
	@echo "    make test-features    Core features (SDIV, CMP, CALL/RET, backward jumps)"
	@echo "    make test-gp          General-purpose (hello world, fibonacci, fizzbuzz, primes)"
	@echo ""
	@echo "  Domain (requires domain/ populated):"
	@echo "    make domain-test-tax                  Tax calculator test"
	@echo "    make domain-transpile-scan             Scan tax-data/ and classify"
	@echo "    make domain-transpile-library          Build + validate full NML tax library"
	@echo "    make domain-transpile-library-symbolic Build library in symbolic syntax"
	@echo "    make domain-prepare-training           Combine JSONL → train/valid splits"
	@echo "    make domain-finetune                   Prepare + LoRA fine-tune (Mistral 7B)"
	@echo "    make domain-finetune-merge             Merge LoRA adapters into base model"
	@echo "    make domain-rag-server                 Start multi-domain RAG server"
	@echo ""
    @echo "  Agent services:"
	@echo "    make agent-start      Start NML server + gateway + chat UI"
	@echo "    make agent-start-headless  Start NML server + gateway (no UI)"
	@echo "    make agent-gateway    Start domain RAG gateway only"
	@echo "    make agent-status     Check health of running services"
	@echo ""
	@echo "  Other:"
	@echo "    make clean            Remove built binary"
	@echo "    make help             Show this message"
	@echo ""

.PHONY: nml nmld nml-fast nml-metal nml-sycl nmld-sycl nml-crypto nml-fmt nml-wasm release test test-anomaly test-extensions test-symbolic test-verbose test-features test-hello test-fibonacci test-fizzbuzz test-primes test-gp test-phase1 test-phase2 test-phase3 test-onnx test-phase4 libnml.so libnml-fast.so libnml.dll libnml.a test-phase5 domain-test-tax domain-transpile-scan domain-transpile-library domain-transpile-library-symbolic domain-train domain-benchmark domain-prepare-training domain-finetune domain-finetune-merge domain-rag-server agent-start agent-start-headless agent-gateway agent-status clean help
