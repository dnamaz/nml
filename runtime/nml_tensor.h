/*
 * nml_tensor.h — Shared type definitions for NML runtime and GPU backends
 *
 * This header is intentionally minimal: only types and constants that backend
 * files (nml_backend_metal.m, nml_backend_sycl.cpp, etc.) need to reference.
 * It is pure C and safe to include from C, C++, or Objective-C translation units.
 *
 * Do NOT add implementation functions here. Keep those in nml.c.
 */

#pragma once
#ifndef NML_TENSOR_H
#define NML_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════
   VERSION
   ═══════════════════════════════════════════ */

#define NML_VERSION_MAJOR  0
#define NML_VERSION_MINOR  10
#define NML_VERSION_PATCH  0

/* ═══════════════════════════════════════════
   RUNTIME LIMITS
   ═══════════════════════════════════════════ */

#define NML_MAX_REGISTERS    32
#define NML_MAX_DIMS         4
#ifndef NML_MAX_TENSOR_SIZE
  /* 16M elements (~64 MB at f32) for general builds.
     Override at compile time for constrained targets:
       gcc ... -DNML_MAX_TENSOR_SIZE=65536  */
  #define NML_MAX_TENSOR_SIZE  16777216
#endif
#ifndef NML_MAX_INSTRUCTIONS
  #define NML_MAX_INSTRUCTIONS 8192
#endif
#ifndef NML_MAX_MEMORY_SLOTS
  #define NML_MAX_MEMORY_SLOTS 64
#endif
#ifndef NML_MAX_LOOP_DEPTH
  #define NML_MAX_LOOP_DEPTH   8
#endif
#ifndef NML_MAX_CALL_DEPTH
  #define NML_MAX_CALL_DEPTH   32
#endif
#define NML_MAX_LINE_LEN     256
#define NML_MAX_LABEL_LEN    64
#define NML_DEFAULT_MAX_CYCLES 1000000

/* ═══════════════════════════════════════════
   ERROR CODES
   ═══════════════════════════════════════════ */

#define NML_OK                0
#define NML_ERR_SHAPE        -1
#define NML_ERR_OOB          -2
#define NML_ERR_UNINIT       -3
#define NML_ERR_OVERFLOW     -4
#define NML_ERR_DIVZERO      -5
#define NML_ERR_OPCODE       -6
#define NML_ERR_MEMORY       -7
#define NML_ERR_ASSEMBLE     -8
#define NML_ERR_CYCLE_LIMIT  -9
#define NML_ERR_TRAP         -10
#define NML_ERR_CALL_DEPTH   -11
#define NML_ERR_RET_EMPTY    -12
#define NML_ERR_EXTENSION    -13
#define NML_ERR_LOOP         -14
#define NML_ERR_FILE         -15

/* ═══════════════════════════════════════════
   DATA TYPES
   ═══════════════════════════════════════════ */

typedef enum { NML_F32 = 0, NML_F64 = 1, NML_I32 = 2 } DType;

/* ═══════════════════════════════════════════
   TENSOR
   ═══════════════════════════════════════════ */

typedef struct {
    union {
        float  *f32;
        double *f64;
        int    *i32;
        void   *raw;
    } data;
    DType dtype;
    int   shape[NML_MAX_DIMS];
    int   ndim;
    int   size;
    int   _capacity;
} Tensor;

/* ═══════════════════════════════════════════
   BACKEND DISPATCH THRESHOLDS
   ═══════════════════════════════════════════ */

/* SYCL: 16384 (128×128) is safe break-even for both iGPU (zero-copy USM,
   breaks even at ~64×64) and dGPU (PCIe round-trip, breaks even at ~256×256).
   Override at compile time for iGPU-only deployments: -DNML_SYCL_MMUL_THRESHOLD=4096 */
#ifndef NML_SYCL_MMUL_THRESHOLD
  #define NML_SYCL_MMUL_THRESHOLD  16384
#endif

/* SYCL elementwise (RELU, SIGMOID, TANH): same submission cost as MMUL. */
#ifndef NML_SYCL_EW_THRESHOLD
  #define NML_SYCL_EW_THRESHOLD    16384
#endif

/* Metal: MPS framework launch cost is high — only profitable at ~1024×1024 */
#ifndef NML_METAL_MMUL_THRESHOLD
  #define NML_METAL_MMUL_THRESHOLD 1048576
#endif

/* CUDA: PCIe round-trip (~600µs for 256×256) dominates until 1024×1024.
   Override for pinned memory or if using cuBLAS on iGPU: -DNML_CUDA_MMUL_THRESHOLD=65536 */
#ifndef NML_CUDA_MMUL_THRESHOLD
  #define NML_CUDA_MMUL_THRESHOLD  1048576
#endif
#ifndef NML_CUDA_EW_THRESHOLD
  #define NML_CUDA_EW_THRESHOLD    1048576
#endif

#ifdef __cplusplus
}
#endif

#endif /* NML_TENSOR_H */
