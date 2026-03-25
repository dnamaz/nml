/*
 * nml_backend_cuda.h — NVIDIA CUDA backend interface for NML
 *
 * Provides cuBLAS-accelerated GEMM/GEMV and custom CUDA kernels for
 * element-wise activations (RELU, SIGMOID, TANH).
 *
 * Build (Linux):
 *   gcc -O2 -std=c99 -DNML_USE_CUDA -c runtime/nml.c -o /tmp/nml_cuda_core.o
 *   nvcc -O3 -DNML_USE_CUDA /tmp/nml_cuda_core.o runtime/nml_backend_cuda.cu \
 *        -o nml-cuda -lm -lcublas
 *
 * Build (Windows, from Developer Command Prompt with CUDA toolkit):
 *   icx /O2 /std:c17 /DNML_USE_CUDA /c runtime\nml.c /Fo:nml_cuda_core.obj
 *   nvcc -O3 -DNML_USE_CUDA nml_cuda_core.obj runtime\nml_backend_cuda.cu \
 *        -o nml-cuda.exe -lcublas -Xlinker /STACK:8388608
 *
 * Thresholds (override with -DNML_CUDA_MMUL_THRESHOLD=N etc.):
 *   NML_CUDA_MMUL_THRESHOLD  min m*n for GEMM/GEMV dispatch (default 4096)
 *   NML_CUDA_EW_THRESHOLD    min elements for activation dispatch (default 4096)
 *
 * Device selection:
 *   CUDA_VISIBLE_DEVICES=0   select GPU by index (default: first available)
 */

#pragma once
#ifndef NML_BACKEND_CUDA_H
#define NML_BACKEND_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Thresholds — also used by nml.c dispatch guards */
#ifndef NML_CUDA_MMUL_THRESHOLD
  /* 1024×1024 = 1M elements — break-even vs CPU BLAS on PCIe 4.0.
     Lower to 65536 (256×256) if using pinned memory or iGPU shared memory.
     Override: -DNML_CUDA_MMUL_THRESHOLD=65536 */
  #define NML_CUDA_MMUL_THRESHOLD 1048576
#endif
#ifndef NML_CUDA_EW_THRESHOLD
  #define NML_CUDA_EW_THRESHOLD   1048576
#endif

/*
 * nml_backend_cuda_matmul — row-major F32 matmul via cuBLAS
 *
 *   dest [m×n] = a [m×k] * b [k×n]
 *
 * Uses cublasSgemv for m==1 (GEMV) and cublasSgemm for m>1 (GEMM).
 * Returns 0 on success, -1 on CUDA/cuBLAS error (caller falls back to CPU).
 */
#include "nml_tensor.h"
int nml_backend_cuda_matmul(Tensor *dest, const Tensor *a,
                             const Tensor *b, int m, int k, int n);

/*
 * Element-wise activation functions — operate in-place on F32 data.
 * Returns 0 on success, -1 on error (caller falls back to scalar loop).
 */
int nml_backend_cuda_relu   (float *data, int n);
int nml_backend_cuda_sigmoid(float *data, int n);
int nml_backend_cuda_tanh   (float *data, int n);

/* CONV2D via im2col + batched cuBLAS GEMM.
 * input:  [N, C_in, H, W]
 * kernel: [C_out, C_in, KH, KW]
 * output: [N, C_out, OH, OW] (caller must allocate with tensor_init_typed)
 * Returns 0 on success, -1 on error (caller falls back to CPU). */
int nml_backend_cuda_conv2d(Tensor *out, const Tensor *input, const Tensor *kernel,
                             int stride, int pad);

/* Scaled dot-product attention: out[seq,dv] = softmax(Q*K^T/sqrt(dk)) * V
 * Q: [seq, dk], K: [seq, dk], V: [seq, dv]
 * Returns 0 on success, -1 on error. */
int nml_backend_cuda_attention(Tensor *out, const Tensor *Q, const Tensor *K, const Tensor *V);

/* Elementwise ops (in-place on device) */
int nml_backend_cuda_gelu(float *data, int n);
int nml_backend_cuda_softmax(float *data, int rows, int cols);
int nml_backend_cuda_layernorm(float *data, int rows, int cols, const float *gamma, const float *beta);

/*
 * nml_backend_cuda_device_count — number of visible CUDA devices.
 * Returns 0 if no CUDA-capable GPU is present.
 */
int nml_backend_cuda_device_count(void);

/*
 * nml_backend_cuda_teardown — release cuBLAS handle and device buffers.
 * Optional: call on clean exit.
 */
void nml_backend_cuda_teardown(void);

#ifdef __cplusplus
}
#endif

#endif /* NML_BACKEND_CUDA_H */
