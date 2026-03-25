/*
 * nml_backend_cuda.cu — NVIDIA CUDA backend for NML
 *
 * Two dispatch tiers:
 *
 *   GEMV (m == 1):
 *     cublasSgemv — matrix-vector product, bandwidth-optimal
 *
 *   GEMM (m > 1):
 *     cublasSgemm — vendor-tuned dense matrix multiply
 *
 *   Activations (RELU, SIGMOID, TANH):
 *     Custom CUDA kernels, 256-thread blocks
 *
 * Persistent device buffers:
 *   cudaMalloc'd once, reused across calls to avoid per-call allocation
 *   overhead on the hot path.  Safe because NML is single-threaded and
 *   all CUDA operations are synchronous (cudaDeviceSynchronize after each).
 *
 * Row-major ↔ cuBLAS column-major mapping:
 *   cuBLAS is natively column-major.  For row-major C[m,n] = A[m,k] * B[k,n]:
 *     Treat as col-major: C^T[n,m] = B^T[n,k] * A^T[k,m]
 *     → cublasSgemm(N, N, n, m, k, B_ptr, n, A_ptr, k, C_ptr, n)
 *
 * Requires: CUDA toolkit >= 11.0, cuBLAS
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "nml_backend_cuda.h"
#include "nml_tensor.h"

/* ═══════════════════════════════════════════════════════════════════
   HANDLE & BUFFER CACHE
   ═══════════════════════════════════════════════════════════════════ */

static cublasHandle_t _handle    = nullptr;

/* Device buffers (VRAM) */
static float         *_da        = nullptr;
static float         *_db        = nullptr;
static float         *_dc        = nullptr;
static size_t         _buf_cap   = 0;

/* Pinned host staging buffers (cudaMallocHost).
 * In WSL2, regular malloc'd memory crosses the Hyper-V boundary on every
 * cudaMemcpy, paying an implicit pin cost each time.  Pinned buffers are
 * locked in physical memory — CUDA can DMA directly, eliminating that overhead. */
static float         *_ha        = nullptr;
static float         *_hb        = nullptr;
static float         *_hc        = nullptr;
static size_t         _host_cap  = 0;

static cublasHandle_t cuda_handle(void) {
    if (!_handle) cublasCreate(&_handle);
    return _handle;
}

static int cuda_ensure(size_t need) {
    if (need <= _buf_cap) return 0;
    cudaFree(_da); cudaFree(_db); cudaFree(_dc);
    if (cudaMalloc(&_da, need * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&_db, need * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&_dc, need * sizeof(float)) != cudaSuccess) goto fail;
    _buf_cap = need;
    return 0;
fail:
    cudaFree(_da); cudaFree(_db); cudaFree(_dc);
    _da = _db = _dc = nullptr; _buf_cap = 0;
    return -1;
}

static int host_ensure(size_t need) {
    if (need <= _host_cap) return 0;
    cudaFreeHost(_ha); cudaFreeHost(_hb); cudaFreeHost(_hc);
    if (cudaMallocHost(&_ha, need * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMallocHost(&_hb, need * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMallocHost(&_hc, need * sizeof(float)) != cudaSuccess) goto fail;
    _host_cap = need;
    return 0;
fail:
    cudaFreeHost(_ha); cudaFreeHost(_hb); cudaFreeHost(_hc);
    _ha = _hb = _hc = nullptr; _host_cap = 0;
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════
   ACTIVATION KERNELS
   ═══════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════
   PHASE 2 KERNELS: IM2COL, GELU, SOFTMAX-ROWS, LAYERNORM
   ═══════════════════════════════════════════════════════════════════ */

/* im2col kernel: reshape input patches into columns */
__global__ static void k_im2col(const float *input, float *col,
                                  int N, int C_in, int H, int W,
                                  int KH, int KW, int OH, int OW,
                                  int stride, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * KH * KW * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW; int tmp = idx / OW;
    int oh = tmp % OH; tmp /= OH;
    int kw = tmp % KW; tmp /= KW;
    int kh = tmp % KH; tmp /= KH;
    int ci = tmp % C_in;
    int n  = tmp / C_in;

    int ih = oh * stride - pad + kh;
    int iw = ow * stride - pad + kw;

    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        val = input[n*(C_in*H*W) + ci*(H*W) + ih*W + iw];

    /* col layout: [N, C_in*KH*KW, OH*OW] */
    col[n*(C_in*KH*KW*OH*OW) + (ci*KH*KW + kh*KW + kw)*(OH*OW) + oh*OW + ow] = val;
}

__global__ static void k_gelu(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = d[i];
        d[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ static void k_softmax_rows(float *d, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float *r = d + row * cols;
    float mx = r[0];
    for (int j = 1; j < cols; j++) if (r[j] > mx) mx = r[j];
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) { r[j] = expf(r[j] - mx); sum += r[j]; }
    for (int j = 0; j < cols; j++) r[j] /= sum;
}

__global__ static void k_layernorm(float *d, int rows, int cols,
                                    const float *gamma, const float *beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float *r = d + row * cols;
    float mean = 0.0f;
    for (int j = 0; j < cols; j++) mean += r[j];
    mean /= cols;
    float var = 0.0f;
    for (int j = 0; j < cols; j++) { float dv = r[j] - mean; var += dv*dv; }
    var = sqrtf(var / cols + 1e-5f);
    for (int j = 0; j < cols; j++) {
        r[j] = (r[j] - mean) / var;
        if (gamma) r[j] *= gamma[j];
        if (beta)  r[j] += beta[j];
    }
}

__global__ static void k_relu(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = d[i] > 0.0f ? d[i] : 0.0f;
}

__global__ static void k_sigmoid(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 1.0f / (1.0f + expf(-d[i]));
}

__global__ static void k_tanh(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = tanhf(d[i]);
}

/* ═══════════════════════════════════════════════════════════════════
   PUBLIC API
   ═══════════════════════════════════════════════════════════════════ */

/* Definitions below have C linkage via the extern "C" declarations in
 * nml_backend_cuda.h — no second extern "C" block needed. */

int nml_backend_cuda_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

int nml_backend_cuda_matmul(Tensor *dest, const Tensor *a, const Tensor *b,
                             int m, int k, int n)
{
    cublasHandle_t h = cuda_handle();
    if (!h) return -1;

    /* need = max(m*k, k*n, m*n) — each persistent buffer must hold the largest operand */
    size_t need = (size_t)m * k;
    if ((size_t)k * n > need) need = (size_t)k * n;
    if ((size_t)m * n > need) need = (size_t)m * n;
    if (cuda_ensure(need) != 0) return -1;
    if (host_ensure(need) != 0) return -1;

    /* Stage through pinned memory: pageable → pinned (CPU), then pinned → VRAM (DMA) */
    memcpy(_ha, a->data.f32, (size_t)m * k * sizeof(float));
    memcpy(_hb, b->data.f32, (size_t)k * n * sizeof(float));
    if (cudaMemcpy(_da, _ha, (size_t)m * k * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(_db, _hb, (size_t)k * n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    const float alpha = 1.0f, beta = 0.0f;

    if (m == 1) {
        /*
         * GEMV: dest[1×n] = a[1×k] * b[k×n]
         *
         * cuBLAS col-major view: b is stored as row-major [k×n],
         * which looks like col-major [n×k].  cublasSgemv with op=N
         * computes y[n] = B_cm[n×k] * x[k], which equals b_rm^T * a,
         * i.e. exactly what we want for row-major GEMV.
         */
        if (cublasSgemv(_handle, CUBLAS_OP_N,
                        n, k,
                        &alpha, _db, n,
                        _da, 1,
                        &beta,  _dc, 1) != CUBLAS_STATUS_SUCCESS) return -1;
    } else {
        /*
         * GEMM: dest[m×n] = a[m×k] * b[k×n]
         *
         * Row-major A*B = col-major (B^T * A^T)^T.
         * cublasSgemm(N, N, n, m, k, B_ptr, ldb=n, A_ptr, lda=k, C_ptr, ldc=n)
         * produces C_cm[n×m] which is C_rm[m×n] in memory — correct.
         */
        if (cublasSgemm(_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k,
                        &alpha, _db, n,
                                _da, k,
                        &beta,  _dc, n) != CUBLAS_STATUS_SUCCESS) return -1;
    }

    /* D2H: VRAM → pinned, then pinned → pageable */
    if (cudaMemcpy(_hc, _dc, (size_t)m * n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    memcpy(dest->data.f32, _hc, (size_t)m * n * sizeof(float));
    return 0;
}

static int ew_dispatch(float *data, int n, void(*kernel)(float*, int)) {
    cublasHandle_t h = cuda_handle();
    if (!h) return -1;
    if (cuda_ensure((size_t)n) != 0) return -1;
    if (host_ensure((size_t)n) != 0) return -1;

    memcpy(_ha, data, (size_t)n * sizeof(float));
    if (cudaMemcpy(_da, _ha, (size_t)n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(_da, n);
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;

    if (cudaMemcpy(_ha, _da, (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    memcpy(data, _ha, (size_t)n * sizeof(float));
    return 0;
}

int nml_backend_cuda_relu   (float *data, int n) { return ew_dispatch(data, n, k_relu);    }
int nml_backend_cuda_sigmoid(float *data, int n) { return ew_dispatch(data, n, k_sigmoid); }
int nml_backend_cuda_tanh   (float *data, int n) { return ew_dispatch(data, n, k_tanh);    }

/* ═══════════════════════════════════════════════════════════════════
   PHASE 2 PUBLIC API
   ═══════════════════════════════════════════════════════════════════ */

int nml_backend_cuda_conv2d(Tensor *out, const Tensor *input, const Tensor *kernel,
                             int stride, int pad) {
    if (input->ndim != 4 || kernel->ndim != 4) return -1;
    int N = input->shape[0], C_in = input->shape[1];
    int H = input->shape[2], W = input->shape[3];
    int C_out = kernel->shape[0];
    int KH = kernel->shape[2], KW = kernel->shape[3];
    int OH = (H + 2*pad - KH) / stride + 1;
    int OW = (W + 2*pad - KW) / stride + 1;

    int col_size = N * C_in * KH * KW * OH * OW;
    int k_flat   = C_out * C_in * KH * KW;
    int out_size = N * C_out * OH * OW;

    size_t need = (size_t)col_size;
    if ((size_t)k_flat   > need) need = (size_t)k_flat;
    if ((size_t)out_size > need) need = (size_t)out_size;
    if ((size_t)(N * C_in * H * W) > need) need = (size_t)(N * C_in * H * W);
    if (cuda_ensure(need) != 0) return -1;
    if (host_ensure(need) != 0) return -1;

    /* Copy input to device */
    float *d_input, *d_col, *d_kernel_flat, *d_out;
    size_t in_bytes = (size_t)(N*C_in*H*W)*sizeof(float);
    if (cudaMalloc(&d_input, in_bytes) != cudaSuccess) return -1;
    memcpy(_ha, input->data.f32, in_bytes);
    cudaMemcpy(d_input, _ha, in_bytes, cudaMemcpyHostToDevice);

    size_t col_bytes = (size_t)col_size*sizeof(float);
    if (cudaMalloc(&d_col, col_bytes) != cudaSuccess) { cudaFree(d_input); return -1; }

    /* im2col */
    int threads = 256, blocks = (col_size + threads - 1) / threads;
    k_im2col<<<blocks, threads>>>(d_input, d_col, N, C_in, H, W, KH, KW, OH, OW, stride, pad);
    cudaDeviceSynchronize();
    cudaFree(d_input);

    /* Copy kernel to device */
    size_t k_bytes = (size_t)k_flat*sizeof(float);
    if (cudaMalloc(&d_kernel_flat, k_bytes) != cudaSuccess) { cudaFree(d_col); return -1; }
    memcpy(_hb, kernel->data.f32, k_bytes);
    cudaMemcpy(d_kernel_flat, _hb, k_bytes, cudaMemcpyHostToDevice);

    /* Output buffer */
    size_t out_bytes = (size_t)out_size*sizeof(float);
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) { cudaFree(d_col); cudaFree(d_kernel_flat); return -1; }

    /* For each n: GEMM kernel[C_out, C_in*KH*KW] x col[C_in*KH*KW, OH*OW] = out[C_out, OH*OW] */
    float alpha = 1.0f, beta = 0.0f;
    int m = C_out, k = C_in*KH*KW, n_gemm = OH*OW;
    for (int b = 0; b < N; b++) {
        float *col_n = d_col + (size_t)b * k * n_gemm;
        float *out_n = d_out + (size_t)b * m * n_gemm;
        cublasSgemm(cuda_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                    n_gemm, m, k,
                    &alpha, col_n, n_gemm,
                            d_kernel_flat, k,
                    &beta,  out_n, n_gemm);
    }
    cudaDeviceSynchronize();

    /* Copy result back */
    int out_shape[] = {N, C_out, OH, OW};
    tensor_init_typed(out, 4, out_shape, NML_F32);
    cudaMemcpy(_hc, d_out, out_bytes, cudaMemcpyDeviceToHost);
    memcpy(out->data.f32, _hc, out_bytes);

    cudaFree(d_col); cudaFree(d_kernel_flat); cudaFree(d_out);
    return 0;
}

int nml_backend_cuda_gelu(float *data, int n) {
    cublasHandle_t h = cuda_handle(); if (!h) return -1;
    if (cuda_ensure((size_t)n) != 0) return -1;
    if (host_ensure((size_t)n) != 0) return -1;
    memcpy(_ha, data, (size_t)n*sizeof(float));
    cudaMemcpy(_da, _ha, (size_t)n*sizeof(float), cudaMemcpyHostToDevice);
    int threads=256, nblocks=(n+threads-1)/threads;
    k_gelu<<<nblocks,threads>>>(_da, n);
    cudaDeviceSynchronize();
    cudaMemcpy(_ha, _da, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(data, _ha, (size_t)n*sizeof(float));
    return 0;
}

int nml_backend_cuda_softmax(float *data, int rows, int cols) {
    cublasHandle_t h = cuda_handle(); if (!h) return -1;
    int n = rows * cols;
    if (cuda_ensure((size_t)n) != 0) return -1;
    if (host_ensure((size_t)n) != 0) return -1;
    memcpy(_ha, data, (size_t)n*sizeof(float));
    cudaMemcpy(_da, _ha, (size_t)n*sizeof(float), cudaMemcpyHostToDevice);
    int threads=256, nblocks=(rows+threads-1)/threads;
    k_softmax_rows<<<nblocks,threads>>>(_da, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(_ha, _da, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(data, _ha, (size_t)n*sizeof(float));
    return 0;
}

int nml_backend_cuda_layernorm(float *data, int rows, int cols,
                                const float *gamma, const float *beta) {
    cublasHandle_t h = cuda_handle(); if (!h) return -1;
    int n = rows * cols;
    if (cuda_ensure((size_t)n) != 0) return -1;
    if (host_ensure((size_t)n) != 0) return -1;
    memcpy(_ha, data, (size_t)n*sizeof(float));
    cudaMemcpy(_da, _ha, (size_t)n*sizeof(float), cudaMemcpyHostToDevice);
    float *d_gamma = nullptr, *d_beta = nullptr;
    if (gamma) { cudaMalloc(&d_gamma, (size_t)cols*sizeof(float)); cudaMemcpy(d_gamma, gamma, (size_t)cols*sizeof(float), cudaMemcpyHostToDevice); }
    if (beta)  { cudaMalloc(&d_beta,  (size_t)cols*sizeof(float)); cudaMemcpy(d_beta,  beta,  (size_t)cols*sizeof(float), cudaMemcpyHostToDevice); }
    int threads=256, nblocks=(rows+threads-1)/threads;
    k_layernorm<<<nblocks,threads>>>(_da, rows, cols, d_gamma, d_beta);
    cudaDeviceSynchronize();
    cudaMemcpy(_ha, _da, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(data, _ha, (size_t)n*sizeof(float));
    if (d_gamma) cudaFree(d_gamma);
    if (d_beta)  cudaFree(d_beta);
    return 0;
}

int nml_backend_cuda_attention(Tensor *out, const Tensor *Q, const Tensor *K, const Tensor *V) {
    if (Q->ndim != 2 || K->ndim != 2 || V->ndim != 2) return -1;
    int seq = Q->shape[0], dk = Q->shape[1], dv = V->shape[1];
    if (K->shape[0] != seq || K->shape[1] != dk) return -1;
    if (V->shape[0] != seq) return -1;
    if (Q->dtype != NML_F32) return -1;

    float *d_q, *d_k, *d_v, *d_scores, *d_out_dev;
    size_t q_bytes = (size_t)seq*dk*sizeof(float);
    size_t k_bytes = (size_t)seq*dk*sizeof(float);
    size_t v_bytes = (size_t)seq*dv*sizeof(float);
    size_t s_bytes = (size_t)seq*seq*sizeof(float);
    size_t o_bytes = (size_t)seq*dv*sizeof(float);

    if (cudaMalloc(&d_q, q_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_k, k_bytes) != cudaSuccess) { cudaFree(d_q); return -1; }
    if (cudaMalloc(&d_v, v_bytes) != cudaSuccess) { cudaFree(d_q); cudaFree(d_k); return -1; }
    if (cudaMalloc(&d_scores, s_bytes) != cudaSuccess) { cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); return -1; }
    if (cudaMalloc(&d_out_dev, o_bytes) != cudaSuccess) { cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_scores); return -1; }

    cudaMemcpy(d_q, Q->data.f32, q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K->data.f32, k_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V->data.f32, v_bytes, cudaMemcpyHostToDevice);

    /* Step 1: scores = Q * K^T / sqrt(dk)  [seq,seq] = [seq,dk] x [dk,seq] */
    float alpha = 1.0f / sqrtf((float)dk), beta_v = 0.0f;
    cublasSgemm(cuda_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                seq, seq, dk,
                &alpha, d_k, dk, d_q, dk,
                &beta_v, d_scores, seq);
    cudaDeviceSynchronize();

    /* Step 2: softmax rows of scores (in-place on device) */
    {
        int athreads=256, ablocks=(seq+athreads-1)/athreads;
        k_softmax_rows<<<ablocks,athreads>>>(d_scores, seq, seq);
        cudaDeviceSynchronize();
    }

    /* Step 3: out = scores * V  [seq,dv] */
    alpha = 1.0f; beta_v = 0.0f;
    cublasSgemm(cuda_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                dv, seq, seq,
                &alpha, d_v, dv, d_scores, seq,
                &beta_v, d_out_dev, dv);
    cudaDeviceSynchronize();

    /* Copy result */
    int out_shape[] = {seq, dv};
    tensor_init_typed(out, 2, out_shape, NML_F32);
    if (host_ensure((size_t)seq*dv) == 0) {
        cudaMemcpy(_hc, d_out_dev, o_bytes, cudaMemcpyDeviceToHost);
        memcpy(out->data.f32, _hc, o_bytes);
    }

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_scores); cudaFree(d_out_dev);
    return 0;
}

void nml_backend_cuda_teardown(void) {
    if (_da) { cudaFree(_da); _da = nullptr; }
    if (_db) { cudaFree(_db); _db = nullptr; }
    if (_dc) { cudaFree(_dc); _dc = nullptr; }
    _buf_cap = 0;
    if (_ha) { cudaFreeHost(_ha); _ha = nullptr; }
    if (_hb) { cudaFreeHost(_hb); _hb = nullptr; }
    if (_hc) { cudaFreeHost(_hc); _hc = nullptr; }
    _host_cap = 0;
    if (_handle) { cublasDestroy(_handle); _handle = nullptr; }
}
