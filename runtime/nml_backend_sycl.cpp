/*
 * nml_backend_sycl.cpp — SYCL GPU backend for NML matmul + activations
 *
 * Compiled separately with icpx -fsycl (Intel oneAPI DPC++) or
 * clang++ -fsycl (LLVM SYCL / AdaptiveCpp):
 *
 *   icpx -fsycl -O3 -DNML_USE_SYCL -c -o nml_backend_sycl.o nml_backend_sycl.cpp
 *
 * Two dispatch tiers (controlled by compile-time macros):
 *
 *   Default (NML_USE_SYCL only):
 *     GEMV — work-group parallel reduction over K, parallelises N output elements
 *     GEMM — naive 2-D parallel_for (correct; portable across all SYCL backends)
 *     Activations — simple 1-D parallel_for (RELU, SIGMOID, TANH)
 *
 *   Optimised (NML_USE_SYCL + NML_USE_ONEMKL):
 *     GEMV — oneapi::mkl::blas::row_major::gemv  (Intel-tuned, bandwidth-optimal)
 *     GEMM — oneapi::mkl::blas::row_major::gemm  (Intel-tuned, BLAS-level throughput)
 *     Activations — unchanged (custom kernels are already memory-bandwidth bound)
 *
 * Only F32 tensors are dispatched here. F64/I32 fall through to BLAS/CPU in nml.c.
 *
 * Device selection at runtime via environment variable:
 *   ONEAPI_DEVICE_SELECTOR=opencl:gpu    (Intel oneAPI)
 *   ONEAPI_DEVICE_SELECTOR=level_zero:gpu
 *   SYCL_DEVICE_FILTER=gpu               (AdaptiveCpp / hipSYCL)
 *
 * Thresholds (override at compile time with -DNML_SYCL_MMUL_THRESHOLD=N etc.):
 *   NML_SYCL_MMUL_THRESHOLD  — min m*n for GEMM/GEMV dispatch (default 4096, 64×64)
 *   NML_SYCL_EW_THRESHOLD    — min elements for activation dispatch (default 4096)
 */

#include <sycl/sycl.hpp>
#ifdef NML_USE_ONEMKL
  #include <oneapi/mkl/blas.hpp>
#endif
#include "nml_tensor.h"

extern "C" {

/* ═══════════════════════════════════════════════════════════════════
   QUEUE
   ═══════════════════════════════════════════════════════════════════ */

/* Lazy-initialised global queue. NML is single-threaded so one queue is correct. */
static sycl::queue *_nml_sycl_queue = nullptr;

static sycl::queue &nml_sycl_queue(void) {
    if (!_nml_sycl_queue) {
        _nml_sycl_queue = new sycl::queue(
            sycl::default_selector_v,
            [](sycl::exception_list el) {
                for (auto &e : el) std::rethrow_exception(e);
            }
        );
    }
    return *_nml_sycl_queue;
}

/* ═══════════════════════════════════════════════════════════════════
   PERSISTENT USM SCRATCH BUFFERS
   Allocated once, reused on every matmul call.
   Avoids malloc_shared/free overhead on the hot path.
   Safe: NML is single-threaded, kernels run synchronously (.wait()).
   ═══════════════════════════════════════════════════════════════════ */

static float  *_da = nullptr, *_db = nullptr, *_dc = nullptr;
static size_t  _buf_cap = 0; /* capacity in floats (each buffer) */

static int usm_ensure(sycl::queue &q, size_t need) {
    if (need <= _buf_cap) return 0;
    sycl::free(_da, q); sycl::free(_db, q); sycl::free(_dc, q);
    /* malloc_device = actual GPU VRAM; kernel reads go at full memory bandwidth.
     * Transfers use q.memcpy() which DMA's via the GPU's copy engine. */
    _da = sycl::malloc_device<float>(need, q);
    _db = sycl::malloc_device<float>(need, q);
    _dc = sycl::malloc_device<float>(need, q);
    if (!_da || !_db || !_dc) { _buf_cap = 0; return -1; }
    _buf_cap = need;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
   GEMM / GEMV
   ═══════════════════════════════════════════════════════════════════ */

/*
 * nml_backend_sycl_matmul — row-major F32 matmul (GEMM or GEMV) via SYCL
 *
 *   dest [m × n]  =  a [m × k]  *  b [k × n]
 *
 * When m == 1 the operation is a matrix-vector product (GEMV).  The naive
 * GEMM kernel only launches N work-items for m=1, leaving the GPU mostly
 * idle.  The GEMV path launches N × WG_K work-items and reduces K in
 * parallel, giving much better occupancy on small N.
 *
 * Returns 0 on success, -1 on SYCL exception or USM allocation failure.
 * Caller (tensor_matmul in nml.c) falls through to BLAS/CPU on -1.
 */
int nml_backend_sycl_matmul(Tensor *dest, const Tensor *a, const Tensor *b,
                             int m, int k, int n) {
    try {
        sycl::queue &q = nml_sycl_queue();

        size_t sz = (size_t)m * k > (size_t)k * n
                    ? ((size_t)m * k > (size_t)m * n ? (size_t)m * k : (size_t)m * n)
                    : ((size_t)k * n > (size_t)m * n ? (size_t)k * n : (size_t)m * n);
        if (usm_ensure(q, sz) != 0) return -1;

        float *da = _da, *db = _db, *dc = _dc;

        /* Stage inputs via DMA — q.memcpy uses GPU copy engine into VRAM */
        q.memcpy(da, a->data.f32, (size_t)m * k * sizeof(float));
        q.memcpy(db, b->data.f32, (size_t)k * n * sizeof(float));
        q.wait();

#ifdef NML_USE_ONEMKL
        /* ── oneMKL path: Intel-tuned BLAS on the SYCL device ─────────── */
        if (m == 1) {
            /*
             * GEMV: dc(N) = db(K,N)^T * da(K)
             * gemv(trans=T, rows=K, cols=N, A=db, lda=N, x=da, y=dc)
             */
            oneapi::mkl::blas::row_major::gemv(q,
                oneapi::mkl::transpose::trans,
                k, n,
                1.0f, db, n,
                da, 1,
                0.0f, dc, 1).wait();
        } else {
            /* GEMM: dc(M,N) = da(M,K) * db(K,N) */
            oneapi::mkl::blas::row_major::gemm(q,
                oneapi::mkl::transpose::nontrans,
                oneapi::mkl::transpose::nontrans,
                m, n, k,
                1.0f, da, k,
                       db, n,
                0.0f, dc, n).wait();
        }

#else
        /* ── Portable SYCL kernels ─────────────────────────────────────── */
        if (m == 1) {
            /*
             * GEMV via work-group parallel reduction.
             *
             * Layout: N work-groups, each of size WG_K.
             *   - Work-group j handles output element dc[j].
             *   - Each of the WG_K threads accumulates a partial dot product
             *     over da[] × db[p*n+j] with stride WG_K.
             *   - A tree reduction in shared (local) memory produces dc[j].
             *
             * This launches N × WG_K work-items versus the naive approach's N,
             * giving ~WG_K× better GPU occupancy.
             */
            constexpr int WG_K = 32; /* must be a power of 2 */
            q.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 1> loc(WG_K, h);
                h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>((size_t)n, (size_t)WG_K),
                                      sycl::range<2>(1,         (size_t)WG_K)),
                    [=](sycl::nd_item<2> it) {
                        int j   = (int)it.get_global_id(0); /* output col */
                        int lid = (int)it.get_local_id(1);  /* reduction lane */
                        float s = 0.0f;
                        for (int p = lid; p < k; p += WG_K)
                            s += da[p] * db[p * n + j];
                        loc[lid] = s;
                        it.barrier(sycl::access::fence_space::local_space);
                        for (int stride = WG_K / 2; stride > 0; stride >>= 1) {
                            if (lid < stride) loc[lid] += loc[lid + stride];
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        if (lid == 0) dc[j] = loc[0];
                    }
                );
            }).wait();
        } else {
            /*
             * GEMM: tiled row-major C[i,j] = sum_p A[i,p] * B[p,j]
             *
             * Each work-group loads a TILE×TILE block of A and B into local
             * memory before computing the partial dot products.  This turns
             * the O(M*N*K) global-memory reads of the naive approach into
             * O(M*N*K/TILE) global reads — TILE× better bandwidth utilisation.
             */
            constexpr int TILE = 16;
            size_t gm = ((size_t)m + TILE - 1) / TILE * TILE;
            size_t gn = ((size_t)n + TILE - 1) / TILE * TILE;
            q.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 2> tA(sycl::range<2>(TILE, TILE), h);
                sycl::local_accessor<float, 2> tB(sycl::range<2>(TILE, TILE), h);
                h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>(gm, gn),
                                      sycl::range<2>(TILE, TILE)),
                    [=](sycl::nd_item<2> it) {
                        int row = (int)it.get_global_id(0);
                        int col = (int)it.get_global_id(1);
                        int lr  = (int)it.get_local_id(0);
                        int lc  = (int)it.get_local_id(1);
                        float sum = 0.0f;
                        int tiles = (k + TILE - 1) / TILE;
                        for (int t = 0; t < tiles; t++) {
                            int ac = t * TILE + lc;
                            int br = t * TILE + lr;
                            tA[lr][lc] = (row < m && ac < k) ? da[row * k + ac] : 0.0f;
                            tB[lr][lc] = (br  < k && col < n) ? db[br  * n + col] : 0.0f;
                            it.barrier(sycl::access::fence_space::local_space);
                            for (int i = 0; i < TILE; i++)
                                sum += tA[lr][i] * tB[i][lc];
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        if (row < m && col < n) dc[row * n + col] = sum;
                    }
                );
            }).wait();
        }
#endif /* NML_USE_ONEMKL */

        /* DMA result back to host */
        q.memcpy(dest->data.f32, dc, (size_t)m * n * sizeof(float)).wait();
        return 0;

    } catch (const sycl::exception &) {
        return -1;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   ELEMENTWISE ACTIVATIONS
   ═══════════════════════════════════════════════════════════════════ */

/*
 * All three activation functions share the same pattern:
 *   - data is already copied into `out` by tensor_copy() in nml.c
 *   - we modify it in-place on the GPU
 *   - threshold guard (NML_SYCL_EW_THRESHOLD) lives in nml.c
 *
 * Returns 0 on success, -1 on SYCL exception or allocation failure.
 * Caller falls through to the scalar CPU loop on -1.
 */

int nml_backend_sycl_relu(float *data, int n) {
    try {
        sycl::queue &q = nml_sycl_queue();
        if (usm_ensure(q, (size_t)n) != 0) return -1;
        float *d = _da;
        q.memcpy(d, data, (size_t)n * sizeof(float)).wait();
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            d[i] = d[i] > 0.0f ? d[i] : 0.0f;
        }).wait();
        q.memcpy(data, d, (size_t)n * sizeof(float)).wait();
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

int nml_backend_sycl_sigmoid(float *data, int n) {
    try {
        sycl::queue &q = nml_sycl_queue();
        if (usm_ensure(q, (size_t)n) != 0) return -1;
        float *d = _da;
        q.memcpy(d, data, (size_t)n * sizeof(float)).wait();
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            d[i] = 1.0f / (1.0f + sycl::exp(-d[i]));
        }).wait();
        q.memcpy(data, d, (size_t)n * sizeof(float)).wait();
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

int nml_backend_sycl_tanh(float *data, int n) {
    try {
        sycl::queue &q = nml_sycl_queue();
        if (usm_ensure(q, (size_t)n) != 0) return -1;
        float *d = _da;
        q.memcpy(d, data, (size_t)n * sizeof(float)).wait();
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            d[i] = sycl::tanh(d[i]);
        }).wait();
        q.memcpy(data, d, (size_t)n * sizeof(float)).wait();
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   PHASE 2: CONV2D, GELU, SOFTMAX, LAYERNORM
   ═══════════════════════════════════════════════════════════════════ */

int nml_backend_sycl_conv2d(Tensor *out, const Tensor *input, const Tensor *kernel,
                              int stride, int pad) {
    if (input->ndim != 4 || kernel->ndim != 4) return -1;
    try {
        sycl::queue &q = nml_sycl_queue();
        int N = input->shape[0], C_in = input->shape[1];
        int H = input->shape[2], W = input->shape[3];
        int C_out = kernel->shape[0];
        int KH = kernel->shape[2], KW = kernel->shape[3];
        int OH = (H + 2*pad - KH) / stride + 1;
        int OW = (W + 2*pad - KW) / stride + 1;

        int col_size = N * C_in * KH * KW * OH * OW;
        int k_flat   = C_out * C_in * KH * KW;
        int out_size = N * C_out * OH * OW;
        int in_size  = N * C_in * H * W;

        size_t need = (size_t)col_size;
        if ((size_t)k_flat   > need) need = (size_t)k_flat;
        if ((size_t)out_size > need) need = (size_t)out_size;
        if ((size_t)in_size  > need) need = (size_t)in_size;
        if (usm_ensure(q, need) != 0) return -1;

        /* Allocate col and output on device */
        float *d_col = sycl::malloc_device<float>((size_t)col_size, q);
        float *d_out = sycl::malloc_device<float>((size_t)out_size, q);
        if (!d_col || !d_out) { sycl::free(d_col, q); sycl::free(d_out, q); return -1; }

        /* Copy input and kernel to persistent buffers */
        q.memcpy(_da, input->data.f32, (size_t)in_size*sizeof(float)).wait();
        q.memcpy(_db, kernel->data.f32, (size_t)k_flat*sizeof(float)).wait();

        /* im2col via parallel_for */
        int total_col = col_size;
        int N_v=N, C_in_v=C_in, H_v=H, W_v=W, KH_v=KH, KW_v=KW, OH_v=OH, OW_v=OW;
        int stride_v=stride, pad_v=pad;
        float *d_in_ptr = _da;
        q.parallel_for(sycl::range<1>((size_t)total_col), [=](sycl::id<1> idx_id) {
            int idx = (int)idx_id[0];
            int ow = idx % OW_v; int tmp = idx / OW_v;
            int oh = tmp % OH_v; tmp /= OH_v;
            int kw = tmp % KW_v; tmp /= KW_v;
            int kh = tmp % KH_v; tmp /= KH_v;
            int ci = tmp % C_in_v;
            int n  = tmp / C_in_v;
            int ih = oh * stride_v - pad_v + kh;
            int iw = ow * stride_v - pad_v + kw;
            float val = 0.0f;
            if (ih >= 0 && ih < H_v && iw >= 0 && iw < W_v)
                val = d_in_ptr[n*(C_in_v*H_v*W_v) + ci*(H_v*W_v) + ih*W_v + iw];
            d_col[n*(C_in_v*KH_v*KW_v*OH_v*OW_v) + (ci*KH_v*KW_v + kh*KW_v + kw)*(OH_v*OW_v) + oh*OW_v + ow] = val;
        }).wait();

#ifdef NML_USE_ONEMKL
        /* Use oneMKL GEMM for each batch item */
        for (int b = 0; b < N; b++) {
            float *col_n = d_col + (size_t)b * C_in*KH*KW * OH*OW;
            float *out_n = d_out + (size_t)b * C_out * OH*OW;
            oneapi::mkl::blas::row_major::gemm(q,
                oneapi::mkl::transpose::nontrans,
                oneapi::mkl::transpose::nontrans,
                C_out, OH*OW, C_in*KH*KW,
                1.0f, _db, C_in*KH*KW,
                      col_n, OH*OW,
                0.0f, out_n, OH*OW).wait();
        }
#else
        /* Naive SYCL GEMM for each batch item */
        for (int b = 0; b < N; b++) {
            float *col_n = d_col + (size_t)b * C_in*KH*KW * OH*OW;
            float *out_n = d_out + (size_t)b * C_out * OH*OW;
            int M_v = C_out, K_v = C_in*KH*KW, Nv = OH*OW;
            float *ker_ptr = _db;
            constexpr int TILE = 16;
            size_t gm = ((size_t)M_v + TILE - 1) / TILE * TILE;
            size_t gn = ((size_t)Nv  + TILE - 1) / TILE * TILE;
            q.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 2> tA(sycl::range<2>(TILE, TILE), h);
                sycl::local_accessor<float, 2> tB(sycl::range<2>(TILE, TILE), h);
                h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>(gm, gn),
                                      sycl::range<2>(TILE, TILE)),
                    [=](sycl::nd_item<2> it) {
                        int row = (int)it.get_global_id(0);
                        int col = (int)it.get_global_id(1);
                        int lr  = (int)it.get_local_id(0);
                        int lc  = (int)it.get_local_id(1);
                        float sum = 0.0f;
                        int tiles = (K_v + TILE - 1) / TILE;
                        for (int t = 0; t < tiles; t++) {
                            int ac = t * TILE + lc;
                            int br = t * TILE + lr;
                            tA[lr][lc] = (row < M_v && ac < K_v) ? ker_ptr[row * K_v + ac] : 0.0f;
                            tB[lr][lc] = (br  < K_v && col < Nv) ? col_n[br  * Nv + col] : 0.0f;
                            it.barrier(sycl::access::fence_space::local_space);
                            for (int i = 0; i < TILE; i++)
                                sum += tA[lr][i] * tB[i][lc];
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        if (row < M_v && col < Nv) out_n[row * Nv + col] = sum;
                    }
                );
            }).wait();
        }
#endif

        int out_shape[] = {N, C_out, OH, OW};
        tensor_init_typed(out, 4, out_shape, NML_F32);
        q.memcpy(out->data.f32, d_out, (size_t)out_size*sizeof(float)).wait();
        sycl::free(d_col, q); sycl::free(d_out, q);
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

int nml_backend_sycl_gelu(float *data, int n) {
    try {
        sycl::queue &q = nml_sycl_queue();
        if (usm_ensure(q, (size_t)n) != 0) return -1;
        float *d = _da;
        q.memcpy(d, data, (size_t)n * sizeof(float)).wait();
        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            float x = d[i];
            d[i] = 0.5f * x * (1.0f + sycl::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        }).wait();
        q.memcpy(data, d, (size_t)n * sizeof(float)).wait();
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

int nml_backend_sycl_softmax(float *data, int rows, int cols) {
    try {
        sycl::queue &q = nml_sycl_queue();
        int n = rows * cols;
        if (usm_ensure(q, (size_t)n) != 0) return -1;
        float *d = _da;
        q.memcpy(d, data, (size_t)n * sizeof(float)).wait();
        /* One work-item per row — sequential within each row */
        q.parallel_for(sycl::range<1>((size_t)rows), [=](sycl::id<1> row_id) {
            int row = (int)row_id[0];
            float *r = d + row * cols;
            float mx = r[0];
            for (int j = 1; j < cols; j++) if (r[j] > mx) mx = r[j];
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) { r[j] = sycl::exp(r[j] - mx); sum += r[j]; }
            for (int j = 0; j < cols; j++) r[j] /= sum;
        }).wait();
        q.memcpy(data, d, (size_t)n * sizeof(float)).wait();
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

int nml_backend_sycl_layernorm(float *data, int rows, int cols,
                                const float *gamma, const float *beta) {
    try {
        sycl::queue &q = nml_sycl_queue();
        int n = rows * cols;
        if (usm_ensure(q, (size_t)n) != 0) return -1;
        float *d = _da;
        q.memcpy(d, data, (size_t)n * sizeof(float)).wait();
        /* Copy gamma/beta to device if provided */
        float *d_gamma = nullptr, *d_beta = nullptr;
        if (gamma) {
            d_gamma = sycl::malloc_device<float>((size_t)cols, q);
            if (d_gamma) q.memcpy(d_gamma, gamma, (size_t)cols*sizeof(float)).wait();
        }
        if (beta) {
            d_beta = sycl::malloc_device<float>((size_t)cols, q);
            if (d_beta) q.memcpy(d_beta, beta, (size_t)cols*sizeof(float)).wait();
        }
        q.parallel_for(sycl::range<1>((size_t)rows), [=](sycl::id<1> row_id) {
            int row = (int)row_id[0];
            float *r = d + row * cols;
            float mean = 0.0f;
            for (int j = 0; j < cols; j++) mean += r[j];
            mean /= cols;
            float var = 0.0f;
            for (int j = 0; j < cols; j++) { float dv = r[j] - mean; var += dv*dv; }
            var = sycl::sqrt(var / cols + 1e-5f);
            for (int j = 0; j < cols; j++) {
                r[j] = (r[j] - mean) / var;
                if (d_gamma) r[j] *= d_gamma[j];
                if (d_beta)  r[j] += d_beta[j];
            }
        }).wait();
        q.memcpy(data, d, (size_t)n * sizeof(float)).wait();
        if (d_gamma) sycl::free(d_gamma, q);
        if (d_beta)  sycl::free(d_beta, q);
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

int nml_backend_sycl_attention(Tensor *out, const Tensor *Q, const Tensor *K, const Tensor *V) {
    if (Q->ndim != 2 || K->ndim != 2 || V->ndim != 2) return -1;
    try {
        sycl::queue &q = nml_sycl_queue();
        int seq = Q->shape[0], dk = Q->shape[1], dv = V->shape[1];
        if (K->shape[0] != seq || K->shape[1] != dk) return -1;
        if (V->shape[0] != seq) return -1;
        if (Q->dtype != NML_F32) return -1;

        size_t need = (size_t)seq*dk;
        if ((size_t)seq*seq > need) need = (size_t)seq*seq;
        if ((size_t)seq*dv  > need) need = (size_t)seq*dv;
        if (usm_ensure(q, need) != 0) return -1;

        float *d_q = sycl::malloc_device<float>((size_t)seq*dk, q);
        float *d_k = sycl::malloc_device<float>((size_t)seq*dk, q);
        float *d_v = sycl::malloc_device<float>((size_t)seq*dv, q);
        float *d_scores = sycl::malloc_device<float>((size_t)seq*seq, q);
        float *d_out_dev = sycl::malloc_device<float>((size_t)seq*dv, q);
        if (!d_q || !d_k || !d_v || !d_scores || !d_out_dev) {
            sycl::free(d_q,q); sycl::free(d_k,q); sycl::free(d_v,q);
            sycl::free(d_scores,q); sycl::free(d_out_dev,q); return -1;
        }

        q.memcpy(d_q, Q->data.f32, (size_t)seq*dk*sizeof(float)).wait();
        q.memcpy(d_k, K->data.f32, (size_t)seq*dk*sizeof(float)).wait();
        q.memcpy(d_v, V->data.f32, (size_t)seq*dv*sizeof(float)).wait();

#ifdef NML_USE_ONEMKL
        float inv_sqrt = 1.0f / sycl::sqrt((float)dk);
        /* scores = Q * K^T */
        oneapi::mkl::blas::row_major::gemm(q,
            oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
            seq, seq, dk, inv_sqrt, d_q, dk, d_k, dk, 0.0f, d_scores, seq).wait();
#else
        /* Naive SYCL: scores[i,j] = sum_p Q[i,p]*K[j,p] / sqrt(dk) */
        {
            float inv_sqrt_dk = 1.0f / sycl::sqrt((float)dk);
            int seq_v = seq, dk_v = dk;
            q.parallel_for(sycl::range<2>((size_t)seq, (size_t)seq), [=](sycl::id<2> id) {
                int i = (int)id[0], j = (int)id[1];
                float s = 0.0f;
                for (int p = 0; p < dk_v; p++) s += d_q[i*dk_v+p] * d_k[j*dk_v+p];
                d_scores[i*seq_v+j] = s * inv_sqrt_dk;
            }).wait();
        }
#endif

        /* Softmax rows in-place */
        {
            int seq_v2 = seq;
            q.parallel_for(sycl::range<1>((size_t)seq), [=](sycl::id<1> row_id) {
                int row = (int)row_id[0];
                float *r = d_scores + row * seq_v2;
                float mx = r[0];
                for (int j = 1; j < seq_v2; j++) if (r[j] > mx) mx = r[j];
                float sum = 0.0f;
                for (int j = 0; j < seq_v2; j++) { r[j] = sycl::exp(r[j] - mx); sum += r[j]; }
                for (int j = 0; j < seq_v2; j++) r[j] /= sum;
            }).wait();
        }

#ifdef NML_USE_ONEMKL
        oneapi::mkl::blas::row_major::gemm(q,
            oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            seq, dv, seq, 1.0f, d_scores, seq, d_v, dv, 0.0f, d_out_dev, dv).wait();
#else
        /* Naive SYCL: out[i,j] = sum_k scores[i,k]*V[k,j] */
        {
            int seq_v3 = seq, dv_v = dv;
            q.parallel_for(sycl::range<2>((size_t)seq, (size_t)dv), [=](sycl::id<2> id) {
                int i = (int)id[0], j = (int)id[1];
                float s = 0.0f;
                for (int kk = 0; kk < seq_v3; kk++) s += d_scores[i*seq_v3+kk] * d_v[kk*dv_v+j];
                d_out_dev[i*dv_v+j] = s;
            }).wait();
        }
#endif

        int out_shape[] = {seq, dv};
        tensor_init_typed(out, 2, out_shape, NML_F32);
        q.memcpy(out->data.f32, d_out_dev, (size_t)seq*dv*sizeof(float)).wait();

        sycl::free(d_q,q); sycl::free(d_k,q); sycl::free(d_v,q);
        sycl::free(d_scores,q); sycl::free(d_out_dev,q);
        return 0;
    } catch (const sycl::exception &) {
        return -1;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   CLEANUP
   ═══════════════════════════════════════════════════════════════════ */

/*
 * nml_backend_sycl_teardown — release the SYCL queue and runtime resources.
 * Optional: call on program exit if clean shutdown is needed.
 */
void nml_backend_sycl_teardown(void) {
    if (_nml_sycl_queue) {
        sycl::free(_da, *_nml_sycl_queue);
        sycl::free(_db, *_nml_sycl_queue);
        sycl::free(_dc, *_nml_sycl_queue);
        _da = _db = _dc = nullptr;
        _buf_cap = 0;
    }
    delete _nml_sycl_queue;
    _nml_sycl_queue = nullptr;
}

} /* extern "C" */
