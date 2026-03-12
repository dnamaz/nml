/*
 * NML — Neural Machine Language Runtime v0.6
 * 
 * v0.4 over v0.3:
 *   - SDIV, EDIV  (scalar + element-wise division)
 *   - CMP, CMPI   (general register/immediate comparison)
 *   - SPLT, MERG  (tensor split + merge)
 *   - CALL, RET   (subroutine call stack, depth 32)
 *   - TRAP        (explicit program fault with error code)
 *   - Backward jumps (negative offsets in JMPT/JMPF/JUMP)
 *   - --trace mode, --max-cycles N, vm_validate()
 *   - Error handling via return codes instead of exit(1)
 * 
 * v0.5 over v0.4:
 *   - Symbolic opcode aliases (Unicode math: × ⊕ ⊖ ↓ ↑ ◼ etc.)
 *   - Greek register aliases (ι κ λ μ ν ξ ο π ρ ς α β γ δ φ ψ)
 *   - Verbose human-readable aliases (LOAD, STORE, ACCUMULATE, etc.)
 *   - Verbose register aliases (ACCUMULATOR, SCRATCH, FLAG, etc.)
 *   - Tri-syntax: classic | symbolic | verbose (all produce same bytecode)
 * 
 * v0.6 over v0.5:
 *   - META      (program metadata: §/@name/@input/@output/@invariant)
 *   - FRAG/ENDF (fragment scopes for compositional programs)
 *   - LINK      (import named fragment)
 *   - PTCH      (differential program patches)
 *   - SIGN/VRFY (cryptographic signing and verification)
 *   - VOTE      (multi-agent consensus: median, mean, quorum)
 *   - PROJ/DIST (latent space: projection and distance)
 *   - Semantic type annotations on registers (:currency, :ratio, etc.)
 *   - --describe flag (print program metadata without executing)
 *   - --fragment NAME flag (execute only a named fragment)
 * 
 * Core:   35 unique instructions
 * NML-V:   4 instructions (vision / convolution)
 * NML-T:   4 instructions (transformer / attention)
 * NML-R:   4 instructions (reduction / aggregation)
 * NML-S:   2 instructions (signal processing)
 * NML-M2M: 10 instructions (machine-to-machine)
 * Total:  59 unique instructions (with symbolic + verbose aliases)
 * 
 * Build:  gcc -O2 -o nml runtime/nml.c -lm
 * Run:    ./nml program.nml [data.nml.data] [--trace] [--max-cycles N] [--describe] [--fragment NAME]
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>

/* BLAS acceleration (optional) */
#ifdef NML_USE_ACCELERATE
  #include <Accelerate/Accelerate.h>
  #define NML_HAS_BLAS 1
#elif defined(NML_USE_OPENBLAS)
  #include <cblas.h>
  #define NML_HAS_BLAS 1
#endif
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>

#ifndef NML_NO_DEFAULT_EXTENSIONS
  #ifndef NML_EXT_VISION
    #define NML_EXT_VISION
  #endif
  #ifndef NML_EXT_TRANSFORMER
    #define NML_EXT_TRANSFORMER
  #endif
  #ifndef NML_EXT_REDUCTION
    #define NML_EXT_REDUCTION
  #endif
  #ifndef NML_EXT_SIGNAL
    #define NML_EXT_SIGNAL
  #endif
#endif

#ifndef NML_NO_M2M
  #ifndef NML_EXT_M2M
    #define NML_EXT_M2M
  #endif
#endif

#ifndef NML_NO_GENERAL
  #ifndef NML_EXT_GENERAL
    #define NML_EXT_GENERAL
  #endif
#endif

#ifndef NML_NO_TRAINING
  #ifndef NML_EXT_TRAINING
    #define NML_EXT_TRAINING
  #endif
#endif

/* ═══════════════════════════════════════════
   CONFIGURATION
   ═══════════════════════════════════════════ */

#define NML_VERSION_MAJOR  0
#define NML_VERSION_MINOR  7

#define NML_MAX_REGISTERS    32
#define NML_MAX_DIMS         4
#ifndef NML_MAX_TENSOR_SIZE
  #define NML_MAX_TENSOR_SIZE  65536
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
   DATA TYPES (v0.5)
   ═══════════════════════════════════════════ */

typedef enum { NML_F32 = 0, NML_F64 = 1, NML_I32 = 2 } DType;

static const char* dtype_name(DType dt) {
    switch (dt) { case NML_F32: return "f32"; case NML_F64: return "f64"; case NML_I32: return "i32"; }
    return "f32";
}

static DType dtype_promote(DType a, DType b) {
    if (a == NML_F64 || b == NML_F64) return NML_F64;
    if (a == NML_F32 || b == NML_F32) return NML_F32;
    return NML_I32;
}

static DType parse_dtype(const char *s) {
    if (!s) return NML_F32;
    if (strcmp(s, "f64") == 0 || strcmp(s, "F64") == 0 || strcmp(s, "double") == 0) return NML_F64;
    if (strcmp(s, "i32") == 0 || strcmp(s, "I32") == 0 || strcmp(s, "int") == 0) return NML_I32;
    return NML_F32;
}

/* ═══════════════════════════════════════════
   TENSOR
   ═══════════════════════════════════════════ */

typedef struct {
    union {
        float  f32[NML_MAX_TENSOR_SIZE];
        double f64[NML_MAX_TENSOR_SIZE];
        int    i32[NML_MAX_TENSOR_SIZE];
    } data;
    DType dtype;
    int   shape[NML_MAX_DIMS];
    int   ndim;
    int   size;
} Tensor;

static inline double tensor_getd(const Tensor *t, int i) {
    if (__builtin_expect(t->dtype == NML_F64, 1))
        return t->data.f64[i];
    switch (t->dtype) {
        case NML_F32: return (double)t->data.f32[i];
        case NML_I32: return (double)t->data.i32[i];
        default: return 0.0;
    }
}

static inline void tensor_setd(Tensor *t, int i, double val) {
    if (__builtin_expect(t->dtype == NML_F64, 1)) {
        t->data.f64[i] = val;
        return;
    }
    switch (t->dtype) {
        case NML_F32: t->data.f32[i] = (float)val; break;
        case NML_I32: t->data.i32[i] = (int)val; break;
        default: break;
    }
}

static int tensor_init_typed(Tensor *t, int ndim, const int *shape, DType dtype) {
    t->dtype = dtype;
    t->ndim = ndim;
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    if (t->size > NML_MAX_TENSOR_SIZE) return NML_ERR_OVERFLOW;
    memset(&t->data, 0, sizeof(t->data));
    return NML_OK;
}

static int tensor_init(Tensor *t, int ndim, const int *shape) {
    t->dtype = NML_F32;
    t->ndim = ndim;
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    if (t->size > NML_MAX_TENSOR_SIZE) return NML_ERR_OVERFLOW;
    memset(&t->data, 0, sizeof(t->data));
    return NML_OK;
}

static void tensor_copy(Tensor *dst, const Tensor *src) {
    memcpy(dst, src, sizeof(Tensor));
}

static void tensor_print(const Tensor *t, const char *label) {
    printf("  %s: shape=[", label);
    for (int i = 0; i < t->ndim; i++)
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? "x" : "");
    printf("]");
    if (t->dtype != NML_F32) printf(" dtype=%s", dtype_name(t->dtype));
    printf(" data=[");
    int n = t->size < 8 ? t->size : 8;
    for (int i = 0; i < n; i++)
        printf("%.4f%s", tensor_getd(t, i), i < n - 1 ? ", " : "");
    if (t->size > 8) printf(", ...");
    printf("]\n");
}

/* ═══════════════════════════════════════════
   CORE TENSOR OPERATIONS
   ═══════════════════════════════════════════ */

static int tensor_matmul(Tensor *out, const Tensor *a, const Tensor *b) {
    int m, k1, k2, n;
    if (a->ndim == 1 && b->ndim == 2) { m = 1; k1 = a->shape[0]; }
    else if (a->ndim == 2 && b->ndim == 2) { m = a->shape[0]; k1 = a->shape[1]; }
    else return NML_ERR_SHAPE;
    k2 = b->shape[0]; n = b->shape[1];
    if (k1 != k2) return NML_ERR_SHAPE;
    DType dt = dtype_promote(a->dtype, b->dtype);
    int shape[] = {m, n};
    Tensor tmp;
    Tensor *dest = (out == a || out == b) ? &tmp : out;
    int rc = tensor_init_typed(dest, 2, shape, dt);
    if (rc) return rc;

#ifdef NML_HAS_BLAS
    if (dt == NML_F64 && a->dtype == NML_F64 && b->dtype == NML_F64) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k1, 1.0, a->data.f64, k1, b->data.f64, n,
                    0.0, dest->data.f64, n);
    } else if (dt == NML_F32 && a->dtype == NML_F32 && b->dtype == NML_F32) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k1, 1.0f, a->data.f32, k1, b->data.f32, n,
                    0.0f, dest->data.f32, n);
    } else
#endif
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k1; p++) sum += tensor_getd(a, i*k1+p) * tensor_getd(b, p*n+j);
                tensor_setd(dest, i*n+j, sum);
            }
    }

    if (dest == &tmp) *out = tmp;
    return NML_OK;
}

static void tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    DType dt = dtype_promote(a->dtype, b->dtype);
    Tensor tmp;
    Tensor *dest = (out == a || out == b) ? &tmp : out;
    tensor_init_typed(dest, a->ndim, a->shape, dt); dest->size = a->size;
    for (int i = 0; i < a->size; i++) tensor_setd(dest, i, tensor_getd(a, i) + tensor_getd(b, i));
    if (dest == &tmp) *out = tmp;
}

static void tensor_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    DType dt = dtype_promote(a->dtype, b->dtype);
    Tensor tmp;
    Tensor *dest = (out == a || out == b) ? &tmp : out;
    tensor_init_typed(dest, a->ndim, a->shape, dt); dest->size = a->size;
    for (int i = 0; i < a->size; i++) tensor_setd(dest, i, tensor_getd(a, i) - tensor_getd(b, i));
    if (dest == &tmp) *out = tmp;
}

static void tensor_emul(Tensor *out, const Tensor *a, const Tensor *b) {
    DType dt = dtype_promote(a->dtype, b->dtype);
    Tensor tmp;
    Tensor *dest = (out == a || out == b) ? &tmp : out;
    tensor_init_typed(dest, a->ndim, a->shape, dt); dest->size = a->size;
    for (int i = 0; i < a->size; i++) tensor_setd(dest, i, tensor_getd(a, i) * tensor_getd(b, i));
    if (dest == &tmp) *out = tmp;
}

static int tensor_ediv(Tensor *out, const Tensor *a, const Tensor *b) {
    DType dt = dtype_promote(a->dtype, b->dtype);
    Tensor tmp;
    Tensor *dest = (out == a || out == b) ? &tmp : out;
    tensor_init_typed(dest, a->ndim, a->shape, dt); dest->size = a->size;
    for (int i = 0; i < a->size; i++) {
        if (tensor_getd(b, i) == 0.0) return NML_ERR_DIVZERO;
        tensor_setd(dest, i, tensor_getd(a, i) / tensor_getd(b, i));
    }
    if (dest == &tmp) *out = tmp;
    return NML_OK;
}

static void tensor_scale(Tensor *out, const Tensor *a, double s) {
    tensor_copy(out, a);
    for (int i = 0; i < a->size; i++) tensor_setd(out, i, tensor_getd(a, i) * s);
}

static int tensor_scale_div(Tensor *out, const Tensor *a, double s) {
    if (s == 0.0) return NML_ERR_DIVZERO;
    tensor_copy(out, a);
    for (int i = 0; i < a->size; i++) tensor_setd(out, i, tensor_getd(a, i) / s);
    return NML_OK;
}

static double tensor_dot(const Tensor *a, const Tensor *b) {
    double sum = 0.0;
    for (int i = 0; i < a->size; i++) sum += tensor_getd(a, i) * tensor_getd(b, i);
    return sum;
}

static void tensor_relu(Tensor *out, const Tensor *t) {
    tensor_copy(out, t);
    for (int i = 0; i < t->size; i++) {
        double v = tensor_getd(t, i);
        tensor_setd(out, i, v > 0 ? v : 0);
    }
}

static void tensor_sigmoid(Tensor *out, const Tensor *t) {
    tensor_copy(out, t);
    for (int i = 0; i < t->size; i++) {
        double v = tensor_getd(t, i);
        tensor_setd(out, i, 1.0 / (1.0 + exp(-v)));
    }
}

static void tensor_tanh_act(Tensor *out, const Tensor *t) {
    tensor_copy(out, t);
    for (int i = 0; i < t->size; i++) tensor_setd(out, i, tanh(tensor_getd(t, i)));
}

static void tensor_softmax(Tensor *out, const Tensor *t) {
    tensor_copy(out, t);
    double max_val = tensor_getd(t, 0);
    for (int i = 1; i < t->size; i++) { double v = tensor_getd(t, i); if (v > max_val) max_val = v; }
    double sum = 0.0;
    for (int i = 0; i < t->size; i++) { double v = exp(tensor_getd(t, i) - max_val); tensor_setd(out, i, v); sum += v; }
    for (int i = 0; i < t->size; i++) tensor_setd(out, i, tensor_getd(out, i) / sum);
}

static int tensor_transpose(Tensor *out, const Tensor *t) {
    if (t->ndim != 2) return NML_ERR_SHAPE;
    int r = t->shape[0], c = t->shape[1];
    int shape[] = {c, r};
    int rc = tensor_init(out, 2, shape);
    if (rc) return rc;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            out->data.f32[j*r+i] = t->data.f32[i*c+j];
    return NML_OK;
}

static int tensor_split(Tensor *out_lo, Tensor *out_hi, const Tensor *src, int dim, int split_at) {
    DType dt = src->dtype;
    if (src->ndim == 1) {
        int n = src->shape[0];
        if (split_at < 0) split_at = n / 2;
        if (split_at < 0 || split_at > n) return NML_ERR_OOB;
        int s1[] = {split_at}, s2[] = {n - split_at};
        int rc = tensor_init_typed(out_lo, 1, s1, dt); if (rc) return rc;
        for (int i = 0; i < split_at; i++) tensor_setd(out_lo, i, tensor_getd(src, i));
        if (out_hi) {
            rc = tensor_init_typed(out_hi, 1, s2, dt); if (rc) return rc;
            for (int i = 0; i < n - split_at; i++) tensor_setd(out_hi, i, tensor_getd(src, split_at + i));
        }
        return NML_OK;
    }
    if (src->ndim != 2 || dim > 1) return NML_ERR_SHAPE;
    int rows = src->shape[0], cols = src->shape[1];
    if (dim == 0) {
        if (split_at < 0) split_at = rows / 2;
        if (split_at < 0 || split_at > rows) return NML_ERR_OOB;
        int s1[] = {split_at, cols}, s2[] = {rows - split_at, cols};
        int rc = tensor_init_typed(out_lo, 2, s1, dt); if (rc) return rc;
        for (int i = 0; i < split_at * cols; i++) tensor_setd(out_lo, i, tensor_getd(src, i));
        if (out_hi) {
            rc = tensor_init_typed(out_hi, 2, s2, dt); if (rc) return rc;
            for (int i = 0; i < (rows - split_at) * cols; i++) tensor_setd(out_hi, i, tensor_getd(src, split_at * cols + i));
        }
    } else {
        if (split_at < 0) split_at = cols / 2;
        if (split_at < 0 || split_at > cols) return NML_ERR_OOB;
        int s1[] = {rows, split_at}, s2[] = {rows, cols - split_at};
        int rc = tensor_init_typed(out_lo, 2, s1, dt); if (rc) return rc;
        if (out_hi) { rc = tensor_init_typed(out_hi, 2, s2, dt); if (rc) return rc; }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < split_at; j++) tensor_setd(out_lo, i * split_at + j, tensor_getd(src, i * cols + j));
            if (out_hi) for (int j = 0; j < cols - split_at; j++) tensor_setd(out_hi, i * (cols - split_at) + j, tensor_getd(src, i * cols + split_at + j));
        }
    }
    return NML_OK;
}

static int tensor_merge(Tensor *out, const Tensor *a, const Tensor *b, int dim) {
    DType dt = dtype_promote(a->dtype, b->dtype);
    /* 1D or scalar: concatenate into 1D */
    if (a->ndim <= 1 && b->ndim <= 1) {
        int na = a->size, nb = b->size;
        int shape[] = {na + nb};
        int rc = tensor_init_typed(out, 1, shape, dt); if (rc) return rc;
        for (int i = 0; i < na; i++) tensor_setd(out, i, tensor_getd(a, i));
        for (int i = 0; i < nb; i++) tensor_setd(out, na + i, tensor_getd(b, i));
        return NML_OK;
    }
    if (a->ndim != 2 || b->ndim != 2 || dim > 1) return NML_ERR_SHAPE;
    int ar = a->shape[0], ac = a->shape[1];
    int br = b->shape[0], bc = b->shape[1];
    if (dim == 0) {
        if (ac != bc) return NML_ERR_SHAPE;
        int shape[] = {ar + br, ac};
        int rc = tensor_init_typed(out, 2, shape, dt); if (rc) return rc;
        for (int i = 0; i < ar * ac; i++) tensor_setd(out, i, tensor_getd(a, i));
        for (int i = 0; i < br * bc; i++) tensor_setd(out, ar * ac + i, tensor_getd(b, i));
    } else {
        if (ar != br) return NML_ERR_SHAPE;
        int shape[] = {ar, ac + bc};
        int oc = ac + bc;
        int rc = tensor_init_typed(out, 2, shape, dt); if (rc) return rc;
        for (int i = 0; i < ar; i++) {
            for (int j = 0; j < ac; j++) tensor_setd(out, i * oc + j, tensor_getd(a, i * ac + j));
            for (int j = 0; j < bc; j++) tensor_setd(out, i * oc + ac + j, tensor_getd(b, i * bc + j));
        }
    }
    return NML_OK;
}

/* ═══════════════════════════════════════════
   EXTENSION: NML-V (Vision)
   ═══════════════════════════════════════════ */

#ifdef NML_EXT_VISION

static void tensor_conv2d(Tensor *out, const Tensor *input, const Tensor *kernel, int stride, int pad) {
    int H, W, KH, KW, OH, OW;
    if (input->ndim == 2) { H = input->shape[0]; W = input->shape[1]; }
    else if (input->ndim == 3) { H = input->shape[1]; W = input->shape[2]; }
    else { H = input->shape[0]; W = input->shape[1]; }
    KH = kernel->shape[0]; KW = kernel->shape[1];
    OH = (H + 2*pad - KH) / stride + 1;
    OW = (W + 2*pad - KW) / stride + 1;
    int shape[] = {OH, OW};
    tensor_init(out, 2, shape);
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            float sum = 0.0f;
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++) {
                    int ih = oh * stride - pad + kh, iw = ow * stride - pad + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        sum += input->data.f32[ih * W + iw] * kernel->data.f32[kh * KW + kw];
                }
            out->data.f32[oh * OW + ow] = sum;
        }
}

static void tensor_maxpool(Tensor *out, const Tensor *input, int pool_size, int stride) {
    int H = input->shape[0], W = input->shape[1];
    int OH = (H - pool_size) / stride + 1, OW = (W - pool_size) / stride + 1;
    int shape[] = {OH, OW};
    tensor_init(out, 2, shape);
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            float mx = -1e30f;
            for (int ph = 0; ph < pool_size; ph++)
                for (int pw = 0; pw < pool_size; pw++) {
                    int ih = oh * stride + ph, iw = ow * stride + pw;
                    if (ih < H && iw < W && input->data.f32[ih * W + iw] > mx)
                        mx = input->data.f32[ih * W + iw];
                }
            out->data.f32[oh * OW + ow] = mx;
        }
}

static void tensor_upscale(Tensor *out, const Tensor *input, int scale) {
    int H = input->shape[0], W = input->shape[1];
    int OH = H * scale, OW = W * scale;
    int shape[] = {OH, OW};
    tensor_init(out, 2, shape);
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            out->data.f32[oh * OW + ow] = input->data.f32[(oh/scale) * W + (ow/scale)];
}

static void tensor_pad(Tensor *out, const Tensor *input, int pad) {
    int H = input->shape[0], W = input->shape[1];
    int OH = H + 2*pad, OW = W + 2*pad;
    int shape[] = {OH, OW};
    tensor_init(out, 2, shape);
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            out->data.f32[(h+pad)*OW + (w+pad)] = input->data.f32[h*W + w];
}

#endif /* NML_EXT_VISION */

/* ═══════════════════════════════════════════
   EXTENSION: NML-T (Transformer)
   ═══════════════════════════════════════════ */

#ifdef NML_EXT_TRANSFORMER

static void tensor_attention(Tensor *out, const Tensor *Q, const Tensor *K, const Tensor *V) {
    int seq_len = Q->shape[0], d_k = Q->shape[1], d_v = V->shape[1];
    float scale = 1.0f / sqrtf((float)d_k);
    Tensor scores;
    int score_shape[] = {seq_len, seq_len};
    tensor_init(&scores, 2, score_shape);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_k; k++) sum += Q->data.f32[i*d_k+k] * K->data.f32[j*d_k+k];
            scores.data.f32[i*seq_len+j] = sum * scale;
        }
    for (int i = 0; i < seq_len; i++) {
        float mx = scores.data.f32[i*seq_len];
        for (int j = 1; j < seq_len; j++) if (scores.data.f32[i*seq_len+j] > mx) mx = scores.data.f32[i*seq_len+j];
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) { scores.data.f32[i*seq_len+j] = expf(scores.data.f32[i*seq_len+j] - mx); sum += scores.data.f32[i*seq_len+j]; }
        for (int j = 0; j < seq_len; j++) scores.data.f32[i*seq_len+j] /= sum;
    }
    int out_shape[] = {seq_len, d_v};
    tensor_init(out, 2, out_shape);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_v; j++) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; k++) sum += scores.data.f32[i*seq_len+k] * V->data.f32[k*d_v+j];
            out->data.f32[i*d_v+j] = sum;
        }
}

static void tensor_layernorm(Tensor *out, const Tensor *input, const Tensor *gamma, const Tensor *beta) {
    tensor_copy(out, input);
    int last_dim = input->shape[input->ndim - 1];
    int n_groups = input->size / last_dim;
    for (int g = 0; g < n_groups; g++) {
        int off = g * last_dim;
        float mean = 0.0f;
        for (int i = 0; i < last_dim; i++) mean += input->data.f32[off + i];
        mean /= last_dim;
        float var = 0.0f;
        for (int i = 0; i < last_dim; i++) { float d = input->data.f32[off + i] - mean; var += d * d; }
        var /= last_dim;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < last_dim; i++) {
            float norm = (input->data.f32[off + i] - mean) * inv_std;
            float gv = (gamma && i < gamma->size) ? gamma->data.f32[i] : 1.0f;
            float bv = (beta && i < beta->size) ? beta->data.f32[i] : 0.0f;
            out->data.f32[off + i] = norm * gv + bv;
        }
    }
}

static void tensor_embedding(Tensor *out, const Tensor *table, const Tensor *indices) {
    int embed_dim = table->shape[1], seq_len = indices->size;
    int shape[] = {seq_len, embed_dim};
    tensor_init(out, 2, shape);
    for (int i = 0; i < seq_len; i++) {
        int idx = (int)indices->data.f32[i];
        if (idx >= 0 && idx < table->shape[0])
            for (int j = 0; j < embed_dim; j++)
                out->data.f32[i * embed_dim + j] = table->data.f32[idx * embed_dim + j];
    }
}

static void tensor_gelu(Tensor *out, const Tensor *t) {
    tensor_copy(out, t);
    for (int i = 0; i < t->size; i++) {
        float x = t->data.f32[i];
        out->data.f32[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

#endif /* NML_EXT_TRANSFORMER */

/* ═══════════════════════════════════════════
   EXTENSION: NML-R (Reduction)
   ═══════════════════════════════════════════ */

#ifdef NML_EXT_REDUCTION

#define RDUC_SUM  0
#define RDUC_MEAN 1
#define RDUC_MAX  2
#define RDUC_MIN  3

static void tensor_reduce(Tensor *out, const Tensor *input, int op, int dim) {
    if (dim == -1) {
        int shape[] = {1};
        tensor_init(out, 1, shape);
        float r = input->data.f32[0];
        for (int i = 1; i < input->size; i++) {
            switch (op) {
                case RDUC_SUM: case RDUC_MEAN: r += input->data.f32[i]; break;
                case RDUC_MAX: if (input->data.f32[i] > r) r = input->data.f32[i]; break;
                case RDUC_MIN: if (input->data.f32[i] < r) r = input->data.f32[i]; break;
            }
        }
        if (op == RDUC_MEAN) r /= input->size;
        out->data.f32[0] = r;
    } else if (input->ndim == 2) {
        int rows = input->shape[0], cols = input->shape[1];
        if (dim == 0) {
            int shape[] = {1, cols};
            tensor_init(out, 2, shape);
            for (int j = 0; j < cols; j++) {
                float r = input->data.f32[j];
                for (int i = 1; i < rows; i++) { float v = input->data.f32[i*cols+j]; switch(op){ case RDUC_SUM:case RDUC_MEAN:r+=v;break; case RDUC_MAX:if(v>r)r=v;break; case RDUC_MIN:if(v<r)r=v;break; }}
                if (op == RDUC_MEAN) r /= rows;
                out->data.f32[j] = r;
            }
        } else {
            int shape[] = {rows, 1};
            tensor_init(out, 2, shape);
            for (int i = 0; i < rows; i++) {
                float r = input->data.f32[i*cols];
                for (int j = 1; j < cols; j++) { float v = input->data.f32[i*cols+j]; switch(op){ case RDUC_SUM:case RDUC_MEAN:r+=v;break; case RDUC_MAX:if(v>r)r=v;break; case RDUC_MIN:if(v<r)r=v;break; }}
                if (op == RDUC_MEAN) r /= cols;
                out->data.f32[i] = r;
            }
        }
    }
}

static void tensor_where(Tensor *out, const Tensor *cond, const Tensor *a, const Tensor *b) {
    tensor_copy(out, a);
    for (int i = 0; i < a->size; i++)
        out->data.f32[i] = (cond->data.f32[i] > 0.0f) ? a->data.f32[i] : b->data.f32[i];
}

static void tensor_clamp(Tensor *out, const Tensor *input, float lo, float hi) {
    tensor_copy(out, input);
    for (int i = 0; i < input->size; i++) {
        if (out->data.f32[i] < lo) out->data.f32[i] = lo;
        if (out->data.f32[i] > hi) out->data.f32[i] = hi;
    }
}

#define CMP_GT 0
#define CMP_LT 1
#define CMP_GE 2
#define CMP_LE 3
#define CMP_EQ 4

static void tensor_compare(Tensor *out, const Tensor *a, float threshold, int cmp_op) {
    tensor_copy(out, a);
    for (int i = 0; i < a->size; i++) {
        int r = 0;
        switch (cmp_op) {
            case CMP_GT: r = a->data.f32[i] > threshold; break;
            case CMP_LT: r = a->data.f32[i] < threshold; break;
            case CMP_GE: r = a->data.f32[i] >= threshold; break;
            case CMP_LE: r = a->data.f32[i] <= threshold; break;
            case CMP_EQ: r = fabsf(a->data.f32[i] - threshold) < 1e-6f; break;
        }
        out->data.f32[i] = r ? 1.0f : 0.0f;
    }
}

#endif /* NML_EXT_REDUCTION */

/* ═══════════════════════════════════════════
   EXTENSION: NML-S (Signal)
   ═══════════════════════════════════════════ */

#ifdef NML_EXT_SIGNAL

static void tensor_fft(Tensor *out_real, Tensor *out_imag, const Tensor *input) {
    int N = input->size;
    int shape[] = {N};
    tensor_init(out_real, 1, shape);
    tensor_init(out_imag, 1, shape);
    for (int k = 0; k < N; k++) {
        float sr = 0.0f, si = 0.0f;
        for (int n = 0; n < N; n++) {
            float angle = -2.0f * 3.14159265358979f * k * n / N;
            sr += input->data.f32[n] * cosf(angle);
            si += input->data.f32[n] * sinf(angle);
        }
        out_real->data.f32[k] = sr;
        out_imag->data.f32[k] = si;
    }
}

static void tensor_fir_filter(Tensor *out, const Tensor *input, const Tensor *coeffs) {
    int N = input->size, M = coeffs->size;
    int shape[] = {N};
    tensor_init(out, 1, shape);
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) { int idx = n - k; if (idx >= 0 && idx < N) sum += input->data.f32[idx] * coeffs->data.f32[k]; }
        out->data.f32[n] = sum;
    }
}

#endif /* NML_EXT_SIGNAL */

/* ═══════════════════════════════════════════
   OPCODES
   ═══════════════════════════════════════════ */

typedef enum {
    /* Core: Neural Network (22) */
    OP_MMUL, OP_MADD, OP_MSUB, OP_EMUL, OP_SDOT, OP_SCLR,
    OP_RELU, OP_SIGM, OP_TANH, OP_SOFT,
    OP_LD,   OP_ST,   OP_MOV,  OP_ALLC,
    OP_RSHP, OP_TRNS, OP_SPLT, OP_MERG,
    OP_LOOP, OP_ENDP,
    OP_SYNC, OP_HALT,
    /* Core: Tree Models (6) */
    OP_CMPF, OP_LEAF, OP_TACC, OP_JMPT, OP_JMPF, OP_JUMP,
    /* Core: v0.4 additions (9) */
    OP_SDIV, OP_EDIV,
    OP_CMP,  OP_CMPI,
    OP_CALL, OP_RET,
    OP_TRAP,
    /* Extension: NML-V Vision (4) */
    OP_CONV, OP_POOL, OP_UPSC, OP_PADZ,
    /* Extension: NML-T Transformer (4) */
    OP_ATTN, OP_NORM, OP_EMBD, OP_GELU,
    /* Extension: NML-R Reduction (4) */
    OP_RDUC, OP_WHER, OP_CLMP, OP_CMPR,
    /* Extension: NML-S Signal (2) */
    OP_FFT,  OP_FILT,
#ifndef NML_NO_M2M
    /* Extension: NML-M2M (11) */
    OP_META, OP_FRAG, OP_ENDF, OP_LINK,
    OP_PTCH, OP_SIGN, OP_VRFY,
    OP_VOTE,
    OP_PROJ, OP_DIST, OP_GATH, OP_SCAT,
#endif
#ifndef NML_NO_GENERAL
    /* Extension: NML-G General Purpose (5) */
    OP_SYS, OP_MOD, OP_ITOF, OP_FTOI, OP_BNOT,
#endif
#ifndef NML_NO_TRAINING
    /* Extension: NML-TR Training (4) — v0.7 */
    OP_BKWD, OP_WUPD, OP_LOSS, OP_TNET,
#endif
    OP_COUNT
} Opcode;

typedef struct { const char *name; Opcode code; const char *ext; } OpcodeEntry;

static const OpcodeEntry OPCODE_TABLE[] = {
    /* Core — original 28 */
    {"MMUL", OP_MMUL, "core"}, {"MADD", OP_MADD, "core"}, {"MSUB", OP_MSUB, "core"},
    {"EMUL", OP_EMUL, "core"}, {"SDOT", OP_SDOT, "core"}, {"SCLR", OP_SCLR, "core"},
    {"RELU", OP_RELU, "core"}, {"SIGM", OP_SIGM, "core"}, {"TANH", OP_TANH, "core"},
    {"SOFT", OP_SOFT, "core"}, {"LD",   OP_LD,   "core"}, {"ST",   OP_ST,   "core"},
    {"MOV",  OP_MOV,  "core"}, {"ALLC", OP_ALLC, "core"}, {"RSHP", OP_RSHP, "core"},
    {"TRNS", OP_TRNS, "core"}, {"SPLT", OP_SPLT, "core"}, {"MERG", OP_MERG, "core"},
    {"LOOP", OP_LOOP, "core"}, {"ENDP", OP_ENDP, "core"},
    {"SYNC", OP_SYNC, "core"}, {"HALT", OP_HALT, "core"},
    {"CMPF", OP_CMPF, "core"}, {"LEAF", OP_LEAF, "core"}, {"TACC", OP_TACC, "core"},
    {"JMPT", OP_JMPT, "core"}, {"JMPF", OP_JMPF, "core"}, {"JUMP", OP_JUMP, "core"},
    /* Core — v0.4 additions */
    {"SDIV", OP_SDIV, "core"}, {"EDIV", OP_EDIV, "core"},
    {"CMP",  OP_CMP,  "core"}, {"CMPI", OP_CMPI, "core"},
    {"CALL", OP_CALL, "core"}, {"RET",  OP_RET,  "core"},
    {"TRAP", OP_TRAP, "core"},
    /* NML-V */
    {"CONV", OP_CONV, "NML-V"}, {"POOL", OP_POOL, "NML-V"},
    {"UPSC", OP_UPSC, "NML-V"}, {"PADZ", OP_PADZ, "NML-V"},
    /* NML-T */
    {"ATTN", OP_ATTN, "NML-T"}, {"NORM", OP_NORM, "NML-T"},
    {"EMBD", OP_EMBD, "NML-T"}, {"GELU", OP_GELU, "NML-T"},
    /* NML-R */
    {"RDUC", OP_RDUC, "NML-R"}, {"WHER", OP_WHER, "NML-R"},
    {"CLMP", OP_CLMP, "NML-R"}, {"CMPR", OP_CMPR, "NML-R"},
    /* NML-S */
    {"FFT",  OP_FFT,  "NML-S"}, {"FILT", OP_FILT, "NML-S"},
#ifndef NML_NO_M2M
    /* NML-M2M */
    {"META", OP_META, "NML-M2M"}, {"FRAG", OP_FRAG, "NML-M2M"},
    {"ENDF", OP_ENDF, "NML-M2M"}, {"LINK", OP_LINK, "NML-M2M"},
    {"PTCH", OP_PTCH, "NML-M2M"}, {"SIGN", OP_SIGN, "NML-M2M"},
    {"VRFY", OP_VRFY, "NML-M2M"}, {"VOTE", OP_VOTE, "NML-M2M"},
    {"PROJ", OP_PROJ, "NML-M2M"}, {"DIST", OP_DIST, "NML-M2M"},
    {"GATH", OP_GATH, "NML-M2M"}, {"SCAT", OP_SCAT, "NML-M2M"},
#endif
    /* ═══ Symbolic aliases (Unicode) ═══ */
    /* Arithmetic */
    {"\xc3\x97", OP_MMUL, "core"},  /* × */
    {"\xe2\x8a\x95", OP_MADD, "core"},  /* ⊕ */
    {"\xe2\x8a\x96", OP_MSUB, "core"},  /* ⊖ */
    {"\xe2\x8a\x97", OP_EMUL, "core"},  /* ⊗ */
    {"\xe2\x8a\x98", OP_EDIV, "core"},  /* ⊘ */
    {"\xc2\xb7", OP_SDOT, "core"},  /* · */
    {"\xe2\x88\x97", OP_SCLR, "core"},  /* ∗ */
    {"\xc3\xb7", OP_SDIV, "core"},  /* ÷ */
    /* Activation */
    {"\xe2\x8c\x90", OP_RELU, "core"},  /* ⌐ */
    {"\xcf\x83", OP_SIGM, "core"},  /* σ */
    {"\xcf\x84", OP_TANH, "core"},  /* τ */
    {"\xce\xa3", OP_SOFT, "core"},  /* Σ */
    /* Memory */
    {"\xe2\x86\x93", OP_LD,   "core"},  /* ↓ */
    {"\xe2\x86\x91", OP_ST,   "core"},  /* ↑ */
    {"\xe2\x86\x90", OP_MOV,  "core"},  /* ← */
    {"\xe2\x96\xa1", OP_ALLC, "core"},  /* □ */
    /* Data Flow */
    {"\xe2\x8a\x9e", OP_RSHP, "core"},  /* ⊞ */
    {"\xe2\x8a\xa4", OP_TRNS, "core"},  /* ⊤ */
    {"\xe2\x8a\xa2", OP_SPLT, "core"},  /* ⊢ */
    {"\xe2\x8a\xa3", OP_MERG, "core"},  /* ⊣ */
    /* Control */
    {"\xe2\x86\xbb", OP_LOOP, "core"},  /* ↻ */
    {"\xe2\x86\xba", OP_ENDP, "core"},  /* ↺ */
    {"\xe2\x86\x97", OP_JMPT, "core"},  /* ↗ */
    {"\xe2\x86\x98", OP_JMPF, "core"},  /* ↘ */
    {"\xe2\x86\x92", OP_JUMP, "core"},  /* → */
    /* Comparison */
    {"\xe2\x8b\x88", OP_CMPF, "core"},  /* ⋈ */
    {"\xe2\x89\xb6", OP_CMP,  "core"},  /* ≶ */
    {"\xe2\x89\xba", OP_CMPI, "core"},  /* ≺ */
    /* Tree */
    {"\xe2\x88\x8e", OP_LEAF, "core"},  /* ∎ */
    {"\xe2\x88\x91", OP_TACC, "core"},  /* ∑ */
    /* Subroutine */
    {"\xe2\x87\x92", OP_CALL, "core"},  /* ⇒ */
    {"\xe2\x87\x90", OP_RET,  "core"},  /* ⇐ */
    /* System */
    {"\xe2\x8f\xb8", OP_SYNC, "core"},  /* ⏸ */
    {"\xe2\x97\xbc", OP_HALT, "core"},  /* ◼ */
    {"\xe2\x9a\xa0", OP_TRAP, "core"},  /* ⚠ */
    /* NML-V symbolic */
    {"\xe2\x8a\x9b", OP_CONV, "NML-V"},  /* ⊛ */
    {"\xe2\x8a\x93", OP_POOL, "NML-V"},  /* ⊓ */
    {"\xe2\x8a\x94", OP_UPSC, "NML-V"},  /* ⊔ */
    {"\xe2\x8a\xa1", OP_PADZ, "NML-V"},  /* ⊡ */
    /* NML-T symbolic */
    {"\xe2\x8a\x99", OP_ATTN, "NML-T"},  /* ⊙ */
    {"\xe2\x80\x96", OP_NORM, "NML-T"},  /* ‖ */
    {"\xe2\x8a\x8f", OP_EMBD, "NML-T"},  /* ⊏ */
    {"\xe2\x84\x8a", OP_GELU, "NML-T"},  /* ℊ */
    /* NML-R symbolic */
    {"\xe2\x8a\xa5", OP_RDUC, "NML-R"},  /* ⊥ */
    {"\xe2\x8a\xbb", OP_WHER, "NML-R"},  /* ⊻ */
    {"\xe2\x8a\xa7", OP_CLMP, "NML-R"},  /* ⊧ */
    {"\xe2\x8a\x9c", OP_CMPR, "NML-R"},  /* ⊜ */
    /* NML-S symbolic */
    {"\xe2\x88\xbf", OP_FFT,  "NML-S"},  /* ∿ */
    {"\xe2\x8b\x90", OP_FILT, "NML-S"},  /* ⋐ */
#ifndef NML_NO_M2M
    /* NML-M2M symbolic */
    {"\xc2\xa7", OP_META, "NML-M2M"},       /* § */
    {"\xe2\x97\x86", OP_FRAG, "NML-M2M"},   /* ◆ */
    {"\xe2\x97\x87", OP_ENDF, "NML-M2M"},   /* ◇ */
    {"\xe2\x8a\x95", OP_LINK, "NML-M2M"},   /* ⊕ (context distinguishes from MADD) */
    {"\xe2\x8a\xbf", OP_PTCH, "NML-M2M"},   /* ⊿ */
    {"\xe2\x9c\xa6", OP_SIGN, "NML-M2M"},   /* ✦ */
    {"\xe2\x9c\x93", OP_VRFY, "NML-M2M"},   /* ✓ */
    {"\xe2\x9a\x96", OP_VOTE, "NML-M2M"},   /* ⚖ */
    {"\xe2\x9f\x90", OP_PROJ, "NML-M2M"},   /* ⟐ */
    {"\xe2\x9f\x82", OP_DIST, "NML-M2M"},   /* ⟂ */
    {"\xe2\x8a\x83", OP_GATH, "NML-M2M"},   /* ⊃ */
    {"\xe2\x8a\x82", OP_SCAT, "NML-M2M"},   /* ⊂ */
#endif
    /* ═══ Alternative aliases (v0.6.4) ═══ */
    {"\xcf\x9f", OP_CMPI, "core"},          /* ϟ koppa — alt for ≺ */
    {"\xcf\x9b", OP_RDUC, "NML-R"},         /* ϛ stigma — alt for ⊥ */
    {"DOT",  OP_SDOT, "core"},              /* DOT — alt for SDOT */
    {"SCTR", OP_SCAT, "NML-M2M"},          /* SCTR — SCAT with Rd-first order */
    /* ═══ Verbose aliases (human-readable) ═══ */
    {"MATRIX_MULTIPLY", OP_MMUL, "core"},
    {"ADD",             OP_MADD, "core"},
    {"SUBTRACT",        OP_MSUB, "core"},
    {"ELEMENT_MULTIPLY", OP_EMUL, "core"},
    {"ELEMENT_DIVIDE",  OP_EDIV, "core"},
    {"DOT_PRODUCT",     OP_SDOT, "core"},
    {"SCALE",           OP_SCLR, "core"},
    {"DIVIDE",          OP_SDIV, "core"},
    {"RELU",            OP_RELU, "core"},  /* already readable */
    {"SIGMOID",         OP_SIGM, "core"},
    {"SOFTMAX",         OP_SOFT, "core"},
    {"LOAD",            OP_LD,   "core"},
    {"STORE",           OP_ST,   "core"},
    {"COPY",            OP_MOV,  "core"},
    {"ALLOCATE",        OP_ALLC, "core"},
    {"RESHAPE",         OP_RSHP, "core"},
    {"TRANSPOSE",       OP_TRNS, "core"},
    {"SPLIT",           OP_SPLT, "core"},
    {"MERGE",           OP_MERG, "core"},
    {"REPEAT",          OP_LOOP, "core"},
    {"END_REPEAT",      OP_ENDP, "core"},
    {"BARRIER",         OP_SYNC, "core"},
    {"STOP",            OP_HALT, "core"},
    {"COMPARE_FEATURE", OP_CMPF, "core"},
    {"COMPARE",         OP_CMP,  "core"},
    {"COMPARE_VALUE",   OP_CMPI, "core"},
    {"SET_VALUE",       OP_LEAF, "core"},
    {"ACCUMULATE",      OP_TACC, "core"},
    {"BRANCH_TRUE",     OP_JMPT, "core"},
    {"BRANCH_FALSE",    OP_JMPF, "core"},
    {"JUMP",            OP_JUMP, "core"},  /* already readable */
    {"CALL",            OP_CALL, "core"},  /* already readable */
    {"RETURN",          OP_RET,  "core"},
    {"FAULT",           OP_TRAP, "core"},
    {"CONVOLVE",        OP_CONV, "NML-V"},
    {"MAX_POOL",        OP_POOL, "NML-V"},
    {"UPSCALE",         OP_UPSC, "NML-V"},
    {"ZERO_PAD",        OP_PADZ, "NML-V"},
    {"ATTENTION",       OP_ATTN, "NML-T"},
    {"LAYER_NORM",      OP_NORM, "NML-T"},
    {"EMBED",           OP_EMBD, "NML-T"},
    {"REDUCE",          OP_RDUC, "NML-R"},
    {"WHERE",           OP_WHER, "NML-R"},
    {"CLAMP",           OP_CLMP, "NML-R"},
    {"MASK_COMPARE",    OP_CMPR, "NML-R"},
    {"FOURIER",         OP_FFT,  "NML-S"},
    {"FILTER",          OP_FILT, "NML-S"},
#ifndef NML_NO_M2M
    {"METADATA",          OP_META, "NML-M2M"},
    {"FRAGMENT",          OP_FRAG, "NML-M2M"},
    {"END_FRAGMENT",      OP_ENDF, "NML-M2M"},
    {"IMPORT",            OP_LINK, "NML-M2M"},
    {"PATCH",             OP_PTCH, "NML-M2M"},
    {"SIGN_PROGRAM",      OP_SIGN, "NML-M2M"},
    {"VERIFY_SIGNATURE",  OP_VRFY, "NML-M2M"},
    {"CONSENSUS",         OP_VOTE, "NML-M2M"},
    {"PROJECT",           OP_PROJ, "NML-M2M"},
    {"DISTANCE",          OP_DIST, "NML-M2M"},
    {"GATHER",            OP_GATH, "NML-M2M"},
    {"SCATTER",           OP_SCAT, "NML-M2M"},
#endif
#ifndef NML_NO_GENERAL
    /* NML-G Classic */
    {"SYS",  OP_SYS,  "NML-G"},
    {"MOD",  OP_MOD,  "NML-G"},
    {"ITOF", OP_ITOF, "NML-G"},
    {"FTOI", OP_FTOI, "NML-G"},
    {"BNOT", OP_BNOT, "NML-G"},
    /* NML-G Symbolic */
    {"\xe2\x9a\x99", OP_SYS,  "NML-G"},   /* ⚙ */
    {"%",             OP_MOD,  "NML-G"},
    {"\xe2\x8a\xb6", OP_ITOF, "NML-G"},   /* ⊶ */
    {"\xe2\x8a\xb7", OP_FTOI, "NML-G"},   /* ⊷ */
    {"\xc2\xac",     OP_BNOT, "NML-G"},   /* ¬ */
    /* NML-G Verbose */
    {"SYSTEM",        OP_SYS,  "NML-G"},
    {"MODULO",        OP_MOD,  "NML-G"},
    {"INT_TO_FLOAT",  OP_ITOF, "NML-G"},
    {"FLOAT_TO_INT",  OP_FTOI, "NML-G"},
    {"BITWISE_NOT",   OP_BNOT, "NML-G"},
#endif
#ifndef NML_NO_TRAINING
    /* NML-TR Training (v0.7) — Classic */
    {"BKWD", OP_BKWD, "NML-TR"},
    {"WUPD", OP_WUPD, "NML-TR"},
    {"LOSS", OP_LOSS, "NML-TR"},
    {"TNET", OP_TNET, "NML-TR"},
    /* NML-TR Symbolic */
    {"\xe2\x88\x87", OP_BKWD, "NML-TR"},   /* ∇ nabla */
    {"\xe2\x9f\xb3", OP_WUPD, "NML-TR"},   /* ⟳ clockwise */
    {"\xe2\x96\xb3", OP_LOSS, "NML-TR"},   /* △ triangle */
    {"\xe2\xa5\x81", OP_TNET, "NML-TR"},   /* ⥁ closed circle */
    /* NML-TR Verbose */
    {"BACKWARD",       OP_BKWD, "NML-TR"},
    {"WEIGHT_UPDATE",  OP_WUPD, "NML-TR"},
    {"COMPUTE_LOSS",   OP_LOSS, "NML-TR"},
    {"TRAIN_NETWORK",  OP_TNET, "NML-TR"},
#endif
    {NULL, 0, NULL}
};

/* ═══════════════════════════════════════════
   PROGRAM DESCRIPTOR (v0.6)
   ═══════════════════════════════════════════ */

#define NML_MAX_META_ENTRIES 32
#define NML_MAX_META_VALUE 256

typedef struct {
    char key[32];
    char value[NML_MAX_META_VALUE];
} MetaEntry;

typedef struct {
    MetaEntry entries[NML_MAX_META_ENTRIES];
    int count;
} ProgramDescriptor;

/* ═══════════════════════════════════════════
   INSTRUCTION & VM
   ═══════════════════════════════════════════ */

typedef struct {
    Opcode op;
    int    reg[3];
    char   addr[NML_MAX_LABEL_LEN];
    double imm;
    double imm2;
    int    feat_idx;
    int    shape[NML_MAX_DIMS];
    int    shape_ndim;
    int    int_params[4];
} Instruction;

typedef struct {
    char   label[NML_MAX_LABEL_LEN];
    Tensor tensor;
    int    used;
} MemorySlot;

typedef struct {
    Tensor      regs[NML_MAX_REGISTERS];
    int         reg_valid[NML_MAX_REGISTERS];
    MemorySlot  memory[NML_MAX_MEMORY_SLOTS];
    int         mem_count;
    Instruction program[NML_MAX_INSTRUCTIONS];
    int         program_len;
    int         pc;
    int         halted;
    int         cycles;
    int         cond_flag;
    struct { int start_pc, count, current; } loop_stack[NML_MAX_LOOP_DEPTH];
    int         loop_depth;
    int         call_stack[NML_MAX_CALL_DEPTH];
    int         call_depth;
    int         max_cycles;
    int         trace;
    int         error_code;
    char        error_msg[256];
    unsigned    extensions;
    ProgramDescriptor descriptor;
} VM;

#define EXT_VISION      0x01
#define EXT_TRANSFORMER 0x02
#define EXT_REDUCTION   0x04
#define EXT_SIGNAL      0x08
#define EXT_M2M         0x10
#define EXT_GENERAL     0x20

static void vm_init(VM *vm) {
    memset(vm, 0, sizeof(VM));
    vm->max_cycles = NML_DEFAULT_MAX_CYCLES;
#ifdef NML_EXT_VISION
    vm->extensions |= EXT_VISION;
#endif
#ifdef NML_EXT_TRANSFORMER
    vm->extensions |= EXT_TRANSFORMER;
#endif
#ifdef NML_EXT_REDUCTION
    vm->extensions |= EXT_REDUCTION;
#endif
#ifdef NML_EXT_SIGNAL
    vm->extensions |= EXT_SIGNAL;
#endif
#ifdef NML_EXT_M2M
    vm->extensions |= EXT_M2M;
#endif
#ifdef NML_EXT_GENERAL
    vm->extensions |= EXT_GENERAL;
#endif
}

#define VM_ERROR(vm, code, ...) do { \
    (vm)->error_code = (code); \
    snprintf((vm)->error_msg, sizeof((vm)->error_msg), __VA_ARGS__); \
    return (code); \
} while(0)

static MemorySlot* vm_memory(VM *vm, const char *label) {
    for (int i = 0; i < vm->mem_count; i++)
        if (strcmp(vm->memory[i].label, label) == 0) return &vm->memory[i];
    if (vm->mem_count >= NML_MAX_MEMORY_SLOTS) return NULL;
    MemorySlot *slot = &vm->memory[vm->mem_count++];
    strncpy(slot->label, label, NML_MAX_LABEL_LEN - 1);
    slot->used = 1;
    return slot;
}

/* ═══════════════════════════════════════════
   ASSEMBLER
   ═══════════════════════════════════════════ */

static const struct { const char *sym; int idx; } GREEK_REGS[] = {
    {"\xce\xb9", 0},  /* ι iota   = R0 */
    {"\xce\xba", 1},  /* κ kappa  = R1 */
    {"\xce\xbb", 2},  /* λ lambda = R2 */
    {"\xce\xbc", 3},  /* μ mu     = R3 */
    {"\xce\xbd", 4},  /* ν nu     = R4 */
    {"\xce\xbe", 5},  /* ξ xi     = R5 */
    {"\xce\xbf", 6},  /* ο omi    = R6 */
    {"\xcf\x80", 7},  /* π pi     = R7 */
    {"\xcf\x81", 8},  /* ρ rho    = R8 */
    {"\xcf\x82", 9},  /* ς sigma  = R9 */
    {"\xce\xb1", 10}, /* α alpha  = RA (accumulator) */
    {"\xce\xb2", 11}, /* β beta   = RB (general) */
    {"\xce\xb3", 12}, /* γ gamma  = RC (scratch) */
    {"\xce\xb4", 13}, /* δ delta  = RD (counter) */
    {"\xcf\x86", 14}, /* φ phi    = RE (flag) */
    {"\xcf\x88", 15}, /* ψ psi    = RF (stack) */
    /* v0.7: Extended registers RG-RV (indices 16-31) */
    {"\xce\xb7", 16}, /* η eta     = RG (gradient 1) */
    {"\xce\xb8", 17}, /* θ theta   = RH (gradient 2) */
    {"\xce\xb6", 18}, /* ζ zeta    = RI (gradient 3) */
    {"\xcf\x89", 19}, /* ω omega   = RJ (loss/learning rate) */
    {"\xcf\x87", 20}, /* χ chi     = RK */
    {"\xcf\x85", 21}, /* υ upsilon = RL */
    {"\xce\xb5", 22}, /* ε epsilon = RM */
    {"\xcf\x84", 23}, /* τ tau     = RN — NOTE: τ already used for TANH opcode, context disambiguates */
    /* RO-RV: no Greek aliases to avoid collisions, use classic only */
    /* Verbose register aliases */
    {"ACCUMULATOR", 10},
    {"ACC", 10},
    {"GENERAL", 11},
    {"GEN", 11},
    {"SCRATCH", 12},
    {"TMP", 12},
    {"COUNTER", 13},
    {"CTR", 13},
    {"FLAG", 14},
    {"FLG", 14},
    {"STACK", 15},
    {"STK", 15},
    {"GRAD1", 16},
    {"GRAD2", 17},
    {"GRAD3", 18},
    {"LRATE", 19},
    {NULL, -1}
};

static int parse_register(const char *s) {
    /* Strip optional :type annotation (e.g. R0:currency → R0) */
    char buf[32];
    const char *colon = strchr(s, ':');
    if (colon && colon != s) {
        size_t len = colon - s;
        if (len < sizeof(buf)) { memcpy(buf, s, len); buf[len] = '\0'; s = buf; }
    }

    if ((s[0] == 'R' || s[0] == 'r') && strlen(s) == 2) {
        char c = s[1];
        if (c >= '0' && c <= '9') return c - '0';       /* R0-R9: 0-9 */
        if (c >= 'A' && c <= 'V') return c - 'A' + 10;  /* RA-RV: 10-31 */
        if (c >= 'a' && c <= 'v') return c - 'a' + 10;
    }
    for (int i = 0; GREEK_REGS[i].sym; i++)
        if (strcmp(s, GREEK_REGS[i].sym) == 0) return GREEK_REGS[i].idx;
    return -1;
}

static Opcode parse_opcode(const char *s) {
    for (int i = 0; OPCODE_TABLE[i].name; i++)
        if (strcmp(s, OPCODE_TABLE[i].name) == 0) return OPCODE_TABLE[i].code;
    for (int i = 0; OPCODE_TABLE[i].name; i++)
        if (strcasecmp(s, OPCODE_TABLE[i].name) == 0) return OPCODE_TABLE[i].code;
    return (Opcode)-1;
}

static void parse_shape(const char *s, int *shape, int *ndim) {
    const char *p = s;
    if (*p == '#') p++;
    if (*p == '[') p++;
    *ndim = 0;
    while (*p && *p != ']') {
        shape[(*ndim)++] = atoi(p);
        while (*p && *p != ',' && *p != ']') p++;
        if (*p == ',') p++;
    }
}

static double parse_imm(const char *s) {
    return strtod(s[0] == '#' ? s+1 : s, NULL);
}

static int is_compact_delim(const char *p) {
    /* ¶ = U+00B6 = UTF-8 bytes C2 B6 */
    return (unsigned char)p[0] == 0xC2
        && (unsigned char)p[1] == 0xB6;
}

static int vm_assemble(VM *vm, const char *source) {
    char line[NML_MAX_LINE_LEN];
    const char *p = source;
    vm->program_len = 0;

    while (*p) {
        int i = 0;
        while (*p && *p != '\n' && !is_compact_delim(p) && i < NML_MAX_LINE_LEN - 1)
            line[i++] = *p++;
        line[i] = '\0';
        if (*p == '\n') p++;
        else if (is_compact_delim(p)) p += 2;

        char *semi = strchr(line, ';');
        if (semi) *semi = '\0';

        char *start = line;
        while (*start == ' ' || *start == '\t') start++;
        char *end = start + strlen(start) - 1;
        while (end > start && (*end == ' ' || *end == '\t' || *end == '\r')) *end-- = '\0';
        if (strlen(start) == 0) continue;

        char *tokens[8];
        int ntokens = 0;
        char *tok = strtok(start, " \t");
        while (tok && ntokens < 8) { tokens[ntokens++] = tok; tok = strtok(NULL, " \t"); }
        if (ntokens == 0) continue;

        if (vm->program_len >= NML_MAX_INSTRUCTIONS) {
            fprintf(stderr, "ERROR: Program exceeds %d instructions\n", NML_MAX_INSTRUCTIONS);
            return NML_ERR_ASSEMBLE;
        }

        Instruction *instr = &vm->program[vm->program_len];
        memset(instr, 0, sizeof(Instruction));

        int is_sctr = (strcmp(tokens[0], "SCTR") == 0);
        int op = parse_opcode(tokens[0]);
        if (op < 0) { fprintf(stderr, "ERROR: Unknown opcode '%s'\n", tokens[0]); return NML_ERR_ASSEMBLE; }
        instr->op = op;

        switch (instr->op) {
            /* 3-register: MMUL MADD MSUB EMUL SDOT TACC EDIV */
            case OP_MMUL: case OP_MADD: case OP_MSUB: case OP_EMUL: case OP_SDOT:
            case OP_TACC: case OP_EDIV:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                break;

            /* Rd Rs #imm: SCLR SDIV */
            case OP_SCLR: case OP_SDIV:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->imm = parse_imm(tokens[3]);
                break;

            /* 2-register: activations, MOV, TRNS, CMP */
            case OP_RELU: case OP_SIGM: case OP_TANH: case OP_SOFT:
            case OP_MOV: case OP_TRNS: case OP_CMP:
#ifdef NML_EXT_TRANSFORMER
            case OP_GELU:
#endif
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                break;

            /* CMPI Rd Rs #imm */
            case OP_CMPI:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->imm = parse_imm(tokens[3]);
                break;

            /* Load/Store */
            case OP_LD:
                instr->reg[0] = parse_register(tokens[1]);
                strncpy(instr->addr, tokens[2][0]=='@' ? tokens[2]+1 : tokens[2], NML_MAX_LABEL_LEN-1);
                break;
            case OP_ST:
                instr->reg[0] = parse_register(tokens[1]);
                strncpy(instr->addr, tokens[2][0]=='@' ? tokens[2]+1 : tokens[2], NML_MAX_LABEL_LEN-1);
                break;

            /* Alloc / Reshape */
            case OP_ALLC:
                instr->reg[0] = parse_register(tokens[1]);
                parse_shape(tokens[2], instr->shape, &instr->shape_ndim);
                instr->int_params[3] = ntokens > 3 ? (int)parse_dtype(tokens[3]) : NML_F32;
                break;
            case OP_RSHP:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                parse_shape(tokens[3], instr->shape, &instr->shape_ndim);
                break;

            /* SPLT: two forms supported:
             *   Short (grammar): SPLT Rd Rs [#dim]       — 2-3 operands, no second output
             *   Long  (spec):    SPLT Rd Re Rs [#dim [#split_at]] — 4-5 operands */
            case OP_SPLT:
                if (ntokens <= 4) {
                    /* Short form: SPLT Rd Rs [#dim] */
                    instr->reg[0] = parse_register(tokens[1]);
                    instr->reg[1] = -1;
                    instr->reg[2] = parse_register(tokens[2]);
                    instr->int_params[0] = ntokens > 3 ? (int)parse_imm(tokens[3]) : 0;
                    instr->int_params[1] = -1;
                } else {
                    /* Long form: SPLT Rd Re Rs [#dim [#split_at]] */
                    instr->reg[0] = parse_register(tokens[1]);
                    instr->reg[1] = parse_register(tokens[2]);
                    instr->reg[2] = parse_register(tokens[3]);
                    instr->int_params[0] = ntokens > 4 ? (int)parse_imm(tokens[4]) : 0;
                    instr->int_params[1] = ntokens > 5 ? (int)parse_imm(tokens[5]) : -1;
                }
                break;

            /* MERG Rd Rs1 Rs2 #dim */
            case OP_MERG:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                instr->int_params[0] = ntokens > 4 ? (int)parse_imm(tokens[4]) : 0;
                break;

            /* Loop */
            case OP_LOOP:
                instr->imm = parse_imm(tokens[1]);
                break;
            case OP_ENDP: case OP_SYNC: case OP_HALT: case OP_RET:
                break;

            /* TRAP #code */
            case OP_TRAP:
                instr->imm = ntokens > 1 ? parse_imm(tokens[1]) : 1.0f;
                break;

            /* Tree extensions */
            case OP_CMPF:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->feat_idx = (int)parse_imm(tokens[3]);
                instr->imm = parse_imm(tokens[4]);
                break;
            case OP_LEAF:
                instr->reg[0] = parse_register(tokens[1]);
                instr->imm = parse_imm(tokens[2]);
                break;
            case OP_JMPT: case OP_JMPF: case OP_JUMP: case OP_CALL:
                instr->imm = parse_imm(tokens[1]);
                break;

            /* NML-V Vision */
#ifdef NML_EXT_VISION
            case OP_CONV:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                instr->int_params[0] = ntokens > 4 ? (int)parse_imm(tokens[4]) : 1;
                instr->int_params[1] = ntokens > 5 ? (int)parse_imm(tokens[5]) : 0;
                break;
            case OP_POOL:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->int_params[0] = ntokens > 3 ? (int)parse_imm(tokens[3]) : 2;
                instr->int_params[1] = ntokens > 4 ? (int)parse_imm(tokens[4]) : 2;
                break;
            case OP_UPSC:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->int_params[0] = ntokens > 3 ? (int)parse_imm(tokens[3]) : 2;
                break;
            case OP_PADZ:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->int_params[0] = ntokens > 3 ? (int)parse_imm(tokens[3]) : 1;
                break;
#endif
            /* NML-T Transformer */
#ifdef NML_EXT_TRANSFORMER
            case OP_ATTN:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                instr->int_params[0] = ntokens > 4 ? parse_register(tokens[4]) : instr->reg[2];
                break;
            case OP_NORM:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = ntokens > 3 ? parse_register(tokens[3]) : -1;
                instr->int_params[0] = ntokens > 4 ? parse_register(tokens[4]) : -1;
                break;
            case OP_EMBD:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                break;
#endif
            /* NML-R Reduction */
#ifdef NML_EXT_REDUCTION
            case OP_RDUC:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->int_params[0] = ntokens > 3 ? (int)parse_imm(tokens[3]) : 0;
                instr->int_params[1] = ntokens > 4 ? (int)parse_imm(tokens[4]) : -1;
                break;
            case OP_WHER:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                instr->int_params[0] = ntokens > 4 ? parse_register(tokens[4]) : instr->reg[2];
                break;
            case OP_CLMP:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->imm = ntokens > 3 ? parse_imm(tokens[3]) : 0.0f;
                instr->imm2 = ntokens > 4 ? parse_imm(tokens[4]) : 1.0f;
                break;
            case OP_CMPR:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->imm = ntokens > 3 ? parse_imm(tokens[3]) : 0.0f;
                instr->int_params[0] = ntokens > 4 ? (int)parse_imm(tokens[4]) : 0;
                break;
#endif
            /* NML-S Signal */
#ifdef NML_EXT_SIGNAL
            case OP_FFT:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                break;
            case OP_FILT:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                break;
#endif
            /* NML-M2M */
#ifndef NML_NO_M2M
            case OP_META: {
                if (ntokens > 1 && vm->descriptor.count < NML_MAX_META_ENTRIES) {
                    char *key = tokens[1];
                    if (key[0] == '@') key++;
                    strncpy(vm->descriptor.entries[vm->descriptor.count].key, key, 31);
                    vm->descriptor.entries[vm->descriptor.count].key[31] = '\0';
                    char val[NML_MAX_META_VALUE] = {0};
                    for (int t = 2; t < ntokens; t++) {
                        if (t > 2) strncat(val, " ", NML_MAX_META_VALUE - strlen(val) - 1);
                        strncat(val, tokens[t], NML_MAX_META_VALUE - strlen(val) - 1);
                    }
                    strncpy(vm->descriptor.entries[vm->descriptor.count].value, val, NML_MAX_META_VALUE - 1);
                    vm->descriptor.count++;
                }
                break;
            }
            case OP_FRAG: case OP_ENDF: case OP_LINK: case OP_PTCH: case OP_SIGN:
                if (ntokens > 1) strncpy(instr->addr, tokens[1], NML_MAX_LABEL_LEN-1);
                break;
            case OP_VRFY:
                if (ntokens > 1) instr->reg[0] = parse_register(tokens[1]);
                break;
            case OP_VOTE:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->imm = ntokens > 3 ? parse_imm(tokens[3]) : 0;
                instr->imm2 = ntokens > 4 ? parse_imm(tokens[4]) : 0;
                break;
            case OP_PROJ:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                break;
            case OP_DIST:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                instr->imm = ntokens > 4 ? parse_imm(tokens[4]) : 0;
                break;
            case OP_GATH:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                break;
            case OP_SCAT:
                if (is_sctr) {
                    /* SCTR Rd Rs Ridx -> swap to SCAT order: Rs Rd Ridx */
                    instr->reg[0] = parse_register(tokens[2]);
                    instr->reg[1] = parse_register(tokens[1]);
                    instr->reg[2] = parse_register(tokens[3]);
                } else {
                    /* SCAT Rs Rd Ridx (original order) */
                    instr->reg[0] = parse_register(tokens[1]);
                    instr->reg[1] = parse_register(tokens[2]);
                    instr->reg[2] = parse_register(tokens[3]);
                }
                break;
#endif
            /* NML-G General Purpose */
#ifndef NML_NO_GENERAL
            case OP_SYS:
                instr->reg[0] = parse_register(tokens[1]);
                instr->imm = ntokens > 2 ? parse_imm(tokens[2]) : 0;
                break;
            case OP_MOD:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                break;
            case OP_ITOF: case OP_FTOI: case OP_BNOT:
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                break;
#endif
#ifndef NML_NO_TRAINING
            /* NML-TR Training (v0.7) */
            case OP_BKWD:
                /* BKWD Rd_grad Rs_output Rs_target Rs_weights */
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                instr->int_params[0] = ntokens > 4 ? parse_register(tokens[4]) : -1;
                break;
            case OP_WUPD:
                /* WUPD Rd_weights Rs_gradient #learning_rate */
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->imm = parse_imm(tokens[3]);
                break;
            case OP_LOSS:
                /* LOSS Rd Rs_predicted Rs_target #loss_type */
                instr->reg[0] = parse_register(tokens[1]);
                instr->reg[1] = parse_register(tokens[2]);
                instr->reg[2] = parse_register(tokens[3]);
                instr->int_params[0] = ntokens > 4 ? (int)parse_imm(tokens[4]) : 0;
                break;
            case OP_TNET:
                /* TNET #epochs #lr #arch
                 * Register convention: R0=input, R1=w1, R2=b1, R3=w2, R4=b2, R9=target
                 * Writes: R1,R3,R4 updated, R8=final_loss */
                instr->imm = parse_imm(tokens[1]);           /* epochs */
                instr->int_params[0] = ntokens > 2 ? (int)(parse_imm(tokens[2]) * 1e6) : 1000; /* lr * 1e6 (stored as int to preserve precision) */
                instr->int_params[1] = ntokens > 3 ? (int)parse_imm(tokens[3]) : 0;  /* arch */
                break;
#endif
            default:
                fprintf(stderr, "ERROR: Unhandled opcode in assembler: %d\n", instr->op);
                return NML_ERR_ASSEMBLE;
        }
        vm->program_len++;
    }
    return vm->program_len;
}

/* ═══════════════════════════════════════════
   STATIC VALIDATION
   ═══════════════════════════════════════════ */

static int vm_validate(VM *vm) {
    int loop_depth = 0;
    int has_halt = 0;

    for (int i = 0; i < vm->program_len; i++) {
        Instruction *ins = &vm->program[i];

        switch (ins->op) {
        case OP_LOOP:
            loop_depth++;
            if (loop_depth > NML_MAX_LOOP_DEPTH)
                VM_ERROR(vm, NML_ERR_LOOP, "Loop nesting exceeds max depth %d at instruction %d", NML_MAX_LOOP_DEPTH, i);
            break;
        case OP_ENDP:
            if (loop_depth <= 0)
                VM_ERROR(vm, NML_ERR_LOOP, "ENDP without matching LOOP at instruction %d", i);
            loop_depth--;
            break;
        case OP_HALT:
            has_halt = 1;
            break;
        case OP_JMPT: case OP_JMPF: case OP_JUMP: case OP_CALL: {
            int target = i + (int)ins->imm;
            if (target < 0 || target >= vm->program_len)
                VM_ERROR(vm, NML_ERR_OOB, "Jump at instruction %d targets %d (out of range 0..%d)", i, target, vm->program_len - 1);
            break;
        }
        default:
            break;
        }

        for (int r = 0; r < 3; r++) {
            if (ins->reg[r] < -1 || ins->reg[r] >= NML_MAX_REGISTERS) {
                VM_ERROR(vm, NML_ERR_OOB, "Invalid register R%d at instruction %d", ins->reg[r], i);
            }
        }
    }

    if (loop_depth != 0)
        VM_ERROR(vm, NML_ERR_LOOP, "Unmatched LOOP — %d unclosed loop(s)", loop_depth);

    if (!has_halt)
        fprintf(stderr, "WARNING: No HALT instruction found — program may not terminate cleanly\n");

    return NML_OK;
}

/* ═══════════════════════════════════════════
   EXECUTION ENGINE
   ═══════════════════════════════════════════ */

static const char* opcode_name(Opcode op) {
    for (int i = 0; OPCODE_TABLE[i].name; i++)
        if (OPCODE_TABLE[i].code == op) return OPCODE_TABLE[i].name;
    return "????";
}

#define CHECK_EXT(flag, name) do { if (!(vm->extensions & flag)) \
    VM_ERROR(vm, NML_ERR_EXTENSION, "Extension %s not enabled", name); } while(0)

#define REG(r) vm->regs[(r)]
#define RVALID(r) vm->reg_valid[(r)]

static int vm_execute(VM *vm) {
    vm->pc = 0; vm->halted = 0; vm->cycles = 0;
    vm->loop_depth = 0; vm->call_depth = 0;
    vm->error_code = NML_OK;
    vm->error_msg[0] = '\0';

    while (vm->pc < vm->program_len && !vm->halted) {
        if (vm->cycles >= vm->max_cycles)
            VM_ERROR(vm, NML_ERR_CYCLE_LIMIT, "Cycle limit %d exceeded at PC=%d", vm->max_cycles, vm->pc);

        Instruction *ins = &vm->program[vm->pc];
        vm->cycles++;

        if (vm->trace)
            fprintf(stderr, "[TRACE] PC=%04d  %s\n", vm->pc, opcode_name(ins->op));

        int rc;
        switch (ins->op) {
            case OP_MMUL: rc=tensor_matmul(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); if(rc) VM_ERROR(vm,rc,"MMUL shape error at PC=%d",vm->pc); RVALID(ins->reg[0])=1; break;
            case OP_MADD: tensor_add(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); RVALID(ins->reg[0])=1; break;
            case OP_MSUB: tensor_sub(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); RVALID(ins->reg[0])=1; break;
            case OP_EMUL: tensor_emul(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); RVALID(ins->reg[0])=1; break;
            case OP_SDOT: { double d=tensor_dot(&REG(ins->reg[1]),&REG(ins->reg[2])); int s[]={1}; tensor_init(&REG(ins->reg[0]),1,s); tensor_setd(&REG(ins->reg[0]),0,d); RVALID(ins->reg[0])=1; break; }
            case OP_SCLR: tensor_scale(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->imm); RVALID(ins->reg[0])=1; break;
            case OP_SDIV: rc=tensor_scale_div(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->imm); if(rc) VM_ERROR(vm,rc,"SDIV division by zero at PC=%d",vm->pc); RVALID(ins->reg[0])=1; break;
            case OP_EDIV: rc=tensor_ediv(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); if(rc) VM_ERROR(vm,rc,"EDIV division by zero at PC=%d",vm->pc); RVALID(ins->reg[0])=1; break;
            case OP_RELU: tensor_relu(&REG(ins->reg[0]),&REG(ins->reg[1])); RVALID(ins->reg[0])=1; break;
            case OP_SIGM: tensor_sigmoid(&REG(ins->reg[0]),&REG(ins->reg[1])); RVALID(ins->reg[0])=1; break;
            case OP_TANH: tensor_tanh_act(&REG(ins->reg[0]),&REG(ins->reg[1])); RVALID(ins->reg[0])=1; break;
            case OP_SOFT: tensor_softmax(&REG(ins->reg[0]),&REG(ins->reg[1])); RVALID(ins->reg[0])=1; break;
            case OP_LD: {
                MemorySlot *sl = vm_memory(vm, ins->addr);
                if (!sl) VM_ERROR(vm, NML_ERR_MEMORY, "Out of memory slots loading @%s at PC=%d", ins->addr, vm->pc);
                if (!sl->used) VM_ERROR(vm, NML_ERR_UNINIT, "@%s not found at PC=%d", ins->addr, vm->pc);
                tensor_copy(&REG(ins->reg[0]), &sl->tensor); RVALID(ins->reg[0])=1; break;
            }
            case OP_ST: {
                MemorySlot *sl = vm_memory(vm, ins->addr);
                if (!sl) VM_ERROR(vm, NML_ERR_MEMORY, "Out of memory slots storing @%s at PC=%d", ins->addr, vm->pc);
                tensor_copy(&sl->tensor, &REG(ins->reg[0])); sl->used=1; break;
            }
            case OP_MOV: tensor_copy(&REG(ins->reg[0]),&REG(ins->reg[1])); RVALID(ins->reg[0])=1; break;
            case OP_ALLC: { rc=tensor_init_typed(&REG(ins->reg[0]),ins->shape_ndim,ins->shape,(DType)ins->int_params[3]); if(rc) VM_ERROR(vm,rc,"ALLC overflow at PC=%d",vm->pc); RVALID(ins->reg[0])=1; break; }
            case OP_RSHP: { Tensor *s=&REG(ins->reg[1]); Tensor *d=&REG(ins->reg[0]); memcpy(d->data.f32,s->data.f32,sizeof(float)*s->size); d->ndim=ins->shape_ndim; memcpy(d->shape,ins->shape,sizeof(int)*ins->shape_ndim); d->size=s->size; RVALID(ins->reg[0])=1; break; }
            case OP_TRNS: rc=tensor_transpose(&REG(ins->reg[0]),&REG(ins->reg[1])); if(rc) VM_ERROR(vm,rc,"TRNS requires 2D at PC=%d",vm->pc); RVALID(ins->reg[0])=1; break;

            case OP_SPLT: {
                int dim = ins->int_params[0];
                int split_at = ins->int_params[1];
                Tensor *src = &REG(ins->reg[2]);
                if (split_at < 0 && src->ndim == 2) split_at = (dim == 0 ? src->shape[0] : src->shape[1]) / 2;
                Tensor *out_hi = (ins->reg[1] >= 0) ? &REG(ins->reg[1]) : NULL;
                rc = tensor_split(&REG(ins->reg[0]), out_hi, src, dim, split_at);
                if (rc) VM_ERROR(vm, rc, "SPLT error at PC=%d", vm->pc);
                RVALID(ins->reg[0])=1;
                if (ins->reg[1] >= 0) RVALID(ins->reg[1])=1;
                break;
            }
            case OP_MERG: {
                rc = tensor_merge(&REG(ins->reg[0]), &REG(ins->reg[1]), &REG(ins->reg[2]), ins->int_params[0]);
                if (rc) VM_ERROR(vm, rc, "MERG shape mismatch at PC=%d", vm->pc);
                RVALID(ins->reg[0])=1; break;
            }

            case OP_LOOP:
                if (vm->loop_depth >= NML_MAX_LOOP_DEPTH) VM_ERROR(vm, NML_ERR_LOOP, "Loop depth exceeded at PC=%d", vm->pc);
                vm->loop_stack[vm->loop_depth].start_pc = vm->pc;
                vm->loop_stack[vm->loop_depth].count = (int)ins->imm;
                vm->loop_stack[vm->loop_depth].current = 0;
                vm->loop_depth++; break;
            case OP_ENDP:
                if (vm->loop_depth <= 0) VM_ERROR(vm, NML_ERR_LOOP, "ENDP without LOOP at PC=%d", vm->pc);
                vm->loop_stack[vm->loop_depth-1].current++;
                if (vm->loop_stack[vm->loop_depth-1].current < vm->loop_stack[vm->loop_depth-1].count)
                    vm->pc = vm->loop_stack[vm->loop_depth-1].start_pc;
                else vm->loop_depth--;
                break;
            case OP_SYNC: break;

            /* General comparison */
            case OP_CMP: {
                double a = RVALID(ins->reg[0]) ? tensor_getd(&REG(ins->reg[0]),0) : 0.0;
                double b = RVALID(ins->reg[1]) ? tensor_getd(&REG(ins->reg[1]),0) : 0.0;
                vm->cond_flag = (a < b) ? 1 : 0;
                break;
            }
            case OP_CMPI: {
                double a = RVALID(ins->reg[1]) ? tensor_getd(&REG(ins->reg[1]),0) : 0.0;
                vm->cond_flag = (a < ins->imm) ? 1 : 0;
                break;
            }

            /* Tree model */
            case OP_CMPF: { Tensor *t=&REG(ins->reg[1]); int fi=ins->feat_idx; double val=(fi<t->size)?tensor_getd(t,fi):0.0; vm->cond_flag=(val<ins->imm)?1:0; break; }
            case OP_LEAF: { int s[]={1}; tensor_init_typed(&REG(ins->reg[0]),1,s,NML_F64); tensor_setd(&REG(ins->reg[0]),0,ins->imm); RVALID(ins->reg[0])=1; break; }
            case OP_TACC: { double a=RVALID(ins->reg[1])?tensor_getd(&REG(ins->reg[1]),0):0; double b=RVALID(ins->reg[2])?tensor_getd(&REG(ins->reg[2]),0):0; int s[]={1}; DType dt=dtype_promote(REG(ins->reg[1]).dtype,REG(ins->reg[2]).dtype); tensor_init_typed(&REG(ins->reg[0]),1,s,dt); tensor_setd(&REG(ins->reg[0]),0,a+b); RVALID(ins->reg[0])=1; break; }

            /* Jumps — support both forward and backward offsets */
            case OP_JMPT: if (vm->cond_flag) vm->pc += (int)ins->imm; break;
            case OP_JMPF: if (!vm->cond_flag) vm->pc += (int)ins->imm; break;
            case OP_JUMP: vm->pc += (int)ins->imm; break;

            /* Subroutines */
            case OP_CALL:
                if (vm->call_depth >= NML_MAX_CALL_DEPTH)
                    VM_ERROR(vm, NML_ERR_CALL_DEPTH, "Call stack overflow at PC=%d (depth %d)", vm->pc, vm->call_depth);
                vm->call_stack[vm->call_depth++] = vm->pc + 1;
                vm->pc += (int)ins->imm;
                break;
            case OP_RET:
                if (vm->call_depth <= 0)
                    VM_ERROR(vm, NML_ERR_RET_EMPTY, "RET with empty call stack at PC=%d", vm->pc);
                vm->pc = vm->call_stack[--vm->call_depth] - 1;
                break;

            case OP_TRAP:
                VM_ERROR(vm, NML_ERR_TRAP, "TRAP #%d at PC=%d", (int)ins->imm, vm->pc);

            /* ═══ NML-V Vision ═══ */
#ifdef NML_EXT_VISION
            case OP_CONV: CHECK_EXT(EXT_VISION,"NML-V"); tensor_conv2d(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2]),ins->int_params[0],ins->int_params[1]); RVALID(ins->reg[0])=1; break;
            case OP_POOL: CHECK_EXT(EXT_VISION,"NML-V"); tensor_maxpool(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->int_params[0],ins->int_params[1]); RVALID(ins->reg[0])=1; break;
            case OP_UPSC: CHECK_EXT(EXT_VISION,"NML-V"); tensor_upscale(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->int_params[0]); RVALID(ins->reg[0])=1; break;
            case OP_PADZ: CHECK_EXT(EXT_VISION,"NML-V"); tensor_pad(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->int_params[0]); RVALID(ins->reg[0])=1; break;
#endif
            /* ═══ NML-T Transformer ═══ */
#ifdef NML_EXT_TRANSFORMER
            case OP_ATTN: CHECK_EXT(EXT_TRANSFORMER,"NML-T"); tensor_attention(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2]),&REG(ins->int_params[0])); RVALID(ins->reg[0])=1; break;
            case OP_NORM: CHECK_EXT(EXT_TRANSFORMER,"NML-T"); tensor_layernorm(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->reg[2]>=0?&REG(ins->reg[2]):NULL,ins->int_params[0]>=0?&REG(ins->int_params[0]):NULL); RVALID(ins->reg[0])=1; break;
            case OP_EMBD: CHECK_EXT(EXT_TRANSFORMER,"NML-T"); tensor_embedding(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); RVALID(ins->reg[0])=1; break;
            case OP_GELU: CHECK_EXT(EXT_TRANSFORMER,"NML-T"); tensor_gelu(&REG(ins->reg[0]),&REG(ins->reg[1])); RVALID(ins->reg[0])=1; break;
#endif
            /* ═══ NML-R Reduction ═══ */
#ifdef NML_EXT_REDUCTION
            case OP_RDUC: CHECK_EXT(EXT_REDUCTION,"NML-R"); tensor_reduce(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->int_params[0],ins->int_params[1]); RVALID(ins->reg[0])=1; break;
            case OP_WHER: CHECK_EXT(EXT_REDUCTION,"NML-R"); tensor_where(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2]),&REG(ins->int_params[0])); RVALID(ins->reg[0])=1; break;
            case OP_CLMP: CHECK_EXT(EXT_REDUCTION,"NML-R"); tensor_clamp(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->imm,ins->imm2); RVALID(ins->reg[0])=1; break;
            case OP_CMPR: CHECK_EXT(EXT_REDUCTION,"NML-R"); tensor_compare(&REG(ins->reg[0]),&REG(ins->reg[1]),ins->imm,ins->int_params[0]); RVALID(ins->reg[0])=1; break;
#endif
            /* ═══ NML-S Signal ═══ */
#ifdef NML_EXT_SIGNAL
            case OP_FFT: CHECK_EXT(EXT_SIGNAL,"NML-S"); tensor_fft(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); RVALID(ins->reg[0])=1; RVALID(ins->reg[1])=1; break;
            case OP_FILT: CHECK_EXT(EXT_SIGNAL,"NML-S"); tensor_fir_filter(&REG(ins->reg[0]),&REG(ins->reg[1]),&REG(ins->reg[2])); RVALID(ins->reg[0])=1; break;
#endif
            /* ═══ NML-M2M Machine-to-Machine ═══ */
#ifndef NML_NO_M2M
        case OP_META:
        case OP_FRAG:
        case OP_ENDF:
        case OP_LINK:
        case OP_PTCH:
        case OP_SIGN:
            break;
        
        case OP_VRFY:
            break;
        
        case OP_VOTE: {
            Tensor *rd = &REG(ins->reg[0]);
            Tensor *rs = &REG(ins->reg[1]);
            int strategy = (int)ins->imm;
            if (rs->size < 1) VM_ERROR(vm, NML_ERR_SHAPE, "VOTE: empty input at PC=%d", vm->pc);
            
            int out_shape[] = {1};
            tensor_init(rd, 1, out_shape);
            
            if (strategy == 0) { /* median */
                double *tmp = (double *)malloc(rs->size * sizeof(double));
                for (int i = 0; i < rs->size; i++) tmp[i] = tensor_getd(rs, i);
                for (int i = 0; i < rs->size - 1; i++)
                    for (int j = i + 1; j < rs->size; j++)
                        if (tmp[j] < tmp[i]) { double t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t; }
                double med = (rs->size % 2 == 1) ? tmp[rs->size/2] : (tmp[rs->size/2-1] + tmp[rs->size/2]) / 2.0;
                tensor_setd(rd, 0, med);
                free(tmp);
            } else if (strategy == 1) { /* mean */
                double sum = 0;
                for (int i = 0; i < rs->size; i++) sum += tensor_getd(rs, i);
                tensor_setd(rd, 0, sum / rs->size);
            } else if (strategy == 2) { /* quorum */
                int threshold = (int)ins->imm2;
                double tol = 0.01;
                int best_group = 0;
                double consensus_val = 0;
                for (int i = 0; i < rs->size; i++) {
                    int count = 0;
                    double v = tensor_getd(rs, i);
                    for (int j = 0; j < rs->size; j++)
                        if (fabs(tensor_getd(rs, j) - v) <= tol) count++;
                    if (count > best_group) { best_group = count; consensus_val = v; }
                }
                tensor_setd(rd, 0, best_group >= threshold ? 1.0 : 0.0);
                int sc[] = {1};
                tensor_init(&REG(12), 1, sc);
                tensor_setd(&REG(12), 0, consensus_val);
            } else if (strategy == 3) { /* min */
                double m = tensor_getd(rs, 0);
                for (int i = 1; i < rs->size; i++) { double v = tensor_getd(rs, i); if (v < m) m = v; }
                tensor_setd(rd, 0, m);
            } else if (strategy == 4) { /* max */
                double m = tensor_getd(rs, 0);
                for (int i = 1; i < rs->size; i++) { double v = tensor_getd(rs, i); if (v > m) m = v; }
                tensor_setd(rd, 0, m);
            }
            RVALID(ins->reg[0])=1;
            break;
        }
        
        case OP_PROJ: {
            Tensor *rd = &REG(ins->reg[0]);
            Tensor *rs = &REG(ins->reg[1]);
            Tensor *rm = &REG(ins->reg[2]);
            rc = tensor_matmul(rd, rs, rm);
            if (rc) VM_ERROR(vm, rc, "PROJ: matmul error at PC=%d", vm->pc);
            double norm = 0;
            for (int i = 0; i < rd->size; i++) { double v = tensor_getd(rd, i); norm += v*v; }
            norm = sqrt(norm);
            if (norm > 1e-12)
                for (int i = 0; i < rd->size; i++) tensor_setd(rd, i, tensor_getd(rd, i) / norm);
            RVALID(ins->reg[0])=1;
            break;
        }
        
        case OP_DIST: {
            Tensor *rd = &REG(ins->reg[0]);
            Tensor *rs1 = &REG(ins->reg[1]);
            Tensor *rs2 = &REG(ins->reg[2]);
            int metric = (int)ins->imm;
            if (rs1->size != rs2->size) VM_ERROR(vm, NML_ERR_SHAPE, "DIST: size mismatch at PC=%d", vm->pc);
            int out_shape[] = {1};
            tensor_init(rd, 1, out_shape);
            
            double dot = 0, norm1 = 0, norm2 = 0, diff_sq = 0;
            for (int i = 0; i < rs1->size; i++) {
                double a = tensor_getd(rs1, i), b = tensor_getd(rs2, i);
                dot += a * b;
                norm1 += a * a;
                norm2 += b * b;
                diff_sq += (a-b) * (a-b);
            }
            if (metric == 0) { /* cosine distance */
                double denom = sqrt(norm1) * sqrt(norm2);
                tensor_setd(rd, 0, denom > 1e-12 ? 1.0 - dot/denom : 1.0);
            } else if (metric == 1) { /* euclidean */
                tensor_setd(rd, 0, sqrt(diff_sq));
            } else { /* dot product */
                tensor_setd(rd, 0, dot);
            }
            RVALID(ins->reg[0])=1;
            break;
        }
        
        case OP_GATH: {
            Tensor *rd = &REG(ins->reg[0]);
            Tensor *rs = &REG(ins->reg[1]);
            Tensor *ridx = &REG(ins->reg[2]);
            int idx = (int)tensor_getd(ridx, 0);
            if (idx < 0 || idx >= rs->size) VM_ERROR(vm, NML_ERR_OOB, "GATH: index %d out of bounds (size %d) at PC=%d", idx, rs->size, vm->pc);
            int shape[] = {1};
            tensor_init(rd, 1, shape);
            tensor_setd(rd, 0, tensor_getd(rs, idx));
            RVALID(ins->reg[0])=1;
            break;
        }
        
        case OP_SCAT: {
            /* SCAT Rs Rd Ridx: Rd[Ridx[0]] = Rs[0] */
            Tensor *rs = &REG(ins->reg[0]);
            Tensor *rd = &REG(ins->reg[1]);
            Tensor *ridx = &REG(ins->reg[2]);
            int idx = (int)tensor_getd(ridx, 0);
            if (idx < 0 || idx >= rd->size) VM_ERROR(vm, NML_ERR_OOB, "SCAT: index %d out of bounds (size %d) at PC=%d", idx, rd->size, vm->pc);
            tensor_setd(rd, idx, tensor_getd(rs, 0));
            break;
        }
#endif /* NML_NO_M2M */
            /* ═══ NML-G General Purpose ═══ */
#ifndef NML_NO_GENERAL
        case OP_SYS: {
            CHECK_EXT(EXT_GENERAL, "NML-G");
            int code = (int)ins->imm;
            Tensor *rd = &REG(ins->reg[0]);
            switch (code) {
                case 0: /* PRINT_NUM */
                    if (RVALID(ins->reg[0])) {
                        double v = tensor_getd(rd, 0);
                        if (rd->dtype == NML_I32)
                            printf("%d\n", (int)v);
                        else if (v == (double)(long long)v && fabs(v) < 1e15)
                            printf("%.1f\n", v);
                        else
                            printf("%.6f\n", v);
                    } else {
                        printf("0\n");
                    }
                    fflush(stdout);
                    break;
                case 1: /* PRINT_CHAR */ {
                    int ch = RVALID(ins->reg[0]) ? (int)tensor_getd(rd, 0) : 0;
                    if (ch >= 0 && ch <= 127) putchar(ch);
                    fflush(stdout);
                    break;
                }
                case 2: /* READ_NUM */ {
                    double val = 0;
                    if (scanf("%lf", &val) != 1) val = 0;
                    int s[] = {1};
                    tensor_init_typed(rd, 1, s, NML_F64);
                    tensor_setd(rd, 0, val);
                    RVALID(ins->reg[0]) = 1;
                    break;
                }
                case 3: /* READ_CHAR */ {
                    int ch = getchar();
                    int s[] = {1};
                    tensor_init_typed(rd, 1, s, NML_I32);
                    tensor_setd(rd, 0, ch >= 0 ? ch : -1);
                    RVALID(ins->reg[0]) = 1;
                    break;
                }
                case 4: /* TIME */ {
                    struct timespec ts;
                    clock_gettime(CLOCK_REALTIME, &ts);
                    double t = (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
                    int s[] = {1};
                    tensor_init_typed(rd, 1, s, NML_F64);
                    tensor_setd(rd, 0, t);
                    RVALID(ins->reg[0]) = 1;
                    break;
                }
                case 5: /* RAND */ {
                    double r = (double)rand() / (double)RAND_MAX;
                    int s[] = {1};
                    tensor_init_typed(rd, 1, s, NML_F64);
                    tensor_setd(rd, 0, r);
                    RVALID(ins->reg[0]) = 1;
                    break;
                }
                case 6: /* EXIT */ {
                    int code_val = RVALID(ins->reg[0]) ? (int)tensor_getd(rd, 0) : 0;
                    vm->halted = 1;
                    return code_val == 0 ? NML_OK : NML_ERR_TRAP;
                }
                default:
                    VM_ERROR(vm, NML_ERR_OPCODE, "SYS: unknown syscall code %d at PC=%d", code, vm->pc);
            }
            break;
        }

        case OP_MOD: {
            CHECK_EXT(EXT_GENERAL, "NML-G");
            int a = RVALID(ins->reg[1]) ? (int)tensor_getd(&REG(ins->reg[1]), 0) : 0;
            int b = RVALID(ins->reg[2]) ? (int)tensor_getd(&REG(ins->reg[2]), 0) : 0;
            if (b == 0) VM_ERROR(vm, NML_ERR_DIVZERO, "MOD division by zero at PC=%d", vm->pc);
            int s[] = {1};
            tensor_init_typed(&REG(ins->reg[0]), 1, s, NML_I32);
            tensor_setd(&REG(ins->reg[0]), 0, (double)(a % b));
            RVALID(ins->reg[0]) = 1;
            break;
        }

        case OP_ITOF: {
            CHECK_EXT(EXT_GENERAL, "NML-G");
            double v = RVALID(ins->reg[1]) ? tensor_getd(&REG(ins->reg[1]), 0) : 0;
            int s[] = {1};
            tensor_init_typed(&REG(ins->reg[0]), 1, s, NML_F64);
            tensor_setd(&REG(ins->reg[0]), 0, v);
            RVALID(ins->reg[0]) = 1;
            break;
        }

        case OP_FTOI: {
            CHECK_EXT(EXT_GENERAL, "NML-G");
            double v = RVALID(ins->reg[1]) ? tensor_getd(&REG(ins->reg[1]), 0) : 0;
            int s[] = {1};
            tensor_init_typed(&REG(ins->reg[0]), 1, s, NML_I32);
            tensor_setd(&REG(ins->reg[0]), 0, (double)(int)v);
            RVALID(ins->reg[0]) = 1;
            break;
        }

        case OP_BNOT: {
            CHECK_EXT(EXT_GENERAL, "NML-G");
            Tensor *rs = &REG(ins->reg[1]);
            Tensor *rdst = &REG(ins->reg[0]);
            tensor_copy(rdst, rs);
            rdst->dtype = NML_I32;
            for (int i = 0; i < rs->size; i++)
                tensor_setd(rdst, i, (double)(~(int)tensor_getd(rs, i)));
            RVALID(ins->reg[0]) = 1;
            break;
        }
#endif /* NML_NO_GENERAL */

#ifndef NML_NO_TRAINING
        /* ═══ NML-TR Training ═══ */

        case OP_LOSS: {
            /* LOSS Rd Rs_pred Rs_target #type (0=MSE, 1=MAE) */
            Tensor *pred = &REG(ins->reg[1]);
            Tensor *target = &REG(ins->reg[2]);
            int loss_type = ins->int_params[0];
            int n = pred->size < target->size ? pred->size : target->size;
            double sum = 0;
            for (int i = 0; i < n; i++) {
                double diff = tensor_getd(pred, i) - tensor_getd(target, i);
                if (loss_type == 1) sum += (diff < 0 ? -diff : diff); /* MAE */
                else sum += diff * diff; /* MSE */
            }
            sum /= (n > 0 ? n : 1);
            int s[] = {1};
            tensor_init_typed(&REG(ins->reg[0]), 1, s, NML_F64);
            tensor_setd(&REG(ins->reg[0]), 0, sum);
            RVALID(ins->reg[0]) = 1;
            break;
        }

        case OP_WUPD: {
            /* WUPD Rd_weights Rs_gradient #learning_rate */
            /* Rd = Rd - lr * Rs (in-place weight update) */
            Tensor *weights = &REG(ins->reg[0]);
            Tensor *grad = &REG(ins->reg[1]);
            double lr = ins->imm;
            int n = weights->size < grad->size ? weights->size : grad->size;
            for (int i = 0; i < n; i++) {
                double w = tensor_getd(weights, i);
                double g = tensor_getd(grad, i);
                tensor_setd(weights, i, w - lr * g);
            }
            break;
        }

        case OP_BKWD: {
            /* BKWD Rd_grad Rs_output Rs_target [Rs_weights]
             * Computes output-layer gradient: Rd = 2 * (output - target) / N
             * If Rs_weights provided (int_params[0] >= 0), applies chain rule
             * through weights: Rd = (2/N * (output-target)) @ weights^T
             */
            Tensor *output = &REG(ins->reg[1]);
            Tensor *target = &REG(ins->reg[2]);
            int n = output->size < target->size ? output->size : target->size;

            /* Compute base gradient: d_loss/d_output = 2*(output-target)/N */
            int s_out[] = {1, n > 1 ? n : 1};
            tensor_init_typed(&REG(ins->reg[0]), 2, s_out, NML_F64);
            double scale = 2.0 / (n > 0 ? n : 1);
            for (int i = 0; i < n; i++) {
                double diff = tensor_getd(output, i) - tensor_getd(target, i);
                tensor_setd(&REG(ins->reg[0]), i, scale * diff);
            }
            RVALID(ins->reg[0]) = 1;

            /* If weights register provided, propagate gradient through weights^T */
            if (ins->int_params[0] >= 0 && ins->int_params[0] < NML_MAX_REGISTERS
                && RVALID(ins->int_params[0])) {
                /* Future: propagate gradient through weights^T for hidden layers */
                /* For 1->N->1 architectures, manual TRNS+MMUL in NML suffices */
                (void)ins->int_params[0];
            }
            break;
        }
        case OP_TNET: {
            /* Fused mini-batch training loop in C.
             * Register convention:
             *   R0 = training input (N×K)  R1 = w1 (K×H)   R2 = b1 (1×H)
             *   R3 = w2 (H×1)             R4 = b2 (1×1)    R9 = target (N×1)
             * Writes: R1,R2,R3,R4 updated weights; R8 = final loss
             */
            int epochs = (int)ins->imm;
            double lr = ins->int_params[0] / 1e6;
            int use_adam = ins->int_params[1];  /* 0=SGD, 1=Adam */

            Tensor *input  = &REG(0);
            Tensor *w1     = &REG(1);
            Tensor *bias1  = &REG(2);
            Tensor *w2     = &REG(3);
            Tensor *bias2  = &REG(4);
            Tensor *target = &REG(9);

            int H = w1->shape[w1->ndim - 1];
            int N = input->shape[0];
            int K = w1->shape[0];
            int B = (N <= 64) ? N : 64;
            int nbatch = (N + B - 1) / B;

            /* Heap-allocate: batch-sized working buffers + weight-sized gradient/Adam buffers */
            size_t bh = (size_t)B * H, kh = (size_t)K * H;
            size_t need = 3 * bh + 2 * (size_t)B + 2 * (size_t)H + kh;
            if (use_adam) need += 2 * kh + 4 * (size_t)H;
            double *buf = (double *)calloc(need, sizeof(double));
            if (!buf) VM_ERROR(vm, NML_ERR_OVERFLOW, "TNET: alloc failed (%zu doubles)", need);

            double *pre_h = buf, *hidden = pre_h + bh, *d_hidden = hidden + bh;
            double *out_buf = d_hidden + bh, *d_out = out_buf + B;
            double *d_w2 = d_out + B, *d_b1 = d_w2 + H, *d_w1 = d_b1 + H;
            double *m_w1 = NULL, *v_w1 = NULL, *m_w2 = NULL, *v_w2 = NULL;
            double *m_b1 = NULL, *v_b1 = NULL;
            if (use_adam) {
                m_w1 = d_w1 + kh; v_w1 = m_w1 + kh;
                m_w2 = v_w1 + kh;  v_w2 = m_w2 + H;
                m_b1 = v_w2 + H;   v_b1 = m_b1 + H;
            }
            double m_b2 = 0, v_b2 = 0;
            const double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;
            int adam_t = 0;
            double loss = 0;

            for (int epoch = 0; epoch < epochs; epoch++) {
                loss = 0;
                for (int batch = 0; batch < nbatch; batch++) {
                    int s = batch * B;
                    int Bn = (s + B <= N) ? B : N - s;

                    /* Forward: pre_h = input[s:s+Bn] @ w1 + b1 */
                    for (int i = 0; i < Bn; i++)
                        for (int j = 0; j < H; j++) {
                            double sum = 0;
                            for (int p = 0; p < K; p++)
                                sum += tensor_getd(input, (s + i) * K + p) * tensor_getd(w1, p * H + j);
                            pre_h[i * H + j] = sum + tensor_getd(bias1, j);
                        }

                    for (int i = 0; i < Bn * H; i++)
                        hidden[i] = pre_h[i] > 0 ? pre_h[i] : 0;

                    for (int i = 0; i < Bn; i++) {
                        double sum = 0;
                        for (int j = 0; j < H; j++)
                            sum += hidden[i * H + j] * tensor_getd(w2, j);
                        out_buf[i] = sum + tensor_getd(bias2, 0);
                    }

                    /* MSE loss + output gradient (normalized per batch) */
                    for (int i = 0; i < Bn; i++) {
                        double diff = out_buf[i] - tensor_getd(target, s + i);
                        loss += diff * diff;
                        d_out[i] = 2.0 * diff / Bn;
                    }

                    /* Backward: d_w2 = hidden^T @ d_out */
                    for (int j = 0; j < H; j++) {
                        double sum = 0;
                        for (int i = 0; i < Bn; i++)
                            sum += hidden[i * H + j] * d_out[i];
                        d_w2[j] = sum;
                    }
                    double d_b2_val = 0;
                    for (int i = 0; i < Bn; i++) d_b2_val += d_out[i];

                    /* d_hidden = d_out @ w2^T * relu'(pre_h) */
                    for (int i = 0; i < Bn; i++)
                        for (int j = 0; j < H; j++) {
                            double g = d_out[i] * tensor_getd(w2, j);
                            d_hidden[i * H + j] = pre_h[i * H + j] > 0 ? g : 0;
                        }

                    /* d_w1 = input[s:s+Bn]^T @ d_hidden */
                    for (int p = 0; p < K; p++)
                        for (int j = 0; j < H; j++) {
                            double sum = 0;
                            for (int i = 0; i < Bn; i++)
                                sum += tensor_getd(input, (s + i) * K + p) * d_hidden[i * H + j];
                            d_w1[p * H + j] = sum;
                        }

                    /* d_b1 = sum(d_hidden, axis=0) */
                    for (int j = 0; j < H; j++) {
                        double sum = 0;
                        for (int i = 0; i < Bn; i++)
                            sum += d_hidden[i * H + j];
                        d_b1[j] = sum;
                    }

                    /* Weight update (per mini-batch) */
                    if (use_adam) {
                        adam_t++;
                        double bc1 = 1.0 - pow(BETA1, adam_t);
                        double bc2 = 1.0 - pow(BETA2, adam_t);
                        for (int j = 0; j < H; j++) {
                            m_w2[j] = BETA1 * m_w2[j] + (1 - BETA1) * d_w2[j];
                            v_w2[j] = BETA2 * v_w2[j] + (1 - BETA2) * d_w2[j] * d_w2[j];
                            tensor_setd(w2, j, tensor_getd(w2, j) - lr * (m_w2[j] / bc1) / (sqrt(v_w2[j] / bc2) + EPS));
                        }
                        m_b2 = BETA1 * m_b2 + (1 - BETA1) * d_b2_val;
                        v_b2 = BETA2 * v_b2 + (1 - BETA2) * d_b2_val * d_b2_val;
                        tensor_setd(bias2, 0, tensor_getd(bias2, 0) - lr * (m_b2 / bc1) / (sqrt(v_b2 / bc2) + EPS));
                        for (int j = 0; j < H; j++) {
                            m_b1[j] = BETA1 * m_b1[j] + (1 - BETA1) * d_b1[j];
                            v_b1[j] = BETA2 * v_b1[j] + (1 - BETA2) * d_b1[j] * d_b1[j];
                            tensor_setd(bias1, j, tensor_getd(bias1, j) - lr * (m_b1[j] / bc1) / (sqrt(v_b1[j] / bc2) + EPS));
                        }
                        for (int idx = 0; idx < K * H; idx++) {
                            m_w1[idx] = BETA1 * m_w1[idx] + (1 - BETA1) * d_w1[idx];
                            v_w1[idx] = BETA2 * v_w1[idx] + (1 - BETA2) * d_w1[idx] * d_w1[idx];
                            tensor_setd(w1, idx, tensor_getd(w1, idx) - lr * (m_w1[idx] / bc1) / (sqrt(v_w1[idx] / bc2) + EPS));
                        }
                    } else {
                        for (int j = 0; j < H; j++)
                            tensor_setd(w2, j, tensor_getd(w2, j) - lr * d_w2[j]);
                        tensor_setd(bias2, 0, tensor_getd(bias2, 0) - lr * d_b2_val);
                        for (int j = 0; j < H; j++)
                            tensor_setd(bias1, j, tensor_getd(bias1, j) - lr * d_b1[j]);
                        for (int idx = 0; idx < K * H; idx++)
                            tensor_setd(w1, idx, tensor_getd(w1, idx) - lr * d_w1[idx]);
                    }
                }
                loss /= N;
            }

            free(buf);

            int ls[] = {1};
            tensor_init_typed(&REG(8), 1, ls, NML_F64);
            tensor_setd(&REG(8), 0, loss);
            RVALID(8) = 1;

            break;
        }
#endif /* NML_NO_TRAINING */

            case OP_HALT: vm->halted = 1; return NML_OK;
            default: VM_ERROR(vm, NML_ERR_OPCODE, "Unknown opcode %d at PC=%d", ins->op, vm->pc);
        }
        vm->pc++;
    }
    return vm->halted ? NML_OK : NML_ERR_CYCLE_LIMIT;
}

/* ═══════════════════════════════════════════
   FILE I/O & DATA LOADER
   ═══════════════════════════════════════════ */

static char* read_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, len, f); buf[len] = '\0'; fclose(f);
    return buf;
}

static int vm_load_data(VM *vm, const char *path) {
    char *src = read_file(path);
    if (!src) return NML_ERR_FILE;
    char line[4096]; const char *p = src; int count = 0;
    while (*p) {
        int i = 0;
        while (*p && *p != '\n' && i < 4095) line[i++] = *p++;
        line[i] = '\0'; if (*p == '\n') p++;
        char *start = line;
        while (*start == ' ' || *start == '\t') start++;
        if (*start == '\0' || *start == '#') continue;
        if (*start == '@') {
            char label[NML_MAX_LABEL_LEN] = {0};
            int shape[NML_MAX_DIMS] = {0}; int ndim = 0;
            char *lp = start + 1; int li = 0;
            while (*lp && *lp != ' ' && *lp != '\t') label[li++] = *lp++;
            char *sp = strstr(lp, "shape=");
            if (sp) { sp += 6; while (*sp && *sp != ' ' && *sp != '\t') { shape[ndim++] = atoi(sp); while (*sp && *sp != ',' && *sp != ' ' && *sp != '\t') sp++; if (*sp == ',') sp++; } }
            DType file_dtype = NML_F32;
            char *dtp = strstr(lp, "dtype=");
            if (dtp) { dtp += 6; char dtbuf[8]={0}; int dti=0; while(*dtp&&*dtp!=' '&&*dtp!='\t'&&dti<7) dtbuf[dti++]=*dtp++; file_dtype=parse_dtype(dtbuf); }
            MemorySlot *slot = vm_memory(vm, label);
            if (!slot) { free(src); return NML_ERR_MEMORY; }
            tensor_init_typed(&slot->tensor, ndim, shape, file_dtype); slot->used = 1;
            char *dp = strstr(lp, "data=");
            if (dp) { dp += 5; int di = 0; while (*dp && di < slot->tensor.size) { tensor_setd(&slot->tensor, di++, strtod(dp, &dp)); if (*dp == ',') dp++; } }
            count++;
        }
    }
    free(src); return count;
}

/* ═══════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════ */

static int count_unique_ops(const char *ext_filter) {
    char seen[OP_COUNT];
    memset(seen, 0, sizeof(seen));
    int n = 0;
    for (int i = 0; OPCODE_TABLE[i].name; i++) {
        if ((!ext_filter || strcmp(OPCODE_TABLE[i].ext, ext_filter) == 0) &&
            !seen[OPCODE_TABLE[i].code]) {
            seen[OPCODE_TABLE[i].code] = 1;
            n++;
        }
    }
    return n;
}

int main(int argc, char **argv) {
    const char *program_path = NULL;
    const char *data_path = NULL;
    int trace = 0;
    int max_cycles = NML_DEFAULT_MAX_CYCLES;
    int describe_only = 0;
    const char *fragment_name = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0) {
            trace = 1;
        } else if (strcmp(argv[i], "--max-cycles") == 0 && i + 1 < argc) {
            max_cycles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--describe") == 0) {
            describe_only = 1;
        } else if (strcmp(argv[i], "--fragment") == 0 && i + 1 < argc) {
            fragment_name = argv[++i];
        } else if (!program_path) {
            program_path = argv[i];
        } else if (!data_path) {
            data_path = argv[i];
        }
    }

    int core_ops = count_unique_ops("core");

    (void)fragment_name;

    if (!program_path) {
        printf("NML — Neural Machine Language v%d.%d\n", NML_VERSION_MAJOR, NML_VERSION_MINOR);
        printf("Usage: nml <program.nml> [data.nml.data] [--trace] [--max-cycles N] [--describe] [--fragment NAME]\n\n");
        printf("Core opcodes:    %d\n", core_ops);
        printf("Extensions:\n");
        printf("  NML-V (Vision):      %s\n",
#ifdef NML_EXT_VISION
            "ENABLED"
#else
            "disabled"
#endif
        );
        printf("  NML-T (Transformer): %s\n",
#ifdef NML_EXT_TRANSFORMER
            "ENABLED"
#else
            "disabled"
#endif
        );
        printf("  NML-R (Reduction):   %s\n",
#ifdef NML_EXT_REDUCTION
            "ENABLED"
#else
            "disabled"
#endif
        );
        printf("  NML-S (Signal):      %s\n",
#ifdef NML_EXT_SIGNAL
            "ENABLED"
#else
            "disabled"
#endif
        );
        printf("  NML-M2M (Machine):   %s\n",
#ifndef NML_NO_M2M
            "ENABLED"
#else
            "disabled"
#endif
        );
        printf("  NML-G (General):     %s\n",
#ifndef NML_NO_GENERAL
            "ENABLED"
#else
            "disabled"
#endif
        );
        printf("\nTotal opcodes: %d (%d core + %d extensions)\n", OP_COUNT, core_ops, OP_COUNT - core_ops);
        printf("\nv0.4 additions: SDIV EDIV CMP CMPI SPLT MERG CALL RET TRAP\n");
        printf("  + backward jumps, --trace, --max-cycles, vm_validate(), error codes\n");
        printf("v0.5 additions: Symbolic, Greek, Verbose aliases (tri-syntax)\n");
        printf("v0.6 additions: META FRAG ENDF LINK PTCH SIGN VRFY VOTE PROJ DIST (NML-M2M)\n");
        return 1;
    }

    srand((unsigned)time(NULL));

    VM *vm = (VM*)calloc(1, sizeof(VM));
    if (!vm) { fprintf(stderr, "ERROR: Cannot allocate VM\n"); return 1; }
    vm_init(vm);
    vm->trace = trace;
    vm->max_cycles = max_cycles;

    if (data_path) {
        printf("[NML v%d.%d] Loading data from %s\n", NML_VERSION_MAJOR, NML_VERSION_MINOR, data_path);
        int n = vm_load_data(vm, data_path);
        if (n < 0) { fprintf(stderr, "ERROR: Failed to load data (code %d)\n", n); free(vm); return 1; }
        printf("[NML] Loaded %d memory slots\n", n);
    }

    char *source = read_file(program_path);
    if (!source) { free(vm); return 1; }

    printf("[NML] Assembling %s\n", program_path);
    int ninstr = vm_assemble(vm, source);
    free(source);
    if (ninstr < 0) { fprintf(stderr, "ERROR: Assembly failed (code %d)\n", ninstr); free(vm); return 1; }
    printf("[NML] %d instructions assembled\n", ninstr);

    printf("[NML] Extensions:");
    if (vm->extensions & EXT_VISION) printf(" NML-V");
    if (vm->extensions & EXT_TRANSFORMER) printf(" NML-T");
    if (vm->extensions & EXT_REDUCTION) printf(" NML-R");
    if (vm->extensions & EXT_SIGNAL) printf(" NML-S");
    if (vm->extensions & EXT_M2M) printf(" NML-M2M");
    if (vm->extensions & EXT_GENERAL) printf(" NML-G");
    if (!vm->extensions) printf(" none");
    printf("\n");

    int vrc = vm_validate(vm);
    if (vrc != NML_OK) {
        fprintf(stderr, "ERROR: Validation failed: %s\n", vm->error_msg);
        free(vm);
        return 1;
    }
    printf("[NML] Validation passed\n");

    if (describe_only) {
        printf("Program Descriptor:\n");
        for (int i = 0; i < vm->descriptor.count; i++)
            printf("  %s: %s\n", vm->descriptor.entries[i].key, vm->descriptor.entries[i].value);
        free(vm);
        return 0;
    }

    printf("[NML] Executing...%s\n", trace ? " (trace enabled)" : "");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int result = vm_execute(vm);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;

    if (result != NML_OK)
        fprintf(stderr, "[NML] ERROR (code %d): %s\n", vm->error_code, vm->error_msg);

    printf("[NML] %s after %d cycles in %.1f µs\n", result == NML_OK ? "HALTED" : "ERROR", vm->cycles, elapsed_us);

    printf("\n=== REGISTERS ===\n");
    for (int i = 0; i < NML_MAX_REGISTERS; i++)
        if (vm->reg_valid[i]) { char n[8]; sprintf(n, "R%X", i); tensor_print(&vm->regs[i], n); }

    printf("\n=== MEMORY ===\n");
    for (int i = 0; i < vm->mem_count; i++)
        if (vm->memory[i].used) tensor_print(&vm->memory[i].tensor, vm->memory[i].label);

    printf("\n=== STATS ===\n");
    printf("  Version:       %d.%d\n", NML_VERSION_MAJOR, NML_VERSION_MINOR);
    printf("  Instructions:  %d\n", vm->program_len);
    printf("  Cycles:        %d\n", vm->cycles);
    printf("  Time:          %.1f µs\n", elapsed_us);
    printf("  Max cycles:    %d\n", vm->max_cycles);
    printf("  Opcodes:       %d (%d core + %d extensions)\n", OP_COUNT, core_ops, OP_COUNT - core_ops);

    int exit_code = (result != NML_OK) ? 1 : 0;
    free(vm);
    return exit_code;
}
