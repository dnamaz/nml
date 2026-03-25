/*
 * nml_api.h — NML public C API
 *
 * Provides a library interface for embedding NML inference in other applications.
 * No global state — each nml_vm_t is fully independent.
 *
 * Minimal usage:
 *
 *   nml_vm_t *vm = nml_vm_create();
 *   nml_vm_load(vm, "model.nml", "model.nml.data");
 *   nml_vm_run(vm);
 *   NMLTensor out = nml_vm_get_output(vm, "output");
 *   printf("result: %f\n", out.data[0]);
 *   nml_vm_destroy(vm);
 *
 * Build as library:
 *   gcc -O2 -shared -fPIC -o libnml.so runtime/nml.c -lm -DNML_BUILD_LIB
 *   gcc -O2 -o libnml.a runtime/nml.c -lm -DNML_BUILD_LIB  (static)
 */

#pragma once
#ifndef NML_API_H
#define NML_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque VM handle ──────────────────────────────────────────────────────── */
/* nml_vm_t is an opaque pointer to the internal VM struct defined in nml.c.
 * Client code must not dereference it — use the API functions below. */
typedef void nml_vm_t;

/* ── Tensor view returned by nml_vm_get_output ─────────────────────────────
 * Points into VM memory — valid until nml_vm_destroy() or next nml_vm_run().
 * Do NOT free data. Copy if you need it beyond that lifetime. */
typedef struct {
    const float *data;  /* F32 element array (NULL if not found or wrong dtype) */
    int          ndim;
    int          shape[4];
    int          size;  /* total elements */
} NMLTensor;

/* ── Lifecycle ─────────────────────────────────────────────────────────────── */

/* Create a new VM instance. Returns NULL on allocation failure. */
nml_vm_t *nml_vm_create(void);

/* Destroy a VM and free all resources. vm may be NULL (no-op). */
void nml_vm_destroy(nml_vm_t *vm);

/* ── Loading ───────────────────────────────────────────────────────────────── */

/* Load a program from a .nml file.
 * data_path may be NULL if the program has no weights.
 * Returns 0 on success, non-zero on error. */
int nml_vm_load(nml_vm_t *vm, const char *program_path, const char *data_path);

/* Load a program from in-memory buffers.
 * program_text: NML source text (null-terminated).
 * data_text:    .nml.data text (null-terminated), or NULL. */
int nml_vm_load_mem(nml_vm_t *vm, const char *program_text, const char *data_text);

/* ── Input ─────────────────────────────────────────────────────────────────── */

/* Set an input tensor in a named memory slot.
 * Copies data — caller retains ownership of the input array.
 * shape_len: number of dimensions (1–4). */
int nml_vm_set_input(nml_vm_t *vm, const char *name,
                     const float *data, const int *shape, int shape_len);

/* ── Execution ─────────────────────────────────────────────────────────────── */

/* Run the loaded program.
 * Returns 0 on success (HALT reached), non-zero on runtime error. */
int nml_vm_run(nml_vm_t *vm);

/* ── Output ────────────────────────────────────────────────────────────────── */

/* Get a named output tensor (set by ST in the NML program).
 * Returns a view into VM memory. Tensor.data is NULL if not found.
 * Do NOT free the returned data pointer. */
NMLTensor nml_vm_get_output(nml_vm_t *vm, const char *name);

/* ── Register access ───────────────────────────────────────────────────────── */

/* Get a register tensor by index (0–31). Returns a view. */
NMLTensor nml_vm_get_register(nml_vm_t *vm, int reg_index);

/* ── Error reporting ───────────────────────────────────────────────────────── */

/* Get the last error message (always valid, never NULL). */
const char *nml_vm_last_error(nml_vm_t *vm);

/* ── Metadata ──────────────────────────────────────────────────────────────── */

/* NML runtime version string (e.g., "0.7.0"). */
const char *nml_version(void);

/* Number of supported opcodes in this build. */
int nml_opcode_count(void);

#ifdef __cplusplus
}
#endif

#endif /* NML_API_H */
