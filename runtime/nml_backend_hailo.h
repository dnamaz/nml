/*
 * nml_backend_hailo.h — Hailo NPU backend interface for NML
 *
 * The Hailo AI HAT+ / AI HAT+ 2 uses a Hailo NPU (Hailo-8 / Hailo-8L / Hailo-10H)
 * connected via PCIe.  Unlike SYCL or BLAS which dispatch individual tensor ops,
 * Hailo works at whole-model level:
 *
 *   Offline  — compile NML program + weights → .hef (Hailo Executable Format)
 *              using tools/nml_to_hailo.py + Hailo Dataflow Compiler
 *
 *   Runtime  — if <program>.hef exists alongside <program>.nml, the entire
 *              inference is dispatched to the Hailo NPU bypassing the NML
 *              interpreter.  Falls back to CPU on -1 return.
 *
 * I/O matching convention:
 *   HEF stream names must match NML memory slot labels (from .nml.data file).
 *   tools/nml_to_hailo.py preserves these names automatically.
 *
 * Float32 I/O:
 *   HailoRT handles quantisation/dequantisation internally when the stream
 *   type is requested as FLOAT32.  NML always uses float32 (NML_F32).
 *
 * Build:
 *   g++ -O3 -DNML_USE_HAILO runtime/nml.c runtime/nml_backend_hailo.cpp \
 *       -lm -lhailort -o nml-rpi-hailo
 */

#pragma once
#ifndef NML_BACKEND_HAILO_H
#define NML_BACKEND_HAILO_H

#ifdef __cplusplus
extern "C" {
#endif

/* ─── I/O tensor descriptor ───────────────────────────────────────────────── */

/*
 * NMLHailoTensor — a named, flat float32 tensor.
 *
 * For inputs:  name must match an HEF input stream name.
 *              data points to the NML memory slot's f32 array (read-only).
 * For outputs: name is filled in by nml_hailo_run() from the HEF stream name.
 *              data is malloc'd by nml_hailo_run(); caller must free().
 */
typedef struct {
    char  name[64]; /* HEF stream name / NML memory slot label */
    float *data;    /* f32 element array                        */
    int    n;       /* element count                            */
} NMLHailoTensor;

/* ─── Runtime API ─────────────────────────────────────────────────────────── */

/*
 * nml_hailo_run — load an HEF and run one synchronous inference pass.
 *
 *   hef_path       : path to the pre-compiled .hef file
 *   inputs         : array of named input tensors (from vm->memory)
 *   n_in           : number of input tensors
 *   outputs        : caller-allocated array; backend fills name + data
 *   n_out_max      : capacity of the outputs array
 *   n_out_written  : set to the number of outputs actually written
 *
 * Returns 0 on success.  On failure returns -1 and writes a human-readable
 * message to errbuf (if non-NULL, errbuf_n bytes).
 *
 * The caller owns outputs[i].data and must free() each after use.
 *
 * Thread safety: single-threaded NML — one call at a time is fine.
 */
/*
 * nml_hailo_run — load HEF from a file path and run one inference pass.
 * HEF is cached after first load (amortises parse cost in nmld).
 */
int nml_hailo_run(const char           *hef_path,
                  const NMLHailoTensor *inputs,        int n_in,
                  NMLHailoTensor       *outputs,        int n_out_max,
                  int                  *n_out_written,
                  char                 *errbuf,         int errbuf_n);

/*
 * nml_hailo_run_mem — load HEF from a byte buffer and run one inference pass.
 *
 * Used when NML_EMBEDDED_HEF is defined: the HEF is compiled into the binary
 * as a C array (via make nml-rpi-hailo-embed) and passed here directly.
 * No file I/O occurs at runtime — fully self-contained deployment.
 *
 *   hef_data : pointer to HEF bytes (e.g. _nml_embedded_hef_data[])
 *   hef_size : byte length          (e.g. _nml_embedded_hef_size)
 *
 * HailoRT copies what it needs during configure(); the caller's buffer
 * does not need to outlive the call.
 *
 * Output tensor ownership is the same as nml_hailo_run: caller must free().
 */
int nml_hailo_run_mem(const void           *hef_data,       size_t hef_size,
                      const NMLHailoTensor *inputs,          int n_in,
                      NMLHailoTensor       *outputs,          int n_out_max,
                      int                  *n_out_written,
                      char                 *errbuf,           int errbuf_n);

/*
 * nml_hailo_device_count — return the number of Hailo devices visible.
 * Returns 0 if no Hailo device is present (or HailoRT is unavailable).
 * Useful for printing diagnostics at startup.
 */
int nml_hailo_device_count(void);

/*
 * nml_hailo_arch — return the architecture string of the first Hailo device.
 *
 * Returns one of: "hailo8", "hailo8l", "hailo10h", "hailo15h", or "unknown".
 * The string is statically allocated — do not free.
 *
 * This is used by the HEF probe in nml.c to prefer chip-specific HEF files
 * (e.g. "program.hailo8.hef") over the generic fallback ("program.hef").
 * The result is cached after the first call.
 */
const char *nml_hailo_arch(void);

/*
 * nml_hailo_teardown — release the cached VDevice and associated resources.
 * Optional: call on clean exit.
 */
void nml_hailo_teardown(void);

#ifdef __cplusplus
}
#endif

#endif /* NML_BACKEND_HAILO_H */
