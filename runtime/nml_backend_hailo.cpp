/*
 * nml_backend_hailo.cpp — Hailo NPU backend for NML (HailoRT 4.x)
 *
 * Two HEF loading modes, same inference core:
 *
 *   File mode (default):
 *     nml_hailo_run(hef_path, ...)
 *     HEF lives on disk; located by the chip-specific probe in nml.c.
 *
 *   Embedded mode (NML_EMBEDDED_HEF):
 *     nml_hailo_run_mem(data, size, ...)
 *     HEF is a C byte-array compiled into the binary.
 *     No file I/O at runtime — fully self-contained deployment.
 *     Built via: make nml-rpi-hailo-embed PROGRAM=... DATA=... ARCH=...
 *
 * Both paths call hailo_run_configured() which owns steps 3–6:
 *   3. Configure network group on VDevice
 *   4. Create float32 vstreams
 *   5. Write named inputs
 *   6. Read outputs → malloc'd NMLHailoTensor.data (caller frees)
 *
 * Requires: HailoRT >= 4.17
 *   sudo apt install hailo-all    # Raspberry Pi OS Bookworm
 */

#include <hailo/hailort.hpp>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "nml_backend_hailo.h"

using namespace hailort;

/* ═══════════════════════════════════════════════════════════════════
   DEVICE CACHE
   ═══════════════════════════════════════════════════════════════════ */

static std::unique_ptr<VDevice> _vdevice;

static VDevice *hailo_get_vdevice(void) {
    if (_vdevice) return _vdevice.get();

    hailo_vdevice_params_t params = {};
    params.device_count = 1;

    const char *dev_id = getenv("HAILO_DEVICE_ID");
    if (dev_id)
        strncpy(params.device_ids->id, dev_id, HAILO_MAX_DEVICE_ID_LENGTH - 1);

    auto vdev = VDevice::create(params);
    if (!vdev) return nullptr;
    _vdevice = std::move(vdev.value());
    return _vdevice.get();
}

/* ═══════════════════════════════════════════════════════════════════
   HEF FILE CACHE
   Amortises file-parse cost across repeated nmld calls.
   Not used in embedded mode (HEF is already in memory).
   ═══════════════════════════════════════════════════════════════════ */

struct HefEntry {
    std::unique_ptr<Hef> hef;
};

static std::unordered_map<std::string, HefEntry> _hef_cache;

static Hef *hailo_get_hef_from_file(const char *path) {
    auto it = _hef_cache.find(path);
    if (it != _hef_cache.end()) return it->second.hef.get();

    auto hef_exp = Hef::create(path);
    if (!hef_exp) return nullptr;

    HefEntry entry;
    entry.hef = std::make_unique<Hef>(std::move(hef_exp.value()));
    _hef_cache[path] = std::move(entry);
    return _hef_cache[path].hef.get();
}

/* ═══════════════════════════════════════════════════════════════════
   INFERENCE CORE  (shared by file and embedded paths)
   ═══════════════════════════════════════════════════════════════════ */

static int hailo_run_configured(
        VDevice              *vdev,
        Hef                  &hef,
        const NMLHailoTensor *inputs,      int n_in,
        NMLHailoTensor       *outputs,     int n_out_max,
        int                  *n_out_written,
        char                 *errbuf,      int errbuf_n)
{
    auto _err = [&](const char *msg) -> int {
        if (errbuf && errbuf_n > 0) snprintf(errbuf, errbuf_n, "%s", msg);
        return -1;
    };

    /* 3. Configure network group */
    auto configured = vdev->configure(hef);
    if (!configured) return _err("hailo: configure failed");
    auto &ng = configured.value()[0];

    /* 4. Create float32 vstreams — HailoRT handles quant/dequant internally */
    auto in_params  = ng->make_input_vstream_params(
                          true, HAILO_FORMAT_TYPE_FLOAT32, 0, 0);
    auto out_params = ng->make_output_vstream_params(
                          false, HAILO_FORMAT_TYPE_FLOAT32, 0, 0);
    if (!in_params || !out_params)
        return _err("hailo: vstream params failed");

    auto vstreams = VStreamsBuilder::create_vstreams(
                        *ng, in_params.value(), out_params.value());
    if (!vstreams) return _err("hailo: create_vstreams failed");

    auto &in_vs  = vstreams->first;
    auto &out_vs = vstreams->second;

    /* 5. Write inputs — match HEF stream name to NMLHailoTensor.name.
     *    Unmatched streams receive zeros; weights are baked into the HEF. */
    for (auto &vs : in_vs) {
        const float *src = nullptr;
        size_t src_bytes = vs.get_frame_size();

        for (int i = 0; i < n_in; i++) {
            if (strcmp(vs.name().c_str(), inputs[i].name) == 0) {
                src       = inputs[i].data;
                src_bytes = (size_t)inputs[i].n * sizeof(float);
                break;
            }
        }

        std::vector<float> zeros;
        if (!src) {
            zeros.assign(vs.get_frame_size() / sizeof(float), 0.0f);
            src       = zeros.data();
            src_bytes = vs.get_frame_size();
        }

        if (HAILO_SUCCESS != vs.write(MemoryView(const_cast<float*>(src), src_bytes)))
            return _err("hailo: input write failed");
    }

    /* 6. Read outputs — malloc; caller must free() */
    int written = 0;
    for (auto &vs : out_vs) {
        if (written >= n_out_max) break;

        size_t frame_bytes = vs.get_frame_size();
        float *out_data = (float*)malloc(frame_bytes);
        if (!out_data) return _err("hailo: malloc failed");

        if (HAILO_SUCCESS != vs.read(MemoryView(out_data, frame_bytes))) {
            free(out_data);
            return _err("hailo: output read failed");
        }

        strncpy(outputs[written].name, vs.name().c_str(), 63);
        outputs[written].name[63] = '\0';
        outputs[written].data = out_data;
        outputs[written].n    = (int)(frame_bytes / sizeof(float));
        written++;
    }
    *n_out_written = written;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
   PUBLIC API
   ═══════════════════════════════════════════════════════════════════ */

extern "C" {

int nml_hailo_device_count(void) {
    auto devices = Device::scan();
    if (!devices) return 0;
    return (int)devices.value().size();
}

const char *nml_hailo_arch(void) {
    static char cached[32] = {0};
    if (cached[0]) return cached;

    VDevice *vdev = hailo_get_vdevice();
    if (!vdev) { strncpy(cached, "unknown", sizeof(cached)-1); return cached; }

    auto phys = vdev->get_physical_devices();
    if (!phys || phys->empty()) {
        strncpy(cached, "unknown", sizeof(cached)-1); return cached;
    }

    auto arch_exp = phys.value()[0].get()->get_architecture();
    if (!arch_exp) { strncpy(cached, "unknown", sizeof(cached)-1); return cached; }

    switch (arch_exp.value()) {
        case HAILO_ARCH_HAILO8:   strncpy(cached, "hailo8",   sizeof(cached)-1); break;
        case HAILO_ARCH_HAILO8L:  strncpy(cached, "hailo8l",  sizeof(cached)-1); break;
#ifdef HAILO_ARCH_HAILO10H
        case HAILO_ARCH_HAILO10H: strncpy(cached, "hailo10h", sizeof(cached)-1); break;
#endif
#ifdef HAILO_ARCH_HAILO15H
        case HAILO_ARCH_HAILO15H: strncpy(cached, "hailo15h", sizeof(cached)-1); break;
#endif
        default: strncpy(cached, "unknown", sizeof(cached)-1); break;
    }
    return cached;
}

/* ── File path ─────────────────────────────────────────────────────────────── */

int nml_hailo_run(const char          *hef_path,
                  const NMLHailoTensor *inputs,       int n_in,
                  NMLHailoTensor       *outputs,       int n_out_max,
                  int                  *n_out_written,
                  char                 *errbuf,        int errbuf_n)
{
    *n_out_written = 0;
    auto _err = [&](const char *msg) -> int {
        if (errbuf && errbuf_n > 0) snprintf(errbuf, errbuf_n, "%s", msg);
        return -1;
    };

    VDevice *vdev = hailo_get_vdevice();
    if (!vdev) return _err("hailo: failed to open device");

    /* Load from file (cached after first call) */
    Hef *hef = hailo_get_hef_from_file(hef_path);
    if (!hef) return _err("hailo: failed to load HEF from file");

    return hailo_run_configured(vdev, *hef,
                                inputs, n_in, outputs, n_out_max,
                                n_out_written, errbuf, errbuf_n);
}

/* ── Memory buffer (embedded HEF) ─────────────────────────────────────────── */

/*
 * nml_hailo_run_mem — same as nml_hailo_run but loads the HEF from a byte
 * buffer instead of a file.  Used when NML_EMBEDDED_HEF is defined and the
 * HEF is compiled directly into the binary as a C array.
 *
 * The buffer is NOT cached (it is already in BSS/rodata — no parse cost to
 * amortise).  HailoRT copies what it needs during configure().
 */
int nml_hailo_run_mem(const void          *hef_data,      size_t hef_size,
                      const NMLHailoTensor *inputs,        int n_in,
                      NMLHailoTensor       *outputs,        int n_out_max,
                      int                  *n_out_written,
                      char                 *errbuf,         int errbuf_n)
{
    *n_out_written = 0;
    auto _err = [&](const char *msg) -> int {
        if (errbuf && errbuf_n > 0) snprintf(errbuf, errbuf_n, "%s", msg);
        return -1;
    };

    VDevice *vdev = hailo_get_vdevice();
    if (!vdev) return _err("hailo: failed to open device");

    /* Load HEF directly from memory — no file I/O */
    auto hef_exp = Hef::create(MemoryView(const_cast<void*>(hef_data), hef_size));
    if (!hef_exp) return _err("hailo: failed to parse embedded HEF");

    return hailo_run_configured(vdev, hef_exp.value(),
                                inputs, n_in, outputs, n_out_max,
                                n_out_written, errbuf, errbuf_n);
}

void nml_hailo_teardown(void) {
    _hef_cache.clear();
    _vdevice.reset();
}

} /* extern "C" */
