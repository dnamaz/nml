/*
 * nmld — NML Daemon (Generic Execution Server)
 *
 * Persistent C99 server that pre-assembles NML programs at startup, then
 * serves execution requests over a Unix domain socket. Pre-fork worker pool
 * for parallel processing. Supports cached and inline program execution.
 *
 * Build:   gcc -O2 -Wall -std=c99 -o nmld runtime/nmld.c -lm
 * Usage:   ./nmld --library domain/output/nml-library-classic [--workers 4]
 *
 * Protocol (newline-delimited JSON over Unix socket):
 *
 *   Cached program:
 *     {"program":"00-000-0000-FIT-000","data":"@gross_pay shape=1 data=78000\n..."}
 *
 *   Inline program:
 *     {"nml":"LEAF R0 #42\nSCLR R1 R0 #2\nST R1 @result\nHALT","data":""}
 *
 *   Batch:
 *     {"batch":[{"program":"...","data":"..."},{"nml":"...","data":"..."}]}
 *
 *   Hot-load:
 *     {"load":"my_program","nml":"LD R0 @input\n...HALT"}
 *
 *   Response:
 *     {"status":"HALTED","outputs":{"result":[84.0]},"cycles":4,"time_us":2.1}
 */

#define NML_LIBRARY_MODE
#include "nml.c"

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <signal.h>
#include <dirent.h>
#include <errno.h>

#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#define NMLD_MAX_PROGRAMS  16384
#define NMLD_MAX_REQUEST   131072
#define NMLD_MAX_RESPONSE  131072
#define NMLD_HASH_SIZE     32749
#define NMLD_MAX_WORKERS   32
#define NMLD_DEFAULT_WORKERS 1

/* Binary cache magic and version */
#define NMLD_CACHE_MAGIC   0x4E4D4C44  /* "NMLD" */
#define NMLD_CACHE_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t instr_size;      /* sizeof(Instruction) for validation */
    uint64_t data_offset;     /* offset to instruction data */
    uint64_t total_size;
} CacheHeader;

typedef struct {
    char name[64];
    uint32_t offset;          /* byte offset into data section */
    uint32_t program_len;     /* number of instructions */
} CacheIndexEntry;

/* ═══════════════════════════════════════════
   Program Cache
   ═══════════════════════════════════════════ */

typedef struct {
    char name[64];
    Instruction *program;
    int program_len;
} CachedProgram;

typedef struct {
    CachedProgram *programs;
    int count;
    int capacity;
    int hash_table[NMLD_HASH_SIZE];
} ProgramCache;

static unsigned nmld_hash(const char *s) {
    unsigned h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return h % NMLD_HASH_SIZE;
}

static void cache_init(ProgramCache *c) {
    c->capacity = NMLD_MAX_PROGRAMS;
    c->programs = (CachedProgram *)calloc(c->capacity, sizeof(CachedProgram));
    c->count = 0;
    for (int i = 0; i < NMLD_HASH_SIZE; i++) c->hash_table[i] = -1;
}

static int cache_lookup(ProgramCache *c, const char *name) {
    unsigned h = nmld_hash(name);
    for (int probe = 0; probe < NMLD_HASH_SIZE; probe++) {
        int idx = (h + probe) % NMLD_HASH_SIZE;
        int pi = c->hash_table[idx];
        if (pi < 0) return -1;
        if (strcmp(c->programs[pi].name, name) == 0) return pi;
    }
    return -1;
}

static int cache_add_from_vm(ProgramCache *c, const char *name, VM *vm) {
    if (c->count >= c->capacity) return -1;
    int pi = c->count++;
    CachedProgram *p = &c->programs[pi];
    strncpy(p->name, name, 63);
    p->program = (Instruction *)malloc(sizeof(Instruction) * vm->program_len);
    if (!p->program) return -1;
    memcpy(p->program, vm->program, sizeof(Instruction) * vm->program_len);
    p->program_len = vm->program_len;

    unsigned h = nmld_hash(name);
    for (int probe = 0; probe < NMLD_HASH_SIZE; probe++) {
        int idx = (h + probe) % NMLD_HASH_SIZE;
        if (c->hash_table[idx] < 0) { c->hash_table[idx] = pi; break; }
    }
    return pi;
}

static int cache_load_file(ProgramCache *c, const char *path, const char *name, VM *temp) {
    char *source = read_file(path);
    if (!source) return -1;
    memset(temp->program, 0, sizeof(Instruction) * 16);
    temp->program_len = 0;
    int n = vm_assemble(temp, source);
    free(source);
    if (n <= 0) return -1;
    return cache_add_from_vm(c, name, temp);
}

static int cache_load_directory(ProgramCache *c, const char *base_dir) {
    DIR *top = opendir(base_dir);
    if (!top) { fprintf(stderr, "[nmld] Cannot open %s\n", base_dir); return -1; }
    struct dirent *type_ent;
    VM *temp = (VM *)calloc(1, sizeof(VM));
    if (!temp) { closedir(top); return -1; }
    vm_init(temp);
    int loaded = 0, failed = 0;
    while ((type_ent = readdir(top)) != NULL) {
        if (type_ent->d_name[0] == '.') continue;
        char type_path[1024];
        snprintf(type_path, sizeof(type_path), "%s/%s", base_dir, type_ent->d_name);
        DIR *sub = opendir(type_path);
        if (!sub) continue;
        struct dirent *ent;
        while ((ent = readdir(sub)) != NULL) {
            size_t len = strlen(ent->d_name);
            if (len < 5 || strcmp(ent->d_name + len - 4, ".nml") != 0) continue;
            if (strstr(ent->d_name, ".nml.data")) continue;
            char file_path[2048], prog_name[64] = {0};
            snprintf(file_path, sizeof(file_path), "%s/%s", type_path, ent->d_name);
            strncpy(prog_name, ent->d_name, (len - 4 < 63) ? len - 4 : 63);
            if (cache_load_file(c, file_path, prog_name, temp) >= 0) loaded++; else failed++;
        }
        closedir(sub);
    }
    free(temp);
    closedir(top);
    fprintf(stderr, "[nmld] Loaded %d programs (%d failed)\n", loaded, failed);
    return loaded;
}

/* ═══════════════════════════════════════════
   Binary Cache: Build + Load
   ═══════════════════════════════════════════ */

static int cache_build_file(ProgramCache *c, const char *cache_path) {
    /* Calculate sizes */
    size_t index_size = sizeof(CacheIndexEntry) * c->count;
    size_t data_size = 0;
    for (int i = 0; i < c->count; i++)
        data_size += sizeof(Instruction) * c->programs[i].program_len;

    CacheHeader hdr;
    hdr.magic = NMLD_CACHE_MAGIC;
    hdr.version = NMLD_CACHE_VERSION;
    hdr.count = c->count;
    hdr.instr_size = sizeof(Instruction);
    hdr.data_offset = sizeof(CacheHeader) + index_size;
    hdr.total_size = hdr.data_offset + data_size;

    FILE *f = fopen(cache_path, "wb");
    if (!f) { perror("[nmld] cache write"); return -1; }

    fwrite(&hdr, sizeof(CacheHeader), 1, f);

    /* Write index */
    unsigned int data_off = 0;
    for (int i = 0; i < c->count; i++) {
        CacheIndexEntry entry;
        memset(&entry, 0, sizeof(entry));
        strncpy(entry.name, c->programs[i].name, 63);
        entry.offset = data_off;
        entry.program_len = c->programs[i].program_len;
        fwrite(&entry, sizeof(CacheIndexEntry), 1, f);
        data_off += sizeof(Instruction) * c->programs[i].program_len;
    }

    /* Write instruction data */
    for (int i = 0; i < c->count; i++)
        fwrite(c->programs[i].program, sizeof(Instruction), c->programs[i].program_len, f);

    fclose(f);
    fprintf(stderr, "[nmld] Cache written: %s (%.1f MB, %d programs)\n",
            cache_path, hdr.total_size / 1e6, c->count);
    return 0;
}

static void *cache_mmap_base = NULL;
static size_t cache_mmap_size = 0;

static int cache_load_from_file(ProgramCache *c, const char *cache_path) {
    int fd = open(cache_path, O_RDONLY);
    if (fd < 0) { perror("[nmld] cache open"); return -1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("[nmld] fstat"); close(fd); return -1; }

    void *mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { perror("[nmld] mmap"); return -1; }

    cache_mmap_base = mapped;
    cache_mmap_size = st.st_size;

    CacheHeader *hdr = (CacheHeader *)mapped;
    if (hdr->magic != NMLD_CACHE_MAGIC) {
        fprintf(stderr, "[nmld] Invalid cache magic\n");
        munmap(mapped, st.st_size); return -1;
    }
    if (hdr->version != NMLD_CACHE_VERSION) {
        fprintf(stderr, "[nmld] Cache version mismatch (got %d, want %d)\n", hdr->version, NMLD_CACHE_VERSION);
        munmap(mapped, st.st_size); return -1;
    }
    if (hdr->instr_size != sizeof(Instruction)) {
        fprintf(stderr, "[nmld] Instruction size mismatch (got %d, compiled %zu)\n", hdr->instr_size, sizeof(Instruction));
        munmap(mapped, st.st_size); return -1;
    }

    CacheIndexEntry *index = (CacheIndexEntry *)((char *)mapped + sizeof(CacheHeader));
    char *data_base = (char *)mapped + hdr->data_offset;

    cache_init(c);
    for (int i = 0; i < (int)hdr->count && i < c->capacity; i++) {
        CachedProgram *p = &c->programs[c->count];
        strncpy(p->name, index[i].name, 63);
        p->program = (Instruction *)(data_base + index[i].offset);
        p->program_len = index[i].program_len;

        /* Add to hash table */
        unsigned h = nmld_hash(p->name);
        for (int probe = 0; probe < NMLD_HASH_SIZE; probe++) {
            int idx = (h + probe) % NMLD_HASH_SIZE;
            if (c->hash_table[idx] < 0) { c->hash_table[idx] = c->count; break; }
        }
        c->count++;
    }

    fprintf(stderr, "[nmld] Cache loaded: %d programs from %s (%.1f MB, mmap)\n",
            c->count, cache_path, st.st_size / 1e6);
    return c->count;
}

/* ═══════════════════════════════════════════
   Output Extraction
   ═══════════════════════════════════════════ */

static int format_outputs(VM *vm, char *buf, int buf_size) {
    int off = 0;
    off += snprintf(buf + off, buf_size - off, "{");
    int first = 1;
    for (int i = 0; i < vm->mem_count; i++) {
        MemorySlot *s = &vm->memory[i];
        if (!s->used) continue;
        int is_input = 0;
        for (int j = 0; j < vm->program_len; j++) {
            if (vm->program[j].op == OP_LD && strcmp(vm->program[j].addr, s->label) == 0) { is_input = 1; break; }
        }
        int is_output = 0;
        for (int j = 0; j < vm->program_len; j++) {
            if (vm->program[j].op == OP_ST && strcmp(vm->program[j].addr, s->label) == 0) { is_output = 1; break; }
        }
        if (!is_output && is_input) continue;

        if (!first) off += snprintf(buf + off, buf_size - off, ",");
        off += snprintf(buf + off, buf_size - off, "\"%s\":[", s->label);
        int n = s->tensor.size < 16 ? s->tensor.size : 16;
        for (int k = 0; k < n; k++) {
            if (k > 0) off += snprintf(buf + off, buf_size - off, ",");
            off += snprintf(buf + off, buf_size - off, "%.6g", tensor_getd(&s->tensor, k));
        }
        off += snprintf(buf + off, buf_size - off, "]");
        first = 0;
    }
    off += snprintf(buf + off, buf_size - off, "}");
    return off;
}

/* ═══════════════════════════════════════════
   Request Handler
   ═══════════════════════════════════════════ */

static void vm_reset(VM *vm) {
    memset(vm->regs, 0, sizeof(vm->regs));
    memset(vm->reg_valid, 0, sizeof(vm->reg_valid));
    for (int i = 0; i < vm->mem_count; i++) vm->memory[i].used = 0;
    vm->mem_count = 0;
    vm->pc = 0; vm->halted = 0; vm->cycles = 0; vm->cond_flag = 0;
    vm->loop_depth = 0; vm->call_depth = 0;
    vm->error_code = NML_OK; vm->error_msg[0] = '\0';
}

static int json_get_string(const char *json, const char *key, char *out, int out_size) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        if (*p == '\\' && *(p+1) == 'n') { out[i++] = '\n'; p += 2; }
        else if (*p == '\\' && *(p+1) == '"') { out[i++] = '"'; p += 2; }
        else if (*p == '\\' && *(p+1) == '\\') { out[i++] = '\\'; p += 2; }
        else out[i++] = *p++;
    }
    out[i] = '\0';
    return 1;
}

static int handle_single(VM *vm, ProgramCache *cache, char *resp, int resp_size,
                          const char *json_obj) {
    char program_name[64] = {0};
    char nml_source[NMLD_MAX_REQUEST] = {0};
    char data[NMLD_MAX_REQUEST] = {0};

    int has_program = json_get_string(json_obj, "program", program_name, 64);
    int has_nml = json_get_string(json_obj, "nml", nml_source, NMLD_MAX_REQUEST);
    json_get_string(json_obj, "data", data, NMLD_MAX_REQUEST);

    vm_reset(vm);

    if (has_program) {
        int pi = cache_lookup(cache, program_name);
        if (pi < 0)
            return snprintf(resp, resp_size,
                "{\"status\":\"ERROR\",\"error\":\"program not found: %s\"}", program_name);
        memcpy(vm->program, cache->programs[pi].program,
               sizeof(Instruction) * cache->programs[pi].program_len);
        vm->program_len = cache->programs[pi].program_len;
    } else if (has_nml) {
        int n = vm_assemble(vm, nml_source);
        if (n <= 0)
            return snprintf(resp, resp_size,
                "{\"status\":\"ERROR\",\"error\":\"assembly failed\"}");
    } else {
        return snprintf(resp, resp_size,
            "{\"status\":\"ERROR\",\"error\":\"missing program or nml field\"}");
    }

    if (data[0]) vm_load_data_from_string(vm, data);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = vm_execute(vm);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;

    const char *status = (rc == NML_OK && vm->halted) ? "HALTED" : "ERROR";
    char outputs[8192];
    format_outputs(vm, outputs, sizeof(outputs));

    return snprintf(resp, resp_size,
        "{\"status\":\"%s\",\"outputs\":%s,\"cycles\":%d,\"time_us\":%.1f}",
        status, outputs, vm->cycles, elapsed_us);
}

static int handle_load(ProgramCache *cache, char *resp, int resp_size, const char *json) {
    char name[64] = {0}, nml_source[NMLD_MAX_REQUEST] = {0};
    if (!json_get_string(json, "load", name, 64) || !json_get_string(json, "nml", nml_source, NMLD_MAX_REQUEST))
        return snprintf(resp, resp_size, "{\"error\":\"load requires 'load' and 'nml' fields\"}");

    VM *temp = (VM *)calloc(1, sizeof(VM));
    if (!temp) return snprintf(resp, resp_size, "{\"error\":\"alloc failed\"}");
    vm_init(temp);
    int n = vm_assemble(temp, nml_source);
    if (n <= 0) { free(temp); return snprintf(resp, resp_size, "{\"error\":\"assembly failed\"}"); }

    int existing = cache_lookup(cache, name);
    if (existing >= 0) {
        free(cache->programs[existing].program);
        cache->programs[existing].program = (Instruction *)malloc(sizeof(Instruction) * temp->program_len);
        memcpy(cache->programs[existing].program, temp->program, sizeof(Instruction) * temp->program_len);
        cache->programs[existing].program_len = temp->program_len;
        free(temp);
        return snprintf(resp, resp_size, "{\"status\":\"updated\",\"program\":\"%s\",\"instructions\":%d}", name, n);
    }

    int pi = cache_add_from_vm(cache, name, temp);
    free(temp);
    if (pi < 0) return snprintf(resp, resp_size, "{\"error\":\"cache full\"}");
    return snprintf(resp, resp_size, "{\"status\":\"loaded\",\"program\":\"%s\",\"instructions\":%d}", name, n);
}

static void handle_request(VM *vm, ProgramCache *cache, const char *request,
                            char *response, int resp_size) {
    if (strstr(request, "\"load\"") && strstr(request, "\"nml\"") && !strstr(request, "\"batch\"")) {
        handle_load(cache, response, resp_size, request);
        return;
    }

    if (strstr(request, "\"batch\"")) {
        struct timespec bt0, bt1;
        clock_gettime(CLOCK_MONOTONIC, &bt0);
        int off = snprintf(response, resp_size, "{\"results\":[");
        const char *p = strstr(request, "[");
        if (!p) { snprintf(response, resp_size, "{\"error\":\"malformed batch\"}"); return; }

        int first = 1;
        while ((p = strstr(p, "{")) != NULL) {
            const char *obj_start = p;
            int depth = 1; p++;
            while (*p && depth > 0) { if (*p == '{') depth++; if (*p == '}') depth--; p++; }
            if (depth != 0) break;

            char obj[NMLD_MAX_REQUEST];
            int olen = (int)(p - obj_start);
            if (olen >= NMLD_MAX_REQUEST) continue;
            memcpy(obj, obj_start, olen); obj[olen] = '\0';

            if (!first) off += snprintf(response + off, resp_size - off, ",");
            char single[NMLD_MAX_RESPONSE / 4];
            handle_single(vm, cache, single, sizeof(single), obj);
            off += snprintf(response + off, resp_size - off, "%s", single);
            first = 0;
        }

        clock_gettime(CLOCK_MONOTONIC, &bt1);
        double batch_us = (bt1.tv_sec - bt0.tv_sec) * 1e6 + (bt1.tv_nsec - bt0.tv_nsec) / 1e3;
        snprintf(response + off, resp_size - off, "],\"total_time_us\":%.1f}", batch_us);
        return;
    }

    handle_single(vm, cache, response, resp_size, request);
}

/* ═══════════════════════════════════════════
   Worker Process
   ═══════════════════════════════════════════ */

static volatile int running = 1;
static volatile int reload_requested = 0;
static void signal_handler(int sig) { (void)sig; running = 0; }
static void sighup_handler(int sig) { (void)sig; reload_requested = 1; }

static void worker_loop(int server_fd, ProgramCache *cache, int worker_id) {
    VM *vm = (VM *)calloc(1, sizeof(VM));
    if (!vm) { fprintf(stderr, "[nmld/w%d] Failed to allocate VM\n", worker_id); return; }
    vm_init(vm);

    char request[NMLD_MAX_REQUEST];
    char response[NMLD_MAX_RESPONSE];

    while (running) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        int total = 0;
        while (total < NMLD_MAX_REQUEST - 1) {
            int n = (int)read(client_fd, request + total, NMLD_MAX_REQUEST - 1 - total);
            if (n <= 0) break;
            total += n;
            if (memchr(request, '\n', total)) break;
        }
        request[total] = '\0';
        char *nl = strchr(request, '\n');
        if (nl) *nl = '\0';

        if (total > 0) {
            handle_request(vm, cache, request, response, NMLD_MAX_RESPONSE);
            strcat(response, "\n");
            write(client_fd, response, strlen(response));
        }
        close(client_fd);
    }
    free(vm);
}

/* ═══════════════════════════════════════════
   Main — Master Process
   ═══════════════════════════════════════════ */

int main(int argc, char **argv) {
    const char *library_dir = NULL;
    const char *socket_path = "/tmp/nmld.sock";
    const char *cache_file = NULL;
    const char *build_cache = NULL;
    int num_workers = NMLD_DEFAULT_WORKERS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--library") == 0 && i + 1 < argc)
            library_dir = argv[++i];
        else if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc)
            socket_path = argv[++i];
        else if (strcmp(argv[i], "--cache-file") == 0 && i + 1 < argc)
            cache_file = argv[++i];
        else if (strcmp(argv[i], "--build-cache") == 0 && i + 1 < argc)
            build_cache = argv[++i];
        else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            const char *val = argv[++i];
            if (strcmp(val, "auto") == 0) {
                long n = 4;
#ifdef _SC_NPROCESSORS_ONLN
                n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
                num_workers = (n > 0 && n <= NMLD_MAX_WORKERS) ? (int)n : 4;
            } else {
                num_workers = atoi(val);
                if (num_workers < 1) num_workers = 1;
                if (num_workers > NMLD_MAX_WORKERS) num_workers = NMLD_MAX_WORKERS;
            }
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("nmld — NML Daemon (Generic Execution Server)\n\n");
            printf("Usage:\n");
            printf("  nmld --library <dir> [--socket <path>] [--workers N|auto]\n");
            printf("  nmld --cache-file <file> [--socket <path>] [--workers N|auto]\n");
            printf("  nmld --library <dir> --build-cache <file>   (build cache, then exit)\n\n");
            printf("Options:\n");
            printf("  --library <dir>       NML program directory to pre-cache\n");
            printf("  --cache-file <file>   Load from binary cache (instant startup)\n");
            printf("  --build-cache <file>  Build binary cache from --library, then exit\n");
            printf("  --socket <path>       Unix socket path (default: /tmp/nmld.sock)\n");
            printf("  --workers N|auto      Worker processes (default: 1, auto = CPU cores)\n");
            return 0;
        }
    }

    if (!library_dir && !cache_file) {
        fprintf(stderr, "Usage: nmld --library <dir> | --cache-file <file>\n");
        return 1;
    }

    fprintf(stderr, "[nmld] NML Daemon v0.7.1\n");

    ProgramCache cache;

    if (cache_file && !build_cache) {
        /* Fast path: load from binary cache via mmap */
        fprintf(stderr, "[nmld] Loading cache: %s\n", cache_file);
        int loaded = cache_load_from_file(&cache, cache_file);
        if (loaded <= 0) { fprintf(stderr, "[nmld] Cache load failed.\n"); return 1; }
    } else if (library_dir) {
        /* Slow path: scan and assemble from directory */
        fprintf(stderr, "[nmld] Scanning %s...\n", library_dir);
        cache_init(&cache);
        int loaded = cache_load_directory(&cache, library_dir);
        if (loaded <= 0) { fprintf(stderr, "[nmld] No programs loaded.\n"); return 1; }

        /* Build cache file if requested */
        if (build_cache) {
            cache_build_file(&cache, build_cache);
            for (int ci = 0; ci < cache.count; ci++) free(cache.programs[ci].program);
            free(cache.programs);
            return 0;
        }
    } else {
        fprintf(stderr, "[nmld] No library or cache specified.\n");
        return 1;
    }

    fprintf(stderr, "[nmld] Cache: %d programs\n", cache.count);

    /* Create socket before fork so all workers share it */
    unlink(socket_path);
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("[nmld] socket"); return 1; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("[nmld] bind"); return 1; }
    if (listen(server_fd, 64) < 0) { perror("[nmld] listen"); return 1; }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGHUP, sighup_handler);

    if (num_workers <= 1) {
        fprintf(stderr, "[nmld] Single worker on %s\n", socket_path);
        fprintf(stderr, "[nmld] Ready. (send SIGHUP to reload library)\n");
        worker_loop(server_fd, &cache, 0);
    } else {
        pid_t pids[NMLD_MAX_WORKERS];

#define SPAWN_WORKERS() do { \
    fprintf(stderr, "[nmld] Forking %d workers\n", num_workers); \
    for (int _i = 0; _i < num_workers; _i++) { \
        pid_t _pid = fork(); \
        if (_pid < 0) { perror("[nmld] fork"); break; } \
        if (_pid == 0) { \
            signal(SIGINT, signal_handler); \
            signal(SIGTERM, signal_handler); \
            signal(SIGHUP, SIG_DFL); \
            worker_loop(server_fd, &cache, _i); \
            close(server_fd); \
            exit(0); \
        } \
        pids[_i] = _pid; \
    } \
} while(0)

        SPAWN_WORKERS();
        fprintf(stderr, "[nmld] Ready. (%d workers, PID %d, SIGHUP to reload)\n", num_workers, getpid());

        while (running) {
            /* SIGHUP: reload library and replace all workers */
            if (reload_requested) {
                reload_requested = 0;
                fprintf(stderr, "[nmld] SIGHUP received — reloading %s...\n", library_dir);

                ProgramCache new_cache;
                cache_init(&new_cache);
                int new_loaded = cache_load_directory(&new_cache, library_dir);
                if (new_loaded > 0) {
                    /* Graceful shutdown of old workers */
                    for (int i = 0; i < num_workers; i++) kill(pids[i], SIGTERM);
                    for (int i = 0; i < num_workers; i++) waitpid(pids[i], NULL, 0);

                    /* Swap cache */
                    for (int ci = 0; ci < cache.count; ci++) free(cache.programs[ci].program);
                    free(cache.programs);
                    cache = new_cache;
                    fprintf(stderr, "[nmld] Reloaded: %d programs\n", cache.count);

                    SPAWN_WORKERS();
                    fprintf(stderr, "[nmld] New workers started.\n");
                } else {
                    fprintf(stderr, "[nmld] Reload failed — keeping old cache (%d programs)\n", cache.count);
                    free(new_cache.programs);
                }
                continue;
            }

            int status;
            pid_t died = waitpid(-1, &status, 0);
            if (died <= 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (!running) break;

            int worker_idx = -1;
            for (int i = 0; i < num_workers; i++) {
                if (pids[i] == died) { worker_idx = i; break; }
            }
            if (worker_idx >= 0 && !reload_requested) {
                fprintf(stderr, "[nmld] Worker %d (PID %d) exited (status %d), respawning...\n",
                        worker_idx, died, WEXITSTATUS(status));
                pid_t pid = fork();
                if (pid == 0) {
                    signal(SIGINT, signal_handler);
                    signal(SIGTERM, signal_handler);
                    signal(SIGHUP, SIG_DFL);
                    worker_loop(server_fd, &cache, worker_idx);
                    close(server_fd);
                    exit(0);
                }
                pids[worker_idx] = pid;
            }
        }

        /* Shutdown: kill all workers */
        for (int i = 0; i < num_workers; i++) kill(pids[i], SIGTERM);
        for (int i = 0; i < num_workers; i++) waitpid(pids[i], NULL, 0);
    }

    close(server_fd);
    unlink(socket_path);
    for (int ci = 0; ci < cache.count; ci++) free(cache.programs[ci].program);
    free(cache.programs);
    fprintf(stderr, "[nmld] Shutdown.\n");
    return 0;
}
