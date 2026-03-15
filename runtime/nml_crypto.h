/*
 * NML Crypto — SHA-256 + HMAC-SHA256 for program signing
 * Minimal, standalone, no external dependencies.
 * Include with -DNML_CRYPTO to enable SIGN/VRFY in the runtime.
 */

#ifndef NML_CRYPTO_H
#define NML_CRYPTO_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════
   SHA-256
   ═══════════════════════════════════════════ */

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buffer[64];
} SHA256_CTX;

static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define ROR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define S0(x) (ROR32(x,2)^ROR32(x,13)^ROR32(x,22))
#define S1(x) (ROR32(x,6)^ROR32(x,11)^ROR32(x,25))
#define s0(x) (ROR32(x,7)^ROR32(x,18)^((x)>>3))
#define s1(x) (ROR32(x,17)^ROR32(x,19)^((x)>>10))

static void sha256_transform(SHA256_CTX *ctx, const uint8_t *data) {
    uint32_t w[64], a, b, c, d, e, f, g, h, t1, t2;
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)data[4*i]<<24)|((uint32_t)data[4*i+1]<<16)|((uint32_t)data[4*i+2]<<8)|(uint32_t)data[4*i+3];
    for (int i = 16; i < 64; i++)
        w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];
    a=ctx->state[0]; b=ctx->state[1]; c=ctx->state[2]; d=ctx->state[3];
    e=ctx->state[4]; f=ctx->state[5]; g=ctx->state[6]; h=ctx->state[7];
    for (int i = 0; i < 64; i++) {
        t1 = h + S1(e) + CH(e,f,g) + SHA256_K[i] + w[i];
        t2 = S0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

static void sha256_init(SHA256_CTX *ctx) {
    ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85;
    ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
    ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c;
    ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
    ctx->count = 0;
    memset(ctx->buffer, 0, 64);
}

static void sha256_update(SHA256_CTX *ctx, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        ctx->buffer[ctx->count % 64] = data[i];
        ctx->count++;
        if (ctx->count % 64 == 0)
            sha256_transform(ctx, ctx->buffer);
    }
}

static void sha256_final(SHA256_CTX *ctx, uint8_t hash[32]) {
    uint64_t bits = ctx->count * 8;
    uint8_t pad = 0x80;
    sha256_update(ctx, &pad, 1);
    pad = 0;
    while (ctx->count % 64 != 56)
        sha256_update(ctx, &pad, 1);
    for (int i = 7; i >= 0; i--) {
        pad = (uint8_t)(bits >> (i * 8));
        sha256_update(ctx, &pad, 1);
    }
    for (int i = 0; i < 8; i++) {
        hash[4*i]   = (uint8_t)(ctx->state[i] >> 24);
        hash[4*i+1] = (uint8_t)(ctx->state[i] >> 16);
        hash[4*i+2] = (uint8_t)(ctx->state[i] >> 8);
        hash[4*i+3] = (uint8_t)(ctx->state[i]);
    }
}

static void sha256(const uint8_t *data, size_t len, uint8_t hash[32]) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, hash);
}

/* ═══════════════════════════════════════════
   HMAC-SHA256
   ═══════════════════════════════════════════ */

static void hmac_sha256(const uint8_t *key, size_t klen,
                        const uint8_t *msg, size_t mlen,
                        uint8_t mac[32]) {
    uint8_t k_pad[64], o_pad[64], i_hash[32];
    uint8_t k_used[32];

    if (klen > 64) {
        sha256(key, klen, k_used);
        key = k_used;
        klen = 32;
    }

    memset(k_pad, 0x36, 64);
    memset(o_pad, 0x5c, 64);
    for (size_t i = 0; i < klen; i++) {
        k_pad[i] ^= key[i];
        o_pad[i] ^= key[i];
    }

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, k_pad, 64);
    sha256_update(&ctx, msg, mlen);
    sha256_final(&ctx, i_hash);

    sha256_init(&ctx);
    sha256_update(&ctx, o_pad, 64);
    sha256_update(&ctx, i_hash, 32);
    sha256_final(&ctx, mac);
}

/* ═══════════════════════════════════════════
   Hex encoding/decoding
   ═══════════════════════════════════════════ */

static void hex_encode(const uint8_t *data, size_t len, char *out) {
    for (size_t i = 0; i < len; i++)
        sprintf(out + i * 2, "%02x", data[i]);
    out[len * 2] = '\0';
}

static int hex_decode(const char *hex, uint8_t *out, size_t max_len) {
    size_t hex_len = strlen(hex);
    if (hex_len % 2 != 0 || hex_len / 2 > max_len) return -1;
    for (size_t i = 0; i < hex_len / 2; i++) {
        unsigned int byte;
        if (sscanf(hex + i * 2, "%2x", &byte) != 1) return -1;
        out[i] = (uint8_t)byte;
    }
    return (int)(hex_len / 2);
}

/* ═══════════════════════════════════════════
   Ed25519 signing (via TweetNaCl)
   ═══════════════════════════════════════════ */

#include "tweetnacl.h"

static int nml_keygen(char *out, size_t out_size) {
    unsigned char pk[32], sk[64];
    crypto_sign_keypair(pk, sk);
    char sk_hex[129], pk_hex[65];
    hex_encode(sk, 64, sk_hex);
    hex_encode(pk, 32, pk_hex);
    int n = snprintf(out, out_size, "%s:%s", sk_hex, pk_hex);
    return (n > 0 && (size_t)n < out_size) ? 0 : -1;
}

static int nml_sign_ed25519(const char *source, const char *agent,
                            const char *sk_hex, char *out, size_t out_size) {
    unsigned char sk[64];
    if (hex_decode(sk_hex, sk, 64) != 64) return -1;

    unsigned char *sm = (unsigned char *)malloc(strlen(source) + 64);
    if (!sm) return -1;
    unsigned long long smlen;
    crypto_sign(sm, &smlen, (const unsigned char *)source, strlen(source), sk);

    char sig_hex[129];
    hex_encode(sm, 64, sig_hex);

    char pk_hex[65];
    hex_encode(sk + 32, 32, pk_hex);

    int written = snprintf(out, out_size,
        "SIGN agent=%s key=ed25519:%s sig=%s\n%s",
        agent, pk_hex, sig_hex, source);
    free(sm);
    return (written > 0 && (size_t)written < out_size) ? 0 : -1;
}

static int nml_verify_ed25519(const char *body, const char *pk_hex, const char *sig_hex) {
    unsigned char pk[32], sig[64];
    if (hex_decode(pk_hex, pk, 32) != 32) return -1;
    if (hex_decode(sig_hex, sig, 64) != 64) return -1;

    size_t body_len = strlen(body);
    unsigned char *sm = (unsigned char *)malloc(body_len + 64);
    if (!sm) return -1;
    memcpy(sm, sig, 64);
    memcpy(sm + 64, body, body_len);

    unsigned char *m = (unsigned char *)malloc(body_len + 64);
    unsigned long long mlen;
    int rc = crypto_sign_open(m, &mlen, sm, body_len + 64, pk);
    free(sm);
    free(m);
    return rc;
}

/* ═══════════════════════════════════════════
   Program signing / verification (dual-mode)
   ═══════════════════════════════════════════ */

static int nml_sign_program(const char *source, const char *agent,
                            const char *key_hex, char *out, size_t out_size) {
    size_t key_hex_len = strlen(key_hex);

    /* Ed25519: 128 hex chars = 64-byte secret key (or file with sk:pk format) */
    /* Check for sk:pk format first */
    const char *colon = strchr(key_hex, ':');
    if (colon && (colon - key_hex) == 128) {
        return nml_sign_ed25519(source, agent, key_hex, out, out_size);
    }
    if (key_hex_len == 128) {
        return nml_sign_ed25519(source, agent, key_hex, out, out_size);
    }

    /* HMAC-SHA256: any shorter key */
    uint8_t key[64];
    int klen = hex_decode(key_hex, key, 64);
    if (klen < 0) return -1;

    uint8_t mac[32];
    hmac_sha256(key, (size_t)klen, (const uint8_t *)source, strlen(source), mac);

    char sig_hex[65];
    hex_encode(mac, 32, sig_hex);

    int written = snprintf(out, out_size,
        "SIGN agent=%s key=hmac-sha256:%s sig=%s\n%s",
        agent, key_hex, sig_hex, source);
    return (written > 0 && (size_t)written < out_size) ? 0 : -1;
}

static int nml_verify_program(const char *signed_source,
                              char *agent_out, size_t agent_size,
                              const char **body_start) {
    if (strncmp(signed_source, "SIGN ", 5) != 0) return -1;

    const char *nl = strchr(signed_source, '\n');
    if (!nl) return -1;

    *body_start = nl + 1;

    char line[512];
    size_t line_len = (size_t)(nl - signed_source);
    if (line_len >= sizeof(line)) return -1;
    memcpy(line, signed_source, line_len);
    line[line_len] = '\0';

    char *agent_p = strstr(line, "agent=");
    char *sig_p = strstr(line, "sig=");
    char *key_p = strstr(line, "key=");
    if (!agent_p || !key_p || !sig_p) return -1;

    agent_p += 6;
    sig_p += 4;

    /* Extract agent name */
    char *sp = strchr(agent_p, ' ');
    if (sp) *sp = '\0';
    strncpy(agent_out, agent_p, agent_size - 1);
    if (sp) *sp = ' ';

    /* Detect key type and extract key value */
    int is_ed25519 = (strstr(line, "key=ed25519:") != NULL);
    char *key_val_p;
    if (is_ed25519) {
        key_val_p = strstr(line, "key=ed25519:") + 12;
    } else {
        key_val_p = strstr(line, "key=hmac-sha256:");
        if (key_val_p) key_val_p += 16;
        else { key_val_p = key_p + 4; }
    }

    /* Extract key hex */
    char key_hex[129] = {0};
    sp = strchr(key_val_p, ' ');
    if (sp) { size_t len = (size_t)(sp - key_val_p); if (len > 128) len = 128; memcpy(key_hex, key_val_p, len); key_hex[len] = '\0'; }
    else strncpy(key_hex, key_val_p, 128);

    /* Extract signature hex */
    char sig_hex[129] = {0};
    strncpy(sig_hex, sig_p, 128);

    if (is_ed25519) {
        if (nml_verify_ed25519(*body_start, key_hex, sig_hex) != 0) return -3;
    } else {
        uint8_t key_bytes[64];
        int klen = hex_decode(key_hex, key_bytes, 64);
        if (klen < 0) return -2;

        uint8_t expected[32];
        hmac_sha256(key_bytes, (size_t)klen, (const uint8_t *)*body_start, strlen(*body_start), expected);

        char computed_hex[65];
        hex_encode(expected, 32, computed_hex);

        if (strncmp(computed_hex, sig_hex, 64) != 0) return -3;
    }

    return 0;
}

/* ═══════════════════════════════════════════
   Differential Patching (PTCH)
   ═══════════════════════════════════════════ */

#define NML_PATCH_MAX_LINES 8192

static int nml_apply_patch(const char *base_source, const char *patch_source,
                           char *out, size_t out_size) {
    /* Split base into lines */
    char *base_lines[NML_PATCH_MAX_LINES];
    int base_count = 0;
    char *base_copy = strdup(base_source);
    char *line = strtok(base_copy, "\n");
    while (line && base_count < NML_PATCH_MAX_LINES) {
        base_lines[base_count++] = strdup(line);
        line = strtok(NULL, "\n");
    }

    /* Parse and apply patch directives */
    char patch_copy[65536];
    strncpy(patch_copy, patch_source, sizeof(patch_copy) - 1);
    patch_copy[sizeof(patch_copy) - 1] = '\0';

    int del_flags[NML_PATCH_MAX_LINES];
    memset(del_flags, 0, sizeof(del_flags));

    /* Insertions: stored as (after_line, text) pairs */
    typedef struct { int after; char text[256]; } InsertOp;
    InsertOp inserts[256];
    int ins_count = 0;

    line = strtok(patch_copy, "\n");
    while (line) {
        while (*line == ' ' || *line == '\t') line++;

        if (strncmp(line, "PTCH", 4) == 0) {
            char *arg = line + 4;
            while (*arg == ' ' || *arg == '\t') arg++;

            if (strncmp(arg, "@base", 5) == 0) {
                /* @base sha256:... — hash verification (TODO: full implementation) */
            } else if (strncmp(arg, "@set", 4) == 0) {
                int ln = 0;
                char replacement[256] = {0};
                if (sscanf(arg + 4, " %d %[^\n]", &ln, replacement) >= 2) {
                    if (ln >= 0 && ln < base_count) {
                        free(base_lines[ln]);
                        base_lines[ln] = strdup(replacement);
                    }
                }
            } else if (strncmp(arg, "@del", 4) == 0) {
                int ln = 0;
                if (sscanf(arg + 4, " %d", &ln) == 1) {
                    if (ln >= 0 && ln < base_count)
                        del_flags[ln] = 1;
                }
            } else if (strncmp(arg, "@ins", 4) == 0) {
                int ln = 0;
                char new_line[256] = {0};
                if (sscanf(arg + 4, " %d %[^\n]", &ln, new_line) >= 2) {
                    if (ins_count < 256) {
                        inserts[ins_count].after = ln;
                        strncpy(inserts[ins_count].text, new_line, 255);
                        ins_count++;
                    }
                }
            } else if (strncmp(arg, "@end", 4) == 0) {
                break;
            }
        }
        line = strtok(NULL, "\n");
    }

    /* Reconstruct the output */
    size_t pos = 0;
    for (int i = 0; i < base_count && pos < out_size - 2; i++) {
        if (!del_flags[i]) {
            int n = snprintf(out + pos, out_size - pos, "%s\n", base_lines[i]);
            if (n > 0) pos += (size_t)n;
        }

        /* Check for insertions after this line */
        for (int j = 0; j < ins_count; j++) {
            if (inserts[j].after == i) {
                int n = snprintf(out + pos, out_size - pos, "%s\n", inserts[j].text);
                if (n > 0) pos += (size_t)n;
            }
        }
    }
    if (pos > 0 && out[pos-1] == '\n') pos--;
    out[pos] = '\0';

    /* Cleanup */
    for (int i = 0; i < base_count; i++) free(base_lines[i]);
    free(base_copy);

    return 0;
}

#endif /* NML_CRYPTO_H */
