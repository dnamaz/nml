/*
 * nml_fmt.c — NML Syntax Converter
 *
 * Converts NML source between classic, symbolic (Unicode), and verbose syntax.
 * Standalone — does not depend on nml.c internals.
 *
 * Build:  gcc -O2 -o nml_fmt runtime/nml_fmt.c
 * Usage:  ./nml_fmt --to-symbolic  program.nml       # print to stdout
 *         ./nml_fmt --to-classic   program.nml -o out.nml
 *         ./nml_fmt --to-verbose   program.nml
 *         ./nml_fmt --list-opcodes
 */

#include "nml_fmt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ═══════════════════════════════════════════
   OPCODE TABLE  {classic, symbolic, verbose}
   ═══════════════════════════════════════════ */

typedef struct { const char *classic; const char *sym; const char *verbose; } OpcodeRow;

static const OpcodeRow OPCODE_TABLE[] = {
    /* ── Core: Arithmetic ── */
    { "MMUL",   "×",   "MATRIX_MULTIPLY"   },
    { "MADD",   "⊕",   "ADD"               },
    { "MSUB",   "⊖",   "SUBTRACT"          },
    { "EMUL",   "⊗",   "ELEMENT_MULTIPLY"  },
    { "EDIV",   "⊘",   "ELEMENT_DIVIDE"    },
    { "SDOT",   "·",   "DOT_PRODUCT"       },
    { "SCLR",   "∗",   "SCALE"             },
    { "SDIV",   "÷",   "DIVIDE"            },
    /* ── Core: Activations ── */
    { "RELU",   "⌐",   "RELU"              },
    { "SIGM",   "σ",   "SIGMOID"           },
    { "TANH",   "τ",   "TANH"              },
    { "SOFT",   "Σ",   "SOFTMAX"           },
    { "GELU",   "ℊ",   "GELU"              },
    /* ── Core: Memory ── */
    { "LD",     "↓",   "LOAD"              },
    { "ST",     "↑",   "STORE"             },
    { "MOV",    "←",   "COPY"              },
    { "ALLC",   "□",   "ALLOCATE"          },
    /* ── Core: Data Flow ── */
    { "RSHP",   "⊞",   "RESHAPE"           },
    { "TRNS",   "⊤",   "TRANSPOSE"         },
    { "SPLT",   "⊢",   "SPLIT"             },
    { "MERG",   "⊣",   "MERGE"             },
    /* ── Core: Control ── */
    { "LOOP",   "↻",   "REPEAT"            },
    { "ENDP",   "↺",   "END_REPEAT"        },
    { "JMPT",   "↗",   "BRANCH_TRUE"       },
    { "JMPF",   "↘",   "BRANCH_FALSE"      },
    { "JUMP",   "→",   "JUMP"              },
    { "CALL",   "⇒",   "CALL"              },
    { "RET",    "⇐",   "RETURN"            },
    { "HALT",   "◼",   "STOP"              },
    { "SYNC",   "⏸",   "BARRIER"           },
    { "TRAP",   "⚠",   "FAULT"             },
    /* ── Core: Comparison ── */
    { "CMP",    "≶",   "COMPARE"           },
    { "CMPI",   "≺",   "COMPARE_VALUE"     },
    { "CMPF",   "⋈",   "COMPARE_FEATURE"   },
    /* ── Core: Tree ── */
    { "LEAF",   "∎",   "SET_VALUE"         },
    { "TACC",   "∑",   "ACCUMULATE"        },
    /* ── NML-V: Vision ── */
    { "CONV",   "⊛",   "CONVOLVE"          },
    { "POOL",   "⊓",   "MAX_POOL"          },
    { "UPSC",   "⊔",   "UPSCALE"           },
    { "PADZ",   "⊡",   "ZERO_PAD"          },
    /* ── NML-T: Transformer ── */
    { "ATTN",   "⊙",   "ATTENTION"         },
    { "NORM",   "‖",   "LAYER_NORM"        },
    { "EMBD",   "⊏",   "EMBED"             },
    /* ── NML-R: Reduction ── */
    { "RDUC",   "⊥",   "REDUCE"            },
    { "WHER",   "⊻",   "WHERE"             },
    { "CLMP",   "⊧",   "CLAMP"             },
    { "CMPR",   "⊜",   "MASK_COMPARE"      },
    /* ── NML-S: Signal ── */
    { "FFT",    "∿",   "FOURIER"           },
    { "FILT",   "⋐",   "FILTER"            },
    /* ── NML-M2M ── */
    { "META",   "§",   "METADATA"          },
    { "FRAG",   "◆",   "FRAGMENT"          },
    { "ENDF",   "◇",   "END_FRAGMENT"      },
    { "LINK",   "⊕",   "IMPORT"            },
    { "PTCH",   "⊿",   "PATCH"             },
    { "SIGN",   "✦",   "SIGN_PROGRAM"      },
    { "VRFY",   "✓",   "VERIFY_SIGNATURE"  },
    { "VOTE",   "⚖",   "CONSENSUS"         },
    { "PROJ",   "⟐",   "PROJECT"           },
    { "DIST",   "⟂",   "DISTANCE"          },
    { "GATH",   "⊃",   "GATHER"            },
    { "SCAT",   "⊂",   "SCATTER"           },
    /* ── NML-G: General Purpose ── */
    { "SYS",    "⚙",   "SYSTEM"            },
    { "MOD",    "%",   "MODULO"            },
    { "ITOF",   "⊶",   "INT_TO_FLOAT"      },
    { "FTOI",   "⊷",   "FLOAT_TO_INT"      },
    { "BNOT",   "¬",   "BITWISE_NOT"       },
    /* ── NML-TR: Training ── */
    { "BKWD",   "∇",   "BACKWARD"          },
    { "WUPD",   "⟳",   "WEIGHT_UPDATE"     },
    { "LOSS",   "△",   "COMPUTE_LOSS"      },
    { "TNET",   "⥁",   "TRAIN_NETWORK"     },
    { "TNDEEP", "⥁ˈ",  "TRAIN_DEEP"        },
    /* ── NML-TR: Backward passes (forward symbol + prime ˈ) ── */
    { "RELUBK", "⌐ˈ",  "RELU_BACKWARD"     },
    { "SIGMBK", "σˈ",  "SIGMOID_BACKWARD"  },
    { "TANHBK", "τˈ",  "TANH_BACKWARD"     },
    { "GELUBK", "ℊˈ",  "GELU_BACKWARD"     },
    { "SOFTBK", "Σˈ",  "SOFTMAX_BACKWARD"  },
    { "MMULBK", "×ˈ",  "MATMUL_BACKWARD"   },
    { "CONVBK", "⊛ˈ",  "CONV_BACKWARD"     },
    { "POOLBK", "⊓ˈ",  "POOL_BACKWARD"     },
    { "NORMBK", "‖ˈ",  "NORM_BACKWARD"     },
    { "ATTNBK", "⊙ˈ",  "ATTN_BACKWARD"     },
    /* ── NML-TR v0.9 ── */
    { "TLOG",   "⧖",   "TRAIN_LOG"         },
    { "TRAIN",  "⟴",   "TRAIN_CONFIG"      },
    { "INFER",  "⟶",   "FORWARD_PASS"      },
    { "WDECAY", "ω",   "WEIGHT_DECAY"      },
    { NULL, NULL, NULL }
};

/* ═══════════════════════════════════════════
   REGISTER TABLE  {classic, symbolic, verbose}
   Indices 0–31 correspond to R0–RV.
   ═══════════════════════════════════════════ */

typedef struct { const char *classic; const char *sym; const char *verbose; } RegRow;

static const RegRow REG_TABLE[32] = {
    { "R0",  "ι",  "R0"          }, /*  0 — input */
    { "R1",  "κ",  "R1"          }, /*  1 — w1 */
    { "R2",  "λ",  "R2"          }, /*  2 — b1 */
    { "R3",  "μ",  "R3"          }, /*  3 — w2 */
    { "R4",  "ν",  "R4"          }, /*  4 — b2 */
    { "R5",  "ξ",  "R5"          }, /*  5 — w3 */
    { "R6",  "ο",  "R6"          }, /*  6 — b3 */
    { "R7",  "π",  "R7"          }, /*  7 — scratch */
    { "R8",  "ρ",  "R8"          }, /*  8 — loss / scratch */
    { "R9",  "ς",  "R9"          }, /*  9 — labels */
    { "RA",  "α",  "ACCUMULATOR" }, /* 10 — accumulator */
    { "RB",  "β",  "GENERAL"     }, /* 11 */
    { "RC",  "γ",  "SCRATCH"     }, /* 12 */
    { "RD",  "δ",  "COUNTER"     }, /* 13 — loop counter */
    { "RE",  "φ",  "FLAG"        }, /* 14 — condition flag */
    { "RF",  "ψ",  "STACK"       }, /* 15 — stack pointer */
    { "RG",  "η",  "GRAD1"       }, /* 16 — gradient 1 */
    { "RH",  "θ",  "GRAD2"       }, /* 17 — gradient 2 */
    { "RI",  "ζ",  "GRAD3"       }, /* 18 — gradient 3 */
    { "RJ",  "ω",  "LRATE"       }, /* 19 — learning rate / loss */
    { "RK",  "χ",  "RK"          }, /* 20 */
    { "RL",  "υ",  "RL"          }, /* 21 */
    { "RM",  "ε",  "RM"          }, /* 22 */
    { "RN",  "RN", "RN"          }, /* 23 — τ conflicts with TANH opcode */
    { "RO",  "RO", "RO"          }, /* 24 */
    { "RP",  "RP", "RP"          }, /* 25 */
    { "RQ",  "RQ", "RQ"          }, /* 26 */
    { "RR",  "RR", "RR"          }, /* 27 */
    { "RS",  "RS", "RS"          }, /* 28 */
    { "RT",  "RT", "RT"          }, /* 29 */
    { "RU",  "RU", "RU"          }, /* 30 */
    { "RV",  "RV", "RV"          }, /* 31 — arch descriptor for TNDEEP */
};

/* Additional verbose register aliases that map back to an index. */
static const struct { const char *name; int idx; } VERBOSE_REGS[] = {
    { "ACCUMULATOR", 10 }, { "ACC", 10 },
    { "GENERAL", 11 },     { "GEN", 11 },
    { "SCRATCH", 12 },     { "TMP", 12 },
    { "COUNTER", 13 },     { "CTR", 13 },
    { "FLAG", 14 },        { "FLG", 14 },
    { "STACK", 15 },       { "STK", 15 },
    { "GRAD1", 16 }, { "GRAD2", 17 }, { "GRAD3", 18 }, { "LRATE", 19 },
    { NULL, -1 }
};

/* Greek register alias → index.
 * Each entry is a UTF-8 string + its register index. */
static const struct { const char *sym; int idx; } GREEK_REGS[] = {
    { "ι",  0  }, { "κ",  1  }, { "λ",  2  }, { "μ",  3  },
    { "ν",  4  }, { "ξ",  5  }, { "ο",  6  }, { "π",  7  },
    { "ρ",  8  }, { "ς",  9  }, { "α",  10 }, { "β",  11 },
    { "γ",  12 }, { "δ",  13 }, { "φ",  14 }, { "ψ",  15 },
    { "η",  16 }, { "θ",  17 }, { "ζ",  18 }, { "ω",  19 },
    { "χ",  20 }, { "υ",  21 }, { "ε",  22 },
    { NULL, -1 }
};

/* ═══════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════ */

/* Case-insensitive token match against a table entry string. */
static int tok_eq(const char *a, const char *b) {
    /* Symbolic tokens are compared case-sensitively (they're Unicode). */
    /* Classic/verbose tokens compared case-insensitively. */
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}

/* Resolve any opcode token (classic, symbolic, or verbose) to its OpcodeRow.
 * Returns NULL if not found. */
static const OpcodeRow *opcode_lookup(const char *token) {
    /* Strip :type annotation if present (e.g. "HALT:exit") */
    char buf[64];
    const char *colon = strchr(token, ':');
    if (colon && (colon - token) < (int)sizeof(buf) - 1) {
        size_t len = (size_t)(colon - token);
        memcpy(buf, token, len);
        buf[len] = '\0';
        token = buf;
    }

    for (const OpcodeRow *r = OPCODE_TABLE; r->classic; r++) {
        if (strcmp(token, r->sym)     == 0) return r;
        if (tok_eq(token, r->classic) == 1) return r;
        if (tok_eq(token, r->verbose) == 1) return r;
    }
    return NULL;
}

/* Resolve any register token to its index 0–31.
 * Returns -1 if not a register. */
static int reg_lookup(const char *token) {
    /* Strip :type annotation */
    char buf[64];
    const char *colon = strchr(token, ':');
    if (colon && (colon - token) < (int)sizeof(buf) - 1) {
        size_t len = (size_t)(colon - token);
        memcpy(buf, token, len);
        buf[len] = '\0';
        token = buf;
    }

    /* Classic: R0-R9 */
    if ((token[0] == 'R' || token[0] == 'r') && token[1] >= '0' && token[1] <= '9' && token[2] == '\0')
        return token[1] - '0';

    /* Classic: RA-RV */
    if ((token[0] == 'R' || token[0] == 'r') &&
        token[1] >= 'A' && token[1] <= 'V' && token[2] == '\0')
        return 10 + (toupper((unsigned char)token[1]) - 'A');

    /* Greek aliases */
    for (int i = 0; GREEK_REGS[i].sym; i++)
        if (strcmp(token, GREEK_REGS[i].sym) == 0) return GREEK_REGS[i].idx;

    /* Verbose aliases */
    for (int i = 0; VERBOSE_REGS[i].name; i++)
        if (tok_eq(token, VERBOSE_REGS[i].name)) return VERBOSE_REGS[i].idx;

    return -1;
}

/* ═══════════════════════════════════════════
   PUBLIC API
   ═══════════════════════════════════════════ */

const char *nml_opcode_convert(const char *token, NmlSyntax to) {
    const OpcodeRow *r = opcode_lookup(token);
    if (!r) return NULL;
    switch (to) {
        case NML_SYNTAX_CLASSIC:  return r->classic;
        case NML_SYNTAX_SYMBOLIC: return r->sym;
        case NML_SYNTAX_VERBOSE:  return r->verbose;
    }
    return r->classic;
}

const char *nml_reg_convert(const char *token, NmlSyntax to) {
    int idx = reg_lookup(token);
    if (idx < 0 || idx > 31) return NULL;
    switch (to) {
        case NML_SYNTAX_CLASSIC:  return REG_TABLE[idx].classic;
        case NML_SYNTAX_SYMBOLIC: return REG_TABLE[idx].sym;
        case NML_SYNTAX_VERBOSE:  return REG_TABLE[idx].verbose;
    }
    return REG_TABLE[idx].classic;
}

/* ═══════════════════════════════════════════
   LINE CONVERTER
   ═══════════════════════════════════════════ */

/* Append a string to dst, advancing *pos. Returns 0 on overflow. */
static int append(char *dst, size_t dst_len, size_t *pos, const char *s) {
    size_t n = strlen(s);
    if (*pos + n >= dst_len) return 0;
    memcpy(dst + *pos, s, n);
    *pos += n;
    dst[*pos] = '\0';
    return 1;
}

/* Convert one line of NML source. dst must be large enough (e.g. 4096 bytes).
 * Returns 0 on success, -1 on overflow. */
static int convert_line(const char *line, char *dst, size_t dst_len, NmlSyntax to) {
    /* Split on first ';' comment */
    const char *comment_start = NULL;
    const char *p = line;

    /* Find ';' that's not inside a quoted string */
    int in_quote = 0;
    for (; *p; p++) {
        if (*p == '"') in_quote = !in_quote;
        if (*p == ';' && !in_quote) { comment_start = p; break; }
    }

    /* Instruction part (may be empty) */
    size_t instr_len = comment_start ? (size_t)(comment_start - line) : strlen(line);
    char instr[4096];
    if (instr_len >= sizeof(instr)) instr_len = sizeof(instr) - 1;
    memcpy(instr, line, instr_len);
    instr[instr_len] = '\0';

    /* Tokenise the instruction part */
    char tokens[64][256];
    int ntok = 0;
    char *tok = strtok(instr, " \t");
    while (tok && ntok < 64) {
        strncpy(tokens[ntok], tok, 255);
        tokens[ntok][255] = '\0';
        ntok++;
        tok = strtok(NULL, " \t");
    }

    size_t pos = 0;
    dst[0] = '\0';

    if (ntok == 0) {
        /* Empty instruction — just write the comment if any */
        if (comment_start) {
            if (!append(dst, dst_len, &pos, comment_start)) return -1;
        }
        return 0;
    }

    /* Token 0: opcode */
    const char *converted_op = nml_opcode_convert(tokens[0], to);
    if (!converted_op) converted_op = tokens[0]; /* unknown token, pass through */
    if (!append(dst, dst_len, &pos, converted_op)) return -1;

    /* Tokens 1+: operands */
    for (int i = 1; i < ntok; i++) {
        if (!append(dst, dst_len, &pos, " ")) return -1;

        char *t = tokens[i];

        /* @label — never convert */
        if (t[0] == '@') { if (!append(dst, dst_len, &pos, t)) return -1; continue; }
        /* #immediate — never convert */
        if (t[0] == '#') { if (!append(dst, dst_len, &pos, t)) return -1; continue; }
        /* Quoted string — never convert */
        if (t[0] == '"') { if (!append(dst, dst_len, &pos, t)) return -1; continue; }

        /* Register? */
        const char *converted_reg = nml_reg_convert(t, to);
        if (converted_reg) {
            if (!append(dst, dst_len, &pos, converted_reg)) return -1;
            continue;
        }

        /* Numeric literal (e.g. bare 0, 1, 3000, 0.005) — pass through */
        if (isdigit((unsigned char)t[0]) || (t[0] == '-' && isdigit((unsigned char)t[1]))) {
            if (!append(dst, dst_len, &pos, t)) return -1;
            continue;
        }

        /* Unknown token (e.g. verbose keyword operand) — pass through */
        if (!append(dst, dst_len, &pos, t)) return -1;
    }

    /* Re-attach comment */
    if (comment_start) {
        if (!append(dst, dst_len, &pos, "  ")) return -1;
        if (!append(dst, dst_len, &pos, comment_start)) return -1;
    }

    return 0;
}

/* ═══════════════════════════════════════════
   PUBLIC: nml_fmt_source
   ═══════════════════════════════════════════ */

int nml_fmt_source(const char *src, char *dst, size_t dst_len, NmlSyntax to) {
    if (!src || !dst || dst_len == 0) return -1;
    dst[0] = '\0';

    char line_buf[4096];
    char out_line[4096];
    size_t pos = 0;
    const char *p = src;

    while (*p) {
        /* Extract one line */
        const char *end = strchr(p, '\n');
        size_t line_len = end ? (size_t)(end - p) : strlen(p);
        if (line_len >= sizeof(line_buf)) line_len = sizeof(line_buf) - 1;
        memcpy(line_buf, p, line_len);
        /* Strip trailing \r */
        if (line_len > 0 && line_buf[line_len - 1] == '\r') line_len--;
        line_buf[line_len] = '\0';
        p += (end ? (size_t)(end - p) + 1 : strlen(p));

        if (convert_line(line_buf, out_line, sizeof(out_line), to) < 0)
            return -1;

        size_t olen = strlen(out_line);
        if (pos + olen + 1 >= dst_len) return -1;
        memcpy(dst + pos, out_line, olen);
        pos += olen;
        dst[pos++] = '\n';
        dst[pos] = '\0';
    }

    return 0;
}

/* ═══════════════════════════════════════════
   CLI
   ═══════════════════════════════════════════ */

static void print_usage(void) {
    printf("nml_fmt — NML syntax converter\n\n");
    printf("Usage:\n");
    printf("  nml_fmt --to-symbolic  <program.nml> [--compact] [-o out.nml]\n");
    printf("  nml_fmt --to-classic   <program.nml> [--compact] [-o out.nml]\n");
    printf("  nml_fmt --to-verbose   <program.nml> [-o out.nml]\n");
    printf("  nml_fmt --list-opcodes\n\n");
    printf("Converts opcodes and register names between NML syntax variants.\n");
    printf("Comments are preserved verbatim. Unknown tokens are passed through.\n");
    printf("\n--compact: strip comments/blank lines, join instructions with ¶ (U+00B6).\n");
    printf("           The NML runtime parses ¶-delimited programs natively.\n");
}

static void list_opcodes(void) {
    printf("%-10s  %-6s  %s\n", "CLASSIC", "SYM", "VERBOSE");
    printf("%-10s  %-6s  %s\n", "-------", "---", "-------");
    for (const OpcodeRow *r = OPCODE_TABLE; r->classic; r++)
        printf("%-10s  %-6s  %s\n", r->classic, r->sym, r->verbose);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nml_fmt: cannot open '%s'\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

/* Compact a converted source: strip blank and comment-only lines, join with ¶.
 * Writes result into a newly malloc'd buffer (caller must free). Returns NULL on OOM. */
static char *compact_output(const char *src) {
    /* Pilcrow: UTF-8 0xC2 0xB6 */
    const char PILCROW[] = "\xc2\xb6";
    size_t out_cap = strlen(src) + 4;
    char *out = malloc(out_cap);
    if (!out) return NULL;
    size_t out_pos = 0;
    int first = 1;

    const char *p = src;
    while (*p) {
        const char *end = strchr(p, '\n');
        size_t line_len = end ? (size_t)(end - p) : strlen(p);

        /* Trim leading whitespace */
        const char *line_start = p;
        while (line_start < p + line_len && (*line_start == ' ' || *line_start == '\t'))
            line_start++;

        /* Skip blank lines and comment-only lines */
        if (line_start >= p + line_len || *line_start == ';') {
            p += (size_t)(end ? (end - p) + 1 : strlen(p));
            continue;
        }

        /* Strip inline comment (;...) */
        const char *line_end = p + line_len;
        {
            int in_q = 0;
            const char *cp = line_start;
            while (cp < line_end) {
                if (*cp == '"') in_q = !in_q;
                if (*cp == ';' && !in_q) { line_end = cp; break; }
                cp++;
            }
        }

        /* Strip trailing whitespace */
        while (line_end > line_start && (*(line_end-1) == ' ' || *(line_end-1) == '\t' || *(line_end-1) == '\r'))
            line_end--;

        /* Skip if nothing left after stripping comment */
        if (line_end <= line_start) {
            p += (size_t)(end ? (end - p) + 1 : strlen(p));
            continue;
        }

        size_t tok_len = (size_t)(line_end - line_start);
        size_t need = out_pos + tok_len + 3 /* pilcrow=2 + nul */;
        if (need > out_cap) {
            out_cap = need * 2;
            char *tmp = realloc(out, out_cap);
            if (!tmp) { free(out); return NULL; }
            out = tmp;
        }

        if (!first) {
            out[out_pos++] = PILCROW[0];
            out[out_pos++] = PILCROW[1];
        }
        memcpy(out + out_pos, line_start, tok_len);
        out_pos += tok_len;
        first = 0;

        p += (size_t)(end ? (end - p) + 1 : strlen(p));
    }

    out[out_pos] = '\0';
    return out;
}

int nml_fmt_main(int argc, char **argv) {
    NmlSyntax to = NML_SYNTAX_SYMBOLIC;
    const char *input_path = NULL;
    const char *output_path = NULL;
    int list_ops = 0;
    int compact = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--to-symbolic") == 0) {
            to = NML_SYNTAX_SYMBOLIC;
        } else if (strcmp(argv[i], "--to-classic") == 0) {
            to = NML_SYNTAX_CLASSIC;
        } else if (strcmp(argv[i], "--to-verbose") == 0) {
            to = NML_SYNTAX_VERBOSE;
        } else if (strcmp(argv[i], "--compact") == 0) {
            compact = 1;
        } else if (strcmp(argv[i], "--list-opcodes") == 0) {
            list_ops = 1;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (argv[i][0] != '-') {
            input_path = argv[i];
        } else {
            fprintf(stderr, "nml_fmt: unknown option '%s'\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    if (list_ops) { list_opcodes(); return 0; }

    if (!input_path) { print_usage(); return 1; }

    char *src = read_file(input_path);
    if (!src) return 1;

    size_t src_len = strlen(src);
    /* Output can be larger than input (classic→verbose expands tokens).
     * 8× is safe for any realistic program. */
    size_t dst_len = src_len * 8 + 4096;
    char *dst = malloc(dst_len);
    if (!dst) { free(src); fprintf(stderr, "nml_fmt: out of memory\n"); return 1; }

    int rc = nml_fmt_source(src, dst, dst_len, to);
    free(src);

    if (rc != 0) {
        fprintf(stderr, "nml_fmt: conversion failed (output buffer too small?)\n");
        free(dst);
        return 1;
    }

    char *out = dst;
    char *compact_buf = NULL;
    if (compact) {
        compact_buf = compact_output(dst);
        if (!compact_buf) {
            fprintf(stderr, "nml_fmt: compact failed (out of memory)\n");
            free(dst);
            return 1;
        }
        out = compact_buf;
    }

    if (output_path) {
        FILE *f = fopen(output_path, "w");
        if (!f) { fprintf(stderr, "nml_fmt: cannot write '%s'\n", output_path); free(dst); free(compact_buf); return 1; }
        fputs(out, f);
        if (!compact) fputc('\n', f);  /* ensure trailing newline for multi-line files */
        fclose(f);
        fprintf(stderr, "nml_fmt: written to %s\n", output_path);
    } else {
        fputs(out, stdout);
        if (!compact) { /* multi-line output already ends with \n */ }
        else printf("\n");
    }

    free(compact_buf);
    free(dst);
    return 0;
}

int main(int argc, char **argv) {
    return nml_fmt_main(argc, argv);
}
