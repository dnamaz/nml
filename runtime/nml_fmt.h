/*
 * nml_fmt.h — NML Syntax Converter
 *
 * Converts NML source between classic, symbolic (Unicode), and verbose syntax.
 * Standalone — does not depend on nml.c internals.
 *
 * Build:  gcc -O2 -o nml_fmt runtime/nml_fmt.c
 * Usage:  ./nml_fmt --to-symbolic  program.nml
 *         ./nml_fmt --to-classic   program.nml
 *         ./nml_fmt --to-verbose   program.nml
 *         ./nml_fmt --list-opcodes
 */

#ifndef NML_FMT_H
#define NML_FMT_H

#include <stddef.h>

typedef enum {
    NML_SYNTAX_CLASSIC  = 0,
    NML_SYNTAX_SYMBOLIC = 1,
    NML_SYNTAX_VERBOSE  = 2,
} NmlSyntax;

/* Convert a single opcode token to target syntax.
 * Returns the converted string, or NULL if the token is not a known opcode.
 * The returned pointer is into a static table — do not free. */
const char *nml_opcode_convert(const char *token, NmlSyntax to);

/* Convert a single register token (R0, RA, ι, α, ACCUMULATOR …) to target syntax.
 * Returns the converted string, or NULL if the token is not a known register.
 * Symbolic target uses Greek aliases; classic target uses R0–RV; verbose uses
 * named aliases where defined (ACCUMULATOR etc.), classic otherwise. */
const char *nml_reg_convert(const char *token, NmlSyntax to);

/* Convert an entire NML source program to target syntax.
 * Comments are preserved verbatim. Blank lines are preserved.
 * Returns 0 on success, -1 if dst is too small. */
int nml_fmt_source(const char *src, char *dst, size_t dst_len, NmlSyntax to);

/* CLI entry point — call from main() if desired. */
int nml_fmt_main(int argc, char **argv);

#endif /* NML_FMT_H */
