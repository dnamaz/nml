# NML v0.6 — Machine-to-Machine Extensions Specification

## Overview

NML v0.6 adds 13 new instructions and a type annotation system for machine-to-machine communication. These extensions enable NML programs to be self-describing, composable, differentially updatable, cryptographically signed, consensus-driven, embedding-aware, and tensor-indexable.

All existing v0.5 programs run unchanged. The new instructions use 13 of the 15 remaining opcode slots (6-bit field supports 64 total; v0.5 uses 49), leaving 2 spare for future use.

### Implementation Status (v0.8.1)

| Opcode | Runtime Status | Notes |
|--------|---------------|-------|
| META | Implemented | Assembly-time metadata, `--describe` flag |
| FRAG/ENDF | **Implemented** | Assembly-time fragment scoping via `vm_resolve_fragments()` |
| LINK | **Implemented** | Inline expansion at assembly time |
| PTCH | **Implemented** | CLI tool: `nml-crypto program.nml --patch patch.ptch` |
| SIGN | **Implemented** | CLI: `nml-crypto --sign program.nml --key <hex> --agent <name>` (HMAC-SHA256) |
| VRFY | **Implemented** | Loader guard: auto-verifies before assembly, TRAP on failure |
| VOTE | Implemented | Runtime opcode: median, mean, quorum, min, max |
| PROJ | Implemented | Runtime opcode: matmul + L2 normalize |
| DIST | Implemented | Runtime opcode: cosine, euclidean, dot product |
| GATH | Implemented | Runtime opcode: tensor index lookup |
| SCAT/SCTR | Implemented | Runtime opcode: tensor index write |

Build with `make nml-crypto` (`-DNML_CRYPTO`) for SIGN/VRFY/PTCH. Without it, SIGN/VRFY are no-ops (backward compatible).

End-to-end demo: `bash demos/distributed_fraud.sh`

## New Instructions Summary

| Opcode | Classic | Symbolic | Verbose | Category | Purpose |
|--------|---------|----------|---------|----------|---------|
| 50 | `META` | `§` | `METADATA` | Structure | Program metadata declaration |
| 51 | `FRAG` | `◆` | `FRAGMENT` | Structure | Open named fragment scope |
| 52 | `ENDF` | `◇` | `END_FRAGMENT` | Structure | Close fragment scope |
| 53 | `LINK` | `⊕` | `IMPORT` | Structure | Import named fragment |
| 54 | `PTCH` | `⊿` | `PATCH` | Communication | Differential program update |
| 55 | `SIGN` | `✦` | `SIGN_PROGRAM` | Communication | Cryptographic signature |
| 56 | `VRFY` | `✓` | `VERIFY_SIGNATURE` | Communication | Signature verification |
| 57 | `VOTE` | `⚖` | `CONSENSUS` | Collective | Multi-agent consensus |
| 58 | `PROJ` | `⟐` | `PROJECT` | Latent | Embedding projection |
| 59 | `DIST` | `⟂` | `DISTANCE` | Latent | Embedding distance |
| 60 | `GATH` | `⊃` | `GATHER` | Data | Tensor index lookup |
| 61 | `SCAT` | `⊂` | `SCATTER` | Data | Tensor index write |

Total instructions: 62 (35 core + 14 existing extensions + 13 M2M extensions).

---

## Extension 1: Self-Describing Programs (META)

### Syntax

```
META  @key  value
§     @key  value
```

META declares program metadata. The runtime parses META lines during assembly and stores them in a program descriptor. During execution, META is a no-op.

### Valid Keys

| Key | Value Format | Required | Description |
|-----|-------------|----------|-------------|
| `@name` | string | No | Program identifier (e.g., jurisdiction key) |
| `@version` | string | No | Version string |
| `@input` | name type [description] | No | Input declaration (repeatable) |
| `@output` | name type [description] | No | Output declaration (repeatable) |
| `@invariant` | expression | No | Runtime assertion (repeatable) |
| `@provenance` | string | No | Source document reference |
| `@author` | string | No | Producing agent or tool |
| `@domain` | string | No | Domain identifier (e.g., "tax", "finance") |
| `@license` | string | No | License identifier |
| `@created` | ISO-8601 | No | Creation timestamp |

### Example

```
§  @name         "00-000-0000-FIT-000"
§  @version      "2025.1"
§  @domain       "tax"
§  @input        gross_pay       currency    "Annual gross pay"
§  @input        filing_status   category    "1=Single, 2=MFJ, 3=HoH"
§  @input        is_exempt       bool        "Exempt from withholding"
§  @output       tax_amount      currency    "Federal income tax"
§  @invariant    "tax_amount >= 0"
§  @invariant    "is_exempt == 1 => tax_amount == 0"
§  @provenance   "IRS Publication 15-T, 2025"
§  @author       "domain_transpiler v2.3"
§  @created      "2025-06-15T00:00:00Z"
↓  ι  @gross_pay
...
```

### CLI

```bash
./nml program.nml --describe    # Print descriptor, don't execute
```

### Encoding

META uses opcode 50. The key and value are stored as string data in the program's metadata section, not in the instruction stream's immediate field. The assembler extracts META lines into a separate descriptor table.

---

## Extension 2: Typed Tensor Annotations

### Syntax

Type annotations appear after register names, separated by `:`.

```
↓  ι:currency     @gross_pay
∗  α:currency     ι  #0.062
↑  α:currency     @tax_amount
```

Types are advisory — they do not change runtime behavior by default. The grammar validator and semantic analyzer use them for static checking. Compile with `-DNML_TYPE_CHECK` to enable runtime enforcement.

### Type Taxonomy

| Type | Code | Description | Valid Operations |
|------|------|-------------|-----------------|
| `float` | 0 | Default untyped | All |
| `currency` | 1 | Dollar/monetary amounts | +, -, * ratio, / count |
| `ratio` | 2 | Rates, percentages [0,1] | * currency, + ratio, - ratio |
| `category` | 3 | Enum/categorical | ==, !=, compare |
| `count` | 4 | Integer counts | +, -, *, / |
| `bool` | 5 | Boolean 0/1 | ==, !=, AND, OR |
| `embedding` | 6 | High-dim vector | PROJ, DIST, dot product |
| `probability` | 7 | Distribution [0,1] summing to 1 | SOFT output, sample |

### Type Compatibility Rules

| Operation | Left | Right | Result |
|-----------|------|-------|--------|
| `SCLR`/`∗` (multiply) | currency | ratio | currency |
| `SCLR`/`∗` (multiply) | ratio | currency | currency |
| `MADD`/`∑` (add) | currency | currency | currency |
| `MSUB`/`⊖` (subtract) | currency | currency | currency |
| `SDIV`/`÷` (divide) | currency | count | currency |
| `CMPF`/`⋈` (compare) | any | any | bool |
| `SOFT`/`Σ` (softmax) | float | — | probability |
| `PROJ`/`⟐` (project) | float | float | embedding |
| `DIST`/`⟂` (distance) | embedding | embedding | float |
| `VOTE`/`⚖` median (#0) | any | — | same as input |
| `VOTE`/`⚖` mean (#1) | any | — | same as input |
| `VOTE`/`⚖` quorum (#2) | any | — | bool (Rd); consensus value → RC, same type as input |
| `VOTE`/`⚖` min (#3) | any | — | same as input |
| `VOTE`/`⚖` max (#4) | any | — | same as input |

Incompatible operations (e.g., `currency * currency`, `category + currency`) produce a semantic warning when type-checked.

### Data File Extension

```
@gross_pay shape=1 dtype=f64 stype=currency data=185000.00
@filing_status shape=1 dtype=i32 stype=category data=1
```

The `stype=` field is optional and defaults to `float`.

---

## Extension 3: Compositional Fragments (FRAG, ENDF, LINK)

### Syntax

```
FRAG  name          ; Open named fragment
◆     name

ENDF                ; Close current fragment
◇

LINK  @name         ; Import fragment by name
⊕    @name
```

### Semantics

- FRAG opens a named scope. All instructions until the matching ENDF belong to this fragment.
- Fragments can contain META declarations (for input/output specification).
- LINK copies a named fragment's instructions inline at assembly time.
- Fragments are independently validatable and executable (`--fragment NAME`).
- A program without FRAG/ENDF is treated as a single implicit fragment.

### Example

```
◆  fica_tax
§  @input   gross_pay   currency
§  @output  fica_amount currency
↓  ι  @gross_pay
□  α  #[1]
⋈  φ  ι  #0  #176100.0
↗  #4
∎  γ  #176100.0
∗  γ  γ  #0.062
∑  α  α  γ
→  #2
∗  γ  ι  #0.062
∑  α  α  γ
↑  α  @fica_amount
◇

◆  medicare_tax
§  @input   gross_pay      currency
§  @output  medicare_amount currency
↓  ι  @gross_pay
∗  α  ι  #0.0145
↑  α  @medicare_amount
◇

◆  total_payroll
⊕  @fica_tax
⊕  @medicare_tax
∑  γ  α  β
↑  γ  @total_payroll_tax
◇
```

### CLI

```bash
./nml program.nml --fragment fica_tax     # Execute only fica_tax fragment
./nml program.nml data.nml.data           # Execute all (LINK resolved)
```

---

## Extension 4: Differential Programs (PTCH)

### Syntax

```
PTCH  @base   hash
PTCH  @set    line  "instruction"
PTCH  @del    line
PTCH  @ins    line  "instruction"
PTCH  @end
```

### Semantics

- `@base` declares the SHA-256 hash of the program being patched.
- `@set` replaces line N with the given instruction.
- `@del` deletes line N.
- `@ins` inserts a new instruction at line N (shifting subsequent lines).
- `@end` closes the patch block.
- The runtime validates the base hash before applying.

### Example

```
⊿  @base  sha256:7fe8412b7c3ecabd6dfdc439f0a8abb23059b871
⊿  @set   9   "∎  γ  #8800.00"
⊿  @set   13  "∎  γ  #13200.00"
⊿  @set   35  "⋈  φ  π  #0  #6550.00"
⊿  @end
```

This patch updates the FIT standard deductions and first bracket threshold — a few hundred bytes instead of the full 201-line program.

### CLI

```bash
./nml --patch base.nml patch.nml          # Apply patch, execute result
./nml --patch base.nml patch.nml --out patched.nml  # Apply and save
```

---

## Extension 5: Cryptographic Signing (SIGN, VRFY)

### Syntax

```
SIGN  agent=NAME  key=ALGORITHM:PUBKEY_HEX  sig=SIGNATURE_HEX
✦     agent=NAME  key=ALGORITHM:PUBKEY_HEX  sig=SIGNATURE_HEX

VRFY  @hash  @signer
✓     @hash  @signer
```

### Semantics

- SIGN is a metadata instruction (no-op at execution). It records a cryptographic signature over the program content (excluding the SIGN line itself).
- VRFY is an executable instruction. It computes the SHA-256 hash of the program, looks up the SIGN metadata, and verifies the signature. If verification fails, VRFY triggers a TRAP.
- Supported algorithms: `ed25519` (default), `hmac-sha256`.

### Example

```
✦  agent=transpiler_v2.3  key=ed25519:a1b2c3d4...  sig=e5f6a7b8...
§  @name  "00-000-0000-FIT-000"
↓  ι  @gross_pay
...
✓  @self  @transpiler_v2.3
◼
```

### Compile-Time Control

```bash
make nml-crypto                                        # With HMAC-SHA256 verification
make nml                                               # SIGN=nop, VRFY=pass (default)

# Or directly:
gcc -O2 -DNML_CRYPTO -o nml-crypto runtime/nml.c -lm  # With verification
gcc -O2 -o nml runtime/nml.c -lm                       # SIGN=nop, VRFY=pass
```

### Signing and Verification (CLI)

```bash
# Sign a program
./nml-crypto --sign program.nml --key deadbeef01020304 --agent authority_v1 > signed.nml

# Execute signed program (auto-verifies)
./nml-crypto signed.nml data.nml.data
# → [NML] Signature verified — signed by agent 'authority_v1'

# Tampered programs are rejected
./nml-crypto tampered.nml data.nml.data
# → [NML] SIGNATURE VERIFICATION FAILED (code -3) — refusing to execute
```

**Note:** The current implementation uses HMAC-SHA256 (symmetric key). The key is embedded in the SIGN header. For production use with untrusted agents, Ed25519 (asymmetric) should be used — the crypto header (`runtime/nml_crypto.h`) is designed to be extended.

---

## Extension 6: Consensus (VOTE)

### Syntax

```
VOTE  Rd  Rs  #strategy  [#threshold]
⚖    Rd  Rs  #strategy  [#threshold]
```

### Strategies

| Code | Name | Description |
|------|------|-------------|
| 0 | median | Rd = median of Rs elements |
| 1 | mean | Rd = arithmetic mean of Rs elements |
| 2 | quorum | Rd = 1 if >= threshold elements agree within tolerance |
| 3 | min | Rd = minimum of Rs elements |
| 4 | max | Rd = maximum of Rs elements |

### Semantics

- Rs must be a 1D tensor of shape [N] containing results from N agents.
- Rd receives a scalar result (shape [1]).
- Quorum (strategy 2) groups values within 0.01 tolerance. If any group has >= threshold members, Rd = 1 and the consensus value is stored in RC (scratch register). Otherwise Rd = 0.

### Example

```
↓  ι  @agent_results      ; [6200.00, 6200.01, 6199.98, 6200.00, 6150.00]
⚖  α  ι  #0               ; α = 6200.00 (median)
⚖  φ  ι  #2  #4           ; φ = 1 (4 of 5 agree within 0.01)
```

---

## Extension 7: Latent Space Primitives (PROJ, DIST)

### Syntax

```
PROJ  Rd  Rs  Rmatrix       ; Rd = normalize(Rs @ Rmatrix)
⟐    Rd  Rs  Rmatrix

DIST  Rd  Rs1  Rs2  #metric  ; Rd = distance(Rs1, Rs2)
⟂    Rd  Rs1  Rs2  #metric
```

### Distance Metrics

| Code | Name | Formula |
|------|------|---------|
| 0 | cosine | 1 - (a . b) / (|a| * |b|) |
| 1 | euclidean | sqrt(sum((a-b)^2)) |
| 2 | dot_product | a . b |

### Semantics

- PROJ performs matrix multiply followed by L2 normalization: `Rd = (Rs @ Rmatrix) / |Rs @ Rmatrix|`. Output has unit norm.
- DIST computes distance/similarity between two vectors. Both inputs should have the same shape.
- PROJ output type is `embedding`. DIST inputs should be `embedding` type.

### Example

```
↓  ι  @feature_vector       ; shape [1, 128]
↓  κ  @projection_matrix    ; shape [128, 64]
⟐  α  ι  κ                  ; α = normalized projection, shape [1, 64]

↓  β  @other_embedding      ; shape [1, 64]
⟂  γ  α  β  #0              ; γ = cosine distance
```

---

## Encoding

All new opcodes use the existing 32-bit instruction word:

```
[OPCODE: 6 bits][Rd: 4 bits][Rs1: 4 bits][Rs2/imm: 18 bits]
```

META, FRAG, ENDF, LINK, PTCH, and SIGN carry string operands that are stored in a side table (not in the 18-bit immediate). The assembler indexes string operands and stores the index in the immediate field.

VOTE, PROJ, and DIST use the standard register + immediate encoding.

## Extension 8: Tensor Indexing (GATH, SCAT)

### Syntax

```
GATH  Rd  Rs  Ridx       ; Rd = Rs[Ridx[0]] — index lookup
⊃     Rd  Rs  Ridx

SCAT  Rs  Rd  Ridx       ; Rd[Ridx[0]] = Rs[0] — index write
⊂     Rs  Rd  Ridx
```

### Semantics

- GATH reads element at position Ridx[0] from tensor Rs into scalar Rd. Out-of-bounds triggers TRAP.
- SCAT writes scalar Rs[0] into tensor Rd at position Ridx[0]. Out-of-bounds triggers TRAP.
- Both enable tensor-table bracket lookups: store thresholds/rates as tensors, find the bracket index via LOOP+CMP, then GATH the rate and base_tax.

### Example: Bracket Table Lookup

```
LD    R3 @thresholds      ; shape [7]
LD    R4 @rates           ; shape [7]
; ... find bracket_index in R6 ...
GATH  R7 R4 R6            ; R7 = rates[bracket_index]
GATH  R8 R3 R6            ; R8 = thresholds[bracket_index]
```

---

## Compact Wire Format

When NML programs are transmitted between agents (via M2M messages, JSON APIs, or embedded in data structures), the **compact form** is the recommended serialization.

Compact form uses `¶` (U+00B6, pilcrow — from Greek *paragraphos*) as the instruction delimiter, producing a single-line string with no newlines:

```
§ @name "00-000-0000-FIT-000"¶§ @domain "tax"¶↓ ι @gross_pay¶∗ γ ι #0.062¶∑ α α γ¶↑ α @tax_amount¶◼
```

### Properties

- **No newlines** — embeds directly in JSON string fields, database columns, message queues
- **Native parsing** — the NML runtime accepts `¶` as an instruction delimiter alongside `\n`
- **Lossless round-trip** — `compact(format(program))` preserves instruction semantics (comments are stripped)
- **META preserved** — META declarations are instructions like any other; they survive compaction
- **FRAG/ENDF preserved** — fragment boundaries are maintained in compact form
- **2 bytes per delimiter** — `¶` is U+00B6 (UTF-8: `C2 B6`), 1 byte more than `\n`

### Agent Message Envelope

When wrapping NML in an agent message, the compact form fits naturally:

```json
{
  "header": { "source": "transpiler", "target": "validator", "timestamp": "..." },
  "payload": "↓ ι @gross_pay¶∗ γ ι #0.062¶∑ α α γ¶↑ α @tax_amount¶◼",
  "syntax": "symbolic-compact"
}
```

The `syntax` field indicates format: `symbolic-compact`, `classic-compact`, or `verbose-compact`. Multi-line forms use `symbolic`, `classic`, or `verbose`.

### Conversion

```bash
python3 nml_format.py compact  program.nml    # multi-line → single line
python3 nml_format.py format   "$COMPACT"     # single line → multi-line
```

---

## Version History Update

| Version | Core | Extensions | M2M | Total | Key Changes |
|---------|------|-----------|-----|-------|-------------|
| v0.5 | 35 | 14 | 0 | 49 | Per-tensor data types |
| v0.6 | 35 | 14 | 13 | 62 | META, typed registers, fragments, patches, signing, consensus, latent space, tensor indexing |
| v0.6.3 | 35 | 14 | 13 | 62 | Compact wire format: `¶` (pilcrow) as native instruction delimiter for single-line NML |
