---
name: nml-codegen
description: Generate, debug, and reason about NML (Neural Machine Language) programs. Use when the user asks to write NML, generate NML code, create an NML program, transpile to NML, debug NML, explain NML, fix NML errors, or work with .nml files. Also triggers on NML opcode names (LEAF, TACC, CMPF, HALT) or symbolic syntax (∎, ∑, ⋈, ◼, ↓, ↑).
---

# NML Code Generation

## Workflow

Follow this sequence when generating NML. Skip steps that don't apply.

1. **Understand the task** — tax calculation, neural net, general algorithm, M2M composition?
2. **Load the spec** — call `nml_spec_lookup` for the relevant section (instruction_set, tree_model, symbolic_syntax, etc.)
3. **Grab examples** — call `nml_library_lookup` or read from `programs/` for reference patterns
4. **Generate** — write the NML following the patterns and conventions below
5. **Validate** — call `nml_validate` with the program (or check manually against the rules below)
6. **Execute** — call `nml_execute` with test data to verify outputs
7. **Iterate** — if validation/execution fails, read the error, fix, re-validate

## Architecture

NML is a 62-opcode register machine. 16 registers (R0–RF), 32-bit fixed-width encoding, three syntax modes.

### Registers

| Register | Greek | Purpose | Convention |
|----------|-------|---------|------------|
| R0–R9 | ι κ λ μ ν ξ ο π ρ ς | General purpose | R0/ι = primary input, R7/π = taxable income |
| RA | α | Accumulator | Tax result, running totals |
| RB | β | General | Auxiliary storage |
| RC | γ | Scratch | Leaf values, temporaries |
| RD | δ | Counter | Loop counters |
| RE | φ | Flag | Set by CMPF/CMP/CMPI, read by JMPT/JMPF |
| RF | ψ | Stack | Call stack pointer (managed by CALL/RET) |

### Three Syntaxes (identical bytecode)

| Classic | Symbolic | Verbose | Meaning |
|---------|----------|---------|---------|
| `LEAF RC #100` | `∎ γ #100` | `SET_VALUE SCRATCH #100` | Store scalar |
| `TACC RA RA RC` | `∑ α α γ` | `ACCUMULATE ACC ACC SCRATCH` | Add |
| `ST RA @tax` | `↑ α @tax` | `STORE ACC @tax` | Store to memory |
| `HALT` | `◼` | `STOP` | Terminate |

Use symbolic for compact output, classic for readability. Ask the user's preference if unclear.

## Jump Offset Rules (Critical)

Jumps are **relative offsets from the current PC**. After applying the offset, PC increments by 1.

- `JMPF #2` at line 5 → skips to line 8 (5 + 2 + 1)
- `JUMP #-5` at line 20 → goes to line 16 (20 + (-5) + 1)
- `JMPT #-8` at line 12 → goes to line 5 (12 + (-8) + 1)

**Count carefully.** Wrong jump offsets are the #1 source of NML bugs. When writing jumps:
1. Number every instruction (0-indexed)
2. Calculate: `offset = target_line - current_line - 1`
3. Double-check by reversing: `current + offset + 1 = target`

## Essential Patterns

### Pattern 1: Tax Bracket Cascade (most common)

Progressive brackets descend from highest threshold. Each tier: compare → base tax + threshold → marginal computation → accumulate.

```
; Symbolic — single filing status bracket
□  α  #[1]                    ; allocate accumulator
⋈  φ  π  #0  #632750.0       ; income < 632750?
↘  #50                        ; no → next lower bracket
⋈  φ  π  #0  #256925.0       ; income < 256925?
↘  #42                        ; ...cascading down...
⋈  φ  π  #0  #6400.0         ; income < 6400 (lowest)?
↘  #2                         ; no → next tier
∗  α  π  #0.0                 ; tier 0: tax = income * 0%
→  #41                        ; jump to store

; Each tier follows this 6-instruction pattern:
∎  α  #1192.5                 ; base tax at this tier floor
∎  γ  #18325.0                ; threshold
⊖  ρ  π  γ                    ; marginal = income - threshold
∗  ρ  ρ  #0.12                ; marginal * rate
∑  α  α  ρ                    ; total = base + marginal
→  #N                         ; jump to store

; Final:
↑  α  @tax_amount             ; store result
◼                             ; halt
```

Key: thresholds descend (highest first), `↘` offsets skip past remaining tiers.

### Pattern 2: Flat Rate

Simple: multiply gross by rate, store.

```
↓  ι  @gross_pay
↓  μ  @is_exempt
⋈  φ  μ  #0  #0.5            ; exempt check
↘  #4                         ; skip if exempt
□  α  #[1]
∗  γ  ι  #0.062              ; gross * rate
∑  α  α  γ
↑  α  @tax_amount
◼
```

### Pattern 3: Loop with Backward Jump

Counter pattern: init → body → increment → compare → backward jump.

```
LEAF  RD #0.0                 ; counter = 0
LEAF  R5 #20.0                ; limit
; --- loop body here (instruction N) ---
LEAF  RC #1.0
TACC  RD RD RC                ; counter++
CMP   RD R5                   ; counter < limit?
JMPT  #-(body_size + 4)       ; back to instruction N
HALT
```

Offset formula for backward jump: `-(instructions_in_body + 4)` where 4 = the increment/compare/jump overhead.

### Pattern 4: Neural Network Layer

```
LD    R0 @input
LD    R1 @weights
LD    R2 @bias
MMUL  R3 R0 R1                ; forward pass
MADD  R3 R3 R2                ; add bias
RELU  R3 R3                   ; activation
ST    R3 @output
HALT
```

### Pattern 5: Conditional Branch (if/else)

```
CMPI  RE R0 #50.0             ; R0 < 50?
JMPF  #2                      ; false → else branch
; true branch (2 instructions)
...
JUMP  #1                      ; skip else
; else branch (1 instruction)
...
```

The `JUMP` after the true branch skips the else. Count instructions precisely.

### Pattern 6: Filing Status Dispatch

Tax programs switch on filing_status (1=single, 2=married, 3=married separate, 4=HoH):

```
↓  κ  @filing_status
⋈  φ  κ  #0  #1.5            ; < 1.5 → single
↘  #N                         ; jump past single block
; ... single bracket cascade ...
⋈  φ  κ  #0  #2.5            ; < 2.5 → married
↘  #M
; ... married bracket cascade ...
; fallback: head of household / married separate
```

### Pattern 7: M2M Metadata

Programs should self-describe with META:

```
META  @name       "fit_single"
META  @domain     "tax"
META  @input      gross_pay   currency
META  @output     tax_amount  currency
```

## Data File Format (.nml.data)

```
@gross_pay shape=1 data=3846.15
@filing_status shape=1 data=1.0
@is_exempt shape=1 data=0.0
@thresholds shape=7 data=6400.0,18325.0,54875.0,109750.0,203700.0,256925.0,632750.0
```

For tax programs, the simplified format also works: `gross_pay 3846.15`

## Common Mistakes to Avoid

1. **Wrong jump offsets** — always count instructions and verify with the formula
2. **Missing HALT/◼** — every program must terminate with HALT
3. **Uninitialized accumulator** — use `ALLC RA #[1]` or `□ α #[1]` before accumulating
4. **Wrong comparison direction** — `CMPF` sets flag if `Rs[feat] < thresh`; `JMPF` jumps when flag is FALSE (value >= threshold)
5. **Forgetting exempt check** — tax programs should check `@is_exempt` and skip calculation if true
6. **Register clobbering** — track which registers are in use; don't overwrite live values
7. **Brackets in wrong order** — descend from highest threshold; the first matching tier wins

## MCP Tools Available

| Tool | When to Use |
|------|-------------|
| `nml_spec_lookup` | Before writing NML — get opcode details, register table, syntax |
| `nml_library_lookup` | Get golden examples for a tax type (FIT, SIT, FICA, CITY, etc.) |
| `nml_transpile` | Generate NML from a jurisdiction key (uses the deterministic transpiler) |
| `nml_validate` | Check grammar + semantics + execution of generated NML |
| `nml_execute` | Run NML against test data and verify outputs |
| `nml_intent` | Parse natural language into structured tax parameters |
| `nml_scan` | List available jurisdictions and tax types |
| `nml_compact` | Convert multi-line NML to single-line ¶-delimited form |
| `nml_format` | Expand compact NML back to readable multi-line form |

## Validation Loop

After generating NML, always validate:

```
1. Call nml_validate(nml_program=..., jurisdiction_key=..., mode="full")
2. If grammar errors → fix syntax (wrong opcode, register name, missing operand)
3. If semantic errors → fix structure (bracket order, jump targets, register usage)
4. If execution errors → fix logic (wrong offsets, missing HALT, bad data format)
5. Call nml_execute with test data to confirm correct output values
```

For tax programs, validate against known inputs: $75,000 single filer is a good baseline.

## Compact Form

Symbolic NML supports a single-line compact representation using `¶` (U+00B6, pilcrow — from Greek *paragraphos*) as the instruction delimiter. The C runtime natively parses both forms.

Multi-line:
```
↓  ι  @gross_pay
∗  γ  ι  #0.062
∑  α  α  γ
↑  α  @tax_amount
◼
```

Compact:
```
↓ ι @gross_pay¶∗ γ ι #0.062¶∑ α α γ¶↑ α @tax_amount¶◼
```

The pair `¶` (next instruction) and `◼` (halt/end) bookend the instruction stream. Use `nml_compact` to collapse and `nml_format` to expand. Prefer compact form when emitting NML for machine consumption (agent pipes, M2M messages, API payloads). Use formatted form for human review and debugging.

## Quick Reference

For the complete opcode table, symbolic mappings, and M2M spec, see:
- `docs/NML_SPEC.md` — full specification
- `docs/NML_M2M_Spec.md` — machine-to-machine extensions
- `output/nml-library-symbolic/` — 7,500+ golden NML programs
- `programs/` — annotated example programs (fibonacci, fizzbuzz, primes, neural nets)
