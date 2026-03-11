# NML — Implementation Document

## What Was Built

This document describes the complete implementation of the NML (Neural Machine Language) system: a 71-instruction machine language for AI workloads with a ~51KB C runtime, three transpiler pipelines (XGBoost, deterministic rules, and domain rule transpilation), a tri-syntax system (classic/symbolic/verbose), a training data generator for LLM fine-tuning, and a general-purpose extension (NML-G) enabling console I/O and integer math.

---

## System Architecture

```mermaid
flowchart TD
    subgraph sources [Source Data]
        XGB[Trained XGBoost Model]
        JSON[JSON Tax Rules]
        DomainRules[Domain Rule Files]
        NN[Neural Network Weights]
    end

    subgraph transpilers [Transpiler Layer]
        XGBTrans[XGBoost Transpiler — tax_pipeline.py]
        RuleTrans[Rule Transpiler — rule_transpiler.py]
        DomainTrans[Domain Transpiler — rule files to NML]
    end

    subgraph nml_programs [NML Programs]
        TaxCalc[tax_calculator.nml — 1,536 instr]
        RulesCalc[tax_rules_calc.nml — 330 instr]
        FIT[fit_2025.nml — 479 instr]
        CASIT[ca_sit_2025.nml — 233 instr]
        FICA[fica_2025.nml — 16 instr]
        Anomaly[anomaly_detector.nml — 18 instr]
        Extensions[extension_demo.nml — 38 instr]
    end

    subgraph runtime [NML Runtime]
        NMLFull["nml (full) — ~68 KB, 71 instructions"]
        NMLCore["nml-core — ~50 KB, 35 instructions"]
        NMLGP["nml-gp — ~68 KB, expanded limits"]
    end

    subgraph training [Training Data Pipeline]
        Validate[Validation Pipeline]
        TrainGen[Training Generator]
        Dataset["Training Dataset — 96,710 pairs"]
    end

    subgraph agents [Agent Services Layer]
        TranspilerSvc["Transpiler Service — port 8083"]
        ValidatorSvc["Validation Service — port 8084"]
        EngineSvc["Execution Service — port 8085"]
        Gateway["Agent Gateway — port 8082"]
        Orchestrator["Intent Router + Pipeline Executor"]
        GrammarVal["Grammar Validator — nml_grammar.py"]
        SemanticVal["Semantic Analyzer — nml_semantic.py"]
        AnomalyDet["Anomaly Detector — nml_anomaly.py"]
    end

    XGB --> XGBTrans --> TaxCalc
    JSON --> RuleTrans --> RulesCalc
    DomainRules --> DomainTrans --> FIT & CASIT & FICA

    NN --> Anomaly
    NN --> Extensions

    TaxCalc & RulesCalc & FIT & Anomaly --> NMLFull
    TaxCalc & RulesCalc & FIT & Anomaly --> NMLCore

    DomainTrans --> Validate
    Validate --> TrainGen --> Dataset

    DomainTrans --> TranspilerSvc
    Validate --> ValidatorSvc
    NMLFull --> EngineSvc
    GrammarVal --> ValidatorSvc
    SemanticVal --> ValidatorSvc
    Gateway --> TranspilerSvc & ValidatorSvc & EngineSvc
    Orchestrator --> Gateway
```

---

## Component 1: NML Runtime (`nml.c`)

**~1,900 lines of C99.** Single-file runtime implementing the full NML instruction set with tri-syntax support, M2M extensions, and general-purpose I/O.

### Build

| Binary | Instructions | Size (stripped) | Build |
|--------|-------------|-----------------|-------|
| `nml` | 71 (35 core + 14 ext + 13 M2M + 5 GP + 4 TR) | ~68 KB | `make` |
| `nml-core` | 35 (core only) | ~50 KB | `make nml-core` |
| `nml-gp` | 71 (expanded limits for GP programs) | ~68 KB | `make nml-gp` |

```bash
make all       # Build full + core binaries
make nml-gp    # Build with expanded resource limits
make release   # Build + strip for smallest size
make test      # Run all test programs
```

### Instruction Set

```mermaid
graph LR
    subgraph core [Core — 35 Instructions]
        Arith["Arithmetic: MMUL MADD MSUB EMUL SDOT SCLR SDIV EDIV"]
        Activ["Activation: RELU SIGM TANH SOFT"]
        Mem["Memory: LD ST MOV ALLC"]
        Data["Data Flow: RSHP TRNS SPLT MERG"]
        Compare["Compare: CMPF CMP CMPI"]
        Ctrl["Control: LOOP ENDP JMPT JMPF JUMP"]
        Sub["Subroutine: CALL RET"]
        Tree["Tree: LEAF TACC"]
        Sys["System: SYNC HALT TRAP"]
    end

    subgraph ext [Extensions — 14 Instructions]
        Vision["NML-V: CONV POOL UPSC PADZ"]
        Trans["NML-T: ATTN NORM EMBD GELU"]
        Reduct["NML-R: RDUC WHER CLMP CMPR"]
        Signal["NML-S: FFT FILT"]
    end

    subgraph m2m [M2M — 13 Instructions]
        Struct["Structure: META FRAG ENDF LINK"]
        Comm["Communication: PTCH SIGN VRFY"]
        Collect["Collective: VOTE PROJ DIST GATH SCAT"]
    end

    subgraph gp [General Purpose — 5 Instructions]
        IO["I/O: SYS"]
        IntMath["Integer: MOD"]
        TypeConv["Type: ITOF FTOI"]
        Bitwise["Bitwise: BNOT"]
    end
```

All 71 instructions support three syntax forms: classic (MMUL), symbolic (×), and verbose (MATRIX_MULTIPLY).

### Register File

| Register | Purpose |
|----------|---------|
| R0–R9 | General-purpose tensor registers |
| RA | Accumulator (tree prediction / federal tax) |
| RB | General-purpose (FICA accumulator) |
| RC | Scratch (leaf values, bracket thresholds) |
| RD | Loop counter |
| RE | Condition flag (set by CMPF, CMP, CMPI) |
| RF | Stack pointer |

### Test Results

| Program | Instructions | Cycles | Time | Result |
|---------|-------------|--------|------|--------|
| Anomaly Detector (NN) | 18 | 18 | 419 µs | Score: 0.5060 |
| Tax Calculator (XGBoost) | 1,536 | 267 | 88 µs | Net pay: $1,721.71 |
| Extension Demo | 38 | 38 | 715 µs | All extensions validated |
| Tax Rules (deterministic) | 330 | 75 | 703 µs | Net pay: $6,205.43 |

### v0.5 Features

- Tri-syntax aliases: all opcodes accept classic (MMUL), symbolic (×), and verbose (MATRIX_MULTIPLY) forms
- Greek register aliases: ι κ λ μ ν ξ ο π ρ ς (R0-R9) and α β γ δ φ ψ (RA-RF)
- Verbose register aliases: ACCUMULATOR, SCRATCH, FLAG, COUNTER, GENERAL, STACK
- `--syntax` flag on transpiler and library builder for output mode selection
- `--no-comments` flag to strip comments for minimal file size

---

## Component 2: XGBoost Transpiler (`tax_pipeline.py`)

**569 lines of Python.** Trains an XGBoost model on synthetic payroll data and transpiles the trained trees to NML.

### Pipeline

```mermaid
flowchart LR
    SynData[Synthetic Data — 5,000 samples] --> Train[XGBoost Training — 20 trees, depth 4]
    Train --> Model[tax_model.json]
    Train --> Transpile[XGBoostToNML class]
    Transpile --> NML[tax_calculator.nml — 1,536 instructions]
    Transpile --> DataFile[employee_test.nml.data]
```

### Validation

20/20 exact match ($0.00 difference) between XGBoost Python predictions and NML runtime output.

| Employee | XGBoost | NML | Difference |
|----------|---------|-----|-----------|
| Junior Developer ($65k) | $1,721.71 | $1,721.71 | $0.00 |
| Senior Manager ($185k) | $4,764.52 | $4,764.52 | $0.00 |
| Executive ($350k) | $8,329.82 | $8,329.82 | $0.00 |
| Part-time Worker ($28k) | $697.25 | $697.25 | $0.00 |

---

## Component 3: Deterministic Rule Transpiler (`rule_transpiler.py`)

**525 lines of Python.** Converts JSON tax rules with exact bracket boundaries into deterministic NML programs. No ML training required.

### Architecture

```mermaid
flowchart LR
    JSON["tax_rules_2024.json\n(brackets, FICA, state rates, credits)"]
    JSON --> Parser[TaxRuleTranspiler]
    Parser --> Builder["NMLProgram\n(label-based jump resolver)"]
    Builder --> NML["tax_rules_calc.nml\n330 instructions"]
    JSON --> Analytical["compute_tax_analytical()\n(validation reference)"]
```

### NML Program Structure

```mermaid
flowchart TD
    Load[Load 7 inputs] --> AGI[Compute AGI]
    AGI --> StdDed[Standard Deduction — branch by filing status]
    StdDed --> Floor[Floor taxable at 0]
    Floor --> FedBrackets[Federal Brackets — 4 filing statuses × 7 brackets]
    FedBrackets --> FICA[FICA — SS + Medicare + Additional Medicare]
    FICA --> State[State Tax — branch by state code]
    State --> Credits[Dependent Credits]
    Credits --> NetPay[Net Pay Per Period]
    NetPay --> Store[Store Results + HALT]
```

### Validation

All 4 test employees match analytical computation exactly:

| Employee | Analytical | NML | Federal | FICA |
|----------|-----------|-----|---------|------|
| Junior Dev ($65k, Single, TX) | $2,112.06 | $2,112.06 | $5,114.00 | $4,972.50 |
| Senior Mgr ($185k, Married, CA) | $6,205.43 | $6,205.43 | $16,242.00 | $13,135.70 |
| Executive ($350k, Single, NY) | $10,248.79 | $10,248.79 | $63,264.75 | $16,878.20 |
| Part-time ($28k, HoH, FL) | $538.46 | $538.46 | $610.00 | $2,142.00 |

### Bugs Found and Fixed

1. **Register aliasing in MSUB**: `MSUB R8 R7 R8` (Rd == Rs2) silently computes 0 because `tensor_sub` copies Rs1 into Rd before reading Rs2. Fixed by using RC as intermediate.
2. **State tax fallthrough**: State code 0 (no tax) was incorrectly applying the code-1 rate due to CMPF threshold grouping. Fixed with explicit early-exit for zero-rate codes.

---

## Component 4: Domain Transpilation (`transpilers/`)

The NML transpiler layer supports a general pattern of **domain transpilation**: converting structured rule files from an external data source into executable NML programs. Given a corpus of domain-specific rule definitions (e.g., JSON files describing progressive brackets, flat rates, thresholds, and constants), the transpiler scans the rule tree, resolves effective dates, classifies each rule into a pattern (bracket-based or flat-rate), and emits the corresponding NML instructions. A builder module (`nml_builder.py`) handles label resolution and tri-syntax translation for the generated programs.

This approach is domain-agnostic. Any rule system that can be expressed as bracket lookups (`tax = addition + (income - threshold) × rate`) or flat-rate computations (`tax = min(wages, wageBase) × rate`) can be transpiled to NML using the same pipeline stages: scan → classify → resolve dates → emit brackets/flat rates → compose into a single program. The resulting NML programs are validated by transpiling, assembling, and executing each one, confirming zero runtime errors across the full rule library.

---

## Component 5: Training Data Pipeline

Training data was generated from the domain transpilation pipeline described in Component 4. Randomized input profiles (varying wage levels, filing statuses, and pay frequencies) were fed through the transpiled NML programs to produce ground-truth outputs, which were then paired with the corresponding NML source for LLM fine-tuning.

### Training Data Generator

Produces training examples in four formats from the transpiled NML programs:

| Format | Count | Description |
|--------|-------|-------------|
| Transpilation | ~500 | Domain rules → NML program |
| Intent-to-NML | ~355 | Natural language → NML + employee data |
| Audit Trail | ~380 | Employee profile → step-by-step tax breakdown |
| Q&A | ~119 | Tax knowledge questions and answers |
| **Base Total** | **~1,354** | (nml_training_large.jsonl) |
| **Expanded (MLX)** | **96,710** | Full Mistral instruction-tuning format (train + valid) |

### Audit Trail Example

```
Gross pay: $95,481.00
Filing status: Married

FICA (Social Security):
  Subject wages: $95,481.00
  Rate: 6.20%
  Tax: $5,919.82

Medicare:
  Subject wages: $95,481.00
  Rate: 1.45%
  Tax: $1,384.47

State Income Tax (RI):
  1.0% on $0 – $77,450 = $774.50
  2.75% on $77,450 – $95,481 = $495.85
  Total state income tax: $1,270.35

Total tax: $8,574.64
Net pay (annual): $86,906.36
```

---

## Component Summary

```mermaid
flowchart TD
    subgraph core [NML Core]
        Runtime["C Runtime\nnml.c — ~1,900 lines\n71 instructions, ~68 KB\nTri-syntax: classic | symbolic | verbose"]
        Spec["NML Spec v0.7.0\n35 core + 14 ext + 13 M2M + 5 GP + 4 TR"]
    end

    subgraph transpilers [Transpilers]
        XGB["XGBoost Transpiler\ntax_pipeline.py — 569 lines\n20/20 exact match"]
        Rules["Rule Transpiler\nrule_transpiler.py — 525 lines\n2024 US tax rules"]
        Domain["Domain Transpiler\ntranspilers/ — ~2,500 lines\nDomain rules, all validated"]
    end

    subgraph programs [Generated Programs]
        Library["NML Rule Library\nDomain-transpiled programs"]
        P1["tax_calculator.nml\n1,536 instr — XGBoost"]
        P6["anomaly_detector.nml\n18 instr — neural net"]
    end

    subgraph training [Training Pipeline]
        T1["96,710 training pairs\nMistral format"]
    end
```

### Performance Summary

| Metric | Value |
|--------|-------|
| NML runtime size (stripped) | ~68 KB (full), ~50 KB (core) |
| NML vocabulary | ~71 symbols (classic) + symbolic + verbose aliases |
| Token count (typical NN program) | 20–50 tokens |
| Inference time (XGBoost, 20 trees) | 88 µs |
| Inference time (FIT bracket lookup) | 145 µs |
| Inference time (FICA flat rate) | 85 µs |
| Domain Rules | Validated |
| Training pairs generated | 96,710 (Mistral format) |
| Syntax modes | 3 (classic, symbolic, verbose) |

---

## Multi-Agent Services Layer

The NML system now includes a multi-agent services layer that wraps the core components as HTTP services with an orchestration pipeline. See [NML_Multi_Agent_Architecture.md](NML_Multi_Agent_Architecture.md) for the full architecture and [NML_Multi_Agent_Implementation_Plan.md](NML_Multi_Agent_Implementation_Plan.md) for the implementation details.

### Agent Services

| Service | Port | Implementation | Status |
|---------|------|---------------|--------|
| Transpiler | 8083 | `serve/transpiler_service.py` | Implemented and tested |
| Validator | 8084 | `serve/validation_service.py` | Implemented and tested |
| Execution Engine | 8085 | `serve/execution_service.py` | Implemented and tested |
| Agent Gateway | 8082 | `transpilers/domain_rag_server.py` | Implemented and tested |

### Validation Tools

| Tool | Implementation | Status |
|------|---------------|--------|
| Grammar Validator | `transpilers/nml_grammar.py` | All programs pass |
| Semantic Analyzer | `transpilers/nml_semantic.py` | Extracts brackets, rates, deductions |
| Regression Suite | `transpilers/nml_regression.py` | Golden test baselines |
| NML Diff Engine | `transpilers/nml_diff.py` | Semantic bracket/rate comparison |
| Anomaly Detector | `transpilers/nml_anomaly.py` | Cross-jurisdiction scanning |

### Orchestration

| Component | Implementation | Status |
|-----------|---------------|--------|
| Intent Router | `serve/intent_router.py` | 7 intent categories |
| Agent Registry | `serve/agent_registry.py` | Health tracking + call stats |
| Pipeline Executor | `serve/pipeline_executor.py` | 6 pipeline types |
| Agent Protocol | `serve/nml_protocol.py` | AgentMessage envelope |
| Provenance Tracker | `serve/provenance_tracker.py` | Instruction-level tracing |
| Audit Log | `serve/audit_log.py` | Append-only JSONL log |

## Component 7: NML v0.6 M2M Extensions

**11 new opcodes** extending NML for machine-to-machine communication. See [NML_M2M_Spec.md](NML_M2M_Spec.md) for the full specification.

### Build

```bash
make nml-v06    # Build v0.6 runtime with M2M extensions
```

| Binary | Instructions | Description |
|--------|-------------|-------------|
| `nml-v06` | 60 (35 core + 14 ext + 11 M2M) | Full runtime with M2M extensions |

### Test Results

```bash
./nml-v06 tests/test_m2m.nml tests/test_m2m.nml.data
```

| Test | Result | Details |
|------|--------|---------|
| VOTE median | 6200.00 | Correct median of [6200, 6200.01, 6199.98, 6200, 6150] |
| VOTE mean | 6189.998 | Correct arithmetic mean |
| VOTE quorum | 1.0 | 4 of 5 values agree within 0.01 tolerance |
| PROJ | [0.7071, 0.7071] | Unit-norm 2D projection |
| DIST cosine | 0.0077 | Small distance (similar vectors) |
| DIST euclidean | 0.1243 | Euclidean distance |
| META --describe | PASS | Prints program descriptor |
| Backward compat | PASS | All v0.5 programs run unchanged |
| Grammar validator | ALL PASS | All existing programs valid under updated validator |

### Runtime Fixes (v0.6.1)

Critical bugs fixed after initial v0.6 release:

| Fix | Problem | Solution |
|-----|---------|----------|
| Register aliasing | `MSUB R0 R0 R2` produced wrong results — tensor_init zeroed output before reading aliased input | Added temporary buffer when output aliases input in tensor_add/sub/emul/ediv/mmul |
| MMUL f64 | Matrix multiply used hardcoded `data.f32[]`, ignoring dtype | Replaced with `tensor_getd()`/`tensor_setd()` for dtype-aware access |
| RELU/SIGM/TANH/SOFT f64 | Activation functions used hardcoded `data.f32[]` | Replaced with dtype-aware `tensor_getd()`/`tensor_setd()` |
| GATH opcode | No way to index into a tensor by position | Added `GATH Rd Rs Ridx` (opcode 60) |
| SCAT opcode | No way to write to a specific tensor index | Added `SCAT Rs Rd Ridx` (opcode 61) |

### Validation Results

| Validation | Result |
|------------|--------|
| Golden regression | **ALL PASS** — zero failures across entire NML library |
| Grammar validator | ALL PASS with v0.6 opcodes |

### Neural Bracket Experiment

Tax brackets encoded as neural network weights (64-neuron ReLU):

| Filing Status | MAE | Max Error |
|---------------|-----|-----------|
| Single | $32 | $145 |
| MFJ | $55 | $317 |
| HoH | $32 | $169 |

After RELU f64 fix, NML execution matches numpy inference: $1-$12 error at most income levels for 128-neuron model.

### Embedding Anomaly Monitor

Trained neural embeddings for 53 bracket jurisdictions, compared 2024 vs 2025:
- **2 ALERTS**: Maryland SIT (brackets expanded 5→10 tiers), PEI PIT (minor drift)
- **51 NORMAL**: Expected inflation adjustments or no change
- Report saved to `output/anomaly_reports/`

### LLM Fine-Tuning (Round 2)

| Metric | Round 1 | Round 2 |
|--------|---------|---------|
| Iterations | 700 | 6,000 |
| Learning rate | 5e-6 | 1.5e-5 |
| Val loss | ~0.8 | **0.396** |
| Training pairs | 149,674 | 149,674 + 15,189 gap pairs |
| Result | Generated pseudocode | **Generates valid NML in classic and symbolic syntax** |

Model saved to `output/model/nml-v06-r2-merged`.

### New Python Tools

| Tool | File | Purpose |
|------|------|---------|
| Type System | `serve/nml_types.py` | Semantic types, compatibility rules |
| Fragment Composer | `transpilers/nml_composer.py` | Extract, resolve, compose fragments |
| Patch Engine | `transpilers/nml_patch.py` | Differential program generation |
| Signing | `serve/nml_signing.py` | HMAC-SHA256 / Ed25519 signing |
| Embedding | `transpilers/nml_embedding.py` | Projection matrices, distance computation |

---

## Component 8: NML-G General Purpose Extension

**5 new opcodes** extending NML for general-purpose computing. See [NML_G_Spec.md](NML_G_Spec.md) for the full specification.

### Build

```bash
make nml-gp    # Build with expanded resource limits
```

| Binary | Instructions | Description |
|--------|-------------|-------------|
| `nml-gp` | 71 (35 core + 14 ext + 13 M2M + 5 GP + 4 TR) | Full runtime with expanded limits |

### New Instructions

| Instruction | Symbolic | Description |
|---|---|---|
| `SYS Rd #code` | `⚙` | Multiplexed syscall (print, read, time, rand, exit) |
| `MOD Rd Rs1 Rs2` | `%` | Integer modulo |
| `ITOF Rd Rs` | `⊶` | Integer → float conversion |
| `FTOI Rd Rs` | `⊷` | Float → integer (truncate) |
| `BNOT Rd Rs` | `¬` | Bitwise NOT |

### Configurable Resource Limits (v0.6.2)

All hard limits now accept compile-time overrides:

```bash
gcc -O2 -o nml-gp nml.c -lm \
    -DNML_MAX_INSTRUCTIONS=65536 \
    -DNML_MAX_MEMORY_SLOTS=256 \
    -DNML_MAX_CALL_DEPTH=128
```

### Test Results

| Program | Instructions | Cycles | Output |
|---------|-------------|--------|--------|
| hello_world.nml | 29 | 29 | "Hello, World!" |
| fibonacci.nml | 13 | 165 | First 20 Fibonacci numbers |
| fizzbuzz.nml | 29 | 418 | FizzBuzz 1-30 |
| primes.nml | 23 | 1,276 | Primes 2-50 |
| calculator.nml | 16 | 14 | Interactive add/sub/mul/div |
| Backward compat | PASS | — | All existing programs run unchanged |

### Example Programs

| Program | File | Demonstrates |
|---------|------|-------------|
| Hello World | `programs/hello_world.nml` | SYS #1 (PRINT_CHAR) |
| Fibonacci | `programs/fibonacci.nml` | SYS #0, backward jumps, CMP |
| Fibonacci (symbolic) | `programs/fibonacci_symbolic.nml` | Same in ⚙/≶/↗ symbolic syntax |
| FizzBuzz | `programs/fizzbuzz.nml` | MOD, conditional branches |
| Prime Finder | `programs/primes.nml` | Nested loops, MOD, trial division |
| Calculator | `programs/calculator.nml` | SYS #2 (READ_NUM), interactive I/O |

---

### Next Steps

1. **Scale training data** — Expand from 1,354 to 200K+ training pairs using the full rule library with varied input profiles.
2. **Fine-tune LLM** — Train CodeLlama/Mistral 7B on the NML dataset using QLoRA (estimated 4-8 hours on 1x A100).
3. **Convergence experiment** — Train matched 125M-param models on NML vs Python, compare convergence to validate the core thesis.
4. **Full golden test generation** — Expand regression suite to cover all transpiled rule files.
5. **Cross-model federation** — Test agent pipeline with multiple LLM backends (Mistral, Llama, Phi).
6. **Self-improving feedback loop** — Use explanation agent outputs as training data for the regulation parser.

---

## Component 9: NML v0.7.0 — Training Extension

NML v0.7.0 extends the register file to 32 registers (R0–RV) and adds 4 training opcodes (BKWD, WUPD, LOSS, TNET) that enable self-training capability within NML programs. The extended register file dedicates RG–RI for gradient tensors, RJ for learning rate, and RK–RV for training workspace and hive collective operations. Optional BLAS acceleration is available for matrix operations during training via compile flag `-DNML_USE_BLAS`.

### New Instructions

| Instruction | Symbolic | Description |
|---|---|---|
| `BKWD Rd Rs Rtarget` | `∇` | Backpropagation: compute gradient of loss w.r.t. Rs into Rd |
| `WUPD Rd Rs Rgrad [Rlr]` | `⟳` | Weight update: Rd = Rs - lr * Rgrad |
| `LOSS Rd Rs Rtarget [#mode]` | `△` | Loss computation (0=MSE, 1=cross-entropy, 2=MAE) |
| `TNET #epochs #lr [#seed]` | `⥁` | Self-training loop using R1–R4 as network, R0 as input, R9 as target |
