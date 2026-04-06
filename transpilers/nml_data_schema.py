"""
nml_data_schema.py — LLM-assisted data schema inference for NML programs

Given a prediction task description and the schema of available raw data,
uses Claude to propose:
  - Which features to include, how to transform them (diff, ratio, absolute)
  - Normalization constants (with validation flags for empirical ones)
  - Tensor shapes for training and inference
  - A ready-to-use .nml.data skeleton

The LLM is the PROPOSER. The human (and data distribution) is the VALIDATOR.
Normalization constants marked "proposed" must be confirmed against actual data.

Usage:
  python nml_data_schema.py \\
      --task    "Predict NCAA March Madness game winner (binary)" \\
      --schema  ../../nml-programs/sports/basketball/march_madness_schema.json \\
      --samples 335 \\
      --target  "1 if team A wins, 0 if team B wins" \\
      --out     /tmp/game_winner_skeleton.nml.data
"""

from __future__ import annotations

import json
import argparse
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

import anthropic

# ── NML hard limits ───────────────────────────────────────────────────────────

NML_MAX_TENSOR_ELEMENTS = 65_536
NML_VALID_DTYPES        = ["f32", "f64", "i32"]


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class FeatureSpec:
    """One feature in the input vector."""
    name:                  str
    source_fields:         list[str]    # dot-path fields from raw data, applied to both teams
    transformation:        str          # "diff" | "ratio" | "absolute" | "custom"
    normalization_divisor: float        # divide transformed value by this
    normalization_basis:   str          # "theoretical" | "empirical" | "proposed"
    expected_range:        list[float]  # [min, max] after normalization
    reasoning:             str
    flag_for_validation:   bool         # True → human must verify divisor against data


@dataclass
class DataSchemaProposal:
    """Full schema proposal returned by the LLM."""
    task:                    str
    features:                list[FeatureSpec]
    training_shape:          list[int]       # [N, K]  — N accounts for augmentation
    inference_shape:         list[int]       # [1, K] or [batch, K]
    dtype:                   str
    augmentation_strategy:   Optional[str]   # "negate_diff_flip_label" | None
    nml_data_skeleton:       str             # populated .nml.data ready to edit
    validation_notes:        list[str]       # things human must verify
    nml_constraint_warnings: list[str]       # NML limit violations


# ── Tool definition (structured output contract) ──────────────────────────────

_PROPOSE_SCHEMA_TOOL = {
    "name": "propose_data_schema",
    "description": (
        "Propose a feature set, tensor shapes, and normalization strategy "
        "for an NML machine learning program given a prediction task and "
        "the schema of available raw data."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "features", "training_shape", "inference_shape",
            "dtype", "augmentation_strategy", "validation_notes",
        ],
        "properties": {
            "features": {
                "type": "array",
                "description": "Ordered list of features in the input vector, smallest useful set.",
                "items": {
                    "type": "object",
                    "required": [
                        "name", "source_fields", "transformation",
                        "normalization_divisor", "normalization_basis",
                        "expected_range", "reasoning", "flag_for_validation",
                    ],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Snake_case name, e.g. adj_margin_diff",
                        },
                        "source_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Dot-path field names from the raw data schema "
                                "(e.g. ['stats.margin']). For 'diff' features the same "
                                "field is read from both team A and team B."
                            ),
                        },
                        "transformation": {
                            "type": "string",
                            "enum": ["diff", "ratio", "absolute", "custom"],
                            "description": (
                                "'diff' = A − B.  'ratio' = A / B.  "
                                "'absolute' = single value (not per-team).  "
                                "'custom' = described in reasoning."
                            ),
                        },
                        "normalization_divisor": {
                            "type": "number",
                            "description": (
                                "Divide the raw transformed value by this to put it in "
                                "approximately [−1, 1].  E.g. 30.0 for scoring margin."
                            ),
                        },
                        "normalization_basis": {
                            "type": "string",
                            "enum": ["theoretical", "empirical", "proposed"],
                            "description": (
                                "'theoretical' = range is mathematically fixed "
                                "(e.g. win_pct always [0,1], diff in [−1,1]).  "
                                "'empirical' = computed from the actual training data.  "
                                "'proposed' = LLM estimate — MUST be validated against data."
                            ),
                        },
                        "expected_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Expected [min, max] after normalization, ideally near [−1, 1].",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Why this feature is predictive and how the "
                                "normalization divisor was chosen."
                            ),
                        },
                        "flag_for_validation": {
                            "type": "boolean",
                            "description": (
                                "True if the divisor is an estimate and must be confirmed "
                                "against the actual training data distribution."
                            ),
                        },
                    },
                },
            },
            "training_shape": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Shape of @training_data as [N_samples, K_features]. "
                    "N must account for augmentation."
                ),
            },
            "inference_shape": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Shape of @predict_input as [batch, K]. Use [1, K] for single prediction.",
            },
            "dtype": {
                "type": "string",
                "enum": ["f32", "f64"],
                "description": (
                    "f32 for most models. f64 if features are very small after normalization "
                    "and you need extra precision."
                ),
            },
            "augmentation_strategy": {
                "type": ["string", "null"],
                "description": (
                    "For diff features: 'negate_diff_flip_label' doubles training data by "
                    "swapping A↔B, negating all features, and flipping the label. "
                    "Null if not applicable."
                ),
            },
            "validation_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Human-readable notes about normalization constants or feature choices "
                    "that must be verified against actual data before training."
                ),
            },
        },
    },
}


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in machine learning feature engineering for NML (Neural Machine Language) programs.

NML DATA FORMAT
Each named tensor in an .nml.data file is declared as:
  @label  shape=d1,d2  dtype=f32  data=val1,val2,...

Example:
  @training_data   shape=652,8  dtype=f32  data=...
  @predict_input   shape=1,8    dtype=f32  data=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

NML HARD CONSTRAINTS
- Maximum tensor elements: 65,536 (product of all shape dimensions).
- Valid dtypes: f32, f64, i32.
- The TNET/TRAIN instruction expects @training_data shape=[N, K]
  where K = number of input features, N = number of training samples.

FEATURE ENGINEERING PRINCIPLES
1. Keep K small (4–12 features). NML models are compact (< 1,000 parameters).
2. For binary matchup prediction (A vs B), prefer DIFFERENCE features (A − B, normalised).
   This makes the model symmetric: swapping A↔B and negating features produces the mirror game,
   enabling free augmentation that doubles training size.
3. Normalization: divide each feature so it falls in approximately [−1, 1].
   - Theoretical ranges (win_pct always 0–1): mark basis = "theoretical", no validation needed.
   - Empirical ranges (scoring margin depends on the data): mark basis = "proposed" and
     flag_for_validation = true. The human must check max(abs(value)) in their training set.
4. Prefer features available BEFORE the game (season stats, seed, conference tier),
   not in-game statistics.
5. Strength-of-schedule (SOS) adjustment by conference tier improves generalization for
   college sports where opponents vary widely in quality.
6. If augmentation applies (all features are differences), set
   augmentation_strategy = "negate_diff_flip_label".

NORMALIZATION GUIDANCE (starting points; always flag proposed ones)
  win_pct diff:    already in [−1, 1] → divisor 1.0,  theoretical
  scoring margin:  typical tournament diff ±25–35 pts → divisor ~30,  proposed
  eFG% diff:       typical diff ±0.10–0.20 → divisor ~0.15–0.20,  proposed
  TOV rate diff:   typical diff ±5–10 per game → divisor ~8,  proposed
  ORB% diff:       typical diff ±5–10 per game → divisor ~8,  proposed
  seed diff:       max = 15 (seed 1 vs 16) → divisor 15,  theoretical
  conference tier: if tiers are [0.87, 1.00, 1.07, 1.18], max diff ≈ 0.31 → divisor 0.31, theoretical

Always include at least one seed- or rank-based feature and at least one efficiency metric
(margin, eFG%, or similar). These are the strongest predictors in NCAA tournament data.
"""


# ── Core function ─────────────────────────────────────────────────────────────

def propose_schema(
    task_description:   str,
    available_fields:   dict,    # {field_path: {type, description, example?, range?}}
    sample_count:       int,     # raw samples before augmentation
    target_description: str,
) -> DataSchemaProposal:
    """
    Call Claude to propose a feature set and tensor shapes for an NML program.

    Args:
        task_description:   Natural language task, e.g. "Predict NCAA game winner (binary)"
        available_fields:   Dict mapping field paths to metadata (type, description, example)
        sample_count:       Number of raw training examples (before augmentation)
        target_description: What the label represents

    Returns:
        DataSchemaProposal with features, shapes, .nml.data skeleton, and validation notes
    """
    client = anthropic.Anthropic()

    user_message = (
        f"Task: {task_description}\n"
        f"Target: {target_description}\n"
        f"Raw training samples (before any augmentation): {sample_count}\n"
        f"NML max tensor elements: {NML_MAX_TENSOR_ELEMENTS}\n\n"
        f"Available fields per team:\n"
        f"{json.dumps(available_fields, indent=2)}\n\n"
        f"Propose the smallest useful feature vector and tensor schema for this NML "
        f"prediction program. Aim for 6–10 features. Explain your normalization choices "
        f"and flag any divisor that must be verified against the actual data distribution."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_PROPOSE_SCHEMA_TOOL],
        tool_choice={"type": "tool", "name": "propose_data_schema"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    raw = tool_use.input

    features = [
        FeatureSpec(
            name=f["name"],
            source_fields=f["source_fields"],
            transformation=f["transformation"],
            normalization_divisor=f["normalization_divisor"],
            normalization_basis=f["normalization_basis"],
            expected_range=f["expected_range"],
            reasoning=f["reasoning"],
            flag_for_validation=f["flag_for_validation"],
        )
        for f in raw["features"]
    ]

    # Validate against NML constraints
    warnings = []
    n, k = raw["training_shape"]
    if n * k > NML_MAX_TENSOR_ELEMENTS:
        warnings.append(
            f"training_data shape={n},{k} = {n*k} elements "
            f"exceeds NML limit of {NML_MAX_TENSOR_ELEMENTS}"
        )

    skeleton = _generate_skeleton(features, n, k, raw["dtype"])

    return DataSchemaProposal(
        task=task_description,
        features=features,
        training_shape=raw["training_shape"],
        inference_shape=raw["inference_shape"],
        dtype=raw["dtype"],
        augmentation_strategy=raw.get("augmentation_strategy"),
        nml_data_skeleton=skeleton,
        validation_notes=raw["validation_notes"],
        nml_constraint_warnings=warnings,
    )


# ── Skeleton generator ────────────────────────────────────────────────────────

def _generate_skeleton(
    features:  list[FeatureSpec],
    n_samples: int,
    k:         int,
    dtype:     str,
) -> str:
    """
    Generate an .nml.data skeleton with documented placeholders.

    Weights are zero-initialised here; the prep script must replace them
    with He-initialised values before training.
    """
    h1 = k * 2          # hidden layer 1 size
    h2 = k              # hidden layer 2 size

    # Feature index table
    feature_table = "\n".join(
        f"; [{i:2d}] {f.name:<35s} ÷{f.normalization_divisor:<8g} "
        f"basis={f.normalization_basis}"
        + ("  ← VALIDATE DIVISOR" if f.flag_for_validation else "")
        for i, f in enumerate(features)
    )

    # Normalization basis legend
    basis_notes = "\n".join(
        f";   [{i:2d}] {f.name}: {f.reasoning[:80]}"
        for i, f in enumerate(features)
    )

    zeros = lambda n: ",".join(["0.0"] * n)

    return f"""; ═══════════════════════════════════════════════════════════
; NML Data Schema Skeleton  (auto-generated by nml_data_schema.py)
; ─────────────────────────────────────────────────────────────
; Feature vector  ({k} features per sample):
{feature_table}
;
; normalization_basis key:
;   theoretical = mathematically fixed range — no validation needed
;   empirical   = computed from actual training data distribution
;   proposed    = LLM estimate — MUST verify against your data
;
; Feature reasoning:
{basis_notes}
; ═══════════════════════════════════════════════════════════

; ── Network architecture ─────────────────────────────────────
; Suggested: {k} → {h1} (ReLU) → {h2} (ReLU) → 1 (Sigmoid)
; arch descriptor: [num_hidden_layers=3, h1={h1}, act1=0(ReLU),
;                   h2={h2}, act2=0(ReLU), out=1, out_act=1(Sigmoid)]
@arch            shape=7         dtype={dtype}  data=3,{h1},0,{h2},0,1,1

; ── Weights  (replace with He-initialised values from prep script) ──
@w1              shape={k},{h1}  dtype={dtype}  data={zeros(k * h1)}
@b1              shape=1,{h1}    dtype={dtype}  data={zeros(h1)}
@w2              shape={h1},{h2} dtype={dtype}  data={zeros(h1 * h2)}
@b2              shape=1,{h2}    dtype={dtype}  data={zeros(h2)}
@w3              shape={h2},1    dtype={dtype}  data={zeros(h2)}
@b3              shape=1,1       dtype={dtype}  data=0.0

; ── Training config ──────────────────────────────────────────
; [epochs, learning_rate, optimizer(1=Adam / 0=SGD), min_delta, patience, decay]
@train_config    shape=6         dtype={dtype}  data=5000,0.003,1,0,0,0

; ── Training data  ({n_samples} samples × {k} features) ─────
; Replace the zeros below with your actual feature matrix.
; Each row: [{", ".join(f.name for f in features)}]
@training_data   shape={n_samples},{k}  dtype={dtype}  data={zeros(n_samples * k)}

; ── Training labels  ({n_samples} × 1) ─────────────────────
; 1.0 = team A wins,  0.0 = team B wins
@training_labels shape={n_samples},1    dtype={dtype}  data={zeros(n_samples)}

; ── Prediction input  (1 × {k}) ─────────────────────────────
; Populate with the feature vector for the game you want to predict.
; Feature order must match @training_data columns exactly.
@predict_input   shape=1,{k}     dtype={dtype}  data={zeros(k)}
"""


# ── Step 4: NML program generation via local fine-tuned LLM ──────────────────

def proposal_to_prompt(proposal: DataSchemaProposal) -> str:
    """
    Convert a DataSchemaProposal into a natural language prompt for the local
    fine-tuned NML LLM.

    The prompt mirrors the format the model was trained on (see nml_realworld_gen.py):
    plain English spec with explicit shape references, architecture, and step list.
    """
    n, k = proposal.training_shape
    h1 = k * 2
    h2 = k
    feature_names = ", ".join(f.name for f in proposal.features)

    lines = [
        "Write NML for binary classification using a neural network.",
        "",
        "Data tensors:",
        f"  @training_data   shape={n},{k}   ({n} samples, {k} features: {feature_names})",
        f"  @training_labels shape={n},1",
        f"  @predict_input   shape=1,{k}",
        "",
        f"Network architecture: {k} → {h1} (ReLU) → {h2} (ReLU) → 1 (Sigmoid)",
        f"Weight tensors: @w1 shape={k},{h1}, @b1 shape=1,{h1},",
        f"                @w2 shape={h1},{h2}, @b2 shape=1,{h2},",
        f"                @w3 shape={h2},1, @b3 shape=1,1",
        "Config tensors: @arch shape=7, @train_config shape=6",
        "",
        "Steps:",
        "  1. Load @arch and all weight/bias tensors into registers",
        "  2. Load @train_config, @training_data, @training_labels — train with TRAIN",
        "  3. Store training loss to @training_loss",
        "  4. Load @predict_input — run INFER",
        "  5. Store result to @win_probability",
        "  6. Compare @win_probability to #0.5 — store 1.0 to @predicted_winner if true, else 0.0",
        "  7. HALT",
    ]

    return "\n".join(lines)


def generate_nml_program(
    proposal:   DataSchemaProposal,
    server_url: str = "http://localhost:8082",
    max_tokens: int = 1024,
) -> str:
    """
    Call the local fine-tuned NML LLM to generate a .nml program from a schema proposal.

    The server uses constrained decoding (Outlines + Lark CFG) which guarantees
    syntactically valid NML output — the LLM cannot produce malformed instructions.

    Args:
        proposal:    DataSchemaProposal from propose_schema()
        server_url:  URL of the running NML serve instance (default: localhost:8082)
        max_tokens:  Max tokens to generate

    Returns:
        NML program text (syntax-valid by construction)

    Raises:
        ConnectionError: if the NML server is not reachable
    """
    prompt = proposal_to_prompt(proposal)

    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "mode": "nml",       # force constrained decoding
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not reach NML server at {server_url}. "
            f"Start it with: python3 nml/serve/nml_server.py --http --port 8082\n"
            f"Original error: {e}"
        ) from e

    return result["choices"][0]["message"]["content"]


# ── Pretty-print a proposal ───────────────────────────────────────────────────

def print_proposal(p: DataSchemaProposal) -> None:
    w = 62
    print(f"\n{'═' * w}")
    print(f"  NML Data Schema Proposal")
    print(f"  {p.task}")
    print(f"{'═' * w}\n")

    print(f"Features ({len(p.features)}):")
    for i, f in enumerate(p.features):
        flag = "  ← VALIDATE" if f.flag_for_validation else ""
        print(
            f"  [{i}] {f.name:<35s}"
            f"  ÷{f.normalization_divisor:<8g}"
            f"  {f.normalization_basis}{flag}"
        )

    print(f"\n  Training shape : {p.training_shape}")
    print(f"  Inference shape: {p.inference_shape}")
    print(f"  Dtype          : {p.dtype}")
    print(f"  Augmentation   : {p.augmentation_strategy or 'none'}")

    if p.validation_notes:
        print(f"\nValidation notes (check before training):")
        for note in p.validation_notes:
            print(f"  • {note}")

    if p.nml_constraint_warnings:
        print(f"\nNML constraint warnings:")
        for w in p.nml_constraint_warnings:
            print(f"  ⚠  {w}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose an NML data schema via LLM (Claude)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task",    required=True,
                        help="Natural language task description")
    parser.add_argument("--schema",  required=True,
                        help="Path to JSON file describing available fields")
    parser.add_argument("--samples", required=True, type=int,
                        help="Number of raw training samples (before augmentation)")
    parser.add_argument("--target",  default="binary label (1 = positive, 0 = negative)",
                        help="Description of the prediction target / label")
    parser.add_argument("--out",      default=None,
                        help="Write .nml.data skeleton to this path (default: stdout)")
    parser.add_argument("--generate", action="store_true",
                        help="Also generate the .nml program via the local fine-tuned LLM")
    parser.add_argument("--server",   default="http://localhost:8082",
                        help="NML server URL for program generation (default: localhost:8082)")
    parser.add_argument("--nml-out",  default=None,
                        help="Write generated .nml program to this path")
    args = parser.parse_args()

    with open(args.schema) as fh:
        schema = json.load(fh)

    # ── Step 2: Claude schema inference ──────────────────────────────────────
    proposal = propose_schema(
        task_description=args.task,
        available_fields=schema,
        sample_count=args.samples,
        target_description=args.target,
    )

    print_proposal(proposal)

    if args.out:
        from pathlib import Path
        Path(args.out).write_text(proposal.nml_data_skeleton)
        print(f"\nSkeleton written → {args.out}")
    else:
        print(f"\n{'─' * 62}")
        print("Skeleton .nml.data:")
        print(f"{'─' * 62}")
        print(proposal.nml_data_skeleton)

    # ── Step 4: Local NML LLM program generation ─────────────────────────────
    if args.generate:
        print(f"\n{'─' * 62}")
        print(f"Generating .nml program via local LLM ({args.server})...")
        print(f"Prompt:\n{proposal_to_prompt(proposal)}\n")

        try:
            nml_program = generate_nml_program(proposal, server_url=args.server)
        except ConnectionError as e:
            print(f"\n⚠  {e}")
            return

        # Validate the generated program
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from nml_grammar import validate_grammar
            report = validate_grammar(nml_program)
            status = "VALID" if report.valid else f"INVALID ({len(report.errors)} errors)"
            print(f"Grammar validation: {status}")
            for err in report.errors:
                print(f"  line {err.line}: {err.message}")
        except ImportError:
            print("Grammar validation skipped (nml_grammar not importable here)")

        nml_out = args.nml_out
        if nml_out:
            from pathlib import Path
            Path(nml_out).write_text(nml_program)
            print(f"Program written → {nml_out}")
        else:
            print(f"\n{'─' * 62}")
            print("Generated .nml program:")
            print(f"{'─' * 62}")
            print(nml_program)


if __name__ == "__main__":
    main()
