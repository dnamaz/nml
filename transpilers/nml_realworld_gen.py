#!/usr/bin/env python3
"""
Generate ~3K real-world ML application training pairs.

Each program is a complete end-to-end pipeline: train with TNET, run inference
with a forward pass, and make a binary decision via CMPI threshold. Covers
6 application domains with varied architectures and tri-syntax coverage.

~30% of outputs include a matching .nml.data file.

Output: domain/output/training/nml_realworld_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_realworld_pairs.jsonl"

random.seed(2026_03_15)

from nml_core_training_gen import (
    _fmt, _pair, _fval,
    apply_syntax, pick_syntax, syntax_tag,
)


def _rand_data(n, lo=-0.5, hi=0.5, decimals=4):
    return ",".join(str(round(random.uniform(lo, hi), decimals)) for _ in range(n))


def _rand_features(n, lo=0.0, hi=1.0, decimals=2):
    return ",".join(str(round(random.uniform(lo, hi), decimals)) for _ in range(n))


DOMAINS = [
    {
        "name": "fraud_detection",
        "prog_name": "fraud_detector",
        "features": ["amount", "time_delta", "distance", "hourly_freq", "is_foreign", "merchant_risk"],
        "input_label": "transactions",
        "target_label": "fraud_labels",
        "new_input_label": "new_transaction",
        "score_label": "fraud_score",
        "decision_label": "fraud_flag",
        "pos_meaning": "fraud",
        "neg_meaning": "legitimate",
        "prompts": [
            "Write NML for credit card fraud detection: train on labeled transactions with TNET, classify a new transaction, flag if fraud score >= {thresh}.",
            "NML fraud detector: {n_feat} features ({feat_str}), {arch} architecture, train {epochs} epochs, threshold at {thresh}.",
            "Write NML to detect fraudulent transactions. Train with TNET, run inference, output binary decision.",
        ],
    },
    {
        "name": "medical_risk",
        "prog_name": "risk_predictor",
        "features": ["blood_pressure", "glucose", "bmi", "age", "cholesterol", "heart_rate"],
        "input_label": "patient_records",
        "target_label": "risk_labels",
        "new_input_label": "new_patient",
        "score_label": "risk_score",
        "decision_label": "high_risk_flag",
        "pos_meaning": "high risk",
        "neg_meaning": "low risk",
        "prompts": [
            "Write NML for medical risk prediction: train on patient data, score a new patient, flag if risk >= {thresh}.",
            "NML health risk model: {n_feat} vitals ({feat_str}), {arch} network, {epochs} epochs, threshold {thresh}.",
            "Write NML to predict patient risk from vitals. Use TNET training and CMPI threshold decision.",
        ],
    },
    {
        "name": "churn_prediction",
        "prog_name": "churn_model",
        "features": ["tenure", "monthly_spend", "support_tickets", "usage_drop", "contract_type"],
        "input_label": "customer_data",
        "target_label": "churn_labels",
        "new_input_label": "new_customer",
        "score_label": "churn_probability",
        "decision_label": "churn_flag",
        "pos_meaning": "will churn",
        "neg_meaning": "will stay",
        "prompts": [
            "Write NML for customer churn prediction: train on historical data, predict churn probability, flag if >= {thresh}.",
            "NML churn model: {n_feat} features ({feat_str}), {arch} architecture, TNET {epochs} epochs.",
            "Write NML to predict customer churn. Train with TNET, sigmoid output, binary decision at {thresh}.",
        ],
    },
    {
        "name": "predictive_maintenance",
        "prog_name": "maintenance_alert",
        "features": ["vibration", "temperature", "pressure", "rpm", "runtime_hours", "error_count", "oil_quality", "noise_level"],
        "input_label": "sensor_readings",
        "target_label": "failure_labels",
        "new_input_label": "current_readings",
        "score_label": "failure_probability",
        "decision_label": "maintenance_alert",
        "pos_meaning": "needs maintenance",
        "neg_meaning": "healthy",
        "prompts": [
            "Write NML for predictive maintenance: train on sensor data, predict failure probability, alert if >= {thresh}.",
            "NML maintenance model: {n_feat} sensors ({feat_str}), {arch} network, {epochs} epochs, alert threshold {thresh}.",
            "Write NML to predict equipment failure from sensor readings. Use TNET and threshold decision.",
        ],
    },
    {
        "name": "spam_classifier",
        "prog_name": "spam_filter",
        "features": ["link_count", "caps_ratio", "word_count", "sender_reputation", "has_attachment"],
        "input_label": "message_features",
        "target_label": "spam_labels",
        "new_input_label": "new_message",
        "score_label": "spam_score",
        "decision_label": "is_spam",
        "pos_meaning": "spam",
        "neg_meaning": "not spam",
        "prompts": [
            "Write NML for spam classification: train on labeled messages, score a new message, block if spam score >= {thresh}.",
            "NML spam filter: {n_feat} features ({feat_str}), {arch} architecture, train {epochs} epochs.",
            "Write NML to classify messages as spam or not. Train with TNET, sigmoid output, CMPI decision.",
        ],
    },
    {
        "name": "credit_scoring",
        "prog_name": "credit_model",
        "features": ["income", "debt_ratio", "credit_history", "delinquencies", "loan_amount", "employment_years"],
        "input_label": "applicant_data",
        "target_label": "default_labels",
        "new_input_label": "new_applicant",
        "score_label": "default_probability",
        "decision_label": "denied",
        "pos_meaning": "deny",
        "neg_meaning": "approve",
        "prompts": [
            "Write NML for credit scoring: train on applicant history, predict default probability, deny if >= {thresh}.",
            "NML credit risk model: {n_feat} features ({feat_str}), {arch} network, TNET {epochs} epochs, threshold {thresh}.",
            "Write NML to score loan applicants. Train with TNET, infer on new applicant, approve or deny.",
        ],
    },
]

ARCHITECTURES = [
    {"hidden": 4, "name_suffix": "4"},
    {"hidden": 8, "name_suffix": "8"},
    {"hidden": 16, "name_suffix": "16"},
]


def _gen_program_lines(domain, arch, epochs, lr, thresh, activation="RELU"):
    """Generate the NML instruction lines for a train+infer+decide pipeline."""
    d = domain
    lines = [
        _fmt("LD", "R1", "@w1"),
        _fmt("LD", "R2", "@b1"),
        _fmt("LD", "R3", "@w2"),
        _fmt("LD", "R4", "@b2"),
        _fmt("LD", "R0", f"@{d['input_label']}"),
        _fmt("LD", "R9", f"@{d['target_label']}"),
        _fmt("TNET", f"#{epochs}", f"#{lr}", "#0"),
        _fmt("ST", "R8", "@training_loss"),
        _fmt("LD", "R0", f"@{d['new_input_label']}"),
        _fmt("MMUL", "R5", "R0", "R1"),
        _fmt("MADD", "R5", "R5", "R2"),
        _fmt(activation, "R5", "R5"),
        _fmt("MMUL", "R6", "R5", "R3"),
        _fmt("MADD", "R6", "R6", "R4"),
        _fmt("SIGM", "RA", "R6"),
        _fmt("ST", "RA", f"@{d['score_label']}"),
        _fmt("CMPI", "RE", "RA", f"#{thresh}"),
        _fmt("JMPF", "#2"),
        _fmt("LEAF", "RB", "#0.0"),
        _fmt("JUMP", "#1"),
        _fmt("LEAF", "RB", "#1.0"),
        _fmt("ST", "RB", f"@{d['decision_label']}"),
        "HALT",
    ]
    return lines


def _gen_data_lines(domain, arch, n_samples):
    """Generate .nml.data file lines for a domain."""
    d = domain
    n_feat = len(d["features"])
    hidden = arch["hidden"]
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    legit_data = _rand_features(n_neg * n_feat, 0.0, 0.5)
    fraud_data = _rand_features(n_pos * n_feat, 0.5, 1.0)
    all_data = legit_data + "," + fraud_data
    labels = ",".join(["0.0"] * n_neg + ["1.0"] * n_pos)

    test_data = _rand_features(n_feat, 0.4, 0.9)

    lines = [
        f"@{d['input_label']} shape={n_samples},{n_feat} data={all_data}",
        f"@{d['target_label']} shape={n_samples},1 data={labels}",
        f"@{d['new_input_label']} shape=1,{n_feat} data={test_data}",
        f"@w1 shape={n_feat},{hidden} data={_rand_data(n_feat * hidden)}",
        f"@b1 shape=1,{hidden} data={_rand_data(hidden, -0.1, 0.1)}",
        f"@w2 shape={hidden},1 data={_rand_data(hidden)}",
        f"@b2 shape=1,1 data={_rand_data(1, -0.1, 0.1)}",
    ]
    return lines


def gen_domain_pairs(domain, count=500):
    """Generate training pairs for a single domain."""
    pairs = []
    d = domain
    n_feat = len(d["features"])
    feat_str = ", ".join(d["features"][:4]) + (", ..." if n_feat > 4 else "")

    for i in range(count):
        syntax = pick_syntax()
        arch = random.choice(ARCHITECTURES)
        epochs = random.choice([500, 1000, 2000, 3000, 5000])
        lr = random.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
        thresh = round(random.choice([0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7]), 1)
        activation = random.choice(["RELU", "RELU", "RELU", "TANH"])
        arch_str = f"{n_feat}-{arch['hidden']}-1"
        n_samples = random.choice([8, 10, 12, 16, 20])

        prompt_template = random.choice(d["prompts"])
        prompt = prompt_template.format(
            thresh=thresh, n_feat=n_feat, feat_str=feat_str,
            arch=arch_str, epochs=epochs,
        ) + syntax_tag(syntax)

        program_lines = _gen_program_lines(d, arch, epochs, lr, thresh, activation)

        include_data = random.random() < 0.30
        if include_data:
            data_lines = _gen_data_lines(d, arch, n_samples)
            program = "\n".join(apply_syntax(program_lines, syntax))
            data_file = "\n".join(data_lines)
            answer = f"Program ({d['prog_name']}.nml):\n{program}\n\nData file ({d['prog_name']}.nml.data):\n{data_file}"
            pairs.append(_pair(prompt, answer))
        else:
            pairs.append(_pair(prompt, apply_syntax(program_lines, syntax)))

    return pairs


def gen_infer_only(count=300):
    """Generate inference-only programs (no TNET, pre-trained weights)."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        d = random.choice(DOMAINS)
        arch = random.choice(ARCHITECTURES)
        thresh = round(random.choice([0.3, 0.4, 0.5, 0.5, 0.6, 0.7]), 1)
        activation = random.choice(["RELU", "RELU", "TANH"])

        lines = [
            _fmt("LD", "R1", "@w1"),
            _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"),
            _fmt("LD", "R4", "@b2"),
            _fmt("LD", "R0", f"@{d['new_input_label']}"),
            _fmt("MMUL", "R5", "R0", "R1"),
            _fmt("MADD", "R5", "R5", "R2"),
            _fmt(activation, "R5", "R5"),
            _fmt("MMUL", "R6", "R5", "R3"),
            _fmt("MADD", "R6", "R6", "R4"),
            _fmt("SIGM", "RA", "R6"),
            _fmt("ST", "RA", f"@{d['score_label']}"),
            _fmt("CMPI", "RE", "RA", f"#{thresh}"),
            _fmt("JMPF", "#2"),
            _fmt("LEAF", "RB", "#0.0"),
            _fmt("JUMP", "#1"),
            _fmt("LEAF", "RB", "#1.0"),
            _fmt("ST", "RB", f"@{d['decision_label']}"),
            "HALT",
        ]

        prompt = random.choice([
            f"Write NML to classify a {d['new_input_label']} using pre-trained weights. Sigmoid output, flag if >= {thresh}.",
            f"NML inference pipeline for {d['name'].replace('_', ' ')}: load weights, forward pass, threshold decision at {thresh}.",
            f"Write NML to run inference on {d['new_input_label']} with a trained {d['name'].replace('_', ' ')} model.",
        ]) + syntax_tag(syntax)

        pairs.append(_pair(prompt, apply_syntax(lines, syntax)))
    return pairs


def gen_fragment_variant(count=200):
    """Generate FRAG/LINK versions of the pipeline."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        d = random.choice(DOMAINS)
        arch = random.choice(ARCHITECTURES)
        epochs = random.choice([500, 1000, 2000])
        lr = random.choice([0.001, 0.01, 0.05])
        thresh = round(random.choice([0.4, 0.5, 0.5, 0.6]), 1)

        lines = [
            _fmt("META", "@name", f'"{d["prog_name"]}"'),
            _fmt("META", "@mode", '"adaptive"'),
            "",
            _fmt("FRAG", "setup"),
            _fmt("LD", "R1", "@w1"),
            _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"),
            _fmt("LD", "R4", "@b2"),
            "ENDF",
            "",
            _fmt("FRAG", "train"),
            _fmt("LD", "R0", f"@{d['input_label']}"),
            _fmt("LD", "R9", f"@{d['target_label']}"),
            _fmt("TNET", f"#{epochs}", f"#{lr}", "#0"),
            _fmt("ST", "R8", "@training_loss"),
            "ENDF",
            "",
            _fmt("FRAG", "infer"),
            _fmt("LD", "R0", f"@{d['new_input_label']}"),
            _fmt("MMUL", "R5", "R0", "R1"),
            _fmt("MADD", "R5", "R5", "R2"),
            _fmt("RELU", "R5", "R5"),
            _fmt("MMUL", "R6", "R5", "R3"),
            _fmt("MADD", "R6", "R6", "R4"),
            _fmt("SIGM", "RA", "R6"),
            _fmt("ST", "RA", f"@{d['score_label']}"),
            "ENDF",
            "",
            _fmt("FRAG", "main"),
            _fmt("LINK", "@setup"),
            _fmt("LINK", "@train"),
            _fmt("LINK", "@infer"),
            "HALT",
            "ENDF",
        ]

        prompt = random.choice([
            f"Write NML for {d['name'].replace('_', ' ')} using FRAG/LINK fragments: setup, train, infer.",
            f"NML adaptive program for {d['name'].replace('_', ' ')} with composable fragments.",
        ]) + syntax_tag(syntax)

        pairs.append(_pair(prompt, apply_syntax(lines, syntax)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate real-world ML application pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Generating real-world ML application pairs...")
    print(f"{'─' * 60}")

    all_pairs = []
    for domain in DOMAINS:
        dp = gen_domain_pairs(domain, count=500)
        all_pairs.extend(dp)
        print(f"  {domain['name']:<28} {len(dp):>5} pairs")

    infer = gen_infer_only(300)
    all_pairs.extend(infer)
    print(f"  {'inference_only':<28} {len(infer):>5} pairs")

    frag = gen_fragment_variant(200)
    all_pairs.extend(frag)
    print(f"  {'fragment_variant':<28} {len(frag):>5} pairs")

    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:                             {len(all_pairs):>5} pairs")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {out_path}")

    # Validate a sample
    print("\n  Validating sample...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import nml_grammar

    valid = 0
    checked = 0
    for p in random.sample(all_pairs, min(200, len(all_pairs))):
        code = p["messages"][1]["content"]
        if code.startswith("Program ("):
            parts = code.split("\n\nData file (")
            prog_section = parts[0]
            first_newline = prog_section.index("\n")
            code = prog_section[first_newline + 1:]
        try:
            report = nml_grammar.validate_grammar(code)
            if report.valid:
                valid += 1
            checked += 1
        except Exception:
            checked += 1

    print(f"  Sample validation: {valid}/{checked} ({valid/checked*100:.0f}%) grammar-valid")


if __name__ == "__main__":
    main()
