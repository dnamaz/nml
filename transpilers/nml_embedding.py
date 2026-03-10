"""
NML Embedding Utilities — latent-space projection and distance helpers.

Provides:
  - Random orthogonal projection matrix generation (QR decomposition)
  - Vector projection with L2 normalisation
  - Cosine and Euclidean distance metrics
  - NML program generation for PROJ / DIST opcodes
  - .nml.data serialisation for projection matrices

CLI:
  python3 nml_embedding.py generate --input-dim 128 --embed-dim 64 --output projection.nml.data
  python3 nml_embedding.py program  --input features --matrix projection --output embedding
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import List

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False


@dataclass
class EmbeddingSpace:
    name: str
    input_dim: int
    embed_dim: int
    projection_matrix: "np.ndarray | list[list[float]]"


# ---------------------------------------------------------------------------
# Projection matrix generation
# ---------------------------------------------------------------------------

def generate_projection_matrix(input_dim: int, embed_dim: int, seed: int = 42):
    """Generate a random orthogonal projection matrix via QR decomposition.

    Returns an (input_dim, embed_dim) array-like.  Falls back to normalised
    random vectors when numpy is unavailable.
    """
    if _HAS_NUMPY:
        rng = np.random.RandomState(seed)
        raw = rng.randn(input_dim, embed_dim)
        q, _ = np.linalg.qr(raw)
        return q[:, :embed_dim]

    rng = random.Random(seed)
    matrix: list[list[float]] = []
    for _ in range(input_dim):
        row = [rng.gauss(0, 1) for _ in range(embed_dim)]
        norm = math.sqrt(sum(v * v for v in row)) or 1.0
        matrix.append([v / norm for v in row])
    return matrix


# ---------------------------------------------------------------------------
# Serialisation (.nml.data)
# ---------------------------------------------------------------------------

def save_projection_data(matrix, name: str, output_path: str) -> None:
    """Save a projection matrix in .nml.data format.

    Format: @{name} shape={rows},{cols} data={comma-separated values}
    """
    if _HAS_NUMPY:
        rows, cols = matrix.shape
        flat = ",".join(f"{v:.8f}" for v in matrix.ravel())
    else:
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0
        flat = ",".join(f"{v:.8f}" for row in matrix for v in row)

    with open(output_path, "w") as fh:
        fh.write(f"@{name} shape={rows},{cols} data={flat}\n")


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def _to_list(v) -> list[float]:
    if _HAS_NUMPY and hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


def project_vector(vector: list[float], matrix) -> list[float]:
    """Project *vector* through *matrix* and L2-normalise the result."""
    if _HAS_NUMPY:
        v = np.asarray(vector, dtype=float)
        m = np.asarray(matrix, dtype=float)
        result = v @ m
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()

    embed_dim = len(matrix[0]) if matrix else 0
    result = [0.0] * embed_dim
    for i, vi in enumerate(vector):
        for j in range(embed_dim):
            result[j] += vi * matrix[i][j]
    norm = math.sqrt(sum(r * r for r in result)) or 1.0
    return [r / norm for r in result]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Return 1 - cos(a, b)."""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a)) or 1.0
    nb = math.sqrt(sum(bi * bi for bi in b)) or 1.0
    return 1.0 - dot / (na * nb)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ---------------------------------------------------------------------------
# NML program generators
# ---------------------------------------------------------------------------

def generate_nml_projection_program(
    input_name: str,
    matrix_name: str,
    output_name: str,
    syntax: str = "symbolic",
) -> str:
    """Generate an NML program that loads, projects, and stores a vector."""
    if syntax == "symbolic":
        return "\n".join([
            f"; Project {input_name} through {matrix_name}",
            f"↓  ι  @{input_name}",
            f"↓  κ  @{matrix_name}",
            f"⟐  λ  ι  κ",
            f"↑  λ  @{output_name}",
            "◼",
        ])
    elif syntax == "verbose":
        return "\n".join([
            f"; Project {input_name} through {matrix_name}",
            f"LOAD              R0 @{input_name}",
            f"LOAD              R1 @{matrix_name}",
            f"PROJECT           R2 R0 R1",
            f"STORE             R2 @{output_name}",
            "STOP",
        ])
    else:
        return "\n".join([
            f"; Project {input_name} through {matrix_name}",
            f"LD    R0 @{input_name}",
            f"LD    R1 @{matrix_name}",
            f"PROJ  R2 R0 R1",
            f"ST    R2 @{output_name}",
            "HALT",
        ])


def generate_nml_distance_program(
    embed1_name: str,
    embed2_name: str,
    output_name: str,
    metric: int = 0,
    syntax: str = "symbolic",
) -> str:
    """Generate an NML program that computes a distance between two embeddings.

    metric 0 = cosine, 1 = euclidean.
    """
    if syntax == "symbolic":
        return "\n".join([
            f"; Distance({embed1_name}, {embed2_name})  metric={metric}",
            f"↓  ι  @{embed1_name}",
            f"↓  κ  @{embed2_name}",
            f"⟂  λ  ι  κ  #{metric}",
            f"↑  λ  @{output_name}",
            "◼",
        ])
    elif syntax == "verbose":
        return "\n".join([
            f"; Distance({embed1_name}, {embed2_name})  metric={metric}",
            f"LOAD              R0 @{embed1_name}",
            f"LOAD              R1 @{embed2_name}",
            f"DISTANCE          R2 R0 R1 #{metric}",
            f"STORE             R2 @{output_name}",
            "STOP",
        ])
    else:
        return "\n".join([
            f"; Distance({embed1_name}, {embed2_name})  metric={metric}",
            f"LD    R0 @{embed1_name}",
            f"LD    R1 @{embed2_name}",
            f"DIST  R2 R0 R1 #{metric}",
            f"ST    R2 @{output_name}",
            "HALT",
        ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NML embedding utilities")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate a projection matrix")
    gen.add_argument("--input-dim", type=int, required=True)
    gen.add_argument("--embed-dim", type=int, required=True)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output", required=True, help="Output .nml.data path")
    gen.add_argument("--name", default="projection", help="Matrix binding name")

    prg = sub.add_parser("program", help="Emit an NML projection program")
    prg.add_argument("--input", required=True, help="Input vector binding name")
    prg.add_argument("--matrix", required=True, help="Matrix binding name")
    prg.add_argument("--output", required=True, help="Output binding name")
    prg.add_argument("--syntax", default="symbolic", choices=["classic", "symbolic", "verbose"])

    args = parser.parse_args()

    if args.command == "generate":
        mat = generate_projection_matrix(args.input_dim, args.embed_dim, args.seed)
        save_projection_data(mat, args.name, args.output)
        print(f"Wrote {args.input_dim}×{args.embed_dim} projection → {args.output}")

    elif args.command == "program":
        prog = generate_nml_projection_program(
            args.input, args.matrix, args.output, args.syntax,
        )
        print(prog)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
