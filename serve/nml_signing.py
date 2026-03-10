"""NML v0.6 – Cryptographic signing for NML programs.

Supports HMAC-SHA256 (always available) and Ed25519 (requires the
``cryptography`` library).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class NMLSignature:
    agent: str
    algorithm: str  # "hmac-sha256" or "ed25519"
    public_key: str
    signature: str


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def generate_keypair(algorithm: str = "hmac-sha256") -> tuple[str, str]:
    """Return ``(public_key_hex, private_key_hex)``.

    For HMAC-SHA256 the "public" and "private" keys are the same shared
    secret.  For Ed25519 a proper asymmetric keypair is generated.
    """
    if algorithm == "hmac-sha256":
        secret = os.urandom(32).hex()
        return secret, secret

    if algorithm == "ed25519":
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
            from cryptography.hazmat.primitives.serialization import (
                Encoding,
                NoEncryption,
                PrivateFormat,
                PublicFormat,
            )
        except ImportError:
            raise RuntimeError(
                "ed25519 requires the 'cryptography' package: "
                "pip install cryptography"
            )

        priv = Ed25519PrivateKey.generate()
        pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        priv_bytes = priv.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )
        return pub_bytes.hex(), priv_bytes.hex()

    raise ValueError(f"Unsupported algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

_SIGN_LINE_RE = re.compile(r"^\s*✦\s+")


def compute_program_hash(nml_program: str) -> str:
    """SHA-256 of the program content, excluding any existing SIGN lines."""
    lines = [
        l for l in nml_program.splitlines()
        if not _SIGN_LINE_RE.match(l)
    ]
    body = "\n".join(lines)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sign / verify
# ---------------------------------------------------------------------------

def sign_program(
    nml_program: str,
    private_key: str,
    agent: str,
    algorithm: str = "hmac-sha256",
) -> str:
    """Return a SIGN line (✦ …) to prepend to the program."""
    prog_hash = compute_program_hash(nml_program)

    if algorithm == "hmac-sha256":
        sig = hmac.new(
            bytes.fromhex(private_key),
            prog_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        pub = private_key
    elif algorithm == "ed25519":
        sig, pub = _ed25519_sign(private_key, prog_hash)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return format_sign_line(NMLSignature(
        agent=agent,
        algorithm=algorithm,
        public_key=pub,
        signature=sig,
    ))


def verify_program(nml_program: str) -> tuple[bool, str]:
    """Extract the SIGN line, verify the signature, and return ``(valid, message)``."""
    sig_obj = None
    for line in nml_program.splitlines():
        if _SIGN_LINE_RE.match(line):
            sig_obj = parse_sign_line(line)
            break

    if sig_obj is None:
        return False, "No signature found"

    prog_hash = compute_program_hash(nml_program)

    if sig_obj.algorithm == "hmac-sha256":
        expected = hmac.new(
            bytes.fromhex(sig_obj.public_key),
            prog_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(expected, sig_obj.signature):
            return True, f"Valid HMAC-SHA256 signature by {sig_obj.agent}"
        return False, "HMAC-SHA256 signature mismatch"

    if sig_obj.algorithm == "ed25519":
        return _ed25519_verify(sig_obj, prog_hash)

    return False, f"Unknown algorithm: {sig_obj.algorithm}"


# ---------------------------------------------------------------------------
# Ed25519 helpers
# ---------------------------------------------------------------------------

def _ed25519_sign(private_key_hex: str, message: str) -> tuple[str, str]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat,
    )

    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    sig = priv.sign(message.encode("utf-8"))
    pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return sig.hex(), pub.hex()


def _ed25519_verify(sig_obj: NMLSignature, prog_hash: str) -> tuple[bool, str]:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        return False, "ed25519 verification requires the 'cryptography' package"

    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sig_obj.public_key))
        pub.verify(bytes.fromhex(sig_obj.signature), prog_hash.encode("utf-8"))
        return True, f"Valid Ed25519 signature by {sig_obj.agent}"
    except Exception as exc:
        return False, f"Ed25519 verification failed: {exc}"


# ---------------------------------------------------------------------------
# Format / Parse
# ---------------------------------------------------------------------------

def format_sign_line(sig: NMLSignature) -> str:
    return (
        f"✦  agent={sig.agent}  "
        f"key={sig.algorithm}:{sig.public_key}  "
        f"sig={sig.signature}"
    )


_SIGN_PARSE = re.compile(
    r"^\s*✦\s+"
    r"agent=(\S+)\s+"
    r"key=(\S+?):(\S+)\s+"
    r"sig=(\S+)"
)


def parse_sign_line(line: str) -> NMLSignature:
    m = _SIGN_PARSE.match(line)
    if not m:
        raise ValueError(f"Cannot parse SIGN line: {line!r}")
    return NMLSignature(
        agent=m.group(1),
        algorithm=m.group(2),
        public_key=m.group(3),
        signature=m.group(4),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NML program signing tool")
    sub = parser.add_subparsers(dest="command")

    p_kg = sub.add_parser("keygen", help="Generate a signing keypair")
    p_kg.add_argument(
        "--algorithm", default="hmac-sha256",
        choices=["hmac-sha256", "ed25519"],
    )
    p_kg.add_argument("-o", "--output", help="Write keys to JSON file")

    p_sign = sub.add_parser("sign", help="Sign an NML program")
    p_sign.add_argument("--program", required=True, help="NML file to sign")
    p_sign.add_argument("--key", required=True, help="Private key file (JSON)")
    p_sign.add_argument("--agent", required=True, help="Agent identity string")
    p_sign.add_argument("-o", "--output", help="Output file (default: stdout)")

    p_ver = sub.add_parser("verify", help="Verify a signed NML program")
    p_ver.add_argument("--program", required=True, help="Signed NML file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "keygen":
        pub, priv = generate_keypair(args.algorithm)
        keys = {"algorithm": args.algorithm, "public_key": pub, "private_key": priv}
        blob = json.dumps(keys, indent=2)
        if args.output:
            Path(args.output).write_text(blob + "\n")
            print(f"Keys written to {args.output}")
        else:
            print(blob)

    elif args.command == "sign":
        program_text = Path(args.program).read_text()
        key_data = json.loads(Path(args.key).read_text())
        sign_line = sign_program(
            program_text,
            key_data["private_key"],
            args.agent,
            algorithm=key_data.get("algorithm", "hmac-sha256"),
        )
        signed = sign_line + "\n" + program_text
        if args.output:
            Path(args.output).write_text(signed)
            print(f"Signed program written to {args.output}")
        else:
            print(signed, end="")

    elif args.command == "verify":
        program_text = Path(args.program).read_text()
        valid, message = verify_program(program_text)
        if valid:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    _cli()
