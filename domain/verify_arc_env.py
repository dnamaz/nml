#!/usr/bin/env python3
"""
verify_arc_env.py — Check that the NML training environment is ready.

Run this after setup_arc_training.bat:
    python verify_arc_env.py

Checks:
  - Python version
  - PyTorch + XPU availability
  - IPEX-LLM installation
  - bitsandbytes (4-bit quantization)
  - transformers / peft / trl / datasets
  - Training data present
  - Tokenizer present
  - Quick XPU smoke test (allocates a tensor on the GPU)
"""

import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).parent
PASS = "  [OK]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    line = f"{icon}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def try_import(name: str):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        return mod, ver
    except ImportError:
        return None, None


print()
print("=" * 60)
print("  NML Training Environment Verification")
print("=" * 60)
print()

all_ok = True

# ── Python ─────────────────────────────────────────────────
print("Python")
pv = sys.version_info
ok = pv >= (3, 10)
all_ok &= check("Python >= 3.10", ok, f"{pv.major}.{pv.minor}.{pv.micro}")

# ── PyTorch ────────────────────────────────────────────────
print()
print("PyTorch")
torch, tv = try_import("torch")
ok = torch is not None
all_ok &= check("torch installed", ok, tv or "not found")

if torch:
    xpu_ok = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
    cuda_ok = torch.cuda.is_available()
    cpu_ok = True

    if xpu_ok:
        dev_count = torch.xpu.device_count()
        dev_name = torch.xpu.get_device_name(0) if dev_count > 0 else "?"
        check("XPU (Intel Arc) available", True, f"{dev_count} device(s): {dev_name}")
    else:
        check("XPU (Intel Arc) available", False, "not found — check oneAPI + driver")
        all_ok = False

    if cuda_ok:
        check("CUDA available (bonus)", True, f"{torch.cuda.get_device_name(0)}")
    elif not xpu_ok:
        check("CUDA available", False, "no accelerator detected!")

    # Quick XPU smoke test
    if xpu_ok:
        try:
            t = torch.ones(4, 4, device="xpu")
            result = (t @ t).sum().item()
            check("XPU tensor smoke test", abs(result - 64.0) < 1e-3, f"result={result:.1f}")
        except Exception as e:
            check("XPU tensor smoke test", False, str(e))
            all_ok = False

# ── IPEX-LLM ───────────────────────────────────────────────
print()
print("IPEX-LLM (Intel optimized LLM loading)")
ipex_llm, ipex_ver = try_import("ipex_llm")
if ipex_llm:
    check("ipex_llm installed", True, ipex_ver)
else:
    # Try plain intel_extension_for_pytorch
    ipex, ipex_ver2 = try_import("intel_extension_for_pytorch")
    if ipex:
        print(f"{WARN}  ipex_llm not found, but intel_extension_for_pytorch present ({ipex_ver2})")
        print(f"       Training will fall back to standard transformers (4-bit still works).")
    else:
        print(f"{WARN}  ipex_llm not found — training will use standard transformers path.")
        print(f"       This is OK but may be slower on XPU.")

# ── bitsandbytes ───────────────────────────────────────────
print()
print("Quantization")
bnb, bnb_ver = try_import("bitsandbytes")
ok = bnb is not None
all_ok &= check("bitsandbytes installed", ok, bnb_ver or "not found")

if bnb:
    try:
        import bitsandbytes as bnb_mod
        from bitsandbytes import functional as F
        check("bitsandbytes functional import", True)
    except Exception as e:
        check("bitsandbytes functional import", False, str(e))

# ── Training libraries ─────────────────────────────────────
print()
print("Training stack")
for lib, min_ver in [
    ("transformers", "4.40"),
    ("peft", "0.10"),
    ("trl", "0.8"),
    ("datasets", "2.18"),
    ("accelerate", "0.28"),
]:
    mod, ver = try_import(lib)
    ok = mod is not None
    if ok:
        # Simple version check: compare first two parts
        try:
            parts = [int(x) for x in ver.split(".")[:2]]
            min_parts = [int(x) for x in min_ver.split(".")[:2]]
            version_ok = parts >= min_parts
            check(f"{lib} >= {min_ver}", version_ok, ver)
            if not version_ok:
                all_ok = False
        except Exception:
            check(f"{lib} installed", True, ver)
    else:
        all_ok &= check(f"{lib} installed", False, "not found")

# ── Training data ──────────────────────────────────────────
print()
print("Training data")
for split in ["train", "valid", "test"]:
    p = ROOT / "domain" / "output" / "training" / f"{split}.jsonl"
    ok = p.exists() and p.stat().st_size > 1_000_000
    size_mb = p.stat().st_size / 1e6 if p.exists() else 0
    check(f"{split}.jsonl present", ok, f"{size_mb:.0f} MB" if p.exists() else "missing")
    if not ok:
        all_ok = False

# ── Tokenizer ──────────────────────────────────────────────
print()
print("Tokenizer")
tok_dir = ROOT / "domain" / "output" / "tokenizer" / "nml_bpe_4096"
tok_json = tok_dir / "tokenizer.json"
ok = tok_json.exists()
check("NML BPE tokenizer built", ok, str(tok_dir) if ok else "run nml_tokenizer_train.py")
if not ok:
    all_ok = False

# ── Summary ────────────────────────────────────────────────
print()
print("=" * 60)
if all_ok:
    print("  All checks passed — ready to train!")
    print()
    print("  Quick test (1.5B, 200 steps):")
    print("    python train_nml.py \\")
    print("        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\")
    print("        --steps 200 --batch 2 --grad-acc 2 \\")
    print("        --output domain/output/model/test-run")
    print()
    print("  Full Stage-1 (7B, 30K steps):")
    print("    python train_nml.py")
else:
    print("  Some checks FAILED — see details above.")
    if torch and not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
        print()
        print("  XPU not available. Common fixes:")
        print("    1. Run setup_arc_training.bat (activates oneAPI setvars)")
        print("    2. Update Intel Arc driver to 32.0.101.6647 or later")
        print("    3. Enable Resizable BAR in BIOS")
        print("       (BIOS > Advanced > PCI Config > Resizable BAR = Enabled)")
        print("    4. Reinstall PyTorch XPU:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/xpu")
print("=" * 60)
print()
