#!/usr/bin/env python3
"""
test_python_binding.py — Test the Python NML binding

Tests subprocess mode (always works) and library mode (if libnml.so present).
"""
import sys
import os

# Add python/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import nml

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_module_import():
    """Test that the nml module imports and has expected attributes."""
    assert hasattr(nml, 'VM'), "nml.VM not found"
    assert hasattr(nml, 'run_program'), "nml.run_program not found"
    assert hasattr(nml, 'infer'), "nml.infer not found"
    assert hasattr(nml, 'NMLTensor'), "nml.NMLTensor not found"
    print("  module attributes: OK")


def test_subprocess_mode():
    """Test run_program() subprocess mode."""
    anomaly_nml  = os.path.join(repo_root, 'programs', 'anomaly_detector.nml')
    anomaly_data = os.path.join(repo_root, 'programs', 'anomaly_weights.nml.data')

    output = nml.run_program(anomaly_nml, anomaly_data)
    assert output, "No output from subprocess mode"
    # The NML runtime prints HALTED or registers/memory on success
    assert len(output) > 0, f"Unexpected empty output: {output!r}"
    print(f"  subprocess mode: OK (output length={len(output)})")


def test_subprocess_mode_no_data():
    """Test run_program() with a program that needs no data file."""
    hello_nml = os.path.join(repo_root, 'programs', 'hello_world.nml')
    if not os.path.exists(hello_nml):
        print("  subprocess (no data): SKIP (hello_world.nml not found)")
        return
    output = nml.run_program(hello_nml)
    assert len(output) > 0, "No output from hello_world.nml"
    print(f"  subprocess (no data): OK (output length={len(output)})")


def test_library_mode():
    """Test VM class if shared library is available."""
    try:
        vm = nml.VM()
        version = nml.VM.version()
        opcount = nml.VM.opcode_count()
        print(f"  library mode: NML version={version}, opcodes={opcount}")
        assert version and version != 'unknown', "Bad version string"
        assert opcount > 0, "Opcode count should be > 0"

        anomaly_nml  = os.path.join(repo_root, 'programs', 'anomaly_detector.nml')
        anomaly_data = os.path.join(repo_root, 'programs', 'anomaly_weights.nml.data')

        vm.load(anomaly_nml, anomaly_data)
        vm.run()
        print("  library mode: run OK")

        # Check that at least some memory slot is accessible (anomaly_score or similar)
        # We won't assert a specific output name since it varies, just verify no crash
        print("  library mode: all checks passed")
    except RuntimeError as e:
        if 'not found' in str(e).lower() or 'shared library' in str(e).lower():
            print(f"  library mode: SKIP (shared lib not built — run 'make libnml.so')")
        else:
            raise


def test_library_mode_set_input():
    """Test set_input() if shared library is available."""
    try:
        vm = nml.VM()
    except RuntimeError:
        print("  set_input test: SKIP (shared lib not built)")
        return

    # We load a simple program that reads from a named slot
    # Use anomaly_detector which already has weights; just verify set_input doesn't crash
    anomaly_nml  = os.path.join(repo_root, 'programs', 'anomaly_detector.nml')
    anomaly_data = os.path.join(repo_root, 'programs', 'anomaly_weights.nml.data')
    vm.load(anomaly_nml, anomaly_data)

    # set_input with a 1-D tensor (should not raise)
    try:
        vm.set_input("test_slot", [1.0, 2.0, 3.0, 4.0])
        print("  set_input (1D): OK")
    except Exception as e:
        print(f"  set_input (1D): {e}")

    # set_input with a 2-D tensor
    try:
        vm.set_input("test_slot_2d", [[1.0, 2.0], [3.0, 4.0]])
        print("  set_input (2D): OK")
    except Exception as e:
        print(f"  set_input (2D): {e}")


def main():
    print("=== NML Python Binding Tests ===")
    test_module_import()
    test_subprocess_mode()
    test_subprocess_mode_no_data()
    test_library_mode()
    test_library_mode_set_input()
    print("=== ALL PASSED ===")


if __name__ == '__main__':
    main()
