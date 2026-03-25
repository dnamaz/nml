"""
nml.py — Python binding for the NML runtime

Uses ctypes to call the NML C library (libnml.so / libnml.dll).
Falls back to subprocess-based execution if the shared library is not found.

Usage:
    import nml

    # Library mode (fast, no subprocess)
    vm = nml.VM()
    vm.load("model.nml", "model.nml.data")
    vm.set_input("input", [[1.0, 2.0, 3.0, 4.0]])
    vm.run()
    result = vm.get_output("output")
    print(result)  # numpy array

    # Subprocess mode (always works, no shared lib needed)
    result = nml.run_program("model.nml", "model.nml.data")
"""

import ctypes
import ctypes.util
import os
import sys
import subprocess
import tempfile
from pathlib import Path

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ── Library discovery ────────────────────────────────────────────────────────

def _find_nml_lib():
    """Find libnml.so / libnml.dll in common locations."""
    candidates = [
        Path(__file__).parent.parent / 'libnml.so',
        Path(__file__).parent.parent / 'libnml.dll',
        Path(__file__).parent.parent / 'nml.so',
        Path(__file__).parent.parent / 'nml.dll',
    ]
    found = ctypes.util.find_library('nml')
    if found:
        candidates.append(Path(found))
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def _probe_binary(path):
    """Return True if the binary at *path* can be executed (not a stub/wrong arch)."""
    try:
        result = subprocess.run(
            [str(path)], capture_output=True, timeout=5
        )
        # Any clean exit (0 or 1 for "no args given") is fine; crash codes are not
        return result.returncode in (0, 1)
    except (OSError, subprocess.TimeoutExpired):
        return False


def _find_nml_binary():
    """Find a working nml CLI binary.

    Checks multiple candidate names and verifies each one actually runs
    before returning it (guards against cross-compiled ELF stubs on Windows).
    """
    root = Path(__file__).parent.parent
    # On Windows prefer .exe variants and check several known names
    if sys.platform == 'win32':
        names = ['nml_new.exe', 'nml.exe', 'nml']
    else:
        names = ['nml', 'nml.exe']

    for name in names:
        c = root / name
        if c.exists() and _probe_binary(c):
            return str(c)
    return None


# ── ctypes structs ───────────────────────────────────────────────────────────

class NMLTensor(ctypes.Structure):
    _fields_ = [
        ('data',  ctypes.POINTER(ctypes.c_float)),
        ('ndim',  ctypes.c_int),
        ('shape', ctypes.c_int * 4),
        ('size',  ctypes.c_int),
    ]

    def to_numpy(self):
        if not _HAS_NUMPY:
            raise RuntimeError("numpy is required for tensor conversion")
        if not self.data:
            return None
        arr = np.ctypeslib.as_array(self.data, shape=(self.size,)).copy()
        if self.ndim > 1:
            arr = arr.reshape([self.shape[i] for i in range(self.ndim)])
        return arr

    def to_list(self):
        """Return tensor data as a plain Python list (no numpy required)."""
        if not self.data:
            return None
        return [self.data[i] for i in range(self.size)]


# ── Library-based VM ─────────────────────────────────────────────────────────

class VM:
    """NML VM instance backed by the shared library (ctypes)."""

    _lib = None  # shared library handle (class-level, loaded once)

    @classmethod
    def _load_lib(cls):
        if cls._lib is not None:
            return cls._lib
        lib_path = _find_nml_lib()
        if not lib_path:
            return None
        try:
            lib = ctypes.CDLL(lib_path)
            # Set up function signatures
            lib.nml_vm_create.restype  = ctypes.c_void_p
            lib.nml_vm_create.argtypes = []

            lib.nml_vm_destroy.restype  = None
            lib.nml_vm_destroy.argtypes = [ctypes.c_void_p]

            lib.nml_vm_load.restype  = ctypes.c_int
            lib.nml_vm_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]

            lib.nml_vm_load_mem.restype  = ctypes.c_int
            lib.nml_vm_load_mem.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]

            lib.nml_vm_set_input.restype  = ctypes.c_int
            lib.nml_vm_set_input.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ]

            lib.nml_vm_run.restype  = ctypes.c_int
            lib.nml_vm_run.argtypes = [ctypes.c_void_p]

            lib.nml_vm_get_output.restype  = NMLTensor
            lib.nml_vm_get_output.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

            lib.nml_vm_get_register.restype  = NMLTensor
            lib.nml_vm_get_register.argtypes = [ctypes.c_void_p, ctypes.c_int]

            lib.nml_vm_last_error.restype  = ctypes.c_char_p
            lib.nml_vm_last_error.argtypes = [ctypes.c_void_p]

            lib.nml_version.restype  = ctypes.c_char_p
            lib.nml_version.argtypes = []

            lib.nml_opcode_count.restype  = ctypes.c_int
            lib.nml_opcode_count.argtypes = []

            cls._lib = lib
            return lib
        except OSError:
            return None

    def __init__(self):
        lib = self._load_lib()
        if lib is None:
            raise RuntimeError(
                "NML shared library not found. "
                "Build with: make libnml.so  (Linux) or  make libnml.dll  (Windows)"
            )
        self._lib = lib
        self._handle = lib.nml_vm_create()
        if not self._handle:
            raise MemoryError("nml_vm_create() returned NULL")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.nml_vm_destroy(self._handle)
            self._handle = None

    def load(self, program_path, data_path=None):
        """Load a program from files. Raises RuntimeError on failure."""
        rc = self._lib.nml_vm_load(
            self._handle,
            str(program_path).encode(),
            str(data_path).encode() if data_path else None,
        )
        if rc != 0:
            raise RuntimeError(f"nml_vm_load failed (code {rc}): {self.last_error()}")
        return self

    def load_str(self, program_text, data_text=None):
        """Load a program from in-memory strings. Raises RuntimeError on failure."""
        rc = self._lib.nml_vm_load_mem(
            self._handle,
            program_text.encode() if isinstance(program_text, str) else program_text,
            data_text.encode()    if isinstance(data_text,    str) else data_text,
        )
        if rc != 0:
            raise RuntimeError(f"nml_vm_load_mem failed (code {rc}): {self.last_error()}")
        return self

    def set_input(self, name, data):
        """Set a named input tensor.

        data may be a list, nested list, or numpy array.
        Copies data — caller retains ownership.
        """
        if _HAS_NUMPY:
            arr = np.asarray(data, dtype=np.float32)
            shape = (ctypes.c_int * len(arr.shape))(*arr.shape)
            flat = arr.flatten()
            c_data = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ndim = len(arr.shape)
        else:
            # Flatten manually without numpy
            def _flatten(x):
                if isinstance(x, (list, tuple)):
                    out = []
                    for item in x:
                        out.extend(_flatten(item))
                    return out
                return [float(x)]

            def _shape(x):
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    return [len(x)] + _shape(x[0])
                return []

            flat_list = _flatten(data)
            shp = _shape(data) if isinstance(data, (list, tuple)) else [len(flat_list)]
            ndim = len(shp)
            shape = (ctypes.c_int * ndim)(*shp)
            FloatArray = ctypes.c_float * len(flat_list)
            c_flat = FloatArray(*flat_list)
            c_data = ctypes.cast(c_flat, ctypes.POINTER(ctypes.c_float))

        rc = self._lib.nml_vm_set_input(
            self._handle, name.encode(), c_data, shape, ndim,
        )
        if rc != 0:
            raise RuntimeError(f"nml_vm_set_input failed (code {rc}): {self.last_error()}")
        return self

    def run(self):
        """Execute the loaded program. Raises RuntimeError on failure."""
        rc = self._lib.nml_vm_run(self._handle)
        if rc != 0:
            raise RuntimeError(f"nml_vm_run failed (code {rc}): {self.last_error()}")
        return self

    def get_output(self, name='output'):
        """Get a named output tensor.

        Returns a numpy array if numpy is available, otherwise a flat Python list.
        Raises KeyError if the output is not found.
        """
        t = self._lib.nml_vm_get_output(self._handle, name.encode())
        if _HAS_NUMPY:
            result = t.to_numpy()
        else:
            result = t.to_list()
        if result is None:
            raise KeyError(f"Output '{name}' not found or not F32")
        return result

    def get_register(self, index):
        """Get a register tensor by index (0–31).

        Returns numpy array / list, or None if the register is empty.
        """
        t = self._lib.nml_vm_get_register(self._handle, index)
        if _HAS_NUMPY:
            return t.to_numpy()
        return t.to_list()

    def last_error(self):
        """Return the last error message string."""
        err = self._lib.nml_vm_last_error(self._handle)
        return err.decode() if err else 'unknown error'

    @staticmethod
    def version():
        """Return the NML runtime version string."""
        lib = VM._load_lib()
        if lib:
            return lib.nml_version().decode()
        return 'unknown'

    @staticmethod
    def opcode_count():
        """Return the number of opcodes compiled into the library."""
        lib = VM._load_lib()
        if lib:
            return lib.nml_opcode_count()
        return 0


# ── Subprocess-based fallback ─────────────────────────────────────────────────

def run_program(program_path, data_path=None, inputs=None):
    """
    Run an NML program via subprocess (no shared library needed).

    inputs: dict of {name: array-like} — written to a temp .nml.data file.
    Returns: stdout string from the NML runtime.
    Raises RuntimeError if the process exits with a non-zero status.
    """
    binary = _find_nml_binary()
    if not binary:
        raise RuntimeError(
            "NML binary not found. Build with: make nml  (or make nml.exe on Windows)"
        )

    cmd = [binary, str(program_path)]
    if data_path:
        cmd.append(str(data_path))

    extra_data = ''
    if inputs:
        for name, arr in inputs.items():
            if _HAS_NUMPY:
                arr = np.asarray(arr, dtype=np.float32)
                shape_str = ','.join(str(s) for s in arr.shape)
                data_str  = ','.join(f'{v:.8g}' for v in arr.flatten())
            else:
                def _flatten(x):
                    if isinstance(x, (list, tuple)):
                        out = []
                        for item in x:
                            out.extend(_flatten(item))
                        return out
                    return [float(x)]
                def _shape(x):
                    if isinstance(x, (list, tuple)) and len(x) > 0:
                        return [len(x)] + _shape(x[0])
                    return []
                flat = _flatten(arr)
                shp  = _shape(arr) if isinstance(arr, (list, tuple)) else [len(flat)]
                shape_str = ','.join(str(s) for s in shp)
                data_str  = ','.join(f'{v:.8g}' for v in flat)
            extra_data += f'@{name} shape={shape_str} dtype=f32 data={data_str}\n'

    tmp_path = None
    if extra_data:
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.nml.data', delete=False) as f:
            f.write(extra_data)
            tmp_path = f.name
        cmd.append(tmp_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"NML runtime error (exit {result.returncode}):\n{result.stderr}"
            )
        return result.stdout
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Convenience function ──────────────────────────────────────────────────────

def infer(program_path, data_path=None, input_data=None,
          input_name='input', output_name='output'):
    """
    One-shot inference: load model, set input, run, return output.

    Uses library mode if the shared library is available; otherwise falls back
    to subprocess mode (output is returned as raw stdout string in that case).

    Returns a numpy array (library mode with numpy) or raw stdout string
    (subprocess fallback).
    """
    try:
        vm = VM()
        vm.load(program_path, data_path)
        if input_data is not None:
            vm.set_input(input_name, input_data)
        vm.run()
        return vm.get_output(output_name)
    except RuntimeError:
        # Fall back to subprocess mode
        return run_program(
            program_path, data_path,
            inputs={input_name: input_data} if input_data is not None else None,
        )
