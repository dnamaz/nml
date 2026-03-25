# NML API Reference

Covers the C library API (`runtime/nml_api.h`) and the Python binding (`python/nml.py`) introduced in Phase 5.

For build instructions see [Building the Library](#building-the-library). For the CLI interface see [docs/NML_Usage_Guide.md](NML_Usage_Guide.md).

---

## C API (runtime/nml_api.h)

The C API provides a clean library interface for embedding NML inference in other applications. Each `nml_vm_t*` handle is fully independent — no global state, thread-safe as long as each thread uses its own handle.

Include the header and link against the shared or static library:

```c
#include "runtime/nml_api.h"
```

### Types

#### nml_vm_t

Opaque handle to a VM instance. Do not dereference — use only through the API functions below.

```c
typedef void nml_vm_t;
```

#### NMLTensor

View into VM memory returned by output and register accessors. The `data` pointer is valid until `nml_vm_destroy()` or the next `nml_vm_run()` call on the same VM. **Do not free `data`.**

```c
typedef struct {
    const float *data;  /* F32 element array; NULL if not found or wrong dtype */
    int          ndim;
    int          shape[4];
    int          size;  /* total element count */
} NMLTensor;
```

Copy the data if you need it beyond the VM lifetime:

```c
NMLTensor t = nml_vm_get_output(vm, "output");
float *my_copy = malloc(t.size * sizeof(float));
memcpy(my_copy, t.data, t.size * sizeof(float));
```

---

### nml_vm_create

```c
nml_vm_t *nml_vm_create(void);
```

Allocate and initialize a new VM instance. Returns `NULL` on allocation failure.

```c
nml_vm_t *vm = nml_vm_create();
if (!vm) { fprintf(stderr, "out of memory\n"); exit(1); }
```

---

### nml_vm_destroy

```c
void nml_vm_destroy(nml_vm_t *vm);
```

Free all resources owned by `vm`. Safe to call with `NULL` (no-op).

```c
nml_vm_destroy(vm);
vm = NULL;
```

---

### nml_vm_load

```c
int nml_vm_load(nml_vm_t *vm, const char *program_path, const char *data_path);
```

Load an NML program from a `.nml` file. `data_path` may be `NULL` if the program has no weight file.

Returns `0` on success, non-zero on error. On failure, `nml_vm_last_error(vm)` contains a description.

```c
int rc = nml_vm_load(vm, "model.nml", "model.nml.data");
if (rc != 0) {
    fprintf(stderr, "load failed: %s\n", nml_vm_last_error(vm));
}
```

---

### nml_vm_load_mem

```c
int nml_vm_load_mem(nml_vm_t *vm, const char *program_text, const char *data_text);
```

Load a program from null-terminated in-memory strings. `data_text` may be `NULL`. Useful for embedding NML programs as string literals or loading from a database.

```c
const char *prog = "LD R0 @x\nSCLR R1 R0 #2.0\nST R1 @y\nHALT\n";
const char *data = "@x shape=1 dtype=f32 data=21.0\n";
nml_vm_load_mem(vm, prog, data);
```

---

### nml_vm_set_input

```c
int nml_vm_set_input(nml_vm_t *vm, const char *name,
                     const float *data, const int *shape, int shape_len);
```

Write a named input tensor into the VM's memory model. The data is copied — the caller retains ownership of `data` and `shape`.

- `name`: memory slot name without the `@` prefix (e.g., `"sensor_data"`)
- `data`: F32 elements in row-major order
- `shape`: array of dimension sizes
- `shape_len`: number of dimensions (1–4)

```c
float input[] = {0.9f, 0.1f, 0.95f, 0.3f};
int   shape[] = {1, 4};
nml_vm_set_input(vm, "sensor_data", input, shape, 2);
```

Call this after `nml_vm_load` and before `nml_vm_run`. You can call it multiple times to update inputs between runs.

---

### nml_vm_run

```c
int nml_vm_run(nml_vm_t *vm);
```

Execute the loaded program until `HALT` is reached. Returns `0` on success (HALT reached within the cycle limit), non-zero on error (bad instruction, cycle limit exceeded, division by zero, etc.).

```c
if (nml_vm_run(vm) != 0) {
    fprintf(stderr, "runtime error: %s\n", nml_vm_last_error(vm));
}
```

The VM can be run multiple times after updating inputs with `nml_vm_set_input`.

---

### nml_vm_get_output

```c
NMLTensor nml_vm_get_output(nml_vm_t *vm, const char *name);
```

Get a named output tensor stored by an `ST` instruction in the NML program. `name` is the memory slot name without `@`.

Returns an `NMLTensor` view into VM memory. `tensor.data` is `NULL` if the slot is not found or has the wrong dtype.

```c
NMLTensor out = nml_vm_get_output(vm, "anomaly_score");
if (out.data) {
    printf("score: %.4f\n", out.data[0]);
}
```

---

### nml_vm_get_register

```c
NMLTensor nml_vm_get_register(nml_vm_t *vm, int reg_index);
```

Get a register tensor by index (0–31). Useful for inspecting intermediate results without modifying the NML program to add `ST` instructions.

```c
NMLTensor r3 = nml_vm_get_register(vm, 3);  /* R3 */
```

---

### nml_vm_last_error

```c
const char *nml_vm_last_error(nml_vm_t *vm);
```

Get the last error message string. Always returns a valid pointer (never `NULL`). Returns `"ok"` when no error has occurred.

---

### nml_version

```c
const char *nml_version(void);
```

Returns the NML runtime version string (e.g., `"1.0.0"`).

---

### nml_opcode_count

```c
int nml_opcode_count(void);
```

Returns the number of opcodes compiled into this build. Standard build returns 89.

---

### Complete C Example

```c
#include <stdio.h>
#include <stdlib.h>
#include "runtime/nml_api.h"

int main(void) {
    nml_vm_t *vm = nml_vm_create();
    if (!vm) return 1;

    if (nml_vm_load(vm, "programs/anomaly_detector.nml",
                        "programs/anomaly_weights.nml.data") != 0) {
        fprintf(stderr, "load: %s\n", nml_vm_last_error(vm));
        nml_vm_destroy(vm);
        return 1;
    }

    float sensor[] = {0.9f, 0.1f, 0.95f, 0.3f};
    int   shape[]  = {1, 4};
    nml_vm_set_input(vm, "sensor_data", sensor, shape, 2);

    if (nml_vm_run(vm) != 0) {
        fprintf(stderr, "run: %s\n", nml_vm_last_error(vm));
        nml_vm_destroy(vm);
        return 1;
    }

    NMLTensor out = nml_vm_get_output(vm, "anomaly_score");
    if (out.data) {
        printf("anomaly_score = %.6f\n", out.data[0]);
    }

    nml_vm_destroy(vm);
    return 0;
}
```

Compile:
```bash
gcc -O2 -o detector detector.c -L. -lnml -Ipath/to/nml
# or with static library:
gcc -O2 -o detector detector.c libnml.a -lm
```

---

## Python Binding (python/nml.py)

```python
import sys
sys.path.insert(0, '/path/to/nml/python')
import nml
```

Requires no installation beyond adding the `python/` directory to `sys.path`. numpy is optional — the binding falls back to plain Python lists.

### VM class

Library mode: uses ctypes to call `libnml.so` / `libnml.dll` directly. Faster than subprocess — no process spawn, no I/O.

The shared library is searched in order:
1. `../libnml.so` relative to `nml.py`
2. `../libnml.dll`
3. System library path via `ctypes.util.find_library('nml')`

If the library is not found, `VM()` raises `RuntimeError` with the build command to use.

#### VM()

```python
vm = nml.VM()
```

Creates a new VM instance. Raises `RuntimeError` if the shared library is not found.

#### VM.load(program_path, data_path=None)

```python
vm.load('model.nml', 'model.nml.data')
vm.load('model.nml')  # no weights
```

Load from files. Returns `self` for chaining. Raises `RuntimeError` on failure.

#### VM.load_str(program_text, data_text=None)

```python
prog = "LD R0 @x\nSCLR R1 R0 #2.0\nST R1 @y\nHALT\n"
data = "@x shape=1 dtype=f32 data=21.0\n"
vm.load_str(prog, data)
```

Load from in-memory strings.

#### VM.set_input(name, data)

```python
vm.set_input('sensor_data', [[0.9, 0.1, 0.95, 0.3]])
vm.set_input('sensor_data', np.array([[0.9, 0.1, 0.95, 0.3]], dtype=np.float32))
```

`data` may be a nested list, tuple, or numpy array. Shape is inferred automatically. Data is copied — caller retains ownership.

#### VM.run()

```python
vm.run()
```

Execute until HALT. Raises `RuntimeError` on runtime error. Returns `self`.

#### VM.get_output(name='output')

```python
result = vm.get_output('anomaly_score')   # numpy array if numpy available
result = vm.get_output()                  # defaults to 'output'
```

Returns a numpy array (or flat list without numpy). Raises `KeyError` if the output is not found.

#### VM.get_register(index)

```python
r3 = vm.get_register(3)   # R3 as numpy array or list; None if empty
```

#### VM.last_error()

```python
msg = vm.last_error()   # str
```

#### VM.version() / VM.opcode_count()

```python
print(nml.VM.version())        # e.g. "1.0.0"
print(nml.VM.opcode_count())   # e.g. 89
```

Static methods — do not require a VM instance.

#### Chained usage

```python
result = (nml.VM()
    .load('model.nml', 'weights.nml.data')
    .set_input('input', data)
    .run()
    .get_output('output'))
```

---

### run_program()

```python
stdout = nml.run_program(program_path, data_path=None, inputs=None)
```

Subprocess fallback. Always works — no shared library required.

- `program_path`: path to `.nml` file
- `data_path`: path to `.nml.data` file (optional)
- `inputs`: dict of `{name: array-like}` — written to a temporary `.nml.data` file and passed as a second data file

Returns the full stdout string from the NML runtime. Raises `RuntimeError` if the process exits non-zero.

```python
stdout = nml.run_program(
    'programs/anomaly_detector.nml',
    'programs/anomaly_weights.nml.data',
)
print(stdout)

# Inject inputs programmatically
stdout = nml.run_program(
    'model.nml', 'weights.nml.data',
    inputs={'sensor_data': [[0.9, 0.1, 0.95, 0.3]]},
)
```

---

### infer()

```python
result = nml.infer(program_path, data_path=None, input_data=None,
                   input_name='input', output_name='output')
```

One-shot convenience function. Tries library mode first; falls back to subprocess transparently.

- Returns a numpy array in library mode (with numpy), or the raw stdout string in subprocess mode.

```python
import nml

score = nml.infer(
    'programs/anomaly_detector.nml',
    'programs/anomaly_weights.nml.data',
    input_data=[[0.9, 0.1, 0.95, 0.3]],
    input_name='sensor_data',
    output_name='anomaly_score',
)
print(f"anomaly score: {score[0][0]:.4f}")
```

---

## Building the Library

### Linux (shared)

```bash
make libnml.so
# or manually:
gcc -O2 -shared -fPIC -DNML_BUILD_LIB -o libnml.so runtime/nml.c -lm
```

### Linux (shared, with OpenBLAS)

```bash
make libnml-fast.so
# or manually:
gcc -O2 -shared -fPIC -DNML_BUILD_LIB -DNML_USE_OPENBLAS -o libnml-fast.so runtime/nml.c -lm -lopenblas
```

### Windows (DLL)

```bash
make libnml.dll
# or manually (MinGW):
gcc -O2 -shared -DNML_BUILD_LIB -o libnml.dll runtime/nml.c -lm
```

### Static Library

```bash
make libnml.a
# or manually:
gcc -O2 -DNML_BUILD_LIB -c -o nml_lib.o runtime/nml.c
ar rcs libnml.a nml_lib.o
```

Link a C program against the static library:

```bash
gcc -O2 -o myapp myapp.c libnml.a -lm
```

### Verifying the build

```bash
# Check the exported symbols are present
nm -D libnml.so | grep nml_vm
# Should list: nml_vm_create, nml_vm_destroy, nml_vm_load, etc.

# Quick Python smoke test (no model needed)
python3 -c "
import sys; sys.path.insert(0,'python')
import nml
print('version:', nml.VM.version())
print('opcodes:', nml.VM.opcode_count())
"
```
