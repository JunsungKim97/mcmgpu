# Benchmark Porting Playbook (OpenCL -> mgpusim)

This document captures the exact workflow used to port Rodinia Hotspot into this tree, and generalizes it for future benchmark ports.

## Scope

- Input: OpenCL kernel source (`kernels.cl`).
- Output:
  - benchmark package under `mgpusim/benchmarks/<suite>/<name>/`
  - sample entrypoint under `mgpusim/samples/<name>/main.go`
  - optional acceptance test hook

## 1) Folder and file layout

For each new benchmark, create:

- `mgpusim/benchmarks/<suite>/<name>/benchmark.go`
- `mgpusim/benchmarks/<suite>/<name>/native/kernels.cl`
- `mgpusim/benchmarks/<suite>/<name>/kernels.hsaco` (generated artifact)
- `mgpusim/benchmarks/<suite>/<name>/makefile`
- `mgpusim/samples/<name>/main.go`

Optional:

- `mgpusim/tests/acceptance/acceptance_test.py` entry

## 2) HSACO build (must be gfx803)

Akita GCN3 path expects gfx803 code object.

```bash
/opt/rocm/bin/clang-ocl -mcpu=gfx803 kernels.cl -o kernels.hsaco
```

Recommended in containerized ROCm environment if host toolchain is missing.

Also useful for inspection:

```bash
/opt/rocm/bin/clang-ocl -mcpu=gfx803 kernels.cl -S -o kernels.asm
readelf -n kernels.hsaco
```

## 3) Host benchmark implementation pattern

In `benchmark.go`:

1. Embed HSACO:
   - `//go:embed kernels.hsaco`
2. Load kernel with:
   - `kernels.LoadProgramFromMemory(hsacoBytes, "<kernel_name>")`
3. Implement benchmark interface methods:
   - `SelectGPU`
   - `Run`
   - `Verify`
   - `SetUnifiedMemory` (and optional LASP variants if needed)
4. Typical run flow:
   - `initMem()` -> `exec()` -> `MemCopyD2H()` -> `Verify()`

## 4) Kernel argument ABI rules (critical)

Most common source of hard-to-debug failures is mismatched kernarg layout.

### Required rules

- Use fixed-width Go types (`int32`, `uint32`, `float32`, `int64`), never `int`.
- Use `driver.GPUPtr` for global pointers.
- Include hidden args emitted by `clang-ocl`:
  - `HiddenGlobalOffsetX/Y/Z` (`int64`, usually set to 0)
- Keep 8-byte fields 8-byte aligned in serialized stream.

### Important mgpusim-specific detail

Driver uses `encoding/binary.Write` to serialize kernel args before H2D copy.
That means serialization is field-by-field packed and does not automatically preserve Go in-memory padding.

So if ABI requires alignment gaps, add explicit padding fields in the struct.

Example pattern:

```go
type KernelArgs struct {
    // user args...
    Step float32

    // explicit 4-byte gap so next int64 starts at 8-byte aligned offset
    PadBeforeHidden int32

    HiddenGlobalOffsetX int64
    HiddenGlobalOffsetY int64
    HiddenGlobalOffsetZ int64
}
```

## 5) Validate kernarg layout before full run

Before large runs, verify:

1. HSACO metadata size:
   - `co.KernargSegmentByteSize`
2. Serialized Go arg size:
   - `binary.Write(..., &KernelArgs{})` buffer length
3. They must match exactly.

If mismatch occurs, fix struct fields/order/padding first.

## 6) Execution and verification strategy

Start with small shape and few iterations:

```bash
./hotspot -platform-type=ideal -verify -rows=8 -cols=8 -iterations=1
```

Then scale up gradually.

Note:
- If platform-specific TLB crashes appear in `mcmgpu`, first confirm correctness in `ideal` to separate benchmark-port issues from platform/configuration issues.

## 7) Hotspot port status in this repo

Implemented here:

- `mgpusim/benchmarks/rodinia/hotspot/benchmark.go`
- `mgpusim/benchmarks/rodinia/hotspot/native/kernels.cl`
- `mgpusim/benchmarks/rodinia/hotspot/makefile`
- `mgpusim/benchmarks/rodinia/hotspot/kernels.hsaco`
- `mgpusim/samples/hotspot/main.go`
- `mgpusim/tests/acceptance/acceptance_test.py` (entry added)

Key fix learned from this port:

- Kernarg struct must include explicit padding before hidden int64 args where ABI alignment requires it, because mgpusim serializes via `binary.Write`.

## 8) Checklist for next benchmark

- [ ] OpenCL kernel compiles with `clang-ocl -mcpu=gfx803`
- [ ] Kernel symbol name confirmed for `LoadProgramFromMemory`
- [ ] `KernelArgs` uses fixed-width types only
- [ ] Hidden offsets present and initialized to 0
- [ ] Explicit padding added where ABI alignment requires
- [ ] Serialized `KernelArgs` size == `KernargSegmentByteSize`
- [ ] Sample runner added under `mgpusim/samples/<name>/main.go`
- [ ] `Verify()` implemented and passing on small case
- [ ] Optional acceptance entry added
