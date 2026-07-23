# wgpu vs CPU

A small Rust example that:

1. Detects and prints the CPU model and whatever GPU backends
   [wgpu](https://github.com/gfx-rs/wgpu) can find on the system
   (Vulkan / Metal / DX12 / GL).
2. Runs a compute-heavy elementwise math kernel on the CPU single-threaded, then again
   split across all logical cores, then on the GPU via a WGSL compute shader.
3. Compares wall-clock time across all three (the GPU timing includes buffer upload,
   dispatch, and readback, since that's the real cost in practice) and reports the
   speedup/slowdown between each pair.

## Requirements

- Rust (stable), install via [rustup](https://rustup.rs) if you don't have it.
- A GPU driver that exposes Vulkan, Metal, DX12, or OpenGL. If none is found, wgpu may
  still find a software (CPU) adapter such as `llvmpipe`, or find no adapter at all —
  the program handles both cases and falls back to CPU-only.

No other setup is needed; `cargo` pulls in the required crates (`wgpu`, `pollster`,
`bytemuck`) automatically.

## Build

```sh
cargo build --release
```

## Run

```sh
cargo run --release
```

(Using `--release` matters a lot here — the CPU path in particular is much slower
in debug builds.)

## Tuning the workload

Two constants at the top of `src/main.rs` control the benchmark size:

- `ELEMENT_COUNT` — how many `f32` elements to process.
- `ITERATIONS` — how many times the math kernel repeats per element. Higher values
  make the workload more compute-bound, which favors the GPU; a very small workload
  may let the CPU win once GPU upload/download overhead is factored in.

## Example output

```
=== wgpu vs CPU benchmark ===

CPU: 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (4 logical cores, arch=x86_64)
Scanning for wgpu-compatible adapters...
  Found: Intel(R) UHD Graphics (TGL GT2) | backend=Vulkan | type=IntegratedGpu | driver=Intel open-source Mesa driver
  Found: llvmpipe (LLVM 20.1.2, 256 bits) | backend=Vulkan | type=Cpu | driver=llvmpipe
  Found: Mesa Intel(R) UHD Graphics (TGL GT2) | backend=Gl | type=IntegratedGpu | driver=

Using adapter: Intel(R) UHD Graphics (TGL GT2) (backend=Vulkan, type=IntegratedGpu)

Workload: 4194304 elements, 60 kernel iterations each

Running CPU (single-threaded)...
  CPU (1 thread) time: 10.458695 s (lower is better)

Running CPU (4 threads)...
  CPU (4 threads) time: 3.683740 s (lower is better)
  Max abs difference vs single-threaded CPU: 0.000000

Running GPU (upload + compute + download)...
  GPU time: 0.133207 s (lower is better)

Max abs difference between CPU and GPU results: 0.000000

=== Result ===
CPU (4 threads) was faster than CPU (1 thread): 2.84x speedup.
GPU was faster than CPU (1 thread): 78.51x speedup.
GPU was faster than CPU (4 threads): 27.65x speedup.
```

## When is wgpu (GPU) actually faster than CPU?

GPU compute wins when a workload is **massively parallel, arithmetic-heavy, and large
enough to amortize fixed overhead**. The traits that make a workload GPU-friendly:

1. **High arithmetic intensity** — lots of FLOPs per byte moved. Memory-bound work
   (e.g. one add per byte) doesn't leave the GPU anything to hide its transfer cost
   behind.
2. **Data parallelism with no cross-element dependencies** — thousands of GPU cores
   only pay off if the work splits into independent lanes.
3. **Uniform control flow** — GPUs execute in lockstep groups (warps/wavefronts);
   heavy per-element branching stalls parallelism.
4. **Enough total work to amortize fixed cost** — every dispatch pays for shader
   compilation, buffer upload, and readback, which is milliseconds of overhead
   regardless of workload size.

The benchmark above hits all four: 4M independent floats, each doing hundreds of
trig/sqrt ops with no dependency on any other element — hence the ~78x speedup.

The flip side — where GPU compute is a poor fit — is small, sequential, or
memory-bound work: parsing/decoding text formats, variable-length or
escape-sequence-based formats (where each byte's position depends on the bytes before
it), or any one-off computation too small to amortize the dispatch/upload/readback
overhead. There, a plain CPU loop (ideally with SIMD) usually wins, sometimes by two or
three orders of magnitude.

Rule of thumb: GPU shines on big, parallel, compute-dense workloads (image/video
processing, ML inference/training, physics simulations, hashing at scale, ray
tracing); it's a poor fit for small, sequential, or memory-bound tasks.
