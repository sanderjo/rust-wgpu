# wgpu vs CPU

A small Rust example that:

1. Detects whatever GPU backends [wgpu](https://github.com/gfx-rs/wgpu) can find on the
   system (Vulkan / Metal / DX12 / GL) and prints each adapter it discovers.
2. Runs a compute-heavy elementwise math kernel on the GPU via a WGSL compute shader.
3. Runs the identical kernel single-threaded on the CPU.
4. Compares wall-clock time (the GPU timing includes buffer upload, dispatch, and
   readback, since that's the real cost in practice) and reports which was faster.

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

Scanning for wgpu-compatible adapters...
  Found: Intel(R) UHD Graphics (TGL GT2) | backend=Vulkan | type=IntegratedGpu | driver=Intel open-source Mesa driver
  Found: llvmpipe (LLVM 20.1.2, 256 bits) | backend=Vulkan | type=Cpu | driver=llvmpipe
  Found: Mesa Intel(R) UHD Graphics (TGL GT2) | backend=Gl | type=IntegratedGpu | driver=

Using adapter: Intel(R) UHD Graphics (TGL GT2) (backend=Vulkan, type=IntegratedGpu)

Workload: 4194304 elements, 60 kernel iterations each

Running CPU (single-threaded)...
  CPU time: 10.331967 s

Running GPU (upload + compute + download)...
  GPU time: 0.153625 s

Max abs difference between CPU and GPU results: 0.000000

=== Result ===
wgpu (GPU) was faster: 67.25x speedup over CPU.
```
