//! Detects whatever GPU backend wgpu can find, runs a compute-heavy
//! workload on it, runs the identical workload on the CPU, and compares
//! wall-clock time (including upload/download for the GPU path, since
//! that's the cost you actually pay in real programs).

use std::time::Instant;
use wgpu::util::DeviceExt;

/// Number of f32 elements to process. 4M elements * 4 bytes = 16 MiB buffers.
const ELEMENT_COUNT: usize = 4 * 1024 * 1024;
/// How many iterations of the math kernel to run per element. Higher =
/// more compute-bound, which is where the GPU has a chance to win.
const ITERATIONS: u32 = 60;
/// Workgroup size used by the shader below; must match `WORKGROUP_SIZE` there.
const WORKGROUP_SIZE: u32 = 256;
/// wgpu/WebGPU requires dispatch_workgroups(x, ..) with x <= 65535. We stay
/// well under that and let each invocation loop over multiple elements
/// (grid-stride loop), so this works for any ELEMENT_COUNT above.
const MAX_WORKGROUPS: u32 = 60_000;

const SHADER_SRC: &str = r#"
struct Params {
    iterations: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
    let stride = num_wg.x * 256u;
    var i = gid.x;
    loop {
        if (i >= arrayLength(&input)) {
            break;
        }
        var y = input[i];
        for (var k: u32 = 0u; k < params.iterations; k = k + 1u) {
            y = sin(y) * cos(y) + sqrt(abs(y)) * 0.5;
        }
        output[i] = y;
        i = i + stride;
    }
}
"#;

/// The same per-element kernel as the shader above, so CPU and GPU do
/// identical work and the timing comparison is meaningful.
#[inline]
fn cpu_kernel(mut y: f32, iterations: u32) -> f32 {
    for _ in 0..iterations {
        y = y.sin() * y.cos() + y.abs().sqrt() * 0.5;
    }
    y
}

fn fmt_secs(d: std::time::Duration) -> String {
    format!("{:.6} s (lower is better)", d.as_secs_f64())
}

/// Reads the CPU model name from /proc/cpuinfo (Linux). Falls back to just
/// the architecture if that's not available (e.g. non-Linux, or sandboxed).
fn cpu_model_name() -> String {
    if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if let Some((key, val)) = line.split_once(':') {
                if key.trim() == "model name" {
                    return val.trim().to_string();
                }
            }
        }
    }
    format!("unknown ({} architecture)", std::env::consts::ARCH)
}

fn make_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.000_001).fract() + 0.1).collect()
}

fn run_cpu_single_threaded(input: &[f32], iterations: u32) -> (Vec<f32>, std::time::Duration) {
    let start = Instant::now();
    let out: Vec<f32> = input.iter().map(|&x| cpu_kernel(x, iterations)).collect();
    (out, start.elapsed())
}

/// Same kernel, split evenly across all available logical cores.
fn run_cpu_multi_threaded(
    input: &[f32],
    iterations: u32,
    num_threads: usize,
) -> (Vec<f32>, std::time::Duration) {
    let mut output = vec![0.0f32; input.len()];
    let chunk_size = input.len().div_ceil(num_threads).max(1);

    let start = Instant::now();
    std::thread::scope(|scope| {
        for (in_chunk, out_chunk) in input
            .chunks(chunk_size)
            .zip(output.chunks_mut(chunk_size))
        {
            scope.spawn(move || {
                for (o, &x) in out_chunk.iter_mut().zip(in_chunk.iter()) {
                    *o = cpu_kernel(x, iterations);
                }
            });
        }
    });
    (output, start.elapsed())
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

fn init_gpu() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    println!("Scanning for wgpu-compatible adapters...");
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    if adapters.is_empty() {
        println!("  No adapters enumerated on any backend (Vulkan/Metal/DX12/GL).");
    } else {
        for a in &adapters {
            let info = a.get_info();
            println!(
                "  Found: {} | backend={:?} | type={:?} | driver={}",
                info.name, info.backend, info.device_type, info.driver
            );
        }
    }

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let adapter_info = adapter.get_info();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("wgpu_vs_cpu device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    Some(GpuContext {
        device,
        queue,
        adapter_info,
    })
}

fn run_gpu(ctx: &GpuContext, input: &[f32], iterations: u32) -> (Vec<f32>, std::time::Duration) {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let start = Instant::now();

    let input_bytes: &[u8] = bytemuck::cast_slice(input);
    let buffer_size = input_bytes.len() as u64;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: input_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        iterations: u32,
        _pad: [u32; 3],
    }
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&Params {
            iterations,
            _pad: [0; 3],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kernel"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (input.len() as u32).div_ceil(WORKGROUP_SIZE).min(MAX_WORKGROUPS);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, buffer_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("failed to map readback buffer");

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buf.unmap();

    (result, start.elapsed())
}

fn main() {
    println!("=== wgpu vs CPU benchmark ===\n");

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "CPU: {} ({} logical cores, arch={})",
        cpu_model_name(),
        cores,
        std::env::consts::ARCH
    );

    let gpu = init_gpu();
    match &gpu {
        Some(ctx) => {
            println!(
                "\nUsing adapter: {} (backend={:?}, type={:?})",
                ctx.adapter_info.name, ctx.adapter_info.backend, ctx.adapter_info.device_type
            );
        }
        None => {
            println!("\nNo usable wgpu adapter available on this system.");
            println!("Falling back to CPU-only; skipping the GPU benchmark.\n");
        }
    }

    println!(
        "\nWorkload: {} elements, {} kernel iterations each\n",
        ELEMENT_COUNT, ITERATIONS
    );

    let input = make_input(ELEMENT_COUNT);

    println!("Running CPU (single-threaded)...");
    let (cpu1_out, cpu1_time) = run_cpu_single_threaded(&input, ITERATIONS);
    println!("  CPU (1 thread) time: {}", fmt_secs(cpu1_time));

    println!("\nRunning CPU ({} threads)...", cores);
    let (cpun_out, cpun_time) = run_cpu_multi_threaded(&input, ITERATIONS, cores);
    println!("  CPU ({} threads) time: {}", cores, fmt_secs(cpun_time));
    let cpu_multi_diff = max_abs_diff(&cpu1_out, &cpun_out);
    println!("  Max abs difference vs single-threaded CPU: {:.6}", cpu_multi_diff);

    let Some(ctx) = gpu else {
        println!("\n=== Result ===");
        print_comparison("CPU (1 thread)", cpu1_time, &format!("CPU ({} threads)", cores), cpun_time);
        println!("\nNo GPU to compare against. Done.");
        return;
    };

    println!("\nRunning GPU (upload + compute + download)...");
    let (gpu_out, gpu_time) = run_gpu(&ctx, &input, ITERATIONS);
    println!("  GPU time: {}", fmt_secs(gpu_time));

    // Sanity-check correctness: GPU f32 math shouldn't drift far from CPU.
    let gpu_diff = max_abs_diff(&cpu1_out, &gpu_out);
    println!("\nMax abs difference between CPU and GPU results: {:.6}", gpu_diff);
    if gpu_diff > 1e-2 {
        println!("  Warning: results diverge more than expected; treat timings with caution.");
    }

    println!("\n=== Result ===");
    print_comparison("CPU (1 thread)", cpu1_time, &format!("CPU ({} threads)", cores), cpun_time);
    print_comparison("CPU (1 thread)", cpu1_time, "GPU", gpu_time);
    print_comparison(&format!("CPU ({} threads)", cores), cpun_time, "GPU", gpu_time);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Prints how `to_time` compares against `from_time`, e.g. "GPU was faster
/// than CPU (1 thread): 78.07x speedup."
fn print_comparison(from_label: &str, from_time: std::time::Duration, to_label: &str, to_time: std::time::Duration) {
    if to_time < from_time {
        let speedup = from_time.as_secs_f64() / to_time.as_secs_f64();
        println!("{} was faster than {}: {:.2}x speedup.", to_label, from_label, speedup);
    } else {
        let slowdown = to_time.as_secs_f64() / from_time.as_secs_f64();
        println!("{} was SLOWER than {}: {:.2}x slower.", to_label, from_label, slowdown);
    }
}
