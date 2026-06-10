# BogoForge

BogoForge is a native Rust client for contributing compute work to the crowd-sourced bogo-sort project at `bogo.swapjs.dev`. It connects to the server over WebSocket, receives shuffle ranges, computes the best permutation found in each range, and reports results back to the service.

The client is designed to run outside the browser and supports multiple compute backends:

- CPU backend with SIMD-assisted shuffle evaluation
- AMD GPU backend through HIP/ROCm
- NVIDIA GPU backend through CUDA
- Vulkan compute backend through `wgpu`
- Terminal user interface with live board, throughput, session, and hardware telemetry

## Status

This project is performance-oriented and hardware-dependent. GPU support may require tuning the selected backend, architecture, block count, and chunk size for your machine.

## Requirements

### General

- Rust toolchain managed by `rustup`
- Git
- A working network connection to the configured bogo server

The project uses Rust features that may require the pinned toolchain in `rust-toolchain.toml`. If the pinned toolchain is not installed, Cargo/rustup should install it automatically when you build.

### Optional backend requirements

| Backend | Requirement |
| --- | --- |
| CPU | No GPU runtime required |
| AMD HIP | ROCm with `hipcc` and `libamdhip64` available |
| NVIDIA CUDA | CUDA Toolkit with `nvcc` available |
| Vulkan | Vulkan-capable GPU and drivers with `wgpu` support |

## Getting started

Clone the repository:

```bash
git clone https://github.com/mnhttn-cafe/bogoforge.git
cd bogoforge
```

Build the default CPU-capable binary:

```bash
cargo build --release
```

Run it:

```bash
cargo run --release
```

On first launch, if `conf.toml` does not exist, BogoForge writes a starter configuration file and exits. Edit `conf.toml`, fill in the `[identity]` section, choose a compute backend, then run the client again.

## Configuration

BogoForge reads `conf.toml` from the current working directory.

A minimal configuration looks like this:

```toml
[identity]
uuid = "01234567-89ab-cdef-0123-456789abcdef"
nickname = "Forge"
code = ""

[server]
url = "wss://bogo.swapjs.dev/ws"

[compute]
use_cpu = true
cpu_threads = 0
cpu_chunk_size = 50000000

use_gpu = false
gpu_profile = ""
gpu_blocks = 4096
gpu_chunk_size = 536870912

[ui]
disable_tui = false

[logging]
enabled = true
kernel_activity = false

[reporting]
report_interval = 1000
```

### Identity

`identity.uuid` must be 16 to 64 hexadecimal characters, with dashes allowed. `identity.nickname` must be 2 to 8 characters.

Leave `identity.code` empty on first registration. If the server issues a recovery code, add it to `identity.code` so future sessions can reconnect with the same identity.

### CPU settings

```toml
[compute]
use_cpu = true
cpu_threads = 0
cpu_chunk_size = 50000000
```

`cpu_threads = 0` means all logical CPU cores. Set it to a fixed number to reserve cores for other work.

### GPU settings

Enable GPU work with:

```toml
[compute]
use_gpu = true
gpu_profile = "hip"
gpu_blocks = 4096
gpu_chunk_size = 536870912
```

Valid `gpu_profile` values are:

| Profile | Backend | Notes |
| --- | --- | --- |
| `hip` | AMD HIP/ROCm | Requires building with `--features hip` |
| `cuda` | NVIDIA CUDA | Requires building with `--features cuda` |
| `vulkan` | Vulkan/wgpu | Requires building with `--features vk` |

The GPU kernels currently use 256 threads per block internally. Configure performance primarily through `gpu_blocks` and `gpu_chunk_size`.

## Running with CPU

```bash
cargo run --release
```

Use this when `use_cpu = true` and `use_gpu = false`.

## Running with NVIDIA CUDA

Install the CUDA Toolkit and confirm that `nvcc` is available:

```bash
nvcc --version
```

Build and run with CUDA support:

```bash
CUDA_ARCH=sm_89 cargo run --release --features cuda
```

Change `CUDA_ARCH` to match your GPU.


## Running with AMD HIP/ROCm

Install ROCm and confirm that `hipcc` is available:

```bash
hipcc --version
rocminfo | grep -m1 -o 'gfx[0-9a-z]*'
```

Build and run with HIP support:

```bash
ROCM_PATH=/usr \
HIP_ARCH=gfx1201 \
cargo run --release --features hip
```

Common architecture examples:

| GPU family | Example `HIP_ARCH` |
| --- | --- |
| RDNA 4 / Radeon RX 9070 XT | `gfx1201` |
| RDNA 3 iGPU / Radeon 760M | `gfx1103` |
| RDNA 2 discrete GPUs | `gfx1030`, `gfx1031`, or related |

If ROCm is installed under `/opt/rocm`, use:

```bash
ROCM_PATH=/opt/rocm \
HIP_ARCH=gfx1201 \
cargo run --release --features hip
```

If your HIP library is in a non-standard location, set `HIP_LIB_DIR` if supported by your local build script, or make sure the directory containing `libamdhip64.so` is visible to the linker.


## Running with Vulkan

Build and run with the Vulkan backend:

```bash
cargo run --release --features vk
```

Set:

```toml
[compute]
use_gpu = true
gpu_profile = "vulkan"
```

The Vulkan backend requires a Vulkan-capable adapter with 64-bit integer shader support.

## Terminal UI and logging

By default, BogoForge starts a terminal UI. To run in log-only mode, set:

```toml
[ui]
disable_tui = true
```

To show detailed kernel activity in the logs, set:

```toml
[logging]
kernel_activity = true
```

The TUI can display CPU, memory, and GPU telemetry when the relevant tools are available. NVIDIA telemetry is read from `nvidia-smi`; AMD telemetry is read from `rocm-smi`.

## License

See `LICENSE`.
