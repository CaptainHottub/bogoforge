// Vectorized PRNG/rejection-sampling in the CPU compute kernel (src/compute/cpu.rs)
// uses `std::simd` (portable_simd) to drive AVX-512 on the target-cpu=native build.
#![feature(portable_simd)]

use std::path::Path;

use anyhow::Context;
use chrono::Local;
use fern::Dispatch;
use log::LevelFilter;

pub mod compute;
pub mod config;
pub mod metrics;
pub mod net;
pub mod runtime;
pub mod scheduler;
pub mod tui;

/// Baked into the binary at build time so a bare release artifact — copied
/// somewhere with no `conf.toml.example` alongside it — can still bootstrap
/// itself on first run instead of just erroring out.
const EXAMPLE_CONFIG: &str = include_str!("../conf.toml.example");

fn main() -> anyhow::Result<()> {
    let config_path = Path::new("conf.toml");
    if !config_path.exists() {
        std::fs::write(config_path, EXAMPLE_CONFIG)
            .context("failed to write starter conf.toml")?;
        println!("no conf.toml found — wrote a starter config to ./conf.toml");
        println!("fill in [identity] (uuid, nickname, code) and re-run bogoforge");
        return Ok(());
    }

    let config = config::Config::load(config_path)?;
    init_logging(&config)?;

    log::info!("starting bogoforge (disable_tui={}, kernel_activity={})", config.ui.disable_tui, config.logging.kernel_activity);

    runtime::ForgeRuntime::new(config).startup()
}

fn init_logging(config: &config::Config) -> anyhow::Result<()> {
    let default_level = if config.logging.kernel_activity {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    let logger = Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{} [{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S.%f"),
                record.level(),
                message
            ))
        })
        .level(default_level)
        .chain(std::io::stdout());

    logger.apply()?;
    Ok(())
}
