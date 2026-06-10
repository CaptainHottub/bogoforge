use std::sync::Arc;
use std::time::Duration;

use anyhow::bail;
use log::{debug, error};
use tokio::runtime::{self, Runtime};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[cfg(feature = "cuda")]
use crate::compute::cuda::CudaBackend;
#[cfg(feature = "hip")]
use crate::compute::amd::AmdBackend;
#[cfg(feature = "vk")]
use crate::compute::vk::VkBackend;
use crate::compute::{cpu::CpuBackend, run_compute_worker};
use crate::config::{Config, GpuBackendKind};
use crate::metrics::Metrics;
use crate::net::{
    types::{Chunk, Lease, RangeResult, Report},
    NetClient,
};
use crate::scheduler::{BackendHandle, Scheduler};
use crate::tui;

pub struct ForgeRuntime {
    runtime: Runtime,
    config: Arc<Config>,
}

impl ForgeRuntime {
    pub fn new(config: Config) -> Self {
        ForgeRuntime {
            runtime: runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Could not build threaded runtime."),
            config: Arc::new(config),
        }
    }

    pub fn startup(&self) -> anyhow::Result<()> {
        if !self.config.compute.use_cpu && !self.config.compute.use_gpu {
            bail!("no compute backend enabled; set use_cpu or use_gpu to true in config.toml");
        }

        let metrics = Metrics::new();
        let cancel = CancellationToken::new();
        let report_interval = self.config.reporting.report_interval;

        let maybe_tui_handle = if self.config.ui.disable_tui {
            let log_metrics = Arc::clone(&metrics);
            let log_cancel = cancel.clone();
            Some(std::thread::spawn(move || {
                loop {
                    if log_cancel.is_cancelled() {
                        break;
                    }

                    debug!(
                        "metrics: rate={:.2} session={} lifetime={} last_best={} session_best={} all_best={} status={}",
                        log_metrics.compute_rate(),
                        log_metrics.session_shuffles.load(std::sync::atomic::Ordering::Relaxed),
                        log_metrics.lifetime_shuffles.load(std::sync::atomic::Ordering::Relaxed),
                        log_metrics.last_report_best.load(std::sync::atomic::Ordering::Relaxed),
                        log_metrics.session_best.load(std::sync::atomic::Ordering::Relaxed),
                        log_metrics.all_time_best.load(std::sync::atomic::Ordering::Relaxed),
                        log_metrics.status.lock().clone(),
                    );

                    std::thread::sleep(Duration::from_millis(report_interval));
                }
            }))
        } else {
            let tui_metrics = Arc::clone(&metrics);
            let tui_cancel = cancel.clone();
            Some(std::thread::spawn(move || {
                tui::run(tui_metrics, tui_cancel);
            }))
        };

        let result = self.runtime.block_on(async {
            let (lease_tx, lease_rx) = mpsc::channel::<Lease>(4);
            let (report_tx, report_rx) = mpsc::channel::<Report>(16);
            let (done_tx, done_rx) = mpsc::channel::<(usize, RangeResult)>(16);

            let mut backends: Vec<BackendHandle> = Vec::new();

            if self.config.compute.use_cpu {
                let id = backends.len();
                let (work_tx, work_rx) = mpsc::channel::<Chunk>(2);
                let cpu_done_tx = done_tx.clone();
                let cpu_threads = self.config.compute.cpu_threads;
                tokio::task::spawn_blocking(move || {
                    run_compute_worker(id, CpuBackend::new(cpu_threads), work_rx, cpu_done_tx);
                });
                backends.push(BackendHandle::new(
                    self.config.compute.cpu_chunk_size,
                    work_tx,
                ));
            }

            if self.config.compute.use_gpu {
                let gpu = self.config.resolve_gpu().expect("invalid gpu_profile");
                let id = backends.len();
                let (work_tx, work_rx) = mpsc::channel::<Chunk>(2);
                let gpu_done_tx = done_tx.clone();
                let metrics_gpu = Arc::clone(&metrics);
                let chunk_size = gpu.chunk_size;
                let (init_tx, init_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

                match gpu.kind {
                    #[cfg(feature = "cuda")]
                    GpuBackendKind::Cuda => {
                        let (blocks, tpb) = (gpu.blocks, gpu.threads_per_block);
                        tokio::task::spawn_blocking(move || match CudaBackend::new(blocks, tpb) {
                            Ok(b) => { let _ = init_tx.send(Ok(())); run_compute_worker(id, b, work_rx, gpu_done_tx); }
                            Err(e) => { let m = format!("gpu: {e:#}"); metrics_gpu.set_status(m.clone()); let _ = init_tx.send(Err(m)); }
                        });
                    }
                    #[cfg(feature = "hip")]
                    GpuBackendKind::Hip => {
                        let (blocks, tpb) = (gpu.blocks, gpu.threads_per_block);
                        tokio::task::spawn_blocking(move || match AmdBackend::new(blocks, tpb) {
                            Ok(b) => { let _ = init_tx.send(Ok(())); run_compute_worker(id, b, work_rx, gpu_done_tx); }
                            Err(e) => { let m = format!("amd: {e:#}"); metrics_gpu.set_status(m.clone()); let _ = init_tx.send(Err(m)); }
                        });
                    }
                    #[cfg(feature = "vk")]
                    GpuBackendKind::Vulkan => {
                        let (blocks, tpb) = (gpu.blocks, gpu.threads_per_block);
                        tokio::task::spawn_blocking(move || match VkBackend::new(blocks, tpb) {
                            Ok(b) => { let _ = init_tx.send(Ok(())); run_compute_worker(id, b, work_rx, gpu_done_tx); }
                            Err(e) => { let m = format!("vk: {e:#}"); metrics_gpu.set_status(m.clone()); let _ = init_tx.send(Err(m)); }
                        });
                    }
                    _ => {
                        let _ = init_tx.send(Err(format!(
                            "gpu_profile \"{}\" requires a backend not compiled in \
                             (rebuild with --features cuda or --features hip or --features vk)",
                            self.config.compute.gpu_profile
                        )));
                    }
                }

                match tokio::time::timeout(Duration::from_secs(15), init_rx).await {
                    Ok(Ok(Ok(()))) => { backends.push(BackendHandle::new(chunk_size, work_tx)); }
                    Ok(Ok(Err(e))) => { error!("[gpu] {e}"); }
                    _ => { metrics.set_status("gpu init timed out"); }
                }
            }

            drop(done_tx);

            let net = NetClient::new(
                Arc::clone(&self.config),
                Arc::clone(&metrics),
                lease_tx,
                report_rx,
            );
            let scheduler = Scheduler::new(
                Arc::clone(&self.config),
                Arc::clone(&metrics),
                lease_rx,
                report_tx,
                backends,
                done_rx,
            );

            let net_handle = tokio::spawn(net.run(cancel.clone()));
            let sched_handle = tokio::spawn(scheduler.run(cancel.clone()));

            tokio::select! {
                res = net_handle => {
                    if let Ok(Err(e)) = res { error!("[net] exited with error: {e}"); }
                }
                res = sched_handle => {
                    if let Ok(Err(e)) = res { error!("[scheduler] exited with error: {e}"); }
                }
                _ = cancel.cancelled() => {}
            }

            cancel.cancel();
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(())
        });

        if let Some(handle) = maybe_tui_handle {
            let _ = handle.join();
        }
        result
    }
}
