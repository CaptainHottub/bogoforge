use std::ffi::CStr;
use std::ptr;

use anyhow::{bail, Result};
use log::debug;

use crate::net::types::RangeResult;
use super::ComputeBackend;

static HSACO: &[u8] = include_bytes!(env!("KERNEL_HSACO_PATH"));

type HipError = i32;
const HIP_SUCCESS: HipError = 0;
const HIP_MEMCPY_D2H: u32 = 2;

#[link(name = "amdhip64")]
extern "C" {
    fn hipInit(flags: u32) -> HipError;
    fn hipSetDevice(dev: i32) -> HipError;
    fn hipModuleLoadData(module: *mut *mut (), image: *const ()) -> HipError;
    fn hipModuleUnload(module: *mut ()) -> HipError;
    fn hipModuleGetFunction(func: *mut *mut (), module: *mut (), name: *const i8) -> HipError;
    fn hipMalloc(ptr: *mut *mut (), size: usize) -> HipError;
    fn hipFree(ptr: *mut ()) -> HipError;
    fn hipMemcpy(dst: *mut (), src: *const (), size: usize, kind: u32) -> HipError;
    fn hipMemset(ptr: *mut (), value: i32, size: usize) -> HipError;
    fn hipModuleLaunchKernel(
        f: *mut (),
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_mem: u32,
        stream: *mut (),
        kernel_params: *mut *mut (),
        extra: *mut *mut (),
    ) -> HipError;
    fn hipStreamCreate(stream: *mut *mut ()) -> HipError;
    fn hipStreamSynchronize(stream: *mut ()) -> HipError;
    fn hipStreamDestroy(stream: *mut ()) -> HipError;
    fn hipGetErrorString(err: HipError) -> *const i8;
}

fn hip_check(err: HipError) -> Result<()> {
    if err == HIP_SUCCESS { return Ok(()); }
    let msg = unsafe { CStr::from_ptr(hipGetErrorString(err)) };
    bail!("HIP error: {}", msg.to_string_lossy())
}

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct AmdBackend {
    module:   *mut (),
    func:     *mut (),
    stream:   *mut (),
    blocks:   u32,
    dev_best:    *mut u64,
    dev_arrays:  *mut u8,
    dev_indices: *mut u64,
    host_arrays:  Vec<u8>,
    host_indices: Vec<u64>,
}

unsafe impl Send for AmdBackend {}

impl AmdBackend {
    pub fn new(blocks: u32, threads_per_block: u32) -> Result<Self> {
        anyhow::ensure!(
            threads_per_block == 256,
            "kernel.hip hardcodes 256 threads/block; set amd_threads_per_block = 256"
        );

        unsafe {
            hip_check(hipInit(0))?;
            hip_check(hipSetDevice(0))?;

            let mut module = ptr::null_mut::<()>();
            hip_check(hipModuleLoadData(
                &mut module,
                HSACO.as_ptr() as *const (),
            ))?;

            let name = c"bogo_range_kernel";
            let mut func = ptr::null_mut::<()>();
            hip_check(hipModuleGetFunction(&mut func, module, name.as_ptr()))?;

            let mut stream = ptr::null_mut::<()>();
            hip_check(hipStreamCreate(&mut stream))?;

            let n = (blocks as usize) * 256;

            let mut dev_best    = ptr::null_mut::<()>();
            let mut dev_arrays  = ptr::null_mut::<()>();
            let mut dev_indices = ptr::null_mut::<()>();
            hip_check(hipMalloc(&mut dev_best,    size_of::<u64>()))?;
            hip_check(hipMalloc(&mut dev_arrays,  n * 25))?;
            hip_check(hipMalloc(&mut dev_indices, n * size_of::<u64>()))?;

            Ok(Self {
                module,
                func,
                stream,
                blocks,
                dev_best:    dev_best    as *mut u64,
                dev_arrays:  dev_arrays  as *mut u8,
                dev_indices: dev_indices as *mut u64,
                host_arrays:  vec![0u8;  n * 25],
                host_indices: vec![0u64; n],
            })
        }
    }
}

impl Drop for AmdBackend {
    fn drop(&mut self) {
        unsafe {
            hipFree(self.dev_best    as *mut ());
            hipFree(self.dev_arrays  as *mut ());
            hipFree(self.dev_indices as *mut ());
            hipStreamDestroy(self.stream);
            hipModuleUnload(self.module);
        }
    }
}

impl ComputeBackend for AmdBackend {
    fn compute_range(&mut self, base_seed: u64, lo: u64, hi: u64) -> RangeResult {
        unsafe {
            // Reset the winner slot to 0.
            hip_check(hipMemset(self.dev_best as *mut (), 0, size_of::<u64>()))
                .expect("hipMemset dev_best");

            // Build the kernel parameter list.
            // hipModuleLaunchKernel expects an array of pointers, each pointing
            // to the corresponding argument value (including device pointers).
            let mut p_base_seed = base_seed;
            let mut p_lo        = lo;
            let mut p_hi        = hi;
            let mut p_dev_best    = self.dev_best;
            let mut p_dev_arrays  = self.dev_arrays;
            let mut p_dev_indices = self.dev_indices;

            let mut kernel_params: [*mut (); 6] = [
                &mut p_base_seed    as *mut u64 as *mut (),
                &mut p_lo           as *mut u64 as *mut (),
                &mut p_hi           as *mut u64 as *mut (),
                &mut p_dev_best     as *mut *mut u64 as *mut (),
                &mut p_dev_arrays   as *mut *mut u8  as *mut (),
                &mut p_dev_indices  as *mut *mut u64 as *mut (),
            ];

            debug!("[hip] launching kernel base_seed={} lo={} hi={} blocks={}", base_seed, lo, hi, self.blocks);
            hip_check(hipModuleLaunchKernel(
                self.func,
                self.blocks, 1, 1,
                256, 1, 1,
                0,
                self.stream,
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )).expect("hipModuleLaunchKernel");

            hip_check(hipStreamSynchronize(self.stream)).expect("hipStreamSynchronize");
            debug!("[hip] kernel complete, copying results to host");

            // Read winner.
            let mut host_best = 0u64;
            hip_check(hipMemcpy(
                &mut host_best as *mut u64 as *mut (),
                self.dev_best as *const (),
                size_of::<u64>(),
                HIP_MEMCPY_D2H,
            )).expect("D2H dev_best");

            if host_best == 0 {
                return RangeResult { lo, hi, best_correct: -1, best_arr: [0u8; 25], best_index: lo };
            }

            let best_score = (host_best >> 32) as i32;
            let best_bid   = (host_best & 0xffff_ffff) as usize;

            // D2H all block results then extract winner's slice.
            hip_check(hipMemcpy(
                self.host_arrays.as_mut_ptr() as *mut (),
                self.dev_arrays as *const (),
                self.host_arrays.len(),
                HIP_MEMCPY_D2H,
            )).expect("D2H arrays");
            hip_check(hipMemcpy(
                self.host_indices.as_mut_ptr() as *mut (),
                self.dev_indices as *const (),
                self.host_indices.len() * size_of::<u64>(),
                HIP_MEMCPY_D2H,
            )).expect("D2H indices");

            let mut best_arr = [0u8; 25];
            best_arr.copy_from_slice(&self.host_arrays[best_bid * 25 .. best_bid * 25 + 25]);

            RangeResult {
                lo,
                hi,
                best_correct: best_score,
                best_arr,
                best_index: self.host_indices[best_bid],
            }
        }
    }
}
