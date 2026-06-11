use std::simd::prelude::*;
use std::simd::Mask;

use orx_parallel::*;

use super::ComputeBackend;
use crate::net::types::RangeResult;

const IDENTITY: [u8; 25] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
];

const GOLDEN: u64 = 0x9e3779b97f4a7c15;

// AVX-512 (we build with target-cpu=native on Zen 5) gives us 512-bit vectors,
// i.e. 16 lanes of u32. We run that many independent shuffle attempts side by
// side, vectorizing the xoshiro128++ PRNG and the rejection-sampling modulo.
// The Fisher-Yates swap itself stays scalar per lane — the swap index is
// data-dependent, and there's no AVX-512 shape for "pick from one of 25 source
// arrays using a different index per lane" that beats just doing 16 scalar
// swaps.
//
// (Tried interleaving two of these streams for extra ILP — measured slower in
// practice, likely register/cache pressure from doubling the live state and
// the per-lane arrays. Reverted; one 16-wide stream per batch wins.)
const LANES: usize = 16;
// Zen 3 CPU's get better performance when LANES is set to 8

type V = Simd<u32, LANES>;
type M = Mask<i32, LANES>;

struct LocalBest {
    score: i32,
    arr: [u8; 25],
    index: u64,
}

pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    /// `threads` = 0 means use all logical cores.
    pub fn new(threads: usize) -> Self {
        Self { num_threads: threads }
    }
}

impl ComputeBackend for CpuBackend {
    fn compute_range(&mut self, base_seed: u64, lo: u64, hi: u64) -> RangeResult {
        let threads = self.num_threads;

        let total = hi - lo;
        let n_batches = total / LANES as u64;
        let batch_end = lo + n_batches * LANES as u64;

        // Vectorized body: LANES (16) independent shuffles per iteration.
        let batched_best = (0usize..n_batches as usize)
            .par()
            .num_threads(threads)
            .map(|b| {
                let base_i = lo + b as u64 * LANES as u64;
                let seeds: [u64; LANES] = core::array::from_fn(|l| {
                    let i = base_i + l as u64;
                    base_seed.wrapping_add(i.wrapping_mul(GOLDEN))
                });

                let mut best = LocalBest { score: -1, arr: [0u8; 25], index: base_i };
                for (lane, (score, arr)) in one_shuffle_batch(seeds).into_iter().enumerate() {
                    if score > best.score {
                        best = LocalBest { score, arr, index: base_i + lane as u64 };
                    }
                }
                best
            })
            .reduce(|a, b| if a.score >= b.score { a } else { b });

        // Scalar tail for whatever's left over — at most LANES-1 items, far
        // too few to bother dispatching to the thread pool over.
        let tail_best = (batch_end..hi)
            .map(|i| {
                let seed_i = base_seed.wrapping_add(i.wrapping_mul(GOLDEN));
                let (score, arr) = one_shuffle(seed_i);
                LocalBest { score, arr, index: i }
            })
            .fold(None::<LocalBest>, |acc, cur| match acc {
                None => Some(cur),
                Some(best) => Some(if best.score >= cur.score { best } else { cur }),
            });

        let result = match (batched_best, tail_best) {
            (Some(a), Some(b)) => if a.score >= b.score { a } else { b },
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => LocalBest { score: -1, arr: [0u8; 25], index: lo },
        };

        RangeResult {
            lo,
            hi,
            best_correct: result.score,
            best_arr: result.arr,
            best_index: result.index,
        }
    }
}

#[inline(always)]
fn splitmix64(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9e3779b97f4a7c15);
    let mut v = *z;
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    v ^ (v >> 31)
}

#[inline(always)]
fn xseed(seed: u64) -> [u32; 4] {
    let mut z = seed;
    let a = splitmix64(&mut z);
    let b = splitmix64(&mut z);
    let mut s = [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32];
    if s == [0, 0, 0, 0] { s[0] = 1; }
    s
}

#[inline(always)]
fn xnext(s: &mut [u32; 4]) -> u32 {
    let result = s[0].wrapping_add(s[3]).rotate_left(7).wrapping_add(s[0]);
    let t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = s[3].rotate_left(11);
    result
}

// MAX is a const generic so LLVM sees a constant divisor and turns both %
// operations into multiply+shift (~3 cycles vs ~30 for integer division) —
// the same trick as Barrett reduction, just emitted for us by the compiler.
#[inline(always)]
fn xint_const<const MAX: u32>(s: &mut [u32; 4]) -> u32 {
    let thr: u32 = (0x100000000u64 % MAX as u64) as u32;
    loop {
        let x = xnext(s);
        if x >= thr { return x % MAX; }
    }
}

macro_rules! fy_step {
    ($s:expr, $arr:expr, $i:literal) => {{
        let j = xint_const::<{ $i + 1 }>(&mut $s) as usize;
        $arr.swap($i, j);
    }};
}

#[inline(always)]
fn one_shuffle(seed: u64) -> (i32, [u8; 25]) {
    let mut s = xseed(seed);
    let mut arr = IDENTITY;

    fy_step!(s, arr, 24);
    fy_step!(s, arr, 23);
    fy_step!(s, arr, 22);
    fy_step!(s, arr, 21);
    fy_step!(s, arr, 20);
    fy_step!(s, arr, 19);
    fy_step!(s, arr, 18);
    fy_step!(s, arr, 17);
    fy_step!(s, arr, 16);
    fy_step!(s, arr, 15);
    fy_step!(s, arr, 14);
    fy_step!(s, arr, 13);
    fy_step!(s, arr, 12);
    fy_step!(s, arr, 11);
    fy_step!(s, arr, 10);
    fy_step!(s, arr,  9);
    fy_step!(s, arr,  8);
    fy_step!(s, arr,  7);
    fy_step!(s, arr,  6);
    fy_step!(s, arr,  5);
    fy_step!(s, arr,  4);
    fy_step!(s, arr,  3);
    fy_step!(s, arr,  2);
    fy_step!(s, arr,  1);

    let correct = arr.iter().zip(1u8..).map(|(&a, b)| (a == b) as i32).sum();
    (correct, arr)
}

// ─── Vectorized hybrid ──────────────────────────────────────────────────────
//
// This has to come out bit-identical to `one_shuffle` above — (seed, index)
// -> (score, arr) gets compared across the CPU/GPU backends and re-verified
// later, so any drift here is a correctness bug, not just a perf regression.
//
// The only data-dependent branch in the scalar algorithm is the redraw inside
// `xint_const` ("if the draw lands in the rejection zone, draw again"). We
// can't vectorize that faithfully without stalling every lane's PRNG in
// lockstep whenever *any* lane needs a redraw — which would desync the lanes
// that didn't need one from what the scalar path would've produced for them.
// Since that only happens for roughly one in several million full shuffles,
// we take the easy way out: run the fast path assuming it never happens, OR a
// "tainted" mask of `x < threshold` hits across all 24 steps, and recompute
// the rare tainted lanes with the scalar `one_shuffle`. Hot path stays fully
// vectorized, output stays exact.

#[inline(always)]
fn rotl_simd(x: V, k: u32) -> V {
    (x << Simd::splat(k)) | (x >> Simd::splat(32 - k))
}

#[inline(always)]
fn xnext_simd(s: &mut [V; 4]) -> V {
    let result = rotl_simd(s[0] + s[3], 7) + s[0];
    let t = s[1] << Simd::splat(9u32);
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl_simd(s[3], 11);
    result
}

/// Seeds LANES independent xoshiro states (scalar splitmix64, same as the
/// scalar path) and transposes them into vector form — one `Simd<u32, LANES>`
/// per state word. Runs once per batch, not once per step, so the scalar
/// seeding cost is negligible next to 24 vectorized rounds.
#[inline(always)]
fn load_state(seeds: [u64; LANES]) -> [V; 4] {
    let scalar: [[u32; 4]; LANES] = core::array::from_fn(|l| xseed(seeds[l]));
    core::array::from_fn(|word| {
        let lanes: [u32; LANES] = core::array::from_fn(|l| scalar[l][word]);
        Simd::from_array(lanes)
    })
}

macro_rules! fy_step_simd {
    ($s:expr, $arrs:expr, $tainted:expr, $i:literal) => {{
        const MAX: u32 = $i + 1;
        const THR: u32 = (0x100000000u64 % MAX as u64) as u32;

        let x = xnext_simd(&mut $s);
        $tainted |= x.simd_lt(Simd::splat(THR));

        // Constant divisor again — multiply+shift, fully vectorized.
        let j = (x % Simd::splat(MAX)).to_array();
        for lane in 0..LANES {
            $arrs[lane].swap($i, j[lane] as usize);
        }
    }};
}

/// Runs LANES (16) independent shuffles vectorized, then patches up the rare
/// "tainted" lanes via the scalar path so the output matches `one_shuffle`
/// exactly, lane for lane.
#[inline(always)]
fn one_shuffle_batch(seeds: [u64; LANES]) -> [(i32, [u8; 25]); LANES] {
    let mut s = load_state(seeds);
    let mut arrs = [IDENTITY; LANES];
    let mut tainted: M = Mask::splat(false);

    fy_step_simd!(s, arrs, tainted, 24);
    fy_step_simd!(s, arrs, tainted, 23);
    fy_step_simd!(s, arrs, tainted, 22);
    fy_step_simd!(s, arrs, tainted, 21);
    fy_step_simd!(s, arrs, tainted, 20);
    fy_step_simd!(s, arrs, tainted, 19);
    fy_step_simd!(s, arrs, tainted, 18);
    fy_step_simd!(s, arrs, tainted, 17);
    fy_step_simd!(s, arrs, tainted, 16);
    fy_step_simd!(s, arrs, tainted, 15);
    fy_step_simd!(s, arrs, tainted, 14);
    fy_step_simd!(s, arrs, tainted, 13);
    fy_step_simd!(s, arrs, tainted, 12);
    fy_step_simd!(s, arrs, tainted, 11);
    fy_step_simd!(s, arrs, tainted, 10);
    fy_step_simd!(s, arrs, tainted,  9);
    fy_step_simd!(s, arrs, tainted,  8);
    fy_step_simd!(s, arrs, tainted,  7);
    fy_step_simd!(s, arrs, tainted,  6);
    fy_step_simd!(s, arrs, tainted,  5);
    fy_step_simd!(s, arrs, tainted,  4);
    fy_step_simd!(s, arrs, tainted,  3);
    fy_step_simd!(s, arrs, tainted,  2);
    fy_step_simd!(s, arrs, tainted,  1);

    let mut results: [(i32, [u8; 25]); LANES] = core::array::from_fn(|lane| {
        let arr = arrs[lane];
        let correct = arr.iter().zip(1u8..).map(|(&a, b)| (a == b) as i32).sum();
        (correct, arr)
    });

    for (lane, was_tainted) in tainted.to_array().into_iter().enumerate() {
        if was_tainted {
            results[lane] = one_shuffle(seeds[lane]);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of this rewrite: the vectorized path has to match the
    /// scalar one exactly, lane for lane, seed for seed.
    #[test]
    fn batch_matches_scalar() {
        for base in [0u64, 1, 12345, u64::MAX / 3, GOLDEN] {
            let seeds: [u64; LANES] =
                core::array::from_fn(|l| base.wrapping_add((l as u64).wrapping_mul(GOLDEN)));

            let batched = one_shuffle_batch(seeds);
            for (lane, seed) in seeds.into_iter().enumerate() {
                assert_eq!(
                    batched[lane],
                    one_shuffle(seed),
                    "lane {lane} (seed {seed}) diverged from scalar one_shuffle"
                );
            }
        }
    }

    /// Sweep consecutive chunks the way `compute_range` actually derives its
    /// seeds, so we're exercising realistic — not just conveniently-aligned —
    /// input, and giving the rare "tainted" rescue path a chance to fire.
    #[test]
    fn batch_matches_scalar_sweep() {
        let base_seed = 0x1234_5678_9abc_def0u64;
        for chunk in 0u64..64 {
            let base_i = chunk * LANES as u64;
            let seeds: [u64; LANES] = core::array::from_fn(|l| {
                let i = base_i + l as u64;
                base_seed.wrapping_add(i.wrapping_mul(GOLDEN))
            });

            let batched = one_shuffle_batch(seeds);
            for (lane, seed) in seeds.into_iter().enumerate() {
                assert_eq!(batched[lane], one_shuffle(seed));
            }
        }
    }
}
