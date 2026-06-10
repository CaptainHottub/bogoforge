#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t *z) {
    *z += 0x9e3779b97f4a7c15ULL;
    uint64_t v = *z;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

__device__ __forceinline__ void xseed(uint64_t seed, uint32_t s[4]) {
    uint64_t z = seed;
    uint64_t a = splitmix64(&z);
    uint64_t b = splitmix64(&z);
    s[0] = (uint32_t)(a & 0xffffffffULL);
    s[1] = (uint32_t)(a >> 32);
    s[2] = (uint32_t)(b & 0xffffffffULL);
    s[3] = (uint32_t)(b >> 32);
    if (!s[0] && !s[1] && !s[2] && !s[3]) s[0] = 1;
}

__device__ __forceinline__ uint32_t xnext(uint32_t s[4]) {
    const uint32_t result = rotl32(s[0] + s[3], 7) + s[0];
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11);
    return result;
}

__device__ __forceinline__ void warp_reduce(
    int *score, unsigned long long *index, uint32_t *lane_out
) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        int                ps = __shfl_xor_sync(0xffffffffull, *score,    d);
        unsigned long long pi = __shfl_xor_sync(0xffffffffull, *index,    d);
        uint32_t           pl = __shfl_xor_sync(0xffffffffull, *lane_out, d);
        if (ps > *score || (ps == *score && pl < *lane_out)) {
            *score    = ps;
            *index    = pi;
            *lane_out = pl;
        }
    }
}


extern "C" __global__
__launch_bounds__(256, 2)
void bogo_range_kernel(
    uint64_t            base_seed,
    uint64_t            lo,
    uint64_t            hi,
    unsigned long long *dev_best_and_bid,
    uint8_t            *dev_all_arrays,
    uint64_t           *dev_all_indices
) {
    __shared__ unsigned int lds[25 * 256];
    unsigned int *arr = lds + threadIdx.x;  // arr[p] = lds[p * 256 + threadIdx.x]

    const uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint32_t lane   = threadIdx.x & 31;
    const uint32_t warpid = threadIdx.x >> 5;

    int      best_score = -1;
    uint8_t  best_arr[25];
    uint64_t best_idx   = lo;

    // 1-cycle stride addition (Replaces 64-bit multiplication inside loop)
    const uint64_t magic_mult = 0x9e3779b97f4a7c15ULL;
    const uint64_t z_stride   = stride * magic_mult;
    uint64_t current_seed     = base_seed + (lo + tid) * magic_mult;

    for (uint64_t iter = lo + tid; iter < hi; iter += stride) {
        uint32_t s[4];
        xseed(current_seed, s);
        current_seed += z_stride; 

        // PHASE 1: PURE ALU
        // Compute all random indices before touching memory to avoid stalls.
        // xint_c is directly embedded to violently force compile-time constant propagation.
        uint32_t js[24];
        #pragma unroll 24
        for (int i = 24; i > 0; i--) {
            const uint32_t max_val = i + 1;
            const uint32_t threshold = (uint32_t)(0x100000000ULL % max_val);
            uint32_t x;
            do { x = xnext(s); } while (x < threshold);
            js[i-1] = x % max_val; 
        }

        #pragma unroll
        for (int p = 0; p < 25; p++) arr[p * 256] = (unsigned int)(p + 1);

        // PHASE 2: MEMORY & SCORING
        int c = 0;
        
        #pragma unroll 24
        for (int i = 24; i > 0; i--) {
            const uint32_t j = js[i-1];
            const unsigned int tmp = arr[i * 256];
            const unsigned int val = arr[j * 256]; // The value moving to position i
            
            arr[i * 256] = val;
            arr[j * 256] = tmp;
            
            // Branchless Tally: arr[i * 256] is now finalized. Score it instantly.
            c += (val == (unsigned int)(i + 1));
        }
        
        // Element 0 is implicitly finalized by the end. Score it.
        c += (arr[0] == 1);

        // Final score check (Original 25-iteration scoring loop is entirely deleted)
        if (c > best_score) {
            best_score = c;
            best_idx   = iter;
            #pragma unroll
            for (int p = 0; p < 25; p++) best_arr[p] = (uint8_t)arr[p * 256];
            if (c == 25) break;
        }
    }

    __syncthreads();

    int                *smem_scores  = (int *)lds;
    unsigned long long *smem_indices = (unsigned long long *)(smem_scores  + 8);
    uint8_t            *smem_arrays  = (uint8_t *)           (smem_indices + 8);

    int                ws = best_score;
    unsigned long long wi = (unsigned long long)best_idx;
    uint32_t           wl = lane;
    warp_reduce(&ws, &wi, &wl);

    uint8_t warp_arr[25];
    #pragma unroll
    for (int p = 0; p < 25; p++) {
        unsigned v = __shfl_sync(0xffffffffull, (unsigned)best_arr[p], wl);
        if (lane == 0) warp_arr[p] = (uint8_t)v;
    }

    if (lane == 0) {
        smem_scores [warpid] = ws;
        smem_indices[warpid] = wi;
        #pragma unroll
        for (int p = 0; p < 25; p++) smem_arrays[warpid * 25 + p] = warp_arr[p];
    }
    
    __syncthreads();

    if (warpid != 0) return;

    const uint32_t n_warps = blockDim.x >> 5;
    int                bs = (lane < n_warps) ? smem_scores [lane] : -1;
    unsigned long long bi = (lane < n_warps) ? smem_indices[lane] : 0ULL;
    uint32_t           bw = lane;
    warp_reduce(&bs, &bi, &bw);

    if (lane == 0 && bs >= 0) {
        const uint32_t blk = blockIdx.x;
        dev_all_indices[blk] = (uint64_t)bi;
        #pragma unroll
        for (int p = 0; p < 25; p++)
            dev_all_arrays[(uint64_t)blk * 25 + p] = smem_arrays[bw * 25 + p];
        atomicMax(dev_best_and_bid,
            ((unsigned long long)(uint32_t)bs << 32) | (unsigned long long)blk);
    }
}