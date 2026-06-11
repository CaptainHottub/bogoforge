#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int ISTOP = 10;

// Move magic numbers to the Constant Cache (K$)
// This frees up Vector Registers for our PRNG state and loop variables.
__constant__ unsigned long long K_GOLDEN = 0x9E3779B97F4A7C15ULL;
__constant__ unsigned long long K_C1     = 0xBF58476D1CE4E5B9ULL;
__constant__ unsigned long long K_C2     = 0x94D049BB133111EBULL;

__device__ __forceinline__ void seed_expand(unsigned long long index, unsigned long long base_seed,
        unsigned int& s0, unsigned int& s1, unsigned int& s2, unsigned int& s3) {
    
    // We access the constants directly from the constant cache
    unsigned long long z = base_seed + index * K_GOLDEN;
    
    z += K_GOLDEN;
    unsigned long long a = (z ^ (z >> 30)) * K_C1;
    a = (a ^ (a >> 27)) * K_C2;
    a = a ^ (a >> 31);
    
    z += K_GOLDEN;
    unsigned long long b = (z ^ (z >> 30)) * K_C1;
    b = (b ^ (b >> 27)) * K_C2;
    b = b ^ (b >> 31);
    
    s0 = (unsigned int)a; 
    s1 = (unsigned int)(a >> 32); 
    s2 = (unsigned int)b; 
    s3 = (unsigned int)(b >> 32);
    
    if ((s0 | s1 | s2 | s3) == 0u) s0 = 1u;
}

__device__ __noinline__ int publish_l(unsigned long long index, unsigned long long base_seed, int c,
        unsigned int tid, unsigned long long* best_and_tid,
        uint8_t* all_arrays, unsigned long long* all_idx) {
    unsigned int s0, s1, s2, s3;

    seed_expand(index, base_seed, s0, s1, s2, s3);

    unsigned char *arr = (unsigned char *)all_arrays + ((unsigned long long)tid * 25ULL);

    for (int t = 0; t < 25; t++) arr[t] = (unsigned int)(t + 1);
    
    for (int i = 24; i > 0; i--) {
        unsigned int bound = (unsigned int)(i + 1);
        unsigned int th = (unsigned int)(0x100000000ULL % (unsigned long long)bound);
        unsigned int j;
        for (;;) {
            unsigned int sum = s0 + s3;
            unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        unsigned int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    
    // 1. Force the pointer to be a byte-pointer (char*)
    uint8_t* byte_arrays = (uint8_t*)all_arrays;

    // 2. Perform explicit byte-offsetting
    for (int t = 0; t < 25; t++) {
        byte_arrays[((unsigned long long)tid * 25ULL) + t] = (uint8_t)arr[t];
    }
    
    all_idx[tid] = index;
    unsigned long long old = atomicMax(best_and_tid,
        ((unsigned long long)(unsigned int)c << 32) | (unsigned long long)tid);
    int oldc = (int)(unsigned int)(old >> 32);
    return c > oldc ? c : oldc;
}

__device__ __noinline__ int exact_redo_l(unsigned long long index, unsigned long long base_seed,
        int lbg, unsigned int tid, unsigned long long* best_and_tid,
        uint8_t* all_arrays, unsigned long long* all_idx) {
    unsigned int s0, s1, s2, s3;

    seed_expand(index, base_seed, s0, s1, s2, s3);

    unsigned char *arr = (unsigned char *)all_arrays + ((unsigned long long)tid * 25ULL);

    for (int t = 0; t < 25; t++) arr[t] = (unsigned int)(t + 1);
    for (int i = 24; i > 0; i--) {
        unsigned int bound = (unsigned int)(i + 1);
        unsigned int th = (unsigned int)(0x100000000ULL % (unsigned long long)bound);
        unsigned int j;
        for (;;) {
            unsigned int sum = s0 + s3;
            unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            if (res >= th) { j = res % bound; break; }
        }
        unsigned int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    int c = 0;
    for (int t = 0; t < 25; t++) if (arr[t] == (unsigned int)(t + 1)) c++;
    if (c > lbg) {
            
        // 1. Force the pointer to be a byte-pointer (char*)
        uint8_t* byte_arrays = (uint8_t*)all_arrays;

        // 2. Perform explicit byte-offsetting
        for (int t = 0; t < 25; t++) {
            byte_arrays[((unsigned long long)tid * 25ULL) + t] = (uint8_t)arr[t];
        }
        all_idx[tid] = index;
        unsigned long long old = atomicMax(best_and_tid,
            ((unsigned long long)(unsigned int)c << 32) | (unsigned long long)tid);
        int oldc = (int)(unsigned int)(old >> 32);
        return c > oldc ? c : oldc;
    }
    return lbg;
}

extern "C" __global__
__attribute__((amdgpu_waves_per_eu(1, 2)))
__launch_bounds__(256, 2)
void bogo_range_kernel(
    uint64_t            base_seed,
    uint64_t            lo,
    uint64_t            hi,
    unsigned long long *dev_best_and_bid,
    uint8_t            *dev_all_arrays,
    uint64_t           *dev_all_indices
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    
    const unsigned int* lbsrc = ((const unsigned int*)dev_best_and_bid) + 1;
    
    // Utilize __ldg to cache the global read in the scalar data cache
    int lbg = __ldg((const int*)lbsrc);
    
    unsigned int ctr = 0;
    for (unsigned long long iter = lo + tid; iter < hi; iter += stride) {
        if ((ctr++ & 7u) == 0u) lbg = __ldg((const int*)lbsrc);
        
        unsigned int s0, s1, s2, s3;
        seed_expand(iter, base_seed, s0, s1, s2, s3);
        
        int c = 0;
        unsigned int H = 0, bad = 0;
        
        // Phase 1: Unchecked hot path
        #pragma unroll
        for (int i = 24; i >= ISTOP + 1; i--) {
            const unsigned int bound = i + 1;
            const unsigned int TH = (unsigned int)(0x100000000ULL % (unsigned long long)bound);
            
            unsigned int sum = s0 + s3;
            unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
            unsigned int t = s1 << 9;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
            s3 = (s3 << 11) | (s3 >> 21);
            
            if (TH != 0u) bad |= (unsigned int)(res < TH);
            unsigned int j = res % bound;
            
            if (j == (unsigned int)i && !(H & (1u << i))) c++;
            H |= (1u << j);
        }
        
        // Phase 2: Pruning path
        bool survived = true;
        if (c + ISTOP + 1 > lbg) {
            #pragma unroll
            for (int i = ISTOP; i >= 1; i--) {
                // Pruning check
                if (c + i + 1 <= lbg) {
                    survived = false;
                    break;
                }
                
                const unsigned int bound = i + 1;
                const unsigned int TH = (unsigned int)(0x100000000ULL % (unsigned long long)bound);
                
                unsigned int sum = s0 + s3;
                unsigned int res = ((sum << 7) | (sum >> 25)) + s0;
                unsigned int t = s1 << 9;
                s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
                s3 = (s3 << 11) | (s3 >> 21);
                
                if (TH != 0u) bad |= (unsigned int)(res < TH);
                unsigned int j = res % bound;
                
                if (j == (unsigned int)i && !(H & (1u << i))) c++;
                H |= (1u << j);
            }
            if (survived) {
                if (!(H & 1u)) c++;
            }
        } else {
            survived = false;
        }
        
        // Phase 3: Resolution
        if (bad)
            lbg = exact_redo_l((unsigned long long)iter, (unsigned long long)base_seed, 
                               lbg, tid, dev_best_and_bid, dev_all_arrays, 
                               (unsigned long long*)dev_all_indices);
        else if (c > lbg && survived)
            lbg = publish_l((unsigned long long)iter, (unsigned long long)base_seed, 
                            c, tid, dev_best_and_bid, dev_all_arrays, 
                            (unsigned long long*)dev_all_indices);
    }
}