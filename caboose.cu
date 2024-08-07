#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include <curand_kernel.h>
#include <curand.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

typedef __uint128_t u128;

// Error checking macro
#define CUDA_CALL(call)                                                   \
do {                                                                             \
    cudaError_t err = call;                                                      \
    if (cudaSuccess != err) {                                                    \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << " : " \
                  << cudaGetErrorString(err) << std::endl;                       \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

#define CURAND_CALL(x) do {  \
    auto err = x; \
    if((err) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

const int num_miller_tests = 1;

__device__ u128 modexp(u128 x, u128 y, u128 p) {
    u128 res = 1;  // Initialize result
    x %= p;  // Update x if it is more than or equal to p

    while (y > 0) {
        // If y is odd, multiply x with the result
        if (y & 1)
            res = (res * x) % p;

        // y must be even now
        y >>= 1;  // y = y/2
        x = (x * x) % p;  // Change x to x^2
    }
    return res;
}

// Miller–Rabin primality test
__device__ bool isPrimeProbabalistic(u128 n, curandStateMtgp32_t *state) {
    if (n == 2 || n == 3) return true;
    if (n == 1 || n % 2 == 0) return false;

     // Write n-1 as d*2^s by factoring powers of 2 from n-1
    int s = 0;
    u128 d = n - 1;
    for (; !(d & 1); ++s, d >>= 1)
        ; // loop

    int upper = num_miller_tests < n-2 ? num_miller_tests : n-2;
    for (int k = 0; k < upper; k++) {

        u_int32_t a32[4];
        for (int i = 0; i < 4; i++) {
            a32[i] = curand(state);
        }
        u128 a = *(u128*)a32;
        a %= n - 1;
        if (a <= 1)
            a += 2;

        assert(a > 1);
        assert(a < n-1);
        u128 x = modexp(a, d, n);

        if (x == 1 || x == n - 1)
            continue;
        
        for (int _ = 0; _ < s - 1; _++) {
            x = (x * x) % n;
            if (x == 1) return false;
            if (x == n - 1) goto NEXT_WITNESS;
        }
        return false;

        NEXT_WITNESS:
        continue;
    }
    return true;
}

__device__ bool isPrimeDeterministic(u128 n) 
{
    if (n <= 1)
        return false; 
    if (n == 2 || n == 3) 
        return true; 
    if (n % 2 == 0 || n % 3 == 0) 
        return false;
      
    for (u128 i = 5; i * i <= n; i += 6) 
        if (n % i == 0 || n % (i + 2) == 0) 
            return false;

    return true; 
}

const u_int64_t thread_count = 256;
const u_int64_t block_count = 128;

const u_int64_t total_threads = thread_count*block_count;
const u_int64_t nums_per_thread = 100;

const u_int64_t nums_per_iter = total_threads * nums_per_thread;
const u_int64_t nums_to_check = 1'000'000'000'000;

__global__ void isCaboose(u_int64_t begin, curandStateMtgp32_t *states, bool *out) {

    u_int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    assert(id < total_threads);
    u_int64_t start = begin + id*nums_per_thread;
    u_int64_t end = start + nums_per_thread;

    assert(blockIdx.x < block_count);

    for (u_int64_t i = start; i < end; i += 2) {

        assert(i - begin < nums_per_iter);
        bool &caboose = out[i - begin];
        assert(!caboose);
        caboose = true;

        for(u128 j = 1; j < i; j++) {
            if(!isPrimeProbabalistic(j*j - j + i, states + blockIdx.x)) {
                caboose = false;
                break;
            }
        }

        if (!caboose)
            continue;
        for (u128 j = 1; j < i; j++) {
            if (!isPrimeDeterministic(j*j - j + i)) {
                caboose = false;
                break;
            }
        }
    }
}

int main() {
    bool *res;
    CUDA_CALL(cudaMalloc(&res, nums_per_iter));

    curandStateMtgp32_t *devMTGPStates;
    CUDA_CALL(cudaMalloc(&devMTGPStates, block_count*sizeof(curandStateMtgp32_t)));

    mtgp32_kernel_params *devKernelParams;
    CUDA_CALL(cudaMalloc(&devKernelParams, sizeof(mtgp32_kernel_params)));

    CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, block_count, 69420));

    bool *h_result = (bool *)malloc(nums_per_iter);
    for (u_int64_t begin = 3; begin <= nums_to_check; begin += nums_per_iter) {
        if (begin % 2 == 0)
            begin--;

        CUDA_CALL(cudaMemset(res, false, nums_per_iter));
        isCaboose<<<block_count, thread_count>>>(begin, devMTGPStates, res);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaGetLastError());

        CUDA_CALL(cudaMemcpy(h_result, res, nums_per_iter, cudaMemcpyDeviceToHost));

        for (int i = 0; i < nums_per_iter; i++) {
            if (h_result[i]) {
                std::cout << "Caboose found: " << i + begin << "\n";
            }
        }
    }

    cudaFree(res);
    free(h_result);

    return 0;
}
