#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include <vector>

#include "fft.h"
#include "fft_param.cuh"

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>(call);                                                                  \
        if (status != cudaSuccess)                                                                                     \
            fprintf(stderr,                                                                                            \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                         \
                    "with "                                                                                            \
                    "%s (%d).\n",                                                                                      \
                    #call, __LINE__, __FILE__, cudaGetErrorString(status), status);                                    \
    }
#endif // CUDA_RT_CALL

__device__ __inline__ cuComplex Get_W_value(int N, int m)
{
    cuComplex ctemp;
    sincosf(-6.283185308f * fdividef((float)m, (float)N), &ctemp.y, &ctemp.x);
    return (ctemp);
}

__device__ __inline__ cuComplex Get_W_value_inverse(int N, int m)
{
    cuComplex ctemp;
    sincosf(6.283185308f * fdividef((float)m, (float)N), &ctemp.y, &ctemp.x);
    return (ctemp);
}

__device__ __inline__ float shfl(float *value, int par)
{
    return (__shfl_sync(0xffffffff, (*value), par));
}

__device__ __inline__ float shfl_xor(float *value, int par)
{
    return (__shfl_xor_sync(0xffffffff, (*value), par));
}

__device__ __inline__ float shfl_down(float *value, int par)
{
    return (__shfl_down_sync(0xffffffff, (*value), par));
}

__device__ __inline__ void reorder_4_register(cuComplex *A_DFT_value, cuComplex *B_DFT_value, cuComplex *C_DFT_value,
                                              cuComplex *D_DFT_value)
{
    cuComplex Af2temp, Bf2temp, Cf2temp, Df2temp;
    unsigned int target = (((unsigned int)__brev((threadIdx.x & 3))) >> (30)) + 4 * (threadIdx.x >> 2);
    Af2temp.x = shfl(&(A_DFT_value->x), target);
    Af2temp.y = shfl(&(A_DFT_value->y), target);
    Bf2temp.x = shfl(&(B_DFT_value->x), target);
    Bf2temp.y = shfl(&(B_DFT_value->y), target);
    Cf2temp.x = shfl(&(C_DFT_value->x), target);
    Cf2temp.y = shfl(&(C_DFT_value->y), target);
    Df2temp.x = shfl(&(D_DFT_value->x), target);
    Df2temp.y = shfl(&(D_DFT_value->y), target);
    __syncwarp();
    (*A_DFT_value) = Af2temp;
    (*B_DFT_value) = Bf2temp;
    (*C_DFT_value) = Cf2temp;
    (*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_8_register(cuComplex *A_DFT_value, cuComplex *B_DFT_value, cuComplex *C_DFT_value,
                                              cuComplex *D_DFT_value, int *local_id)
{
    cuComplex Af2temp, Bf2temp, Cf2temp, Df2temp;
    unsigned int target = (((unsigned int)__brev(((*local_id) & 7))) >> (29)) + 8 * ((*local_id) >> 3);
    Af2temp.x = shfl(&(A_DFT_value->x), target);
    Af2temp.y = shfl(&(A_DFT_value->y), target);
    Bf2temp.x = shfl(&(B_DFT_value->x), target);
    Bf2temp.y = shfl(&(B_DFT_value->y), target);
    Cf2temp.x = shfl(&(C_DFT_value->x), target);
    Cf2temp.y = shfl(&(C_DFT_value->y), target);
    Df2temp.x = shfl(&(D_DFT_value->x), target);
    Df2temp.y = shfl(&(D_DFT_value->y), target);
    __syncwarp();
    (*A_DFT_value) = Af2temp;
    (*B_DFT_value) = Bf2temp;
    (*C_DFT_value) = Cf2temp;
    (*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_16_register(cuComplex *A_DFT_value, cuComplex *B_DFT_value, cuComplex *C_DFT_value,
                                               cuComplex *D_DFT_value, int *local_id)
{
    cuComplex Af2temp, Bf2temp, Cf2temp, Df2temp;
    unsigned int target = (((unsigned int)__brev(((*local_id) & 15))) >> (28)) + 16 * ((*local_id) >> 4);
    Af2temp.x = shfl(&(A_DFT_value->x), target);
    Af2temp.y = shfl(&(A_DFT_value->y), target);
    Bf2temp.x = shfl(&(B_DFT_value->x), target);
    Bf2temp.y = shfl(&(B_DFT_value->y), target);
    Cf2temp.x = shfl(&(C_DFT_value->x), target);
    Cf2temp.y = shfl(&(C_DFT_value->y), target);
    Df2temp.x = shfl(&(D_DFT_value->x), target);
    Df2temp.y = shfl(&(D_DFT_value->y), target);
    __syncwarp();
    (*A_DFT_value) = Af2temp;
    (*B_DFT_value) = Bf2temp;
    (*C_DFT_value) = Cf2temp;
    (*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_32_register(cuComplex *A_DFT_value, cuComplex *B_DFT_value, cuComplex *C_DFT_value,
                                               cuComplex *D_DFT_value)
{
    cuComplex Af2temp, Bf2temp, Cf2temp, Df2temp;
    unsigned int target = ((unsigned int)__brev(threadIdx.x)) >> (27);
    Af2temp.x = shfl(&(A_DFT_value->x), target);
    Af2temp.y = shfl(&(A_DFT_value->y), target);
    Bf2temp.x = shfl(&(B_DFT_value->x), target);
    Bf2temp.y = shfl(&(B_DFT_value->y), target);
    Cf2temp.x = shfl(&(C_DFT_value->x), target);
    Cf2temp.y = shfl(&(C_DFT_value->y), target);
    Df2temp.x = shfl(&(D_DFT_value->x), target);
    Df2temp.y = shfl(&(D_DFT_value->y), target);
    __syncwarp();
    (*A_DFT_value) = Af2temp;
    (*B_DFT_value) = Bf2temp;
    (*C_DFT_value) = Cf2temp;
    (*D_DFT_value) = Df2temp;
}

template <class const_params>
__device__ __inline__ void reorder_32(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                      cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

template <class const_params>
__device__ __inline__ void reorder_64(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                      cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    __syncthreads();
    unsigned int sm_store_pos = (local_id >> 4) + 2 * (local_id & 15) + warp_id * 132;
    s_input[sm_store_pos] = *A_DFT_value;
    s_input[sm_store_pos + 33] = *B_DFT_value;
    s_input[66 + sm_store_pos] = *C_DFT_value;
    s_input[66 + sm_store_pos + 33] = *D_DFT_value;

    // Read shared memory to get reordered input
    unsigned int sm_read_pos = (local_id & 1) * 32 + local_id + warp_id * 132;
    __syncthreads();
    *A_DFT_value = s_input[sm_read_pos];
    *B_DFT_value = s_input[sm_read_pos + 1];
    *C_DFT_value = s_input[sm_read_pos + 66];
    *D_DFT_value = s_input[sm_read_pos + 66 + 1];
}

template <class const_params>
__device__ __inline__ void reorder_128(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                       cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

    __syncwarp();
    unsigned int sm_store_pos = (local_id >> 3) + 4 * (local_id & 7) + warp_id * 132;
    s_input[sm_store_pos] = *A_DFT_value;
    s_input[sm_store_pos + 33] = *B_DFT_value;
    s_input[66 + sm_store_pos] = *C_DFT_value;
    s_input[66 + sm_store_pos + 33] = *D_DFT_value;

    // Read shared memory to get reordered input
    __syncwarp();
    unsigned int sm_read_pos = (local_id & 3) * 32 + local_id + warp_id * 132;
    *A_DFT_value = s_input[sm_read_pos];
    *B_DFT_value = s_input[sm_read_pos + 1];
    *C_DFT_value = s_input[sm_read_pos + 2];
    *D_DFT_value = s_input[sm_read_pos + 3];

    __syncwarp();
    reorder_4_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

template <class const_params>
__device__ __inline__ void reorder_256(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                       cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    __syncthreads();
    unsigned int sm_store_pos = (local_id >> 2) + 8 * (local_id & 3) + warp_id * 132;
    s_input[sm_store_pos] = *A_DFT_value;
    s_input[sm_store_pos + 33] = *B_DFT_value;
    s_input[66 + sm_store_pos] = *C_DFT_value;
    s_input[66 + sm_store_pos + 33] = *D_DFT_value;

    // Read shared memory to get reordered input
    __syncthreads();
    unsigned int sm_read_pos = (local_id & 7) * 32 + local_id;
    *A_DFT_value = s_input[sm_read_pos + warp_id * 4 + 0];
    *B_DFT_value = s_input[sm_read_pos + warp_id * 4 + 1];
    *C_DFT_value = s_input[sm_read_pos + warp_id * 4 + 2];
    *D_DFT_value = s_input[sm_read_pos + warp_id * 4 + 3];

    __syncthreads();
    reorder_8_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value, &local_id);
}

template <class const_params>
__device__ __inline__ void reorder_512(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                       cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    __syncthreads();
    unsigned int sm_store_pos = (local_id >> 1) + 16 * (local_id & 1) + warp_id * 132;
    s_input[sm_store_pos] = *A_DFT_value;
    s_input[sm_store_pos + 33] = *B_DFT_value;
    s_input[66 + sm_store_pos] = *C_DFT_value;
    s_input[66 + sm_store_pos + 33] = *D_DFT_value;

    // Read shared memory to get reordered input
    unsigned int sm_read_pos = (local_id & 15) * 32 + local_id + warp_id * 4;
    __syncthreads();
    *A_DFT_value = s_input[sm_read_pos + 0];
    *B_DFT_value = s_input[sm_read_pos + 1];
    *C_DFT_value = s_input[sm_read_pos + 2];
    *D_DFT_value = s_input[sm_read_pos + 3];

    __syncthreads();
    reorder_16_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value, &local_id);
}

template <class const_params>
__device__ __inline__ void reorder_1024(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                        cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    __syncthreads();
    unsigned int sm_store_pos = (local_id >> 0) + 32 * (local_id & 0) + warp_id * 132;
    s_input[sm_store_pos] = *A_DFT_value;
    s_input[sm_store_pos + 33] = *B_DFT_value;
    s_input[66 + sm_store_pos] = *C_DFT_value;
    s_input[66 + sm_store_pos + 33] = *D_DFT_value;

    // Read shared memory to get reordered input
    unsigned int sm_read_pos = (local_id & 31) * 32 + local_id + warp_id * 4;
    __syncthreads();
    *A_DFT_value = s_input[sm_read_pos + 0];
    *B_DFT_value = s_input[sm_read_pos + 1];
    *C_DFT_value = s_input[sm_read_pos + 2];
    *D_DFT_value = s_input[sm_read_pos + 3];

    __syncthreads();
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

template <class const_params>
__device__ __inline__ void reorder_2048(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                        cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

    __syncthreads();
    // unsigned int sm_store_pos = (local_id>>0) + 32*(local_id&0) + warp_id*132;
    unsigned int sm_store_pos = local_id + warp_id * 132;
    s_input[sm_store_pos] = *A_DFT_value;
    s_input[sm_store_pos + 33] = *B_DFT_value;
    s_input[sm_store_pos + 66] = *C_DFT_value;
    s_input[sm_store_pos + 99] = *D_DFT_value;

    // Read shared memory to get reordered input
    __syncthreads();
    // unsigned int sm_read_pos = (local_id&31)*33 + warp_id*2;
    unsigned int sm_read_pos = local_id * 33 + warp_id * 2;
    *A_DFT_value = s_input[sm_read_pos + 0];
    *B_DFT_value = s_input[sm_read_pos + 1056];
    *C_DFT_value = s_input[sm_read_pos + 1];
    *D_DFT_value = s_input[sm_read_pos + 1056 + 1];

    __syncthreads();
    reorder_64<const_params>(s_input, A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

template <class const_params>
__device__ __inline__ void reorder_4096(cuComplex *s_input, cuComplex *A_DFT_value, cuComplex *B_DFT_value,
                                        cuComplex *C_DFT_value, cuComplex *D_DFT_value)
{
    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;

    // reorder elements within warp so we can save them in semi-transposed manner into shared memory
    reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);

    __syncthreads();
    unsigned int sm_store_pos = local_id + warp_id * 132;
    s_input[sm_store_pos] = *A_DFT_value;
    s_input[sm_store_pos + 33] = *B_DFT_value;
    s_input[sm_store_pos + 66] = *C_DFT_value;
    s_input[sm_store_pos + 99] = *D_DFT_value;

    // Read shared memory to get reordered input
    __syncthreads();
    unsigned int sm_read_pos = local_id * 33 + warp_id;
    *A_DFT_value = s_input[sm_read_pos + 0];
    *B_DFT_value = s_input[sm_read_pos + 1056];
    *C_DFT_value = s_input[sm_read_pos + 2112];
    *D_DFT_value = s_input[sm_read_pos + 3168];

    __syncthreads();
    reorder_128<const_params>(s_input, A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

template <class const_params> __device__ void execute_shared_memory_fft(cuComplex *s_input)
{
    cuComplex A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
    cuComplex W;
    cuComplex Aftemp, Bftemp, Cftemp, Dftemp;

    int j, m_param;
    int parity, itemp;
    int A_read_index, B_read_index, C_read_index, D_read_index;
    int PoT, PoTp1, q;

    int local_id = threadIdx.x & (const_params::warp - 1);
    int warp_id = threadIdx.x / const_params::warp;
    A_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp];
    B_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp + const_params::warp];
    C_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp + 2 * const_params::warp];
    D_DFT_value = s_input[local_id + (warp_id << 2) * const_params::warp + 3 * const_params::warp];

    if constexpr (const_params::fft_reorder)
    {
        if constexpr (const_params::fft_exp == 5)
            reorder_32<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
        else if constexpr (const_params::fft_exp == 6)
            reorder_64<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
        else if constexpr (const_params::fft_exp == 7)
            reorder_128<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
        else if constexpr (const_params::fft_exp == 8)
            reorder_256<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
        else if constexpr (const_params::fft_exp == 9)
            reorder_512<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
        else if constexpr (const_params::fft_exp == 10)
            reorder_1024<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
        else if constexpr (const_params::fft_exp == 11)
            reorder_2048<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
        else if constexpr (const_params::fft_exp == 12)
            reorder_4096<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
    }

    PoT = 1;
    PoTp1 = 2;

    itemp = local_id & 1;
    parity = (1 - itemp * 2);

    A_DFT_value.x = parity * A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
    A_DFT_value.y = parity * A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
    B_DFT_value.x = parity * B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
    B_DFT_value.y = parity * B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
    C_DFT_value.x = parity * C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
    C_DFT_value.y = parity * C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
    D_DFT_value.x = parity * D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
    D_DFT_value.y = parity * D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);

    PoT = 2;
    PoTp1 = 4;

#pragma unroll
    for (q = 1; q < 5; q++)
    {
        m_param = (local_id & (PoTp1 - 1));
        itemp = m_param >> q;
        parity = ((itemp << 1) - 1);

        if constexpr (const_params::fft_direction)
            W = Get_W_value_inverse(PoTp1, itemp * m_param);
        else
            W = Get_W_value(PoTp1, itemp * m_param);

        Aftemp.x = W.x * A_DFT_value.x - W.y * A_DFT_value.y;
        Aftemp.y = W.x * A_DFT_value.y + W.y * A_DFT_value.x;
        Bftemp.x = W.x * B_DFT_value.x - W.y * B_DFT_value.y;
        Bftemp.y = W.x * B_DFT_value.y + W.y * B_DFT_value.x;
        Cftemp.x = W.x * C_DFT_value.x - W.y * C_DFT_value.y;
        Cftemp.y = W.x * C_DFT_value.y + W.y * C_DFT_value.x;
        Dftemp.x = W.x * D_DFT_value.x - W.y * D_DFT_value.y;
        Dftemp.y = W.x * D_DFT_value.y + W.y * D_DFT_value.x;

        A_DFT_value.x = Aftemp.x + parity * shfl_xor(&Aftemp.x, PoT);
        A_DFT_value.y = Aftemp.y + parity * shfl_xor(&Aftemp.y, PoT);
        B_DFT_value.x = Bftemp.x + parity * shfl_xor(&Bftemp.x, PoT);
        B_DFT_value.y = Bftemp.y + parity * shfl_xor(&Bftemp.y, PoT);
        C_DFT_value.x = Cftemp.x + parity * shfl_xor(&Cftemp.x, PoT);
        C_DFT_value.y = Cftemp.y + parity * shfl_xor(&Cftemp.y, PoT);
        D_DFT_value.x = Dftemp.x + parity * shfl_xor(&Dftemp.x, PoT);
        D_DFT_value.y = Dftemp.y + parity * shfl_xor(&Dftemp.y, PoT);

        PoT = PoT << 1;
        PoTp1 = PoTp1 << 1;
    }

    itemp = local_id + (warp_id << 2) * const_params::warp;
    s_input[itemp] = A_DFT_value;
    s_input[itemp + const_params::warp] = B_DFT_value;
    s_input[itemp + 2 * const_params::warp] = C_DFT_value;
    s_input[itemp + 3 * const_params::warp] = D_DFT_value;

    if (const_params::fft_exp == 6)
    {
        __syncthreads();
        q = 5;
        m_param = threadIdx.x & (PoT - 1);
        j = threadIdx.x >> q;

        if (const_params::fft_direction)
            W = Get_W_value_inverse(PoTp1, m_param);
        else
            W = Get_W_value(PoTp1, m_param);

        A_read_index = j * (PoTp1 << 1) + m_param;
        B_read_index = j * (PoTp1 << 1) + m_param + PoT;
        C_read_index = j * (PoTp1 << 1) + m_param + PoTp1;
        D_read_index = j * (PoTp1 << 1) + m_param + 3 * PoT;

        Aftemp = s_input[A_read_index];
        Bftemp = s_input[B_read_index];
        A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
        A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
        B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
        B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

        Cftemp = s_input[C_read_index];
        Dftemp = s_input[D_read_index];
        C_DFT_value.x = Cftemp.x + W.x * Dftemp.x - W.y * Dftemp.y;
        C_DFT_value.y = Cftemp.y + W.x * Dftemp.y + W.y * Dftemp.x;
        D_DFT_value.x = Cftemp.x - W.x * Dftemp.x + W.y * Dftemp.y;
        D_DFT_value.y = Cftemp.y - W.x * Dftemp.y - W.y * Dftemp.x;

        s_input[A_read_index] = A_DFT_value;
        s_input[B_read_index] = B_DFT_value;
        s_input[C_read_index] = C_DFT_value;
        s_input[D_read_index] = D_DFT_value;

        PoT = PoT << 1;
        PoTp1 = PoTp1 << 1;
    }

    for (q = 5; q < (const_params::fft_exp - 1); q++)
    {
        __syncthreads();
        m_param = threadIdx.x & (PoT - 1);
        j = threadIdx.x >> q;

        if (const_params::fft_direction)
            W = Get_W_value_inverse(PoTp1, m_param);
        else
            W = Get_W_value(PoTp1, m_param);

        A_read_index = j * (PoTp1 << 1) + m_param;
        B_read_index = j * (PoTp1 << 1) + m_param + PoT;
        C_read_index = j * (PoTp1 << 1) + m_param + PoTp1;
        D_read_index = j * (PoTp1 << 1) + m_param + 3 * PoT;

        Aftemp = s_input[A_read_index];
        Bftemp = s_input[B_read_index];
        A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
        A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
        B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
        B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

        Cftemp = s_input[C_read_index];
        Dftemp = s_input[D_read_index];
        C_DFT_value.x = Cftemp.x + W.x * Dftemp.x - W.y * Dftemp.y;
        C_DFT_value.y = Cftemp.y + W.x * Dftemp.y + W.y * Dftemp.x;
        D_DFT_value.x = Cftemp.x - W.x * Dftemp.x + W.y * Dftemp.y;
        D_DFT_value.y = Cftemp.y - W.x * Dftemp.y - W.y * Dftemp.x;

        s_input[A_read_index] = A_DFT_value;
        s_input[B_read_index] = B_DFT_value;
        s_input[C_read_index] = C_DFT_value;
        s_input[D_read_index] = D_DFT_value;

        PoT = PoT << 1;
        PoTp1 = PoTp1 << 1;
    }

    if (const_params::fft_exp > 6)
    {
        __syncthreads();
        m_param = threadIdx.x;

        if (const_params::fft_direction)
            W = Get_W_value_inverse(PoTp1, m_param);
        else
            W = Get_W_value(PoTp1, m_param);

        A_read_index = m_param;
        B_read_index = m_param + PoT;
        C_read_index = m_param + (PoT >> 1);
        D_read_index = m_param + 3 * (PoT >> 1);

        Aftemp = s_input[A_read_index];
        Bftemp = s_input[B_read_index];
        A_DFT_value.x = Aftemp.x + W.x * Bftemp.x - W.y * Bftemp.y;
        A_DFT_value.y = Aftemp.y + W.x * Bftemp.y + W.y * Bftemp.x;
        B_DFT_value.x = Aftemp.x - W.x * Bftemp.x + W.y * Bftemp.y;
        B_DFT_value.y = Aftemp.y - W.x * Bftemp.y - W.y * Bftemp.x;

        Cftemp = s_input[C_read_index];
        Dftemp = s_input[D_read_index];
        if (const_params::fft_direction)
        {
            C_DFT_value.x = Cftemp.x - W.y * Dftemp.x - W.x * Dftemp.y;
            C_DFT_value.y = Cftemp.y - W.y * Dftemp.y + W.x * Dftemp.x;
            D_DFT_value.x = Cftemp.x + W.y * Dftemp.x + W.x * Dftemp.y;
            D_DFT_value.y = Cftemp.y + W.y * Dftemp.y - W.x * Dftemp.x;
        }
        else
        {
            C_DFT_value.x = Cftemp.x + W.y * Dftemp.x + W.x * Dftemp.y;
            C_DFT_value.y = Cftemp.y + W.y * Dftemp.y - W.x * Dftemp.x;
            D_DFT_value.x = Cftemp.x - W.y * Dftemp.x - W.x * Dftemp.y;
            D_DFT_value.y = Cftemp.y - W.y * Dftemp.y + W.x * Dftemp.x;
        }

        s_input[A_read_index] = A_DFT_value;
        s_input[B_read_index] = B_DFT_value;
        s_input[C_read_index] = C_DFT_value;
        s_input[D_read_index] = D_DFT_value;
    }
}

template <class const_params> __global__ void shared_memory_fft(cuComplex *d_output, cuComplex *d_input)
{
    __shared__ cuComplex s_input[const_params::fft_shared_memory_required];

    s_input[threadIdx.x] = d_input[threadIdx.x + blockIdx.x * const_params::fft_length];
    s_input[threadIdx.x + const_params::fft_length_quarter] =
        d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_quarter];
    s_input[threadIdx.x + const_params::fft_length_half] =
        d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_half];
    s_input[threadIdx.x + const_params::fft_length_three_quarters] =
        d_input[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_three_quarters];

    __syncthreads();
    execute_shared_memory_fft<const_params>(s_input);

    __syncthreads();
    d_output[threadIdx.x + blockIdx.x * const_params::fft_length] = s_input[threadIdx.x];
    d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_quarter] =
        s_input[threadIdx.x + const_params::fft_length_quarter];
    d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_half] =
        s_input[threadIdx.x + const_params::fft_length_half];
    d_output[threadIdx.x + blockIdx.x * const_params::fft_length + const_params::fft_length_three_quarters] =
        s_input[threadIdx.x + const_params::fft_length_three_quarters];
}

std::vector<std::complex<float>> _manual_fft_impl(std::vector<std::complex<float>> input_signal)
{
    int fft_size = 1024;

    if (input_signal.size() % fft_size != 0)
    {
        printf("Input signal must be a power of 2\n");
        exit(EXIT_FAILURE);
    }

    std::vector<std::complex<float>> output_signal(input_signal.size());
    cuComplex *d_input = nullptr, *d_output = nullptr;
    int number_of_ffts = input_signal.size() / fft_size;

    // Create device data arrays
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(cuComplex) * input_signal.size()));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(cuComplex) * output_signal.size()));

    CUDA_RT_CALL(cudaMemcpy(d_input, input_signal.data(), sizeof(std::complex<float>) * input_signal.size(),
                            cudaMemcpyHostToDevice));

    dim3 gridSize(number_of_ffts, 1, 1);
    dim3 blockSize(fft_size / 4, 1, 1);
    if (fft_size == 32)
    {
        gridSize.x = number_of_ffts / 4;
        blockSize.x = 32;
    }
    if (fft_size == 64)
    {
        gridSize.x = number_of_ffts / 2;
        blockSize.x = 32;
    }

    //---------> FFT part
    switch (fft_size)
    {
    case 32:
        shared_memory_fft<FFT_32_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    case 64:
        shared_memory_fft<FFT_64_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    case 128:
        shared_memory_fft<FFT_128_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    case 256:
        shared_memory_fft<FFT_256_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    case 512:
        shared_memory_fft<FFT_512_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    case 1024:
        shared_memory_fft<FFT_1024_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    case 2048:
        shared_memory_fft<FFT_2048_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    case 4096:
        shared_memory_fft<FFT_4096_forward><<<gridSize, blockSize>>>(d_input, d_output);
        break;

    default:
        printf("Error wrong FFT length!\n");
        break;
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaMemcpy(output_signal.data(), d_output, sizeof(std::complex<float>) * output_signal.size(),
                            cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaFree(d_input))
    CUDA_RT_CALL(cudaFree(d_output));

    return output_signal;
}