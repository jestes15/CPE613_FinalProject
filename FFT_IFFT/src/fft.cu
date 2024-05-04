#include "fft.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "helper_cuda.h"

#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                                                               \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>(call);                                                                  \
        if (status != CUFFT_SUCCESS)                                                                                   \
            fprintf(stderr,                                                                                            \
                    "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                           \
                    "with "                                                                                            \
                    "code (%d).\n",                                                                                    \
                    #call, __LINE__, __FILE__, status);                                                                \
    }
#endif // CUFFT_CALL

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

uint32_t reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

__device__ inline uint32_t reverse_bits_gpu(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

int _fft_backend(const std::complex<float> *x, std::complex<float> *Y, uint32_t N)
{
    using namespace std::complex_literals;
    if (N & (N - 1))
    {
        fprintf(stderr,
                "N=%u must be a power of 2.  "
                "This implementation of the Cooley-Tukey FFT algorithm "
                "does not support input that is not a power of 2.\n",
                N);

        return -1;
    }

    int logN = (int)log2f((float)N);

    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t rev = reverse_bits(i);
        rev = rev >> (32 - logN);
        Y[i] = x[rev];
    }

    // Set m to 2, 4, 8, 16, ..., N
    for (int s = 1; s <= logN; s++)
    {
        int m = 1 << s;
        int mh = 1 << (s - 1);

        std::complex<float> twiddle = std::exp(-2if * static_cast<float>(M_PI / m));

        for (uint32_t k = 0; k < N; k += m)
        {
            std::complex<float> twiddle_factor = 1;
            for (int j = 0; j < mh; j++)
            {
                std::complex<float> a = Y[k + j];
                std::complex<float> b = twiddle_factor * Y[k + j + mh];
                twiddle_factor *= twiddle;
                Y[k + j] = a + b;
                Y[k + j + mh] = a - b;
            }
        }
    }
    return EXIT_SUCCESS;
}

std::vector<float> _calculate_relative_error(int size, std::vector<std::complex<float>> reference_solution,
                               std::vector<std::complex<float>> computed_solution)
{
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

	cuFloatComplex *d_reference_solution = nullptr;
	cuFloatComplex *d_computed_solution = nullptr;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_reference_solution), sizeof(cuFloatComplex) * size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_computed_solution), sizeof(cuFloatComplex) * size));

    checkCudaErrors(cudaMemcpy(d_reference_solution, reinterpret_cast<cuFloatComplex *>(reference_solution.data()),
                               sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_computed_solution, reinterpret_cast<cuFloatComplex *>(computed_solution.data()),
                               sizeof(cuFloatComplex) * size, cudaMemcpyHostToDevice));

    cuFloatComplex alpha_axpy = make_cuFloatComplex(-1.0, -1.0);
    float norm_axpy = 0.0;
    float norm_ref = 0.0;

    checkCudaErrors(cublasScnrm2(handle, size, d_reference_solution, 1, &norm_ref));
    checkCudaErrors(cublasCaxpy(handle, size, &alpha_axpy, d_computed_solution, 1, d_reference_solution, 1));
    checkCudaErrors(cublasScnrm2(handle, size, d_reference_solution, 1, &norm_axpy));

    cudaDeviceSynchronize();

	cudaFree(d_reference_solution);
	cudaFree(d_computed_solution);

    std::vector<float> result = {norm_axpy, norm_ref, (norm_axpy - norm_ref) / norm_ref};
    return result;
}

__global__ void fft_kernel(const cuFloatComplex *x, cuFloatComplex *Y, uint32_t N, int logN)
{
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t rev;

    rev = reverse_bits_gpu(2 * i);
    rev >>= (32 - logN);
    Y[2 * i] = x[rev];

    rev = reverse_bits_gpu(2 * i + 1);
    rev >>= (32 - logN);
    Y[2 * i + 1] = x[rev];

    __syncthreads();

    for (int s = 1; s <= logN; s++)
    {
        int mh = 1 << (s - 1);

        int k = threadIdx.x / mh * (1 << s);
        int j = threadIdx.x % mh;
        int kj = k + j;

        cuFloatComplex a = Y[kj];
        float tr;
        float ti;

        sincosf(-M_PI * j / mh, &ti, &tr);
        cuFloatComplex twiddle = make_cuFloatComplex(tr, ti);

        cuFloatComplex b = cuCmulf(twiddle, Y[kj + mh]);

        Y[kj] = cuCaddf(a, b);
        Y[kj + mh] = cuCsubf(a, b);

        __syncthreads();
    }
}

std::vector<std::complex<float>> _manual_fft_impl2(std::vector<std::complex<float>> input_signal, int fft_size)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t input_size = input_signal.size();
    size_t output_size = input_signal.size();

    std::vector<std::complex<float>> output(output_size);

    cuComplex *d_input;
    cuComplex *d_output;
    CUDA_RT_CALL(cudaMalloc((void **)&d_input, sizeof(cuComplex) * input_size));
    CUDA_RT_CALL(cudaMalloc((void **)&d_output, sizeof(cuComplex) * output_size));
    CUDA_RT_CALL(cudaMemcpy(d_input, input_signal.data(), input_size * sizeof(cuComplex), cudaMemcpyHostToDevice));

    int size = fft_size >> 1;
    int block_size = min(size, prop.maxThreadsPerBlock);

    dim3 gridSize((size + block_size - 1) / block_size, 1, 1);
    dim3 blockSize(block_size, 1, 1);

    fft_kernel<<<gridSize, blockSize>>>(d_input, d_output, fft_size, (int)log2f((float)fft_size));

    CUDA_RT_CALL(cudaMemcpy(output.data(), d_output, output_size * sizeof(cuComplex), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    return output;
}

std::vector<std::complex<float>> _fft_cpu(std::vector<std::complex<float>> input_signal)
{
    std::vector<std::complex<float>> output(input_signal.size());
    _fft_backend(input_signal.data(), output.data(), input_signal.size());
    return output;
}

std::vector<std::complex<float>> _fft_cuda_reference(std::vector<std::complex<float>> input_signal)
{
    cufftHandle plan;
    cudaStream_t stream = NULL;

    std::vector<std::complex<float>> output_signal(input_signal.size());

    cufftComplex *d_data = nullptr;

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan1d(&plan, input_signal.size(), CUFFT_C2C, 1));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    // Create device data arrays
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(std::complex<float>) * input_signal.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_data, input_signal.data(), sizeof(std::complex<float>) * input_signal.size(),
                                 cudaMemcpyHostToDevice, stream));

    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    CUDA_RT_CALL(cudaMemcpyAsync(output_signal.data(), d_data, sizeof(std::complex<float>) * output_signal.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    CUDA_RT_CALL(cudaFree(d_data))
    CUFFT_CALL(cufftDestroy(plan));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaDeviceReset());

    return output_signal;
}