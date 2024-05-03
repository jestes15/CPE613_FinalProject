#include <complex>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>

std::vector<std::complex<float>> _manual_fft_impl(std::vector<std::complex<float>> input, int fft_size);
std::vector<std::complex<float>> _fft_cpu(std::vector<std::complex<float>> input_signal);
std::vector<std::complex<float>> _fft_cuda_reference(std::vector<std::complex<float>> input_signal);

__global__ void _fill_with_signal(cuFloatComplex *signal, int size, int fs, int f1, int f2, int f3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float t = (float)idx / fs;
        signal[idx] = make_cuFloatComplex(cosf(2 * M_PI * f1 * t) + cosf(2 * M_PI * f2 * t) + cosf(2 * M_PI * f3 * t), 0);
    }
}

int main()
{
    std::vector<std::complex<float>> input(4096), output_host(4096), output_cuda_ref(4096), output_manual(4096);

    // Allocate memory on the device
    cuFloatComplex *d_input;
    cudaMalloc(&d_input, input.size() * sizeof(std::complex<float>));

    // Fill the input signal with a sum of 3 sinusoids
    _fill_with_signal<<<(input.size() + 255) / 256, 256>>>(d_input, input.size(), 4096, 10, 20, 30);

    // Copy signal from device to host
    cudaMemcpy(input.data(), d_input, input.size() * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_input);

    // Perform FFT on the CPU
    output_host = _fft_cpu(input);
    output_cuda_ref = _fft_cuda_reference(input);
    output_manual = _manual_fft_impl(input, 4096);

    // Compare the results
    float max_diff = 0;
    for (int i = 0; i < 4096; i++)
    {
        float diff = abs(output_host[i] - output_cuda_ref[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }

    std::cout << "Max difference between CPU and CUDA reference: " << max_diff << std::endl;

    max_diff = 0;
    for (int i = 0; i < 4096; i++)
    {
        float diff = abs(output_host[i] - output_manual[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }

    std::cout << "Max difference between CPU and manual FFT: " << max_diff << std::endl;

    return 0;
}