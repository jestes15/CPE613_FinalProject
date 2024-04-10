#include "fft.h"
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <vector>

uint32_t reverse_bits(uint32_t x)
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

std::vector<std::complex<float>> _fft(std::vector<std::complex<float>> input_signal)
{
    std::vector<std::complex<float>> output(input_signal.size());
    _fft_backend(input_signal.data(), output.data(), input_signal.size());
    return output;
}

std::vector<std::complex<float>> _fft_cuda_reference(std::vector<std::complex<float>> input_signal)
{
	
}