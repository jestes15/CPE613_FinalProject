#include <cmath>
#include <complex>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "combined_signal.hpp"

typedef int (*algorithm_t)(const void *, void *, uint32_t);

using namespace std::complex_literals;

#define CHECK_RET(ret)                                                                                                 \
    if (ret == EXIT_FAILURE)                                                                                           \
    {                                                                                                                  \
        return EXIT_FAILURE;                                                                                           \
    }

#define CHECK(condition, err_fmt, err_msg)                                                                             \
    if (condition)                                                                                                     \
    {                                                                                                                  \
        printf(err_fmt " (%s:%d)\n", err_msg, __FILE__, __LINE__);                                                     \
        return EXIT_FAILURE;                                                                                           \
    }

#define CHECK_MALLOC(p, name) CHECK(!(p), "Failed to allocate %s", name)

int dft(const std::complex<float> *x, std::complex<float> *Y, uint32_t N)
{
    for (size_t k = 0; k < N; k++)
    {
        std::complex<float> sum = 0;
        float c = -2 * M_PI * k / N;
        for (size_t n = 0; n < N; n++)
        {
            float a = c * n;
            sum = sum + x[n] * (std::cos(a) + 1if * std::sin(a));
        }
        Y[k] = sum;
    }
    return EXIT_SUCCESS;
}

uint32_t reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

int fft(const std::complex<float> *x, std::complex<float> *Y, uint32_t N)
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

void show_complex_vector(std::complex<float> *v, uint32_t N)
{
    printf("# TOTAL PROCESSED SAMPLES: %u\n", N);
    printf("# %s\n", "================================");
    printf("import cmath\n\ndata = [\n");

    // Set the output precision
    int prec = 10;
    for (uint32_t k = 0; k < N; k++)
    {
        if (k < N - 1)
            printf("%10f + %10fj, ", std::real(v[k]), std::imag(v[k]));
        else
            printf("%10f + %10fj ]", std::real(v[k]), std::imag(v[k]));
    }
}

static const size_t sample_size = sizeof(signal) / sizeof(*signal);

int setup_data(std::complex<float> **in, std::complex<float> **out, uint32_t N)
{
    *in = (std::complex<float> *)calloc(N, sizeof(std::complex<float>));
    CHECK_MALLOC(in, "input");
    *out = (std::complex<float> *)calloc(N, sizeof(std::complex<float>));
    CHECK_MALLOC(out, "output");

    for (size_t i = 0; i < ((N < sample_size) ? N : sample_size); i++)
    {
        (*in)[i] = signal[i];
    }
    return EXIT_SUCCESS;
}

int run(const char *algorithm_name, algorithm_t f, const void *in, void *out, uint32_t N)
{
    fprintf(stderr, "Running %s with N=%d\n", algorithm_name, N);
    CHECK_RET(f(in, out, N));
    return EXIT_SUCCESS;
}

int main(int argc, const char **argv)
{
    uint32_t N = 262144;

    bool no_print = false;

    std::complex<float> *in;
    std::complex<float> *out;

    CHECK_RET(setup_data(&in, &out, N));
    CHECK_RET(run("Cooley-Tukey FFT", (algorithm_t)fft, in, out, N));

    // Print the results
    if (!no_print)
    {
        show_complex_vector(out, N);
    }
    free(in);
    free(out);
    return 0;
}