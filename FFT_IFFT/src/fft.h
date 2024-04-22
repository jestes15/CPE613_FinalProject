#ifndef FFT_INTERFACE
#define FFT_INTERFACE

#include <complex>
#include <iostream>
#include <mkl.h>
#include <vector>

std::vector<std::complex<float>> _fft_cpu(std::vector<std::complex<float>> input_signal);
std::vector<std::complex<float>> _fft_cuda_reference(std::vector<std::complex<float>> input_signal, int polling_rate);

std::vector<std::complex<float>> _forward_fft_R2C(std::vector<float> in);
std::vector<std::complex<float>> _forward_fft_C2C(std::vector<std::complex<float>> in);

std::vector<float> _backward_fft_C2R(std::vector<std::complex<float>> in, int original_size);
std::vector<float> _backward_fft_C2R_Complex(std::vector<std::complex<float>> in);

std::vector<std::complex<float>> _manual_fft_impl(std::vector<std::complex<float>> input, int fft_size);

#endif