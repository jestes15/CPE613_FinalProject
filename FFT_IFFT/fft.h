#ifndef FFT_INTERFACE
#define FFT_INTERFACE

#include <complex>
#include <vector>

std::vector<std::complex<float>> _fft(std::vector<std::complex<float>> input_signal);
std::vector<std::complex<float>> _fft_cuda_reference(std::vector<std::complex<float>> input_signal);

#endif