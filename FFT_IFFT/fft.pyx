from libcpp.vector cimport vector
from libcpp.complex cimport complex as cpp_complex

cdef extern from "fft.h":
	vector[cpp_complex[float]] _fft(vector[cpp_complex[float]] input_signal)
	vector[cpp_complex[float]] _fft_cuda_reference(vector[cpp_complex[float]] input_signal, int polling_rate)

def fft(input_signal):
	cdef vector[cpp_complex[float]] signal = input_signal
	return _fft(signal)

def fft_reference_gpu(input_signal, polling_rate):
	cdef vector[cpp_complex[float]] signal = input_signal
	cdef int pol_rate = polling_rate
	return _fft_cuda_reference(signal, pol_rate)