from libcpp.vector cimport vector
from libcpp.complex cimport complex as cpp_complex

cdef extern from "fft.h":
	vector[float] _calculate_relative_error(int size, vector[cpp_complex[float]] reference, vector[cpp_complex[float]] computed)
	vector[cpp_complex[float]] _fft_cpu(vector[cpp_complex[float]] input_signal)
	vector[cpp_complex[float]] _fft_cuda_reference(vector[cpp_complex[float]] input_signal)
	vector[cpp_complex[float]] _forward_fft_R2C(vector[float] input)
	vector[cpp_complex[float]] _forward_fft_C2C(vector[cpp_complex[float]] input)
	vector[cpp_complex[float]] _manual_fft_impl(vector[cpp_complex[float]] input, int fft_size)
	vector[cpp_complex[float]] _manual_fft_impl2(vector[cpp_complex[float]] input, int fft_size)
	vector[float] _backward_fft_C2R(vector[cpp_complex[float]] input, int original_size)
	vector[float] _backward_fft_C2R_Complex(vector[cpp_complex[float]] input)

def fft(input_signal):
	cdef vector[cpp_complex[float]] signal = input_signal
	return _fft_cpu(signal)

def fft_reference_gpu(input_signal):
	cdef vector[cpp_complex[float]] signal = input_signal
	return _fft_cuda_reference(signal)

def MKL_forward_fft_R2C(input_signal):
	cdef vector[float] signal = input_signal
	return _forward_fft_R2C(signal)

def MKL_forward_fft_C2C(input_signal):
	cdef vector[cpp_complex[float]] signal = input_signal
	return _forward_fft_C2C(signal)

def MKL_backward_fft_C2R(input_signal, orig_size):
	cdef vector[cpp_complex[float]] signal = input_signal
	cdef int size = orig_size
	return _backward_fft_C2R(signal, size)

def MKL_backward_fft_C2R_Complex(input_signal):
	cdef vector[cpp_complex[float]] signal = input_signal
	return _backward_fft_C2R_Complex(signal)

def manual_fft_impl(input_signal, fft_size):
	cdef vector[cpp_complex[float]] signal = input_signal
	cdef int size = fft_size
	return _manual_fft_impl(signal, size)

def manual_fft_impl2(input_signal, fft_size):
	cdef vector[cpp_complex[float]] signal = input_signal
	cdef int size = fft_size
	return _manual_fft_impl2(signal, size)

def calculate_relative_error(size, reference, computed):
	cdef int _size = size
	cdef vector[cpp_complex[float]] _reference = reference
	cdef vector[cpp_complex[float]] _computed = computed
	return _calculate_relative_error(_size, _reference, _computed)