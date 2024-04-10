from libcpp.vector cimport vector
from libcpp.complex cimport complex as cpp_complex

cdef extern from "fft.h":
	vector[cpp_complex[float]] _fft(vector[cpp_complex[float]] input_signal)

def fft(input_signal):
	cdef vector[cpp_complex[float]] signal = input_signal
	return _fft(signal)