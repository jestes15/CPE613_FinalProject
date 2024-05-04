import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft as np_fft
from scipy.signal import chirp
from custom_fft import fft, fft_reference_gpu, MKL_forward_fft_R2C, MKL_forward_fft_C2C, manual_fft_impl, calculate_relative_error  # type: ignore

Fs = 4096
time = np.arange(0, 1, 1/Fs)

lower_freq = 1
upper_freq = 50

combined_signals = np.append(
    chirp(time[0:int(Fs/2)], f0=lower_freq, f1=upper_freq, t1=1, method='linear'),
    chirp(time[0:int(Fs/2)], f0=lower_freq, f1=upper_freq, t1=1, method='linear'))

np_fft = np_fft(combined_signals.astype(complex))
fft_sig = fft(combined_signals.astype(complex))
fft_cuda = fft_reference_gpu(combined_signals.astype(complex))
fft_r2c_mkl = MKL_forward_fft_R2C(combined_signals)
fft_c2c_mkl = MKL_forward_fft_C2C(combined_signals.astype(complex))
fft_c2c_manual_impl = manual_fft_impl(combined_signals.astype(complex), 4096)

relative_error = calculate_relative_error(
    len(combined_signals), fft_cuda, fft_c2c_manual_impl)
print(f"Relative Error between CUDA and MKL C2C: {relative_error}")
print(f"First four elements of the the CUDA Result: {fft_cuda[0:4]}")
print(f"First four elements of the the MKL Result: {fft_c2c_manual_impl[0:4]}")

N_custom = len(fft_sig)
n_custom = np.arange(N_custom)
T_custom = N_custom/Fs
freq_custom = n_custom/T_custom

N_np = len(np_fft)
n_np = np.arange(N_np)
T_np = N_np/Fs
freq_np = n_np/T_np

N_cuda = len(fft_cuda)
n_cuda = np.arange(N_cuda)
T_cuda = N_cuda/Fs
freq_cuda = n_cuda/T_cuda

N_r2c_mkl = len(fft_r2c_mkl)
n_r2c_mkl = np.arange(N_r2c_mkl)
T_r2c_mkl = N_r2c_mkl/Fs
freq_r2c_mkl = n_r2c_mkl/T_r2c_mkl

N_c2c_mkl = len(fft_c2c_mkl)
n_c2c_mkl = np.arange(N_c2c_mkl)
T_c2c_mkl = N_c2c_mkl/Fs
freq_c2c_mkl = n_c2c_mkl/T_c2c_mkl

N_c2c_manual_impl = len(fft_c2c_manual_impl)
n_c2c_manual_impl = np.arange(N_c2c_manual_impl)
T_c2c_manual_impl = N_c2c_manual_impl/Fs
freq_c2c_manual_impl = n_c2c_manual_impl/T_c2c_manual_impl

plt.figure(figsize=(20, 11))
ax1 = plt.subplot(313)
ax1.plot(time, combined_signals)
ax1.set_title("Original Signal")

ax2 = plt.subplot(331)
ax2.stem(freq_custom, np.abs(fft_sig))
ax2.set_xlim([0.0, upper_freq + 1])
ax2.set_title('CPU FFT Result')

ax3 = plt.subplot(332)
ax3.stem(freq_cuda, np.abs(fft_cuda))
ax3.set_xlim([0.0, upper_freq + 1])
ax3.set_title('CUDA Reference FFT Result')

ax4 = plt.subplot(333)
ax4.stem(freq_np, np.abs(np_fft))
ax4.set_xlim([0.0, upper_freq + 1])
ax4.set_title('Numpy Reference FFT Result')

ax4 = plt.subplot(334)
ax4.stem(freq_c2c_mkl, np.abs(fft_c2c_mkl))
ax4.set_xlim([0.0, upper_freq + 1])
ax4.set_title('MKL Complex-to-Complex Reference FFT Result')

ax4 = plt.subplot(335)
ax4.stem(freq_r2c_mkl, np.abs(fft_r2c_mkl))
ax4.set_xlim([0.0, upper_freq + 1])
ax4.set_title('MKL Real-to-Complex Reference FFT Result')

ax4 = plt.subplot(336)
ax4.stem(freq_c2c_manual_impl, np.abs(fft_c2c_manual_impl))
ax4.set_xlim([0.0, upper_freq + 1])
ax4.set_title('Manual CUDA Implementation w/ Shared Memory')


plt.show()
