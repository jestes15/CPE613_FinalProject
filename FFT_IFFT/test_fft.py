import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft as np_fft
from scipy.signal import chirp
from custom_fft import fft, fft_reference_gpu, MKL_forward_fft_R2C, MKL_forward_fft_C2C, manual_fft_impl, calculate_relative_error  # type: ignore

Fs = 4096
time = np.arange(0, 1, 1/Fs)

F1 = 1540
F2 = 1900
F3 = 32
F4 = 64
F5 = 128
F6 = 256
F7 = 512
F8 = 1024

Signal1 = np.sin(2 * np.pi * time * F1)
Signal2 = np.sin(2 * np.pi * time * F2)
Signal3 = np.sin(2 * np.pi * time * F3)
Signal4 = np.sin(2 * np.pi * time * F4)
Signal5 = np.sin(2 * np.pi * time * F5)
Signal6 = np.sin(2 * np.pi * time * F6)
Signal7 = np.sin(2 * np.pi * time * F7)
Signal8 = np.sin(2 * np.pi * time * F8)

# Generate wideband noise with extremely low noise floor
upper_limit = 0.5
lower_limit = 0
noise = np.random.normal(lower_limit, upper_limit, len(
    time)) + np.random.normal(lower_limit, upper_limit, len(time)) + np.random.normal(lower_limit, upper_limit, len(time))

combined_signals = Signal1 + Signal2 + Signal3 + \
    Signal4 + Signal5 + Signal6 + Signal7 + Signal8

plt.figure(1)
ax = plt.subplot(111)
ax.plot(time, combined_signals)
ax.plot(time, noise)
ax.set_xlim([0.0, 0.05])

combined_signals += noise

np_fft = np_fft(combined_signals.astype(complex))
fft_sig = fft(combined_signals.astype(complex))
fft_cuda = fft_reference_gpu(combined_signals.astype(complex))
fft_r2c_mkl = MKL_forward_fft_R2C(combined_signals)
fft_c2c_mkl = MKL_forward_fft_C2C(combined_signals.astype(complex))
fft_c2c_manual_impl = manual_fft_impl(combined_signals.astype(complex), 4096)

relative_error = calculate_relative_error(
    len(combined_signals), fft_cuda, fft_c2c_manual_impl)
print(
    f"Relative Error between CUDA and the Shared Memory Impl: {relative_error[2]}")

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
ax1.set_xlim([0.0, 0.05])
ax1.set_title("Original Signal")

ax2 = plt.subplot(331)
ax2.stem(freq_custom, np.abs(fft_sig))
ax2.set_xlim([0.0, Fs/2])
ax2.set_title('CPU FFT Result')

ax3 = plt.subplot(332)
ax3.stem(freq_cuda, np.abs(fft_cuda))
ax3.set_xlim([0.0, Fs/2])
ax3.set_title('CUDA Reference FFT Result')

ax4 = plt.subplot(333)
ax4.stem(freq_np, np.abs(np_fft))
ax4.set_xlim([0.0, Fs/2])
ax4.set_title('Numpy Reference FFT Result')

ax4 = plt.subplot(334)
ax4.stem(freq_c2c_mkl, np.abs(fft_c2c_mkl))
ax4.set_xlim([0.0, Fs/2])
ax4.set_title('MKL Complex-to-Complex Reference FFT Result')

ax4 = plt.subplot(335)
ax4.stem(freq_r2c_mkl, np.abs(fft_r2c_mkl))
ax4.set_xlim([0.0, Fs/2])
ax4.set_title('MKL Real-to-Complex Reference FFT Result')

ax4 = plt.subplot(336)
ax4.stem(freq_c2c_manual_impl, np.abs(fft_c2c_manual_impl))
ax4.set_xlim([0.0, Fs/2])
ax4.set_title('Manual CUDA Implementation w/ Shared Memory')


plt.show()
