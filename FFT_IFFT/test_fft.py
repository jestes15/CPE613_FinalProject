import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft as np_fft
from custom_fft import fft, fft_reference_gpu # type: ignore

Fs = 2 ** 13
time = np.arange(0, 1, 1/Fs)

F1 = 500
F2 = 1000
F3 = 1500
F4 = 2000
F5 = 2500
F6 = 3000
F7 = 3500
F8 = 4000

Signal1 = np.sin(2 * np.pi * time * F1).astype(complex)
Signal2 = np.sin(2 * np.pi * time * F2).astype(complex)
Signal3 = np.sin(2 * np.pi * time * F3).astype(complex)
Signal4 = np.sin(2 * np.pi * time * F4).astype(complex)
Signal5 = np.sin(2 * np.pi * time * F5).astype(complex)
Signal6 = np.sin(2 * np.pi * time * F6).astype(complex)
Signal7 = np.sin(2 * np.pi * time * F7).astype(complex)
Signal8 = np.sin(2 * np.pi * time * F8).astype(complex)

combined_signals = Signal1 + Signal2 + Signal3 + \
    Signal4 + Signal5 + Signal6 + Signal7 + Signal8

np_fft = np_fft(combined_signals)
fft_sig = fft(combined_signals)
fft_cuda = fft_reference_gpu(combined_signals, Fs)

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

plt.figure(figsize=(15, 12))

ax1 = plt.subplot(122)
ax1.margins(0, 0.1)
ax1.plot(time, combined_signals)
ax1.set_xlim([0.0, 0.01])
ax1.set_title("Original Signal")

ax2 = plt.subplot(321)
ax2.stem(freq_custom, np.abs(fft_sig))
ax2.set_xlim([0.0, Fs/2])
ax2.set_title('CPU FFT Result')

ax3 = plt.subplot(323)
ax3.stem(freq_cuda, np.abs(fft_cuda))
ax3.set_xlim([0.0, Fs/2])
ax3.set_title('CUDA Reference FFT Result')

ax4 = plt.subplot(325)
ax4.stem(freq_np, np.abs(np_fft))
ax4.set_xlim([0.0, Fs/2])
ax4.set_title('Numpy Reference FFT Result')

plt.show()
