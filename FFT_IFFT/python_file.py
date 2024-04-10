import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from cython_file import fft as custom_fft
from cython import floatcomplex

Fs = 2 ** 13
time = np.arange(0, 1, 1/Fs)

F1 = 5
F2 = 67
F3 = 86
F4 = 800
F5 = 1200
F6 = 1600
F7 = 2600
F8 = 3500

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

np_fft = fft(combined_signals)
fft_sig = custom_fft(combined_signals)

N_custom = len(fft_sig)
n_custom = np.arange(N_custom)
T_custom = N_custom/Fs
freq_custom = n_custom/T_custom

plt.subplot(2, 1, 1)
plt.stem(freq_custom, np.abs(fft_sig))
plt.xlim(-200, Fs / 2)

N_np = len(np_fft)
n_np = np.arange(N_np)
T_np = N_np/Fs
freq_np = n_np/T_np

plt.subplot(2, 1, 2)
plt.stem(freq_np, np.abs(np_fft))
plt.xlim(-200, Fs / 2)
plt.show()
