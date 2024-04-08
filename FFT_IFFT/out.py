import numpy as np
from pylab import *

dt = 1 / 144000
t = np.arange(0, 1, dt)
x = np.sin(2 * pi * t * 16000)
fftx = np.fft.fft(x)
fftx = fftx[range(int(len(fftx)/2))]
freq_fftx = np.linspace(0,2/dt,len(fftx))
plot(freq_fftx,abs(fftx)**2)
show()