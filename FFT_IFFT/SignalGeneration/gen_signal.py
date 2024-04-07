import math
import numpy as np
import matplotlib.pyplot as plt

Fs = 2 ** 18
time = np.arange(0, 1, 1/Fs)

F1 = 1900
Signal1 = np.sin(2 * np.pi * time * F1)

F2 = 2600
Signal2 = np.sin(2 * np.pi * time * F2)

F3 = 4800
Signal3 = np.sin(2 * np.pi * time * F3)

F4 = 10000
Signal4 = np.sin(2 * np.pi * time * F4)

F5 = 21000
Signal5 = np.sin(2 * np.pi * time * F5)

F6 = 27500
Signal6 = np.sin(2 * np.pi * time * F6)

F7 = 32000
Signal7 = np.sin(2 * np.pi * time * F7)

F8 = 44100
Signal8 = np.sin(2 * np.pi * time * F8)

combined_signals = Signal1 + Signal2 + Signal3 + \
    Signal4 + Signal5 + Signal6 + Signal7 + Signal8

plt.subplot(2, 1, 1)
plt.plot(time, Signal1)
plt.plot(time, Signal2)
plt.plot(time, Signal3)
plt.plot(time, Signal4)
plt.plot(time, Signal5)
plt.plot(time, Signal6)
plt.plot(time, Signal7)
plt.plot(time, Signal8)

plt.subplot(2, 1, 2)
plt.plot(time, combined_signals)
plt.show()


with open("../combined_signal.hpp", "w") as header_file:
	header_file.write("double signal[] = {")
	for idx, x in enumerate(combined_signals):
		if idx == (len(combined_signals) - 1):
			header_file.write(f"{x}\n")
		else:
			header_file.write(f"{x}, ")
	header_file.write("};\n")