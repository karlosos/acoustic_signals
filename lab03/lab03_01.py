from scipy import signal
import numpy as np


my_file = open("Lab_03-Flt_03_DK.txt", "r")
b = my_file.readlines()

# b = signal.firwin(80, 0.5, window=('kaiser', 8))
w, h = signal.freqz(b, fs=48000)

signalPSD = 20 * np.log10(abs(h))
# signalPSD = (signalPSD - min(signalPSD)) / (max(signalPSD - min(signalPSD)))

signalPhase = np.unwrap(np.angle(h))
# signalPhase = (signalPhase - min(signalPhase)) / (max(signalPhase - min(signalPhase)))

l = signalPSD[signalPSD >= -3][0]
r = signalPSD[signalPSD >= -3][-1]

distance = np.abs(l - r)

print(l, r)
print(distance)
print("test")

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 4))
ax1 = plt.subplot()
plt.title('Digital filter frequency response')


ax1.plot(w, signalPSD, 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [Hz]')

ax2 = ax1.twinx()
ax2.plot(w, signalPhase, 'g')
ax2.set_ylabel('Phase (rad)', color='g')
ax2.grid()
# ax2.axis('tight')
plt.show()

