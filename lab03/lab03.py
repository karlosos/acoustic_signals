import matplotlib.pyplot as plt
import numpy as np
import cmath

freq = 1 / 48000

my_file = open("Lab_03-Flt_03_DK.txt", "r")
signal = my_file.readlines()
t = np.linspace(0, len(signal), len(signal))

signalFFT = np.fft.rfft(signal)

signalPSD = np.abs(signalFFT) ** 2
signalPSD /= len(signalFFT) ** 2
signalPhase = np.angle(signalFFT)
newSignalFFT = signalFFT * cmath.rect(1.0, np.pi / 2)
newSignal = np.fft.irfft(newSignalFFT)

fftFreq = np.fft.rfftfreq(len(signal), freq)

plt.figure(figsize=(10, 4))

ax1 = plt.subplot(1, 1, 1)
ax1.plot(fftFreq, signalPSD)
ax1.set_ylabel("Power")
ax1.set_xlabel("frequency")

ax2 = ax1.twinx()
ax2.plot(fftFreq, signalPhase, alpha=0.25, color="r")
ax2.set_ylabel("Phase", color="r")


plt.tight_layout()
plt.show()
