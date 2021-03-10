import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from funcy import print_durations
from numba import jit


def generate_sin(sr=44100, frequency=440, length=5):
    t = np.linspace(0, length, sr * length)
    y = np.sin(frequency * 2 * np.pi * t)

    return y, sr


@print_durations()
@jit
def autocorellation(y):
    length = y.shape[0]
    g = np.zeros(length)
    for k in range(length):
        for n in range(length - 1 - k):
            g[k] += y[n, 0] * y[n-k, 1]
    return g


def gcc_phat():
    pass


def ild():
    pass


def main():
    y, sr = generate_sin(44100, 440, 5)

    wavfile.write("./output/440hz_mono.wav", sr, y)

    # Shift channel
    shift_seconds = 0.002
    shift = int(shift_seconds * sr)
    y_shifted = np.roll(y, shift=shift)
    # y_shifted[:shift] = np.nan

    # Plot signals with shift
    fig, axes = plt.subplots(1, 1)
    axes.plot(y[:1000])
    axes.plot(y_shifted[:1000])
    plt.show()

    # Stereo
    y_stereo = np.column_stack((y, y))
    wavfile.write("./output/440hz_stereo.wav", sr, y_stereo)

    # Stereo shifted
    y_stereo_shifted = np.column_stack((y, y_shifted))
    wavfile.write("./output/440hz_stereo_shifted.wav", sr, y_stereo_shifted)

    # Crosscorellation 
    g = autocorellation(y_stereo_shifted)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(g)
    axes[1].plot(g[:1000])
    plt.show()
    
    # Crosscorellation numpu
    g = np.correlate(y_stereo_shifted[:, 0], y_stereo_shifted[:, 1], "full")
    plt.plot(g)
    plt.show()
    breakpoint()


if __name__ == "__main__":
    main()
