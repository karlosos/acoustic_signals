import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def generate_sin(sr=44100, frequency=440, length=5):
    t = np.linspace(0, length, sr * length)  #  Produces a 5 second Audio-File
    y = np.sin(frequency * 2 * np.pi * t)  #  Has frequency of 440Hz

    return y, sr

def main():
    y, sr = generate_sin(44100, 440, 5)

    wavfile.write('./output/440hz_mono.wav', sr, y)

    # Shift channel
    shift_seconds = 0.002
    shift = int(shift_seconds * sr)
    y_shifted = np.roll(y, shift=shift)
    y_shifted[:shift] = np.nan

    # Plot signals with shift
    fig, axes = plt.subplots(1, 1)
    axes.plot(y[:1000])
    axes.plot(y_shifted[:1000])
    plt.show()

    # Stereo 
    y_stereo = np.column_stack((y, y))
    wavfile.write('./output/440hz_stereo.wav', sr, y_stereo)

    # Stereo shifted
    y_stereo_shifted = np.column_stack((y, y_shifted))
    wavfile.write('./output/440hz_stereo_shifted.wav', sr, y_stereo_shifted)

if __name__ == "__main__":
    main()
