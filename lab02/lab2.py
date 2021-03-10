import numpy as np
from scipy.io import wavfile

def generate_sin(sr=44100, frequency=440, length=5):
    t = np.linspace(0, length, sr * length)  #  Produces a 5 second Audio-File
    y = np.sin(frequency * 2 * np.pi * t)  #  Has frequency of 440Hz

    return y, sr

def main():
    y, sr = generate_sin(44100, 440, 5)

    wavfile.write('./output/440hz_mono.wav', sr, y)

    # To write multiple-channels, use a 2-D array of shape (Nsamples, Nchannels).
    # Stworzenie sygnału stereo
    y_stereo = np.column_stack((y, y))
    wavfile.write('./output/440hz_stereo.wav', sr, y_stereo)

if __name__ == "__main__":
    main()
