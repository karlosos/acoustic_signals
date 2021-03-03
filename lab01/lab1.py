import librosa
import soundfile

import matplotlib.pyplot as plt
import numpy as np


def main():
    y, sr = librosa.load('./data/a_C4_ugp44.wav')
    print("Częstotliwość próbkowania", sr)
    print("Czas trwania sygnału", librosa.get_duration(y=y, sr=sr))

    # Zapis do pliku WAV (mono i stereo)
    soundfile.write('output.wav', y, sr) 

    # Normalizacja sygnalu
    y_max = np.max(y)
    y_norm = y/y_max

    # Normalizacja wzgleden zadanej wielkosci
    plt.plot(y[:50], label="y")
    y_norm = y/librosa.db_to_amplitude(0.1)
    plt.plot(y_norm[:50], label="-0.1dB")
    y_norm = y/librosa.db_to_amplitude(0.3)
    plt.plot(y_norm[:50], label="-0.3dB")
    y_norm = y/librosa.db_to_amplitude(3)
    plt.plot(y_norm[:50], label="-3dB")

    plt.legend()
    plt.show()

    # Monofonizacja
    y_mono = librosa.to_mono(y)
    soundfile.write('./data/output.wav', y_mono, sr) 

    # Wyznaczanie i eliminacja składowej stałej
    y_mean = np.mean(y_mono)
    y_mono_eliminated = y_mono - y_mean

    # Zmiana czestotliwosci probkowania za pomoca dostepnych bibliotek
    y_8k = librosa.resample(y, sr, 8000)
    soundfile.write('./data/output_8k.wav', y_8k, 8000) 
    y_44k = librosa.resample(y, sr, 44000)
    soundfile.write('./data/output_44k.wav', y_44k, 44000) 

    # Wyswietlenie sygnalu
    x = np.arange(0, len(y_mono))/sr
    plt.plot(x, y_mono)
    plt.show()


if __name__ == "__main__":
    main()
