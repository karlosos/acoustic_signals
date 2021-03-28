from scipy import signal
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def main():
    b = get_filter()
    y, sr = librosa.load("./data/dziendobry.wav", mono=True)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(fft.fft(y))
    axes[0].set_title('Widmo przed filtracją')

    filtered_y = signal.convolve(y.astype('float32'), b.astype('float32'))
    axes[1].plot(fft.fft(filtered_y))
    axes[1].set_title('Widmo po filtracji')
    plt.tight_layout()
    plt.show()

    spectogram(y)
    plt.title("Spektogram przed filtracją")
    plt.show()

    spectogram(filtered_y)
    plt.title("Spektogram po filtracji")
    plt.show()


def spectogram(y):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()


def get_filter():
    b_file = open("Lab_03-Flt_03_DK.txt", "r")
    b = b_file.readlines()
    # FFT
    w, h = signal.freqz(b, fs=48000)
    # Charakterystyka częstotliwościowa i fazowa
    filter_freq = 20 * np.log10(abs(h))
    filter_phase = np.unwrap(np.angle(h))
    # Częstotliwości graniczne
    lower = w[filter_freq >= -3][0]
    upper = w[filter_freq >= -3][-1]
    bandwidth = np.abs(lower - upper)
    print(lower, upper)
    print(bandwidth)
    # Wykresy dla charakterystyk częstotliwościowych
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    axes[0].plot(w, filter_freq, 'b')
    axes[0].set_ylabel('Amplituda [dB]', color='b')
    axes[0].set_xlabel('Częsotliwość [Hz]')
    axes[0].axhline(y=-3, linestyle='--', label='-3 dB')
    axes[0].axvline(x=lower)
    axes[0].axvline(x=upper)
    axes[0].set_xticks([lower, upper])
    axes[0].legend()
    axes[1].plot(w, filter_phase, 'g')
    axes[1].set_ylabel('Faza (rad)', color='g')
    axes[1].grid()
    plt.tight_layout()
    plt.show()
    # Parametry filtra
    print("Szerokość pasma:", bandwidth)
    print("Częstotliwości graniczne:", lower, upper)

    return np.array(b)


if __name__ == "__main__":
    main()


