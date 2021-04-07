"""
Lab 04:

wygenerowac funkcje sinc
przesunąć ją w osi x (tak żeby osie dodatnie)
wybrać okno (podane na wykładzie)
porównać charakterystyki częstotliwościowe w skali
logarytmicznej i w skali liniowej

TODO:
- to co zroibłem niżej to tylko PoC przy użyciu gotowych funkcji
- zaimplementowanie analogicznie tylko ręcznie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift


def main():
    fs = 48000  # czestotliwosc probkowania
    n = 101  # rzad filtra
    fc = 1000  # czestotliwosc odciecia

    # Tworzenie okna hamminga
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hamming.html#scipy.signal.windows.hamming
    window = signal.windows.hamming(n)

    # Wykres okna hamminga
    plt.plot(window)
    plt.title("Okno haminga")
    plt.ylabel("Amplituda")
    plt.xlabel("Próbka")
    plt.show()
    # Wykres odpowiedzi czestotliwosciowej okna hamminga
    A = fft(window, 2048) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    plt.plot(freq, response)
    plt.axis([-0.5, 0.5, -120, 0])
    plt.title("Odpowiedź częstotliwościowa okna haminga")
    plt.ylabel("[dB]")
    plt.xlabel("Znormalizowana częstotliwość")
    plt.show()

    # Tworzenie filtra
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    b = signal.firwin(n, cutoff=fc, fs=fs, window='hamming', pass_zero="lowpass")
    w, h = signal.freqz(b, [1])

    # Wykres odpowiedzi impulsowej
    plt.title("Odpowiedź impulsowa filtra")
    plt.plot(b)
    plt.show()

    # Wykresy charakterystyk
    fig, axes = plt.subplots(2, 1)
    # Wykres charakterystyki amplitudowej
    x = w * fs / (2 * np.pi)
    y = 20 * np.log10(np.abs(h))
    axes[0].plot(x, y)
    axes[0].set_xlabel("Częstotliwość [Hz]")
    # Wykres charakterystyki fazowej
    y = np.arctan2(np.imag(h), np.real(h))
    y = np.degrees(np.unwrap(y))
    axes[1].plot(x, y)
    axes[1].set_xlabel("Częstotliwość [Hz]")
    fig.suptitle('Charakterystyka filtra z oknem hamminga', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
