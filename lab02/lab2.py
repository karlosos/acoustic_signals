import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import correlate
from funcy import print_durations
from numba import jit
import pandas as pd


def generate_sin(sr=44100, frequency=440, length=5):
    t = np.linspace(0, length, sr * length)
    y = np.sin(frequency * 2 * np.pi * t)

    return y, sr


@print_durations()
@jit
def crosscorrelation_mine(y):
    length = y.shape[0]
    g = np.zeros(length)
    for k in range(length):
        for n in range(length - 1 - k):
            g[k] += y[n, 1] * y[n - k, 0]
    return g


@print_durations()
def crosscorrelation_scipy(y1, y2, sr):
    length = y1.size
    xcorr = correlate(y1, y2)
    dt = np.arange(1 - length, length)
    shift = dt[xcorr.argmax()]
    return shift


def gcc_phat(y1, y2):
    n = y1.size + y2.size

    x_l = np.fft.rfft(y1)
    x_r = np.fft.rfft(y2)
    x_conj = np.conj(x_r)

    g = np.fft.irfft((x_l * x_conj) / (np.abs(x_l * x_conj)))
    dt = np.arange(-y1.size, y1.size)
    shift = dt[g.argmax()]
    return shift


def ild(y1, y2):
    left_channel = np.average(np.fft.fft(y1) ** 2)
    right_channel = np.average(np.fft.fft(y2) ** 2)
    ild = 10 * np.log10(left_channel / right_channel)
    return ild


def find_angle(shift, speed=343, ear_distance=20):
    return np.arcsin((shift * speed) / ear_distance)


def main():
    y, sr = generate_sin(44100, 440, 5)

    wavfile.write("./output/440hz_mono.wav", sr, y)

    # Shift channel
    shift_seconds = 0.002
    shift = int(shift_seconds * sr)
    print(f"Shift: {shift} samples, {shift_seconds} seconds")
    y_shifted = np.roll(y, shift=shift)
    # y_shifted[:shift] = np.nan

    # Plot signals with shift
    fig, axes = plt.subplots(1, 1)
    axes.plot(y[:1000])
    axes.plot(y_shifted[:1000])
    plt.show()

    # Create stereo
    y_stereo = np.column_stack((y, y))
    wavfile.write("./output/440hz_stereo.wav", sr, y_stereo)

    # Create shifted stereo
    y_stereo_shifted = np.column_stack((y, y_shifted))
    wavfile.write("./output/440hz_stereo_shifted.wav", sr, y_stereo_shifted)

    # # # Find shift (mine implementation)
    # g = crosscorrelation_mine(y_stereo_shifted)
    # recovered_shift = np.argmax(g)
    # print(
    #     f"Recovered shift my implementation: {recovered_shift}, {recovered_shift/sr} seconds"
    # )

    # # Find shift (scipy implementation)
    recovered_shift = crosscorrelation_scipy(
        y_stereo_shifted[:, 0], y_stereo_shifted[:, 1], sr
    )
    print(
        f"Recovered shift scipy implementation: {recovered_shift}, {recovered_shift/sr} seconds"
    )

    # Angle
    angle = find_angle(0.03)
    print(f"Angle: {angle}")

    # GCC PHAT
    recovered_shift = gcc_phat(y_stereo_shifted[:, 0], y_stereo_shifted[:, 1])
    print(f"Recovered shift gccphat: {recovered_shift}, {recovered_shift/sr} seconds")

    # ILD
    print(f"ILD: {ild(y_stereo_shifted[:, 0], y_stereo_shifted[:, 1])}")


def experiment_shifts():
    y, sr = generate_sin(44100, 3000, 25)
    data = {
        "shift": [],
        "recovered_shift_xcorr": [],
        # "recovered_shift_xcorr_2": [],
        "recovered_shift_gcc": [],
        "angle": [],
        "ild": [],
    }

    for shift_seconds in np.linspace(0.0001, 0.002, 5):
        shift = int(shift_seconds * sr)
        y_shifted = np.roll(y, shift=shift)

        # Plot signals with shift
        fig, axes = plt.subplots(1, 1)
        axes.plot(y[:1000], label="left")
        axes.plot(y_shifted[:1000], label="right")
        plt.legend()
        plt.show()

        y_stereo_shifted = np.column_stack((y, y_shifted))

        # g = crosscorrelation_mine(y_stereo_shifted)
        # recovered_shift_xcorr_mine = np.argmax(g)
        recovered_shift_xcorr_scipy = crosscorrelation_scipy(
            y_stereo_shifted[:, 0], y_stereo_shifted[:, 1], sr
        )
        angle = find_angle(shift_seconds)
        recovered_shift_gcc = gcc_phat(y_stereo_shifted[:, 0], y_stereo_shifted[:, 1])
        ild_val = ild(y_stereo_shifted[:, 0], y_stereo_shifted[:, 1])
        data["shift"].append(shift)
        data["recovered_shift_xcorr"].append(recovered_shift_xcorr_scipy)
        # data["recovered_shift_xcorr_2"].append(recovered_shift_xcorr_mine)
        data["recovered_shift_gcc"].append(recovered_shift_gcc)
        data["angle"].append(angle)
        data["ild"].append(ild_val)

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("shifts.csv")


if __name__ == "__main__":
    # main()
    experiment_shifts()
