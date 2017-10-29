from scipy.io import wavfile
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
import numpy as np


# Función open_audio
def open_audio(audio):
    return wavfile.read(audio)


# Función plot_signal
def plot_signal(sample_rate, data):
    time = np.linspace(0, len(data)/sample_rate, num=len(data))
    plt.ioff()
    plt.figure()
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(time, data)
    plt.show()


# Función amp_modulation
# data: Information or modulation signal
# percentage: Modulation percentage
# freq: Frequency of carrier signal (cosine).
def amp_modulation(data, percentage, freq):
    information = np.array(data[1])
    time_inf = len(information)/data[0]
    samples_cos = np.linspace(0, time_inf, len(information))
    carrier = np.cos(2*np.pi*freq*samples_cos)
    m = percentage/100
    mod_signal = (1+m*information)*carrier
    return mod_signal


# Función freq_modulation
def freq_modulation(data, percentage):
    return data


a = open_audio("handel.wav")
test = amp_modulation(a, 125, a[0]*10)
plt.plot(test)
plt.show()
print("lol")
