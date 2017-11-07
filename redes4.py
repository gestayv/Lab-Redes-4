from scipy.io import wavfile
from scipy.integrate import cumtrapz
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
import numpy as np
import os


# Función open_audio
def open_audio(audio):
    return wavfile.read(audio)


# Función amp_modulation
# data: Information or modulation signal
# percentage: Modulation percentage
# freq: Frequency of carrier signal (cosine).
def am_modulation(data, percentage, freq):
    print("Modulación de amplitud...")
    signal_time = len(data[1])/data[0]
    # Nro de samples del vector que representa la portadora
    samples_carrier = np.arange(0, signal_time, 1/(4*freq))
    carrier = np.cos(2 * np.pi * freq * samples_carrier)
    # Como la moduladora tiene menos muestras que la portadora, se interpola para que tengan el mismo largo
    # Nro de samples de la onda moduladora
    samples_signal = np.linspace(0, signal_time, len(data[1]))
    information = np.interp(samples_carrier, samples_signal, data[1])

    m = percentage/100
    mod_signal = (1+m*information)*carrier
    plot_waves(samples_carrier, information, carrier, mod_signal, "AM_Modulation")
    print("Calculando transformadas de AM...")
    plot_spectrums(information, mod_signal, carrier, samples_carrier, "TF_AM")

    return mod_signal, samples_carrier


# Función fm_modulation
def fm_modulation(data, percentage, freq):
    print("Modulación de frecuencia...")
    signal_time = len(data[1]) / data[0]
    samples_carrier = np.arange(0, signal_time, 1 / (4 * freq))
    samples_signal = np.linspace(0, signal_time, len(data[1]))
    carrier = np.cos(2 * np.pi * freq * samples_carrier)
    information = np.interp(samples_carrier, samples_signal, data[1])   # Interpolación de moduladora
    m = percentage/100                                                  # Cálculo del índice de modulación
    integral_info = cumtrapz(information, samples_carrier, initial=0)   # Integral acumulativa del mensaje
    mod_signal = np.cos(2*np.pi*freq*samples_carrier+m*integral_info)

    plot_waves(samples_carrier, information, carrier, mod_signal, "FM_Modulation")
    print("Calculando transformadas de FM...")
    plot_spectrums(information, mod_signal, carrier, samples_carrier, "TF_FM")
    return mod_signal, samples_carrier


def am_demodulation(modulated, samples_carrier, freq):
    print("Demodulando amplitud...")
    carrier = np.cos(2 * np.pi * freq * samples_carrier)
    demod_signal = modulated*carrier
    xf = fftp.fftfreq(len(modulated), samples_carrier[2]-samples_carrier[1])
    plt.plot(xf, fftp.fft(demod_signal))
    # aplicar filtro paso bajo y amplificar
    return demod_signal


def plot_spectrums(information, modulated, carrier, samples_carrier, name):
    # Cálculo de las transformadas de la moduladora, portadora y modulada.
    ft_original = fftp.fft(information)
    ft_modulated = fftp.fft(modulated)
    ft_carrier = fftp.fft(carrier)
    # Se emplea fftfreq para graficar la transformada de manera apropiada.
    samples = len(ft_original)
    xf = fftp.fftfreq(samples, samples_carrier[2] - samples_carrier[1])
    xf = xf[0:samples//2]
    # Creación de una figura con subplots
    plt.figure(figsize=(10.24, 7.20), dpi=100)
    plt.subplot(221)
    plt.title("FT de la moduladora")
    plt.plot(xf, np.abs(ft_original[0:samples//2]))
    plt.subplot(222)
    plt.title("FT de la portadora")
    plt.plot(xf, np.abs(ft_carrier[0:samples//2]))
    plt.subplot(223)
    plt.title("FT de la señal modulada")
    plt.plot(xf, np.abs(ft_modulated[0:samples//2]), "g")
    plt.tight_layout()
    dir_name = "./graphs/"
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(dir_name+name+'.png', bbox_inches='tight', dpi=100)


def plot_waves(time, information, carrier, modulated, name):
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure(figsize=(10.24, 7.20), dpi=100)
    plt.subplot(221)
    plt.title("Onda moduladora")
    plt.plot(time, information)
    plt.subplot(222)
    plt.title("Onda portadora")
    plt.plot(time, carrier)
    plt.subplot(223)
    plt.title("Onda modulada")
    plt.plot(time, modulated, "g")
    plt.tight_layout()
    dir_name = "./graphs/"
    os.makedirs(dir_name, exist_ok=True)
    plt.draw()
    plt.savefig(dir_name+name+'.png', bbox_inches='tight', dpi=200)


def plot_signal(time, data, name, xAxis, yAxis):
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure(figsize=(10.24, 7.20), dpi=100)
    plt.plot(time, data)
    plt.title(name)
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.tight_layout()
    dir_name = "./graphs/"
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(dir_name + name + '.png', bbox_inches='tight', dpi=200)

signal = open_audio("handel.wav")
# frecuencia carrier 30Khz
signal_2 = [signal[0], np.cos(2*np.pi*1*np.linspace(0, len(signal[1])/signal[0], len(signal[1])))]
#### AM ####
am, sc = am_modulation(signal, 15, 30000)
am_demodulation(am, sc, 30000)
#### FM ####
fm_modulation(signal, 100, 30000)
print("Proceso finalizado!")