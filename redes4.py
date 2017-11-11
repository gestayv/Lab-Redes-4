from scipy.io import wavfile
from scipy.integrate import cumtrapz
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
import numpy as np
import os
import lab2lib

# Función open_audio: Se encarga de abrir un archivo de audio en formato wav.
# Entrada:
#           - audio: String, nombre del archivo de audio con extensión incluida.
# Salida:
#           - Arreglo, en su primera posición se ubica un entero que indica la frecuencia de muestreo del audio
#             mientras que en su segunda posición se encuentra otro arreglo que representa el audio abierto.
def open_audio(audio):
    return wavfile.read(audio)

# Función save_audio: Se encarga de guardar un archivo de audio en formato wav.
# Entrada:
#           - name: String, nombre con el que se guarda el archivo de audio
#           - rate: Entero, frecuencia de muestreo del archivo de audio.
#           - data: Arreglo, contiene los datos que componen el archivo de audio
# Salida:
#           - Archivo de audio generado en la carpeta "/audio/" ubicada en el mismo directorio que el código.
def save_audio(name, rate, data):
    dir_name = "./audio/"
    os.makedirs(dir_name, exist_ok=True)
    wavfile.write(dir_name + name, rate, data)


# Función am_modulation: Recibe un mensaje y lo emplea para modular la amplitud de una señal portadora.
# Entradas:
#           - data: Arreglo, contiene un entero en su primera posición correspondiente a la frecuencia de muestreo
#                   de la señal moduladora y en su segunda posición un arreglo con los datos del mensaje.
#           - percentage: Entero, porcentaje de modulación.
#           - freq: Frecuencia de la señal portadora.
# Salida:
#           - mod_signal: arreglos, corresponde a la señal modulada
#           - samples_carrier: muestras empleadas para generar la señal portadora.
def am_modulation(data, percentage, freq):
    signal_time = len(data[1])/data[0]

    # Se generan las muestras de la portadora, a una frecuencia de muestreo de 4 veces la frecuencia a la que se modula.
    samples_carrier = np.arange(0, signal_time, 1/(4*freq))
    carrier = np.cos(2 * np.pi * freq * samples_carrier)

    # Se generan las muestras iniciales del mensaje, las que se emplean para interpolar posteriormente.
    samples_signal = np.linspace(0, signal_time, len(data[1]))
    # Interpolación del mensaje para igualar el número de muestras con la portadora.
    information = np.interp(samples_carrier, samples_signal, data[1])
    m = percentage/100

    # Cálculo de la señal modulada.
    mod_signal = m*information*carrier

    # Se interpola la señal modulada para tener una frecuencia de muestreo menor, soportada por wavfile.write()
    mod_signal_small = np.interp(samples_signal, samples_carrier, mod_signal)
    save_audio("audio_am_" + str(percentage) + ".wav", data[0], mod_signal_small/500)

    # Generación de gráficos (modulada y espectros de frecuencia)
    plot_signal(samples_carrier, mod_signal, "Modulación AM "+str(percentage)+"%", "T[s]", "Amplitud[dB]")
    plot_spectrums(information, mod_signal, carrier, samples_carrier, "TF_AM_"+str(percentage)+"%")
    return mod_signal, samples_carrier


# Función fm_modulation: Recibe un mensaje y lo emplea para modular la frecuencia de una señal portadora.
# Entradas:
#           - data: Arreglo, contiene un entero en su primera posición correspondiente a la frecuencia de muestreo
#                   de la señal moduladora y en su segunda posición un arreglo con los datos del mensaje.
#           - percentage: Entero, porcentaje de modulación.
#           - freq: Frecuencia de la señal portadora.
# Salida:
#           - mod_signal: arreglos, corresponde a la señal modulada
#           - samples_carrier: muestras empleadas para generar la señal portadora.
def fm_modulation(data, percentage, freq):
    signal_time = len(data[1]) / data[0]
    # Se generan las muestras de la portadora, a una frecuencia de muestreo de 4 veces la frecuencia a la que se modula.
    samples_carrier = np.arange(0, signal_time, 1 / (4 * freq))
    # Se general las muestras del mensaje original, las que luego son empleadas para interpolar el audio original.
    samples_signal = np.linspace(0, signal_time, len(data[1]))
    carrier = np.cos(2 * np.pi * freq * samples_carrier)

    # Interpolación de moduladora para tener el mismo numero de muestras que la portadora
    information = np.interp(samples_carrier, samples_signal, data[1])
    m = percentage/100

    # Integral acumulativa del mensaje, esta se emplea posteriormente para calcular la señal modulada.
    integral_info = cumtrapz(information, samples_carrier, initial=0)
    mod_signal = np.cos(2*np.pi*freq*samples_carrier+m*integral_info)

    # Interpolación de la señal para tener una frecuencia de muestreo soportada por wavfile.write()
    mod_signal_small = np.interp(samples_signal, samples_carrier, mod_signal)
    save_audio("audio_fm_"+str(percentage)+".wav", data[0], mod_signal_small*5)

    # Generación de gráficos (modulada y espectros de frecuencia)
    plot_signal(samples_carrier, mod_signal, "Modulación FM "+str(percentage)+"%", "T[s]"
                , "Amplitud[dB]", [0, 0.0015])
    plot_spectrums(information, mod_signal, carrier, samples_carrier, "TF_FM_" + str(percentage) + "%")
    return mod_signal, samples_carrier


# Función am_demodulation: Se encarga de demodular una señal que ya ha sido modulada en su amplitud para obtener el
# mensaje que esta porta.
# Entradas:
#           - modulated: Arreglo, corresponde a la señal modulada.
#           - samples_carrier: Arreglo, corresponde a las muestras que se emplearon para crear la señal portadora.
#           - freq: Entero, corresponde a la frecuencia de la señal portadora.
#           - percentage: Entero, corresponde al porcentaje de modulación de la señal portadora.
#           - original_freq: Entero, corresponde a la frecuencia de muestreo de la señal original.
# Salida:
#           - demod_signal: Arreglo, corresponde a la señal demodulada.
def am_demodulation(modulated, samples_carrier, freq, percentage, meta_data):
    # Creación de una portadora, que se multiplica con la modulada para obtener la señal original
    carrier = np.cos(2 * np.pi * freq * samples_carrier)
    demod_signal = modulated*carrier

    ft_am_0 = np.abs(fftp.fftshift(fftp.fft(demod_signal)))
    # Se aplica un filtro paso bajo (diseñado en el lab 2) sobre la señal demodulada para obtener la señal original.
    demod_signal = lab2lib.filter_signal([4*freq, demod_signal], freq/2)

    # Se emplea fftshift sobre fftfreq para obtener un eje x en el dominio de las frecuencias.
    xf = fftp.fftshift(fftp.fftfreq(len(modulated), samples_carrier[2]-samples_carrier[1]))
    ft_am_1 = np.abs(fftp.fftshift(fftp.fft(demod_signal[1])))

    signal_time = len(demod_signal[1])/demod_signal[0]
    samples_signal = np.linspace(0, signal_time, meta_data[1])
    # Se interpola la señal demodulada, para tener una frecuencia de muestreo soportada por wavfile.write()
    demod_signal_small = np.interp(samples_signal, samples_carrier, demod_signal[1])
    save_audio("audio_demod_"+str(percentage)+".wav", meta_data[0], demod_signal_small/2000)

    # Se grafica la señal demodulada y su transformada de fourier
    plot_signal(np.linspace(0, len(demod_signal[1])/demod_signal[0], len(demod_signal[1])), demod_signal[1],
                "Señal demodulada "+str(percentage)+"%", "T[s]", "Amplitud[dB]")
    plot_signal(xf, ft_am_0, "TF Demodulación AM (sin filtro)" + str(percentage)+ "%", "Frecuencia[Hz]", "|F(w)|")
    plot_signal(xf, ft_am_1, "TF Demodulación AM (con filtro)" + str(percentage)+ "%", "Frecuencia[Hz]", "|F(w)|")
    return demod_signal


# Función plot_spectrums: Se encarga de graficar los espectros de las señales entregadas
# Entrada:
#           - information: Arreglo, señal que contiene el mensaje que se desea modular.
#           - modulated: Arreglo, señal correspondiente al resultado de modular la señal portadora.
#           - carrier: Arreglo, señal correspondiente a la señal portadora.
#           - samples_carrier: Arreglo, muestras del eje x de la señal portadora.
#           - name: String, nombre con el que se guarda el gráfico.
def plot_spectrums(information, modulated, carrier, samples_carrier, name):
    # Cálculo de las transformadas de la moduladora, portadora y modulada.
    ft_original = np.abs(fftp.fft(information))
    ft_modulated = np.abs(fftp.fft(modulated))
    ft_carrier = np.abs(fftp.fft(carrier))
    samples = len(ft_original)
    # Se cambia un parametro de matplotlib para poder graficar vectores de gran tamaño
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure(figsize=(10.24, 7.20), dpi=100)
    xf = fftp.fftfreq(samples, samples_carrier[2] - samples_carrier[1])
    # Creación de una figura con subplots, correspondientes a cada transformada
    plt.subplot(221)
    plt.title("FT de la moduladora")
    plt.xlabel("|F(w)|")
    plt.ylabel("Frecuencia[Hz]")
    plt.grid()
    plt.plot(xf[0:samples//2], np.abs(ft_original[0:samples//2]))
    plt.subplot(222)
    plt.title("FT de la portadora")
    plt.xlabel("|F(w)|")
    plt.ylabel("Frecuencia[Hz]")
    plt.grid()
    plt.plot(xf[0:samples//2], np.abs(ft_carrier[0:samples//2]), "r")
    plt.subplot(223)
    plt.title("FT de la señal modulada")
    plt.xlabel("|F(w)|")
    plt.ylabel("Frecuencia[Hz]")
    plt.grid()
    plt.plot(xf[0:samples//2], np.abs(ft_modulated[0:samples//2]), "g")
    plt.tight_layout()
    dir_name = "./graphs/"
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(dir_name+name+'.png', bbox_inches='tight', dpi=100)


# Función plot_signal: Se encarga de graficar una señal.
# Entradas:
#           - time: Arreglo, contiene los datos del eje x.
#           - data: Arreglo, contiene los datos del eje y.
#           - name: String, corresponde al título de la figura y al nombre con el que se guarda el gráfico.
#           - x_axis: String, unidad de medida empleada en el eje x.
#           - y_axis: String, unidad de medida empleada en el eje y.
#           - x_limit: Arreglo, indica desde qué punto y hasta qué punto se grafica en el eje x.
def plot_signal(time, data, name, x_axis, y_axis, x_limit=[]):
    # Se cambia un parametro de matplotlib para poder graficar vectores de gran tamaño
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure(figsize=(10.24, 7.20), dpi=100)
    if len(x_limit) == 2:
        plt.xlim(x_limit[0], x_limit[1])
    plt.plot(time, data)
    plt.grid()
    plt.title(name)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()
    dir_name = "./graphs/"
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(dir_name + name + '.png', bbox_inches='tight', dpi=100)


# Función lab4_modulation: Se encarga de aplicar modulaciones AM y FM con un porcentaje de modulación de 15%, 100% y
# 125% sobre un archivo de audio a cierta frecuencia.
# Entradas:
#           - file_name: String, nombre del archivo de audio que será modulado.
#           - frequency: Entero, frecuencia a la que se modula el audio.
def lab4_modulation(file_name, frequency):
    if os.path.isfile(file_name):
        signal = open_audio(file_name)
        print("AM 15% en progreso... ", end="", flush=True)
        am, sc = am_modulation(signal, 15, frequency)
        am_demodulation(am, sc, frequency, 15, [signal[0], len(signal[1])])
        print("OK!", flush=True)
        print("AM 100% en progreso... ", end="", flush=True)
        am, sc = am_modulation(signal, 100, frequency)
        am_demodulation(am, sc, frequency, 100, [signal[0], len(signal[1])])
        print("OK!", flush=True)
        print("AM 125% en progreso... ", end="", flush=True)
        am, sc = am_modulation(signal, 125, frequency)
        am_demodulation(am, sc, frequency, 125, [signal[0], len(signal[1])])
        print("OK!", flush=True)
        print("FM 15% en progreso... ", end="", flush=True)
        fm, sc = fm_modulation(signal, 15, frequency)
        lab2lib.plot_spectrogram([frequency*4, fm], save_fig="./graphs/spec15")
        print("OK!", flush=True)
        print("FM 100% en progreso... ", end="", flush=True)
        fm, sc = fm_modulation(signal, 100, frequency)
        lab2lib.plot_spectrogram([frequency * 4, fm], save_fig="./graphs/spec100")
        print("OK!", flush=True)
        print("FM 125% en progreso... ", end="", flush=True)
        fm, sc = fm_modulation(signal, 125, frequency)
        lab2lib.plot_spectrogram([frequency * 4, fm], save_fig="./graphs/spec125")
        print("OK!", flush=True)
        print("Proceso finalizado!")
    else:
        print("El archivo indicado no existe.\nVerifique si el nombre ingresado es correcto.")
