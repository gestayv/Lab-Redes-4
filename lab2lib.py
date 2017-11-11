import scipy.signal as sg
import scipy.fftpack as fftp
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def read_wav(name):
    # Función para leer el archivo wav. Wrapper de scipy.io.wavfile.read(name)
    # Retorna una tupla con el sample rate y los datos leidos del wav_data
    # *
    return wavfile.read(name)


def write_wav(wav_data, file_name):
    # Funcion para escribir un archivo de audio .wav en base a los datos de la señal
    # wav_data: objetos con los datos del wav (sample_rate, datos)
    # file_name: String con el nombre del archivo a escribir
    if isinstance(file_name, str):
        if not file_name.endswith(".wav"):
            file_name = file_name + ".wav"
        wavfile.write(file_name, wav_data[0], wav_data[1])


def fft(wav_data):
    # Función que calcula la transformada de fourier. Encapsula scipy.fftpack.fft y scipy.fftpack.fftfreq
    # wav_data: obtejo con los datos del wav (leidos con read_wav)
    # retorna un arreglo con los valores de la transformada y otro con las frecuencias correspondientes
    data = wav_data[1]
    t = fftp.fft(data)
    f = fftp.fftfreq(len(data))
    return t, f


def plot_wav(wav_data, abs_values=False, save_fig=None, title="None"):
    # Función para graficar el archivo de audio wav
    # wav_data: objeto con los datos del wav (leidos con read_wav)
    # abs_values: aplicar valor absoluto a todos los valores
    # save_fig: String con el nombre con el que se desea guardar la imagen con el grafico. De no
    # ingresarse algún valor, los graficos se muestran en pantalla
    sample_rate = wav_data[0]
    if(abs_values):
        data = np.abs(wav_data[1])
    else:
        data = wav_data[1]
    time = np.linspace(0, len(data)/sample_rate, num=len(data))

    plt.ioff()
    plt.figure()
    plt.xlabel("Tiempo [s]")
    plt.plot(time, data)

    if isinstance(title, str):
        plt.title(title)

    if save_fig is None:
        plt.show()
        plt.close()
    elif isinstance(save_fig, str):
        plt.savefig(save_fig)
        plt.close()


def plot_fourier(wav_data, abs_values=True, hertz=True, pos_freq=False, save_fig=None, title=None):
    # Función para graficar la transformada de fourier
    # wav_data: objeto con los datos del wav (leidos con read_wav)
    # abs_values: aplicar valor absoluto a la Magnitud
    # hertz: Graficar en hertz
    # pos_freq: Solo graficar frecuencias positivas
    # save_fig: String con el nombre con el que se desea guardar la imagen con el grafico. De no
    # ingresarse algún valor, los graficos se muestran en pantalla

    # Nota: Tiene problemas al graficar

    # Toma los valores de la transformada de fourier y las frecuencias correspondientes
    t, f = fft(wav_data)

    length = len(t)

    plt.ioff()
    plt.figure()
    if hertz:
        f = f * wav_data[0]
        plt.xlabel("Frecuencia [Hz] ")
    if pos_freq:
        t = t[0:int(length/2)]
        f = f[0:int(length/2)]

    if abs_values:
        t = np.abs(t)
        plt.ylabel("Amplitud [db]")
    else:
        plt.ylabel("F(w)")

    if isinstance(title, str):
        plt.title(title)

    plt.plot(f, t.real)

    if save_fig is None:
        plt.show()
        plt.close()
    elif isinstance(save_fig, str):
        plt.savefig(save_fig)
        plt.close()


def spectrogram(wav_data, scaling="spectrum"):
    # Funcion para generar los datos del espectrograma
    # Retorna una tupla con el arreglo de las frecuencis de muestreo, arreglo
    # de los segmentos de tiempo y los datos del espectrograma

    return sg.spectrogram(wav_data[1], fs=wav_data[0], nperseg=1024, scaling=scaling)


def plot_spectrogram(wav_data, cmap="gist_stern", save_fig=None, title=None):
    # Función para graficar el espectrograma
    # wav_data: datos del wav (sample_rate, datos)
    # cmap: String con el color-map a utilizar

    f, t, Sxx = spectrogram(wav_data)
    # Sxx = Sxx/np.max(Sxx)
    Sxx = 10 * np.log10(Sxx)

    plt.ioff()
    plt.figure()

    plt.pcolormesh(t, f, Sxx, cmap=plt.get_cmap(cmap))
    plt.colorbar(label="Amplitud [db]")

    if isinstance(title, str):
        plt.title(title)

    plt.ylabel('Frequencia [Hz]')
    plt.xlabel('Tiempo [sec]')

    if save_fig is None:
        plt.show()
        plt.close()
    elif isinstance(save_fig, str):
        plt.savefig(save_fig)
        plt.close()


def filter_data(wav_data, freq, filter_type="butter", btype="low", order=3, cheb_rp=1):
    # Función para generar los arreglos de numeradores y denominadores para filtrar
    # Retorna una tupla de arreglos (numerador, denominador)
    # wav_data: objeto con los datos del wav (leidos con read_wav)
    # freq: frequencia de corte en hertz. Dato escalar para low o high pass, lista de dos
    # datos para band-pass
    # filter_type: String con el filtro a utilizarse. "butter" para Butterworth.
    # btype: forma de aplicar el filtro ("low", "high" o "band")
    # order: Orden del filtro si lo requiere.
    # cheb_rp: factor de rizado (ripple factor) usado por el filtro chebyshev si lo requiere.

    sample_rate = wav_data[0]
    nyq = sample_rate * 0.5

    if filter_type == "butter" or filter_type.lower() == "butterworth":
        res = sg.butter(order, Wn=np.divide(freq, nyq), btype=btype, output="ba")
        b, a = res[0], res[1]
    elif filter_type == "cheb" or filter_type.lower() == "chebyshev":
        res = sg.cheby1(order, cheb_rp, Wn=np.divide(freq, nyq), btype=btype, output="ba")
        b, a = res[0], res[1]
    elif filter_type.lower() == "bessel":
        res = sg.bessel(order, Wn=np.divide(freq, nyq), btype=btype, output="ba")
        b, a = res[0], res[1]
    else:
        b, a = None, None

    return b, a


def filter_signal(wav_data, freq, filter_type="butter", btype="low", order=3, cheb_rp=1):
    # Función para aplicar un filtro a una señal directamente.
    # Retorna una tupla (frecuencia de muestreo, datos de la señal)
    # wav_data: objeto con los datos del wav (leidos con read_wav)
    # freq: frequencia de corte en hertz. Dato escalar para low o high pass, lista de dos
    # datos para band-pass
    # filter_type: String con el filtro a utilizarse. "butter" para Butterworth.
    # btype: forma de aplicar el filtro ("low", "high" o "band")
    # order: Orden del filtro si lo requiere.
    # cheb_rp: factor de rizado (ripple factor) usado por el filtro chebyshev si lo requiere.

    if isinstance(freq, list):
        if len(freq) == 2:
            # Se ajusta la elección a Band-pass
            btype = "band"
        else:
            print("Se debe ingresar una lista de dos frecuencias para usar band-pass")
            return None
    else:
        if btype == "bandpass" or btype == "band":
            print("Se debe ingresar una lista de dos frecuencias para usar band-pass")
            return None

    b, a = filter_data(wav_data, freq, filter_type, btype, order, cheb_rp)

    # Se aplica el filtro
    new_data = sg.lfilter(b, a, wav_data[1])

    return wav_data[0], new_data


def plot_filter(wav_data, freq, filter_type="butter", btype="low", order=3, cheb_rp=1):
    # Función para graficar el filtro
    # wav_data: objeto con los datos del wav (leidos con read_wav)
    # freq: frequencia de corte en hertz. Dato escalar para low o high pass, lista de dos
    # datos para band-pass
    # filter_type: String con el filtro a utilizarse. "butter" para Butterworth.
    # btype: forma de aplicar el filtro ("low", "high" o "band")
    # order: Orden del filtro si lo requiere.
    # cheb_rp: factor de rizado (ripple factor) usado por el filtro chebyshev si lo requiere.

    if isinstance(freq, list):
        if len(freq) == 2:
            # Se ajusta la elección a Band-pass
            btype = "band"
        else:
            print("Se debe ingresar una lista de dos frecuencias para usar band-pass")
            return None
    else:
        if btype == "bandpass" or btype == "band":
            print("Se debe ingresar una lista de dos frecuencias para usar band-pass")
            return None

    b, a = filter_data(wav_data, freq, filter_type, btype, order, cheb_rp)

    #ESTO ES PARA GRAFICAR LOS FILTROS.

    # nyq = np.multiply(wav_data[0], 0.5)
    w, h = sg.freqz(b, a, worN=8000)
    f = wav_data[0] * 0.5 * w / np.pi
    h = np.abs(h)

    plt.plot(f, h)
    plt.show()

