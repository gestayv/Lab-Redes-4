import redes4 as r4


def main():
    audio_name = input('Ingrese el nombre del archivo de audio (ejemplo: audio.wav): \n')
    # Se usan 30 kHz para simular VHF
    r4.lab4_modulation(audio_name, 30000)


main()