#Proyecto final: Cesar Zapata, Nicolas Garnica

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import scipy.io as sciio
import scipy.io.wavfile as wav

# Se carga la colección de archivos de audio Para ello se cargan todos los archivos .wav del directorio definido en la variable str_AudioDir
def load_audio():
    str_AudioDir = './Datos/Musica'
    str_AudioExt = 'wav'
    v_AudioFiles = np.sort(glob.glob(f'{str_AudioDir}/*.{str_AudioExt}'))
    v_AudioSigArray = []
    v_AudioSigFsArray = []

    for s_AudioFile in v_AudioFiles:
        print('[signal2sound] - Loading file: ', s_AudioFile)
        s_FsHzAudio, v_AudioSig = wav.read(s_AudioFile)
        print('[signal2sound] - FsHz: ', s_FsHzAudio)
        v_Dim = v_AudioSig.shape
        if len(v_Dim) > 1:
            v_AudioSigArray.append(v_AudioSig[:, 0])
        else:
            v_AudioSigArray.append(v_AudioSig)
        v_AudioSigFsArray.append(s_FsHzAudio)

    return v_AudioSigArray, v_AudioSigFsArray

def load_signal(fil_name):
    st_File = sciio.loadmat(fil_name)
    v_Sig = np.double(st_File['v_Sig'])
    s_FsHz = np.double(st_File['s_FsHz'])
    v_Sig = v_Sig[0]
    v_Time = np.linspace(0.0, len(v_Sig)/s_FsHz, len(v_Sig))
    return v_Sig, s_FsHz, v_Time


