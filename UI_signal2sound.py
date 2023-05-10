#Procesamiento de Señales y Biointrumentación
#Proyecto final: Cesar Zapata, Nicolas Garnica

from tkinter import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import numpy as np
import scipy.interpolate as interpol
import glob
from LoadFunctions import load_signal, load_audio
import soundfile as sf
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy import signal
from tkinter import messagebox
from tqdm import tqdm


# mpl.rcParams['text.color'] = 'blue'
# mpl.rcParams['axes.labelcolor'] = 'blue'
# mpl.rcParams['xtick.color'] = 'blue'
# mpl.rcParams['ytick.color'] = 'blue'
# mpl.rc('axes',edgecolor='blue')

#Cración de la ventana, grid y frame para la interfaz gráfica
window = Tk()
window.geometry('1130x570')
window.wm_title('Proyecto')
window.minsize(width=1130,height=570)
window.maxsize(width=1130,height=570)
frame = Frame(window, bg='blue')
frame.grid(column=0,row=0, sticky='nsew',padx=(0,30), rowspan=14)

#Gráficas por defecto si no hay archivos cargados
x1 = []
t1 = []

x2 = []
t2 = []

x3 = []
t3 = []

x4 = []
t4 = []
global fig1
global ax1
fig1, ax1=plt.subplots(4,1, sharex=True, dpi=120,facecolor='#90ABC6')

#Funciion que abre archivos y los grafica

def open_file():
    window.filename = filedialog.askopenfilename(initialdir=glob.glob('./Datos/Señales'),
                                                 filetypes=(('Archivos .mat','*.mat'),('Archivos .np','*.np')))

    fig1.suptitle('  Calculando transformada TF de la señal...', fontsize=10,fontweight='bold')
    canvas = FigureCanvasTkAgg(fig1, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=0)
    fig1.tight_layout()
    messagebox.showinfo('Alerta','Seleccione aceptar para calcular la transformada TF de la señal. Este proceso demorará unos minutos...')

    global audio
    global signal_an
    audio = load_audio()
    print(window.filename)
    signal_an=load_signal(window.filename)
    v_Sig = signal_an[0]
    v_Time = signal_an[2]
    ax1[0].plot(v_Time,v_Sig)
    ax1[0].set_title('Señal fisiológica',fontsize=10)
    ax1[0].set_ylabel("Freq (Hz)", fontsize=10)
    ax1[0].grid()
    res_conv = T_tiempo_frec(v_Sig,v_Time,'signal')
    ConvMatPlot = res_conv[0]
    immat = ax1[1].imshow(ConvMatPlot, cmap='hot', interpolation='none', origin='lower', aspect='auto', extent=[v_Time[0], v_Time[-1],v_Sig[0], v_Sig[-1]])
    ax1[1].set_ylabel("Freq (Hz)", fontsize=10)
    ax1[1].set_xlim([v_Time[0], v_Time[len(v_Time)-1]])
    # ax1[0].colorbar(immat, ax=ax1[1])
    fig1.suptitle('              Señal Calculada', fontsize=10,fontweight='bold')
    canvas = FigureCanvasTkAgg(fig1, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=0)
    fig1.tight_layout()
    messagebox.showinfo('Alerta','Insertar los valores de amplitud y pista para generar el audio (entre 0 y 4)')

  
#Boton de abrir archivo
btn_OpnFile = Button(window, text='Open File', command=open_file)
btn_OpnFile.grid(row=0,column=1,columnspan=3)

#Labels de frecuencia, amplitud y pista, colocados en una grid
lbl_frec = Label(window, text='Frecuencia').grid(row=1,column=1)
lbl_Amp = Label(window, text='Amplitud').grid(row=1,column=2,padx=30)
lbl_Pista = Label(window, text='Pista').grid(row=1,column=3)

#Lables de frecuencias.
Label(window, text='4-8 Hz').grid(row=2,column=1)
Label(window, text='8-12 Hz').grid(row=3,column=1)
Label(window, text='12-18 Hz').grid(row=4,column=1)
Label(window, text='18-30 Hz').grid(row=5,column=1)
Label(window, text='30-50 Hz').grid(row=6,column=1)

#Cajas de entrada de texto para amplitud
txt_Amplitud1 = Entry(window,width=10)
txt_Amplitud1.grid(row=2,column=2,padx=10)
txt_Amplitud2 = Entry(window,width=10)
txt_Amplitud2.grid(row=3,column=2,padx=10)
txt_Amplitud3 = Entry(window,width=10)
txt_Amplitud3.grid(row=4,column=2,padx=10)
txt_Amplitud4 = Entry(window,width=10)
txt_Amplitud4.grid(row=5,column=2,padx=10)
txt_Amplitud5 = Entry(window,width=10)
txt_Amplitud5.grid(row=6,column=2,padx=10)

#Cajas de entrada de texto de pista
txt_Pista1 = Entry(window,width=10)
txt_Pista1.grid(row=2,column=3, padx=10)
txt_Pista2 = Entry(window,width=10)
txt_Pista2.grid(row=3,column=3, padx=10)
txt_Pista3 = Entry(window,width=10)
txt_Pista3.grid(row=4,column=3, padx=10)
txt_Pista4 = Entry(window,width=10)
txt_Pista4.grid(row=5,column=3, padx=10)
txt_Pista5 = Entry(window,width=10)
txt_Pista5.grid(row=6,column=3, padx=10)

#Función para generar el audio llamando la funciones del script signal2sound.py
def get_audio():

    fig1.suptitle('       Calculando transformada TF de la señal de audio...', fontsize=10,fontweight='bold')
    canvas = FigureCanvasTkAgg(fig1, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=0)
    fig1.tight_layout()
    messagebox.showinfo('Alerta','Seleccione aceptar para calcular la transformada TF de la señal de audio. Este proceso demorará unos minutos...')

    v_AmpBand = np.array([float(txt_Amplitud1.get()[0]),
                          float(txt_Amplitud2.get()[0]),
                          float(txt_Amplitud3.get()[0]),
                          float(txt_Amplitud4.get()[0]),
                          float(txt_Amplitud5.get()[0])])
  
    v_AudioByFreq = np.array([int(txt_Pista1.get()[0]),
                              int(txt_Pista2.get()[0]),
                              int(txt_Pista3.get()[0]),
                              int(txt_Pista4.get()[0]),
                              int(txt_Pista5.get()[0])])
    
    global new_audio_file
    new_audio_file = audio_gen(signal_an[0],signal_an[1],audio[0],audio[1],v_AmpBand,v_AudioByFreq)
    v_Time_Audio = new_audio_file[2]
    v_Audio = new_audio_file[1]
    ax1[2].plot(v_Time_Audio,v_Audio)
    ax1[2].set_title('Audio generado',fontsize=10)
    ax1[2].set_ylabel("Freq (Hz)", fontsize=10)
    ax1[2].grid()
    fig1.suptitle('           Resultados', fontsize=10,fontweight='bold')
    res_conv = T_tiempo_frec(v_Audio,v_Time_Audio,'audio')
    ConvMatPlot = res_conv[0]
    immat = ax1[3].imshow(ConvMatPlot, cmap='hot', interpolation='none', origin='lower', aspect='auto', extent=[v_Time_Audio[0], v_Time_Audio[-1],res_conv[1][0], res_conv[1][-1]])
    ax1[3].set_xlabel("Time (sec)", fontsize=10)
    ax1[3].set_ylabel("Freq (Hz)", fontsize=10)
    ax1[3].set_xlim([v_Time_Audio[0], v_Time_Audio[len(v_Time_Audio)-1]])
    canvas = FigureCanvasTkAgg(fig1, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=0)
    fig1.tight_layout()
        


#Función que reproduce el audio creado
def play_audio():
    filename = new_audio_file[0]
    data, fs = sf.read(filename, dtype='float32')
    sd.play(data,fs)


#Funcion de transformada tiempo-frec.
def T_tiempo_frec(v_Sig_,v_Time_,que_es):
    fact = 0
    v_Sig_new = v_Sig_
    v_Time_new = v_Time_

    #Disminucion de la resolucion del audio
    if que_es == 'audio':
        v_Sig_new = []
        v_Time_new = []
        fact = int(np.round(len(v_Sig_)/100000,0))
        ant = 0
        fin = fact
        for i in range(0,len(v_Time_)):
            fin = ant + fact
            if fin <= (len(v_Sig_)-1):
                v_Sig_new.append(np.mean(v_Sig_[ant:fin]))
                v_Time_new.append(np.mean(v_Time_[ant:fin]))
                ant = fin
            else:
                break

        v_Sig_new = np.array(v_Sig_new)
        v_Time_new = np.array(v_Time_new)

    #Patrones de frecuencias entre 1 y 50 Hz
    FreqTestHz = np.arange(0,40+0.6,0.6)
    ConvMat = np.zeros([np.size(FreqTestHz),np.size(v_Sig_new)],dtype=complex)
    NumCycles = 5

    TimeArrayGauss = v_Time_new-(v_Time_new[-1]/2.0)
    for FreqIter in  tqdm(range(np.size(FreqTestHz)),'Calculando TTF: '):
        xtest = np.exp(1j*2.0*np.pi*FreqTestHz[FreqIter]*v_Time_new)
        xtestwinstd = ((1.0/FreqTestHz[FreqIter])*NumCycles)/2.0
        xtestwin = np.exp(-0.5*(TimeArrayGauss/xtestwinstd)**2.0)
        xtest = xtest*xtestwin
        ConvMat[FreqIter,:]=np.convolve(v_Sig_new,np.real(xtest),'same')+\
                            1j*np.convolve(v_Sig_new,np.imag(xtest),'same')
    
    ConvMatPlot = np.abs(ConvMat)
    ConvMatPlot_new = np.zeros((ConvMatPlot.shape[0],ConvMatPlot.shape[1]))
    suma_Conv_filas = np.zeros(ConvMatPlot.shape[0])

    for i in tqdm(range(0,ConvMatPlot.shape[0])):
        suma_Conv_filas[i] = np.sum(ConvMatPlot[i,:])
    for i  in range(0,ConvMatPlot.shape[0]):
        for j in range(0,ConvMatPlot.shape[1]):
            ConvMatPlot_new[i][j] = ConvMatPlot[i][j]/suma_Conv_filas[i]
    
    return ConvMatPlot_new, FreqTestHz


#Funcion generadora de audio
global rep
rep = False
def audio_gen(v_Sig,s_FsHz,v_AudioSigArray,v_AudioSigFsArray,v_amp,v_freq):
    rep = True
    # Se ajusta cada pista de audio a la duración de la señal fisiológica y se normaliza la amplitud
    s_FsHzAudioRef = 44100
    s_TimeDurSig = len(v_Sig) / s_FsHz
    s_Len = 0
    for s_Count in range(len(v_AudioSigArray)):
        if v_AudioSigFsArray[s_Count] != s_FsHzAudioRef:
            v_Time = np.arange(0, len(v_AudioSigArray[s_Count])) / v_AudioSigFsArray[s_Count]
            v_TimeAudioAux = np.arange(0, v_Time[-1] + 1 / s_FsHzAudioRef, 1 / s_FsHzAudioRef)
            fun_F1 = interpol.CubicSpline(v_Time, v_AudioSigArray[s_Count], bc_type='clamped')
            v_SigAux = fun_F1(v_TimeAudioAux)
        else:
            v_SigAux = np.array(v_AudioSigArray[s_Count])

        s_Dur = len(v_SigAux) / s_FsHzAudioRef
        if s_Dur < s_TimeDurSig:
            s_Num = np.ceil(s_TimeDurSig / s_Dur)
            for s_Count1 in range(int(s_Num) - 1):
                v_SigAux = np.append(v_SigAux, np.array(v_AudioSigArray[s_Count]))

        if s_Len == 0:
            s_Len = int(s_TimeDurSig * s_FsHzAudioRef)
        v_SigAux = v_SigAux[0:s_Len]

        v_AudioSigArray[s_Count] = v_SigAux / np.max(np.abs(v_SigAux))


    # Funcion de filtrado por FFT en rango especifico
    def filt_FFT(x, FsHz, freqrange):
        even = 0 # booleano para indicar paridad
        size = np.size(x)

        if (size % 2) == 0: # si la senal tiene un numero de datos par
            x = x[0:- 1]
            even = 1

        v_half = int((size - 1) / 2) # se toma la mitad de los datos 
        v_Freq = np.arange(0, size) * FsHz / size
        v_Freq = v_Freq[0:v_half+1]

        v_FFT = np.fft.fft(x) # sacando la transformada de Fourier
        v_FFT = v_FFT[0:v_half+1] # se toma solo la mitad de los datos, pues se refleja la misma senal en la otra mitad

        v_Ind = np.zeros(v_half + 1)
        v_Ind = v_Ind > 0.0

        v_Ind1 = v_Freq >= freqrange[0] 
        v_Ind2 = v_Freq <= freqrange[1]
        v_Ind = v_Ind + (v_Ind1 & v_Ind2)

        v_FFT[~v_Ind] = (10.0 ** -10.0) * np.exp(1j * np.angle(v_FFT[~v_Ind]))
        v_FFT = np.concatenate((v_FFT,np.flip(np.conjugate(v_FFT[1:]))))
        y = np.real(np.fft.ifft(v_FFT))

        if even:
            y = np.concatenate((y, [y[-1]])) # 

        return y

    # Funcion para obtener la envolvente de una senal
    def envolvente(x):
        H = signal.hilbert(x)
        y = np.abs(H)

        return y

    # Se definen las bandas de frecuencia de la señal fisiológica (EEG) para su representación musical
    v_FreqBandHz = np.array([[4, 8], [8, 12], [12, 18], [18, 30], [30, 50]])
    # Se define la amplitud que aportará cada banda
    # v_AmpBand = np.array([1, 0.5, 1, 0.2, 0.3])
    v_AmpBand = v_amp
    # Se define la correspondencia entre la pista de audio y la banda de frecuencia
    v_AudioByFreq = v_freq
    # v_AudioByFreq = np.array([4, 1, 2, 0, 3])


    v_TimeAudioSamples = np.arange(0, s_Len) / s_FsHzAudioRef
    v_TimeSig = np.arange(0, len(v_Sig)) / s_FsHz
    v_SigAudioNew = []
    v_SigFiltAll = []
    # Se le resta la media a la señal fisiológica y se normaliza la amplitud
    v_SigNorm = (v_Sig - np.mean(v_Sig))
    v_SigNorm /= np.max(np.abs(v_SigNorm))
    for s_Count in range(len(v_FreqBandHz)):
        # Se filtra la señal fisiológica en cada una de las bandas de frecuencia y se multiplica por la amplitud correspondiente
        # TODO:
        #  Incluir en esta parte el llamado a la función de filtrado en la banda v_FreqBandHz[s_Count] y asignar el resultado a la variable v_SigFilt
        v_SigFilt = filt_FFT(v_Sig, s_FsHz, v_FreqBandHz[s_Count])

        # Se multiplica la señal filtrada por la amplitud correspondiente
        v_SigFilt *= v_AmpBand[s_Count]
        # TODO: Incluir en esta parte la obtención de la envolvente de la señal
        #  filtrada a través del valor absoluto de la señal analítica y asignarlo a la varible v_SigFiltAbs
        v_SigFiltAbs = envolvente(v_SigFilt)

        # Se interpola la envolvente para que quede a la misma frecuencia de muestreo de las pistas de audio
        fun_F1 = interpol.CubicSpline(v_TimeSig, v_SigFiltAbs, bc_type='clamped')
        v_SigAux = fun_F1(v_TimeAudioSamples)

        # Se multiplica la envolvente con la señal de la pista de audio correspondiente y se obtiene la suma acumulada para todas las pistas
        # de audio y bandas de frecuencia
        if len(v_SigAudioNew) == 0:
            v_SigFiltAll = v_SigAux
            v_SigAudioNew = v_AudioSigArray[v_AudioByFreq[s_Count]] * v_SigAux  
        else:
            v_SigFiltAll += v_SigAux
            v_SigAudioNew += v_AudioSigArray[v_AudioByFreq[s_Count]] * v_SigAux
            
    np.float16(v_SigAudioNew)
    np.float16(s_FsHzAudioRef)
    # Se genera un archivo de audio .wav con la señal resultante
    str_FileNameOut = 'Audio_generado/Audio_generated_.wav'
    wav.write(str_FileNameOut, s_FsHzAudioRef, v_SigAudioNew)
    print('[signal2sound] - Wav file generated: ', str_FileNameOut)
    return str_FileNameOut, v_SigAudioNew, v_TimeAudioSamples
    


#Botones de genrar audio y reproducción
btn_GenerarAudio = Button(window, text='Generar audio', command=get_audio)
btn_GenerarAudio.grid(row=7,column=1,columnspan=3)
btn_RepAudio = Button(window, text='Reproducir audio', command=play_audio)
btn_RepAudio.grid(row=8,column=1,columnspan=3)

lbl_cesar = Label(window,text='Proyecto final').grid(row=9,column=1,columnspan=3)
lbl_cesar = Label(window,text='César Zapata ()').grid(row=10,column=1,columnspan=3)
lbl_cesar = Label(window,text='Nicolás Garnica (201713127)').grid(row=11,column=1,columnspan=3)

canvas = FigureCanvasTkAgg(fig1, master=frame)
canvas.draw()
canvas.get_tk_widget().grid(column=0, row=0)

window.mainloop()