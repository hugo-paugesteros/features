import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

y, sr = librosa.load('../exp1/processed/V1F1D1T1.wav', sr=22050)

'''
Timbre toolbox :
    - https://github.com/VincentPerreault0/timbretoolbox/blob/66b4c51fea07c987281748f436e6f8b20717113a/classes/Representations/TEE.m
'''
def envelope(y):
    y_a         = scipy.signal.hilbert(y)
    envelope    = np.abs(y_a)

    f_c         = 10
    w           = 2*f_c / sr
    b, a        = scipy.signal.butter(3, w, 'lowpass')
    filtered    = scipy.signal.filtfilt(b, a, envelope)
    filtered    /= np.max(filtered) 

    return filtered

# def envelope(y):
#     RMS = librosa.feature.rms(y=y, frame_length=1024, hop_length=1)[0]
#     print(RMS.shape)
#     f_c         = 5
#     w           = 2*f_c / sr
#     b, a        = scipy.signal.butter(3, w, 'lowpass')
#     filtered    = scipy.signal.filtfilt(b, a, RMS)
#     filtered    /= np.max(filtered)
#     return filtered

plt.plot(y/np.max(y))
plt.plot(envelope(y))
plt.show()
