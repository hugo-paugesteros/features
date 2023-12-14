import numpy as np
from ZCR import ZCR, zcr

# y, sr = librosa.load('../exp1/processed/V1F1D1T1.wav', sr=22050)
t = np.linspace(0,10,1000)
y = np.sin(2*np.pi*11*t)

print(zcr(y))

zcr = ZCR(frame_size=500, hop_size=250)
print(zcr.compute(y))
zcr.visualize()
