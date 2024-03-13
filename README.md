# Audio Features
Audio feature extraction in Python

## Installation
1. Clone this repository
```bash
git clone https://github.com/hugo-paugesteros/features.git
```
2. Install dependencies
```bash
pip install -r ./requirements.txt
```
3. Install :
```bash
pip install -e .
```

## Usage
```python
import librosa
from features import Waveform, RMS

y, sr = librosa.load('path/to/file.wav')
waveform = Waveform(y=y, sr=sr)
print(RMS().compute(waveform))
```