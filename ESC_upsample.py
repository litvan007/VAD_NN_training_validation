import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
import os

def int16_to_float32(signal):
    return (signal / 32768).astype(np.float32)

df = pd.read_csv('./ESC-50-master/meta/esc50.csv')
flist = df['filename'].to_list()
flist = map(lambda x: os.path.join('./ESC-50-master/audio/', x), flist)
target_rate = 48000
outpath = './ESC_{}'.format(target_rate)

for fname in flist:
    source_rate, signal = wavfile.read(fname)
    resampled = int16_to_float32(resample_poly(signal, target_rate, source_rate))
    wavfile.write(os.path.join(outpath,
                               fname.split('/')[-1]), target_rate, resampled)
