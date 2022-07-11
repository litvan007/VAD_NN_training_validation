import numpy as np
import pandas as pd
from scipy.io import wavfile
import random
import os
from pathlib import Path
import glob
import scipy.signal as si

np.seterr(divide='raise')

def mean_energy(signal):
    return (signal ** 2).mean()


def get_whitenoise(length):
    return np.random.normal(loc=0.0, scale=1.0, size=length)


def gen_indata(i, df, get_noise_f, length, max_words, custom_snr):
    noise, noise_type_name = get_noise_f(length, i)
    noise = noise.astype(np.float32)
    assert(len(noise) == length)
    noise_mean_energy = mean_energy(noise)
    labels = np.zeros(shape=length, dtype=np.int8)
    word_count = random.randint(1, max_words)
    length_sum = length + 1
    while length_sum >= length:
        idx = np.random.choice(len(df), size=word_count, replace=False)
        words = df.iloc[idx]
        word_lengths = words['length'].values
        length_sum = word_lengths.sum()
    free_samples = length - length_sum
    pauses = word_count + 1
    pause_length_dist = np.random.uniform(low=0.0, high=1.0, size=pauses).astype(np.float32)
    pause_length_dist = pause_length_dist / pause_length_dist.sum()
    pause_lengths = (free_samples.astype(np.float32) * pause_length_dist).astype(np.int)
    pause_lengths[-1] += free_samples - pause_lengths.sum()
    word_starts = pause_lengths[:-1] + np.concatenate([[0], word_lengths[:-1]])
    word_starts = word_starts.cumsum()
    word_data = words['fpath'].map(lambda x: wavfile.read(x)[1])
    word_data = word_data.values
    assert(type(word_data) == np.ndarray)
    assert(len(word_data) == word_count)
    word_ends = word_starts + word_lengths
    snrs = []
    custom_snrs = []
    for i, word in enumerate(word_data):
        assert(type(word) == np.ndarray)
        assert(word.dtype == np.int16)
        word = word.astype(np.float32)
        word = word * np.sqrt(custom_snr) * np.sqrt(noise_mean_energy / mean_energy(word))
        local_noise_mean_energy = mean_energy(noise[word_starts[i]:word_ends[i]])
        current_word_mean_energy = mean_energy(word)
        try:
            tmp_snr = current_word_mean_energy / local_noise_mean_energy
        except FloatingPointError:
            tmp_snr = -1
        snrs.append(tmp_snr)
        custom_snrs.append(current_word_mean_energy / noise_mean_energy)
        noise[word_starts[i]:word_ends[i]] += word
        labels[word_starts[i]:word_ends[i]] = 1
    noise = noise / np.abs(noise).max()
    return noise, labels, word_count, snrs, custom_snrs, noise_type_name

ESC_flist = glob.glob('./ESC_48000/*')
def get_ESC_noise(length):
    idx = np.random.choice(len(ESC_flist), replace=False)
    sr, signal = wavfile.read(ESC_flist[idx])
    assert(len(signal) == length)
    return signal

def get_rand_noise(length, i):
    if i % 6 == 0:
        return (get_whitenoise(length), 'white_noise')
    else:
        return (get_ESC_noise(length), 'ESC_noise')

wav_path_list = glob.glob('./all_single_cut/*.wav')
get_noise_f = get_rand_noise
indata_time_length = 5
max_words = 6
amnt_per_snr = 12000
custom_snr_set = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
threshold = 1e-10
global_path = os.path.join('./usb_datasets', 'esc_white_{}_{}'.format(amnt_per_snr,
                                                                      len(custom_snr_set)))
Path(global_path).mkdir(exist_ok=False)
train_ratio = 0.7
valid_ratio = 0.1

train_amnt = int(train_ratio * amnt_per_snr)
valid_amnt = int(valid_ratio * amnt_per_snr)
test_amnt = amnt_per_snr - valid_amnt - train_amnt

total_df = pd.DataFrame(wav_path_list)
total_df.rename(columns={0: 'fpath'}, inplace=True)
total_df['rate'] = total_df['fpath'].map(lambda x: wavfile.read(x)[0])
rate = total_df['rate'].min()
assert(rate == total_df['rate'].max())
total_df['length'] = total_df['fpath'].map(lambda x: len(wavfile.read(x)[1]))
total_df['time_length'] = total_df['length'] / total_df['rate']

indata_length = indata_time_length * rate

rand_idx_permut = np.random.choice(len(total_df), size=len(total_df), replace=False)
train_end = int(len(rand_idx_permut) * train_ratio)
valid_end = int(len(rand_idx_permut) * (train_ratio + valid_ratio))
train_idx = rand_idx_permut[:train_end]
valid_idx = rand_idx_permut[train_end:valid_end]
test_idx = rand_idx_permut[valid_end:]
assert(len(train_idx) + len(valid_idx) + len(test_idx)) == len(total_df)
assert(np.all(np.sort(np.concatenate((train_idx, valid_idx, test_idx))) == np.arange(len(total_df))))
train_df = total_df.iloc[train_idx]
valid_df = total_df.iloc[valid_idx]
test_df = total_df.iloc[test_idx]

parts = {
    'train': (train_amnt, train_df),
    'valid': (valid_amnt, valid_df),
    'test': (test_amnt, test_df),
}

for (postfix, (amnt, df)) in parts.items():
    local_path = os.path.join(global_path, postfix)
    Path(local_path).mkdir(exist_ok=False)
    desc = []
    for custom_snr in custom_snr_set:
        for i in range(amnt):
            data, labels, word_count, snrs, custom_snrs, noise_type_name = gen_indata(i, df, get_noise_f, indata_length, max_words, custom_snr)
            name = '{}_{}'.format(custom_snr, i)
            desc.append((name, custom_snr, word_count, snrs, custom_snrs, noise_type_name))
            wavfile.write(os.path.join(local_path, name + '.wav'), rate, data)
            sgram = si.spectrogram(data, rate, window='hamming', nperseg=2048,
                                     noverlap=1024, detrend=False,
                                      scaling='spectrum')[2]
            sgram[sgram < threshold] = threshold
            sgram = np.log(sgram)
            np.save(os.path.join(local_path, name + '.sgram'), sgram, allow_pickle=False)
            np.save(os.path.join(local_path, name + '.labels'), labels, allow_pickle=False)
    desc_df = pd.DataFrame(desc)
    desc_df.rename(columns={0: 'name', 1: 'custom_snr', 2: 'word_count', 3: 'snrs', 4: 'custom_snrs', 5: 'noise_type'}, inplace=True)
    desc_df.to_csv(os.path.join(local_path, 'desc.csv'), sep='\t', index=False)
