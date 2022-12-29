import numpy as np
import torch
import torchaudio.transforms as T
import librosa
import opensmile
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).to(device)

class baseline_dataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, subset, transform=ToTensor()):
        self.transform = transform
        self.rootdir = rootdir
        self.subset = subset

    def __getitem__(self, index):
        xname, y = self.subset[index].values()
        X, _ = librosa.load(f"{self.rootdir}/{xname}.wav", sr=8000)
        if self.transform:
            X = torch.unsqueeze(self.transform(X.astype('float32')), 0)
            y = torch.unsqueeze(self.transform(y.astype('float32')), 0)
        return xname, X, y, -1

    def __len__(self):
        return len(self.subset)

class mfcc_dataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, subset, transform=ToTensor()):
        self.transform = transform
        self.rootdir = rootdir
        self.subset = subset
        self.mfcc_transform = T.MFCC(
            sample_rate=8000,
            n_mfcc=256,
            melkwargs={
                "n_fft": 2048,
                "n_mels": 256,
                "hop_length": 512,
                "mel_scale": "htk",
            },
        )

    def __getitem__(self, index):
        xname, y = self.subset[index].values()
        signal, sr = librosa.load(f"{self.rootdir}/{xname}.wav", sr=8000)
        if self.transform:
            X = torch.unsqueeze(self.mfcc_transform(self.transform(signal)), 0)
            y = torch.unsqueeze(self.transform(y.astype('float32')), 0)
            signal = torch.unsqueeze(self.transform(signal.astype('float32')), 0)

        return xname, X, y, signal

    def __len__(self):
        return len(self.subset)

class eGMAPSv02_dataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, subset, transform=ToTensor()):
        self.transform = transform
        self.rootdir = rootdir
        self.subset = subset
        self.n_subset = len(self.subset)
        self.smile = opensmile.Smile(
                        feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                        )
        # self.data = []
        # self.preprocessing()

    def __getitem__(self, index):
        xname, y = self.subset[index].values()
        if self.transform:
            X = torch.unsqueeze(torch.transpose(self.transform(self.smile.process_file(f"{self.rootdir}/{xname}.wav").to_numpy().astype('float32')).to(device), 0, 1), 0)
            y = torch.unsqueeze(self.transform(y.astype('float32')).to(device), 0)
            # signal = torch.unsqueeze(signal, 0)

        return xname, X, y, -1

    def __len__(self):
        return len(self.subset)

    # def preprocessing(self):
    #     for i in tqdm(np.arange(self.n_subset)):

    #         # signal, sr = librosa.load(f"{self.rootdir}/{xname}.wav", sr=8000)
    #         data_features =
    #         self.data.append(data_features)