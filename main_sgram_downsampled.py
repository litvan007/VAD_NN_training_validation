import os
import random
import inspect
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import clear_output

random.seed(123456)
np.random.seed(123456)
t.manual_seed(123456)

class arr_label_dataset(t.utils.data.Dataset):
    def __init__(self, arrnames, labelnames, extradatas=None):
        assert(len(arrnames) == len(labelnames))
        if extradatas is not None:
            assert(len(arrnames) == len(extradatas))
        self.len = len(arrnames)
        self.arrnames = arrnames.copy()
        self.labelnames = labelnames.copy()
        if extradatas is not None:
            self.extradatas = extradatas.copy()
        else:
            self.extradatas = None
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        arrname = self.arrnames[idx]
        labelname = self.labelnames[idx]
        try:
            extradata = self.extradatas[idx]
        except TypeError:
            extradata = None
        label = np.load(labelname, allow_pickle=False)
        return (arrname, extradata,
                t.unsqueeze(t.from_numpy(np.load(arrname, allow_pickle=False)), 0),
                t.from_numpy(np.expand_dims(label.astype(np.float32), 0)))

def train_one_cycle(model, init_func, epochs, batch_size,
          train_loader, valid_loader, savedir,
          loss_tuple, max_lr=0.01, div_factor=50,
          weight_decay=0.001):
    assert(Path(savedir).is_dir())
    loss_name = loss_tuple[0]
    loss = loss_tuple[1]
    def save(path):
        t.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'mean_valid_loss': mean_valid_loss,
                    'train_loss_epochs': train_loss_epochs,
                    'valid_loss_epochs': valid_loss_epochs,
                }, path)
    model.apply(init_func)
    optimizer = t.optim.AdamW(model.parameters(), lr=max_lr / div_factor,
                             weight_decay=weight_decay, amsgrad=True)
    scheduler = t.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                               steps_per_epoch=len(train_loader),
                                               epochs=epochs, div_factor=div_factor)
    train_loss_epochs = []
    valid_loss_epochs = []
    best_epoch = None
    best_mean_valid_loss = None
    best_path = None
    for epoch in range(epochs):
        clear_output(wait=True)
        losses = []
        model.train()
        for bnum, (wavname,
                   extradata,
                   X, y) in enumerate(train_loader):
            model.zero_grad()
            prediction = model(X)
            loss_batch = loss(prediction, y)
            losses.append(loss_batch.item())
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
        train_loss_epochs.append(np.mean(losses))
        model.zero_grad()
        with t.no_grad():
            losses = []
            model.eval()
            for bnum, (wavname,
                   extradata,
                   X, y) in enumerate(valid_loader):
                prediction = model(X)
                loss_batch = loss(prediction, y)
                losses.append(loss_batch.item())
            mean_valid_loss = np.mean(losses)
            if epoch == 0 or mean_valid_loss < np.min(valid_loss_epochs):
                best_epoch = epoch
                best_mean_valid_loss = mean_valid_loss
                if epoch > 0:
                    prev_best_path = best_path
                best_path = os.path.join(savedir,
                                         datetime.now().strftime('{0}_{1:.3f}_%Y-%m-%d_%H-%M-%S'.format(epoch,
                                                                                                        mean_valid_loss)))
                save(best_path)
                if epoch > 0:
                    Path(prev_best_path).unlink()
            valid_loss_epochs.append(mean_valid_loss)
            print('Epoch {0}, (Train / Valid) ({1}): {2:.3f} / {3:.3f}'.format(epoch,
                                                                             loss_name,
                                                                             train_loss_epochs[-1],
                                                                             valid_loss_epochs[-1]))
            plt.plot(train_loss_epochs)
            plt.scatter(range(len(train_loss_epochs)), train_loss_epochs)
            plt.plot(valid_loss_epochs)
            plt.scatter(range(len(valid_loss_epochs)), valid_loss_epochs)
            plt.show()
    print('best_mean_valid_loss ({}): {:.8f},\nbest_epoch: {},\nbest_path: {}'.format(loss_name,
                                                                                  best_mean_valid_loss,
                                                                                  best_epoch,
                                                                                  best_path))
    return train_loss_epochs, valid_loss_epochs, best_epoch, best_mean_valid_loss, best_path

data_name = 'esc_white_12000_6_downsampled'
data_global_path = os.path.join('./local_datasets', data_name)

datatype = {
    'dsclass': arr_label_dataset,
    'xext': '.wav.downsampled.sgram.npy',
}

snrs = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

datasets = {
    'train': None,
    'valid': None,
}

for postfix in ('train', 'valid'):
    data_local_path = os.path.join(data_global_path, postfix)
    desc = pd.read_csv(os.path.join(data_local_path, 'desc.csv'), sep='\t')
    xnames = []
    labelnames = []
    custom_snrs = []
    for idx, n, c_s in desc[['name', 'custom_snr']].itertuples():
        xnames.append(os.path.join(data_local_path, n + datatype['xext']))
        labelnames.append(os.path.join(data_local_path, n + '.labels.npy.downsampled.npy'))
        custom_snrs.append(c_s)
    datasets[postfix] = datatype['dsclass'](xnames, labelnames, custom_snrs)

batch_size = 1024  # also see learning_rate
train_loader = t.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
valid_loader = t.utils.data.DataLoader(datasets['valid'], batch_size=batch_size, shuffle=True)

class ConvConvT2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(20, 5),
                      stride=(10, 2),
                      padding=(10, 2)),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(13, 3),
                      stride=(1, 2),
                      padding=(0, 1)),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=256),
        )
        self.convT_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256,
                               out_channels=128,
                               kernel_size=101,
                               stride=10,
                               padding=50,
                               output_padding=9),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=128),
            nn.ConvTranspose1d(in_channels=128,
                               out_channels=64,
                               kernel_size=101,
                               stride=10,
                               padding=50,
                               output_padding=9),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=64),
            nn.ConvTranspose1d(in_channels=64,
                               out_channels=1,
                               kernel_size=101,
                               stride=20,
                               padding=50,
                               output_padding=19),
        )
    
    def forward(self, x):
        return self.convT_layers(t.squeeze(self.conv_layers(x), -2))

def init_weights(m):
    if type(m) in (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d):
        nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
    else:
        print('Not setting weights for type {}'.format(type(m)))

model_name = 'sgram'
run_iter = '2'
model = ConvConvT2D()
epochs = 100
loss_tuple = ('BCEWLL(pos_weight=3)',
              nn.BCEWithLogitsLoss(pos_weight=t.ones(60000,
                                                     dtype=t.float32) * 3))
max_lr = 0.05
div_factor = 250
weight_decay = 0.0005
run_name = model_name + '_' + data_name + '_' + run_iter
savedir = os.path.join('./results/', run_name)

Path(savedir).mkdir(exist_ok=False)
train_loss_epochs,valid_loss_epochs,best_epoch,best_mean_valid_loss,best_path = train_one_cycle(model, init_weights,
                  epochs, batch_size,
                  train_loader, valid_loader,
                  savedir, loss_tuple=loss_tuple,
                  max_lr=max_lr, div_factor=div_factor,
                  weight_decay=weight_decay)
