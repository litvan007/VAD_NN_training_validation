import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import logging

import pickle
import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from NN_stracture import SoundDataset
from NN_stracture import ToTensor
from NN_stracture import ConvConvT1D
from NN_stracture import ConvConvT2D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sound_show(inp, title=None):
    inp = inp.numpy()
    time = np.linspace(0., inp.size/8000, inp.size)
    plt.figure(figsize=(16, 8))
    plt.plot(time, inp)
    plt.title(title)
    plt.show()

def init_weights(m):
    if type(m) in (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d):
        nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
    else:
        print('Not setting weights for type {}'.format(type(m)))

def train(model, optim, loss_tuple, scheduler, dataloader, epoch, device, init_weights):
    total = 0
    correct = 0
    train_loss = 0
    losses = []
    loss_name = loss_tuple[0]
    loss = loss_tuple[1]
    model.apply(init_weights)

    model.train()
    print('Here0')
    for names_signals, signals, marks in dataloader:
        signals, marks = signals.to(device), marks.to(device)
        print(signals.shape, marks.shape)
        optim.zero_grad()
        prediction = model(signals)
        print(prediction.shape)
        loss_batch = loss(prediction, marks)
        losses.append(loss_batch.item())
        loss_batch.backward()
        optim.step()
        scheduler.step()
        print('Here2')
        train_loss += loss_batch.item()
        total += prediction.shape[0]
        correct += prediction.eq(marks).sum().item()
    return train_loss / total, 100. * correct / total

def test(model, loss_tuple, dataloader, epoch, device, best_acc, model_name='model'):
    total = 0
    correct = 0
    valid_loss = 0

    loss_name = loss_tuple[0]
    loss = loss_tuple[1]

    model.eval()
    with torch.no_grad():
        for names_signals, signals, marks in dataloader:
            signals, marks = signals.to(device), marks.to(device)
            prediction = model(signals)
            loss_batch = loss(prediction, marks)

            valid_loss += loss_batch.item()
            # pred = output.argmax(1)
            correct += torch.eq(prediction, marks).sum().item()
            total += signals.shape[0]

    acc = 100. * correct / total
    if acc > best_acc:
        # print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{}.pth'.format(model_name))
    return valid_loss / total, acc


if __name__ == '__main__':
    path_data = '/Users/litvan007/NN_sound_data_base/data_2'
    path_data_list = '/Users/litvan007/NN_sound_data_base/data_base_2.pickle'

    t = time.time()
    with open(path_data_list, 'rb') as fh:
        data_list = pickle.load(fh)
    logger.info(f"time of data init: {time.time() - t}")

    end_idx_tv = 87000
    train_valid_subset = data_list[:end_idx_tv]
    test_subset = data_list[end_idx_tv:]

    lengths = [int(len(train_valid_subset) * 0.9), int(len(train_valid_subset) * 0.1)]
    train_subset, valid_subset = torch.utils.data.random_split(train_valid_subset, lengths)

    transform_baseline = transforms.Compose([ToTensor()])
    data_set = {
                'train': SoundDataset(rootdir=path_data, subset=train_subset, transform=transform_baseline),
                'valid': SoundDataset(rootdir=path_data, subset=valid_subset, transform=transform_baseline),
                'test': SoundDataset(rootdir=path_data, subset=test_subset[:10], transform=transform_baseline)
    }

    batch_size = 10
    loaders = {
                'train': torch.utils.data.DataLoader(data_set['train'], batch_size=batch_size, shuffle=True),
                'valid': torch.utils.data.DataLoader(data_set['valid'], batch_size=batch_size, shuffle=True),
                'test': torch.utils.data.DataLoader(data_set['test'], batch_size=batch_size, shuffle=False)
    }

    dataset_sizes = {x: len(data_set[x]) for x in ['train', 'valid', 'test']}
    xname, X, y = data_set['train'][0]
    print(X.shape, y.shape)
    print(xname)
    # sound_show(X)

# Training
    model_name = 'baseline'
    max_lr = 0.05
    div_factor = 250
    weight_decay = 0.0005

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ConvConvT1D().to(device)

    epochs = 100
    loss_tuple = ('BCEWLL(pos_weight=3)',
                  nn.BCEWithLogitsLoss(pos_weight=torch.ones(48000,
                                                         dtype=torch.float16) * 3))
    optim = torch.optim.AdamW(model.parameters(), lr=max_lr / div_factor,
                             weight_decay=weight_decay, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr,
                                                steps_per_epoch=len(loaders['train']),
                                                epochs=epochs, div_factor=div_factor)

    train_losses = []
    test_losses = []
    best_acc = 0
    bect_epoch = -1
    for i in range(epochs):
        train_loss, train_acc = train(model, optim, loss_tuple, scheduler, loaders['train'], i, device, init_weights)
        train_losses.append(train_loss)

        test_loss, test_acc = test(model, loss_tuple, loaders['valid'], i, device, best_acc)
        test_losses.append(test_loss)
        best_acc = max(best_acc, test_acc)
        best_epoch = i if best_acc == test_acc else best_epoch

        plt.figure(figsize=(18, 9))
        plt.plot(np.arange(len(train_losses)), train_losses, label=f'Train, loss: {train_loss:.4f}, Acc: {train_acc}')
        plt.plot(np.arange(len(test_losses)), train_losses, label=f'Train, loss: {test_loss:.4f}, Acc: {test_acc}')
        plt.title(f'Epoch {i}')
        plt.legend(loc='best')
        plt.show()





