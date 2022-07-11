import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import logging

import pickle
import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from NN_stracture import SoundDataset
from NN_stracture import ToTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sound_show(inp, title=None):
    inp = inp.numpy()
    time = np.linspace(0., inp.size/8000, inp.size)
    plt.figure(figsize=(16, 8))
    plt.plot(time, inp)
    plt.title(title)
    plt.show()

def train(model, optim, criterion, dataloader, epoch, device):
    total = 0
    correct = 0
    train_loss = 0

    model.train()
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        optim.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        pred = output.argmax(1)
        total += output.shape[0]
        correct += pred.eq(label).sum().item()
    return train_loss / total, 100. * correct / total

def test(model, criterion, dataloader, epoch, device, best_acc, model_name='model'):
    total = 0
    correct = 0
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)

            loss = criterion(output, label)

            test_loss += loss
            pred = output.argmax(1)
            correct += torch.eq(pred, label).sum().item()
            total += data.shape[0]

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
    return test_loss / total, acc


if __name__ == '__main__':
    path_data = '/Users/litvan007/NN_sound_data_base/data'
    path_data_list = '/Users/litvan007/NN_sound_data_base/data_base.pickle'

    t = time.time()
    with open(path_data_list, 'rb') as fh:
        data_list = pickle.load(fh)
    logger.info(f"time of data init: {time.time() - t}")

    end_idx_tv = 120000
    train_valid_subset = data_list[:end_idx_tv]
    test_subset = data_list[end_idx_tv:]

    lengths = [int(len(train_valid_subset) * 0.77), int(len(train_valid_subset) * 0.23)]
    train_subset, valid_subset = torch.utils.data.random_split(train_valid_subset, lengths)

    transform_baseline = transforms.Compose([ToTensor()])
    data_set = {
                'train': SoundDataset(rootdir=path_data, subset=train_subset, transform=transform_baseline),
                'valid': SoundDataset(rootdir=path_data, subset=valid_subset, transform=transform_baseline),
                'test': SoundDataset(rootdir=path_data, subset=test_subset[:10], transform=transform_baseline)
    }

    batch_size = 1024
    loaders = {
                'train': torch.utils.data.DataLoader(data_set['train'], batch_size=batch_size, shuffle=True),
                'valid': torch.utils.data.DataLoader(data_set['valid'], batch_size=batch_size, shuffle=True),
                'test': torch.utils.data.DataLoader(data_set['test'], batch_size=batch_size, shuffle=False)
    }

    dataset_sizes = {x: len(data_set[x]) for x in ['train', 'valid', 'test']}
    xname, X, y = data_set['train'][0]
    print(xname)
    sound_show(X)
