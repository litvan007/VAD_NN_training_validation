import sys
import os
import gc
import logging
import pickle
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from nn_sound_tools.Datasets.segmentation import baseline_dataset, mfcc_dataset, eGMAPSv02_dataset
from nn_sound_tools.Architectures.segmentation import baseline_model, mfcc_model, eGMAPSv02_model

class learn():
    def __init__(self, **kwargs) -> None:
        assert list(kwargs.keys()) == ['--dataset', '--arch', '--path'], 'Wrong keys'
        assert kwargs['--dataset'] in ['baseline', 'mfcc', 'egmapsv02'], 'Wrong dataset'
        assert kwargs['--arch'] in ['v1'], 'Wrong architecture' # TODO add new versions in near future

        self.type_dataset = kwargs['--dataset']
        self.model_version = kwargs['--arch']
        self.path_data = kwargs['--path']
        self.path_save = './models/'
        self.model_name = f"model_{kwargs['--dataset']}_{kwargs['--arch']}"

        self.end_idx_tv = 87000
        self.batch_size = 512
        self.epochs = 30

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.subsets = self.__create_subsets()
        self.loaders = self.__create_loader(*self.__create_subsets())
        logging.info('Loaders are created')

        logging.info('Learning...')
        self.process()

    def process(self):
        # os.system("wandb login a4fc0db907801a21173615cb84708e1774b986a4")
        # wandb.init(project=self.model_name)
        # wandb.config = {
        #     "epochs": self.epochs,
        #     "batch_size": self.batch_size
        # }
        # logging.info('Wandb is loaded')

        torch.cuda.empty_cache()
        gc.collect()
        model, loss, optim, \
        scheduler, epoch, \
        train_losses, valid_losses, \
        train_metrics, valid_metrics = self.__init_model()

        best_path = None
        # wandb.watch(model)
        for i in range(epoch, self.epochs):
            train_loss, metrics = self.__train(model, optim, loss, scheduler, self.loaders['train'])

            train_losses.append(train_loss)
            train_acc = metrics['Accuracy'].item()
            train_metrics['Accuracy'].append(train_acc)

            valid_loss, metrics = self.__valid(model, loss, self.loaders['valid'])
            valid_acc = metrics['Accuracy'].item()
            valid_metrics['Accuracy'].append(valid_acc)

            print(f'{i} epoch')
            print(valid_loss, valid_losses)
            if i == 0 or valid_loss < np.min(valid_losses) or True:
                best_path = os.path.join(self.path_data, datetime.now().strftime('{0}_{1}_{2:.3f}_%Y-%m-%d_%H'.format(self.model_name, i, valid_loss)))
                valid_losses.append(valid_loss)
                self.__save(best_path, loss, epoch, model, optim, scheduler, train_losses, valid_losses, train_metrics, valid_metrics)
            else:
                valid_losses.append(valid_loss)

            for train_loss, valid_loss, train_acc, valid_acc in zip(train_losses, valid_losses, train_metrics['Accuracy'], valid_metrics['Accuracy']):
                wandb.log({"train_loss": train_loss})
                wandb.log({"valid_loss": valid_loss})
                wandb.log({"train_accuracy": train_acc})
                wandb.log({"valid_accuracy": valid_acc})
        wandb.finish()

    def __save(self, path, *args):
        torch.save({
        'loss': args[0],
        'epoch': args[1],
        'model_state_dict': args[2].state_dict(),
        'optimizer_state_dict': args[3].state_dict(),
        'scheduler_state_dict': args[4].state_dict(),
        'train_loss_epochs': args[5],
        'valid_loss_epochs': args[6],
        'train_metrics': args[7],
        'valid_metrics': args[8]
    }, path)

    def __create_subsets(self):
        data_list = []
        with open(f'{self.path_data}/data_base_2.pickle', 'rb') as fh:
            data_list = pickle.load(fh)

        train_valid_subset = data_list[:self.end_idx_tv]
        test_subset = data_list[self.end_idx_tv:]

        lengths = [int(len(train_valid_subset) * 0.9), int(len(train_valid_subset) * 0.1)]
        train_subset, valid_subset = torch.utils.data.random_split(train_valid_subset, lengths)

        return train_subset, valid_subset, test_subset

    def __create_loader(self, *subset):
        if self.type_dataset == 'baseline':
            data_set = {
                'train': baseline_dataset(rootdir=self.path_data, subset=subset[0]),
                'valid': baseline_dataset(rootdir=self.path_data, subset=subset[1]),
                'test': baseline_dataset(rootdir=self.path_data, subset=subset[2])
            }
        elif self.type_dataset == 'mfcc':
            data_set = {
                'train': mfcc_dataset(rootdir=self.path_data, subset=subset[0]),
                'valid': mfcc_dataset(rootdir=self.path_data, subset=subset[1]),
                'test': mfcc_dataset(rootdir=self.path_data, subset=subset[2])
            }
        elif self.type_dataset == 'egmapsv02':
            data_set = {
                'train': eGMAPSv02_dataset(rootdir=self.path_data, subset=subset[0]),
                'valid': eGMAPSv02_dataset(rootdir=self.path_data, subset=subset[1]),
                'test': eGMAPSv02_dataset(rootdir=self.path_data, subset=subset[2])
            }

        loaders = {
            'train': torch.utils.data.DataLoader(data_set['train'], batch_size=self.batch_size, shuffle=True),
            'valid': torch.utils.data.DataLoader(data_set['valid'], batch_size=self.batch_size, shuffle=False),
            'test': torch.utils.data.DataLoader(data_set['test'], batch_size=1, shuffle=False)
            }
        self.dataset_sizes = {x: len(data_set[x]) for x in ['train', 'valid', 'test']}

        return loaders

    def __init_model(self, path=None):
        max_lr = 0.05
        div_factor = 250
        weight_decay = 0.0005

        loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones(48000, dtype=torch.float16).to(self.device) * 3).to(self.device)

        if self.model_name == 'model_baseline_v1':
            model = baseline_model().to(self.device)
            optim = torch.optim.AdamW(model.parameters(), lr=max_lr / div_factor,
                                        weight_decay=weight_decay, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr,
                                    steps_per_epoch=len(self.loaders['train']),
                                    epochs=self.epochs, div_factor=div_factor)
        elif self.model_name == 'model_mfcc_v1':
            model = mfcc_model().to(self.device)
            optim = torch.optim.AdamW(model.parameters(), lr=max_lr / div_factor,
                                        weight_decay=weight_decay, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr,
                                    steps_per_epoch=len(self.loaders['train']),
                                    epochs=self.epochs, div_factor=div_factor)
        elif self.model_name == 'model_egmapsv02_v1':
            model = eGMAPSv02_model().to(self.device)
            optim = torch.optim.AdamW(model.parameters(), lr=max_lr / div_factor,
                                        weight_decay=weight_decay, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr,
                                    steps_per_epoch=len(self.loaders['train']),
                                    epochs=self.epochs, div_factor=div_factor)

        if path is None:
            def init_weights(m):
                if type(m) in (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d):
                    nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                # else:
                    # print('Not setting weights for type {}'.format(type(m)))
            model.apply(init_weights)

        epoch = 0
        train_losses = []
        valid_losses = []
        train_metrics = {
            'Accuracy': []
        }
        valid_metrics = {
            'Accuracy': []
        }

        if path is not None:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            train_losses = checkpoint['train_loss_epochs']
            valid_losses = checkpoint['valid_loss_epochs']
            epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            train_metrics = checkpoint['train_metrics']
            valid_metrics = checkpoint['valid_metrics']

        return model, loss, optim, scheduler, epoch, train_losses, valid_losses, train_metrics, valid_metrics

    def __binary_acc(self, y_pred, y_test):
            correct_results_sum = (y_pred == y_test).sum().float()
            print(torch.unique(y_pred))
            acc = correct_results_sum/(y_test.shape[0]*y_test.shape[2])
            return acc

    def __train(self, model, optim, loss, scheduler, dataloader):
        losses = []
        model.train()
        for i, batch in enumerate(tqdm(dataloader)):
            xname, X, marks, _ = batch
            X, marks = X.to(self.device), marks.to(self.device)
            print(X.shape, marks.shape)
            optim.zero_grad()
            out = model(X)
            print(out.shape)
            loss_batch = loss(out, marks)
            losses.append(loss_batch.item())
            loss_batch.backward()
            optim.step()
            scheduler.step()

            prediction = torch.round(torch.sigmoid(out)).type(torch.int8)
            marks = marks.type(torch.int8)

        return np.mean(losses), {'Accuracy': self.__binary_acc(prediction, marks)}

    def __valid(self, model, loss, dataloader):
        losses = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                xname, X, marks, _ = batch
                X, marks = X.to(self.device), marks.to(self.device)
                out = model(X)
                loss_batch = loss(out, marks)
                losses.append(loss_batch.item())
                prediction = torch.round(torch.sigmoid(out)).type(torch.int8)
                marks = marks.type(torch.int8)

        return np.mean(losses), {'Accuracy': self.__binary_acc(prediction, marks)}

if __name__ == '__main__':
    logging.basicConfig(filename='nn_segmentation.log', level=logging.INFO)
    input_args = dict(arg.split('=') for arg in sys.argv[1:])
    print(input_args)
    logging.info('Initialization')
    learn(**input_args)
