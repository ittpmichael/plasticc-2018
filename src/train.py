# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from constants import *

LOGGER = logging.getLogger(__name__)


class PlasticcTrainingDataset(Dataset):
    """Plasticc training dataset"""

    def __init__(self, training_data_file_normed, training_metadata_file_normed, transform=None):
        """"

        :param training_data_file_normed: Path to training dataset that has been grouped
               and normalized with preprocessing.py.
        :type training_data_file_normed: str
        :param training_metadata_file_normed: Path to training metadata that has been
               normalized with preprocessing.py.
        :type training_metadata_file_normed: str
        :param transform: Optional transform to be applied
        :type transform: object
        """
        self.training_data = pd.read_csv(training_data_file_normed, index_col=[0,1,2])
        self.training_metadata = pd.read_csv(training_metadata_file_normed)
        self.transform = transform

    def __len__(self):
        """
        :return: number of objects
        """
        return len(self.training_data.index.unique(0))

    def __getitem__(self, idx):
        """
        :param idx: number index starting from 0 not the object id
        :type idx: object
        :return: sample: dictionary of sample data as large as specified batch size
        """
        object_id = self.training_data.index.unique(0)[idx]
        lightcurve = self.training_data.loc[object_id, :, :]

        metadata_feature = self.training_metadata[
            ['ra', 'decl', 'hostgal_specz', 'distmod',
             'mwebv', 'mjd_mean', 'flux_mean', 'flux_std']].loc[idx].as_matrix()

        target_id = self.training_metadata.target.loc[idx]
        target = np.zeros(NUMBER_OF_CLASSES)
        target[CLASSES_DICT[target_id]] = 1

        sample = {'object_id': object_id,
                  'lightcurve': lightcurve,
                  'metadata_feature': metadata_feature,
                  'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MakePassbandGroup(object):
    """Group data for each object_id by each passband as numpy array"""

    def __call__(self, sample):
        global passbands
        object_id, lightcurve, metadata_feature, target = sample['object_id'], sample['lightcurve'], \
                                                          sample['metadata_feature'], sample['target']
        for key in passbands_dict.keys():
            # put ['mjd','flux'] from one passband into numpy array
            each_passband = lightcurve.loc[object_id, key, :].transpose().as_matrix()

            if key == 0:
                passbands = np.array([each_passband])
            else:
                passbands = np.append(passbands, [each_passband], axis=0)

        return {'object_id': object_id, 'passbands': passbands,
                'metadata_feature': metadata_feature, 'target': target}


class ToTensor(object):
    """Convert ndarray in sample to Tensors"""
    def __call__(self, sample):
        object_id, passbands, metadata_feature, target = sample['object_id'], sample['passbands'],\
                                                sample['metadata_feature'], sample['target']
        return {'object_id': object_id,
                'passbands': torch.from_numpy(passbands),
                'metadata_feature': torch.from_numpy(metadata_feature),
                'target': torch.from_numpy(target)}


'''def TestLoadData():
    time_series_data = PlasticcTrainingDataset(
        training_file=DATA_TRAINING_SET_CSV,
        training_metadata_file=DATA_TRAINING_SET_METADATA_CSV,
        transform=transforms.Compose([
            MakePassbandGroup(),
            FeatureScale(),
        ]))

    for i in range(len(time_series_data.training_data.object_id)):
        sample = time_series_data[i]

        print(i, sample['object_id'],
              sample['passbands'].shape,
              sample['passbands'].min(),
              sample['passbands'].max(),
              )

        if i == 3:
            break'''


def train(model, **kwargs):
    time_series_data = PlasticcTrainingDataset(
        training_data_file_normed=DATA_TRAINING_SET_B_NORMED_CSV,
        training_metadata_file_normed=DATA_TRAINING_SET_METADATA_B_NORMED_CSV,
        transform=transforms.Compose([
            MakePassbandGroup(),
            ToTensor(),
        ]))

    time_series_test = PlasticcTrainingDataset(
        training_data_file_normed=DATA_TRAINING_SET_NORMED_CSV,
        training_metadata_file_normed=DATA_TRAINING_SET_METADATA_NORMED_CSV,
        transform=transforms.Compose([
            MakePassbandGroup(),
            ToTensor(),
        ])
    )

    dataloader = DataLoader(
        time_series_data,
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=kwargs['using_gpu'],
    )
    testloader = DataLoader(
        time_series_test,
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=kwargs['using_gpu'],
    )
    # Choose an optimizer
    optimizer = optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
    # start training
    for epoch in range(1, kwargs['epochs'] + 1):
        if kwargs['use_learning_rate_decay']:
            adjust_learning_rate(optimizer, epoch, kwargs['learning_rate_decay'], kwargs['start_learning_rate_decay'],
                                 kwargs['learning_rate'])
        train_epoch(epoch, model, dataloader, optimizer, kwargs['log_interval'], kwargs['using_gpu'])

    class_correct = list(0. for i in range(NUMBER_OF_CLASSES))
    class_total = list(0. for i in range(NUMBER_OF_CLASSES))
    with torch.no_grad():
        for sample in testloader:
            passbands, metadata_feature, target = sample['passbands'], sample['metadata_feature'], sample['target']
            outputs = model(passbands, metadata_feature)
            _, target = torch.max(target, 1)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target.cuda()).squeeze()
            for i in range(4):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_list = list(CLASSES_DICT.items())
    for i in range(NUMBER_OF_CLASSES):
        LOGGER.info('Accuracy of {} : {:.1f}%'.format(
            class_list[i][0],
            100 * class_correct[i] / (class_total[i]+0.01))
        )
    LOGGER.info('Training finished.')


def train_epoch(epoch, model, dataloader, optimizer, log_interval, using_gpu):
    model.train()
    for batch_idx, sample in enumerate(dataloader):
        object_id, passbands, target = Variable(sample['object_id']), \
                                       Variable(sample['passbands']), \
                                       Variable(sample['target'])
        metadata_feature = Variable(sample['metadata_feature'])
        # zeros the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(passbands, metadata_feature)
        if using_gpu:
            loss = F.mse_loss(outputs, target.cuda())
        else:
            loss = F.mse_loss(outputs, target)

        # Criterion is defined here
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and batch_idx > 1:
            LOGGER.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                epoch,
                batch_idx * len(object_id),
                len(dataloader.dataset),
                100. * batch_idx / len(dataloader),
                loss.data[0])
            )


def adjust_learning_rate(optimizer, epoch, learning_rate_decay, start_learning_rate_decay, learning_rate):
    """ Sets the learning rate to the initial LR decayed  """
    lr_decay = learning_rate_decay ** max(epoch + 1 - start_learning_rate_decay, 0.0)
    new_learning_rate = learning_rate * lr_decay
    LOGGER.info('New learning rate: {}'.format(new_learning_rate))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = new_learning_rate