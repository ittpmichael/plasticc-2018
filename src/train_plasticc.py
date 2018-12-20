
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from constants import *
from train import train


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 6 input channels
        self.conv1 = nn.Conv1d(NUMBER_CHANNELS, 36, kernel_size=(4, 2), stride=(2, 2), padding=1).double()
        self.conv2 = nn.Conv1d(36, 60, kernel_size=(4, 2), stride=(2, 2), padding=1).double()
        self.conv3 = nn.Conv1d(60, 12, kernel_size=(4, 2), stride=(2, 2), padding=1).double()
        self.fc1 = nn.Linear(216, 100).double()
        self.batchnorm1 = nn.BatchNorm1d(100).double()
        self.fc2 = nn.Linear(100, 50).double()
        self.batchnorm2 = nn.BatchNorm1d(50).double()
        self.fc3 = nn.Linear(50, 20).double()
        self.batchnorm3 = nn.BatchNorm1d(20).double()
        self.fc4 = nn.Linear(28, 200).double()
        self.fc5 = nn.Linear(200, 100).double()
        self.batchnorm5 = nn.BatchNorm1d(100).double()
        self.fc6 = nn.Linear(100, 50).double()
        self.batchnorm6 = nn.BatchNorm1d(50).double()
        self.fc7 = nn.Linear(50, NUMBER_OF_CLASSES).double()

    def forward(self, input_data_values, metadata_feature):
        x = F.leaky_relu(self.conv1(input_data_values))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.batchnorm1(F.leaky_relu(self.fc1(x)))
        x = self.batchnorm2(F.leaky_relu(self.fc2(x)))
        x = self.batchnorm3(F.leaky_relu(self.fc3(x)))
        x = torch.cat((x, metadata_feature), dim=1)

        x = F.leaky_relu(self.fc4(x))
        x = self.batchnorm5(F.leaky_relu(self.fc5(x)))
        x = self.batchnorm6(F.leaky_relu(self.fc6(x)))
        x = F.softmax(self.fc7(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net2(nn.Module):
    """
    Net2 put more features on light curve data
    """
    def __init__(self):
        super(Net2, self).__init__()
        # 6 input channels
        self.conv1 = nn.Conv1d(NUMBER_CHANNELS, 36, kernel_size=(4, 2), stride=(2, 2), padding=1).double()
        self.conv2 = nn.Conv1d(36, 60, kernel_size=(4, 2), stride=(2, 2), padding=1).double()
        self.conv3 = nn.Conv1d(60, 12, kernel_size=(4, 2), stride=(2, 2), padding=1).double()
        self.fc1 = nn.Linear(216, 100).double()
        self.batchnorm1 = nn.BatchNorm1d(100).double()
        self.fc2 = nn.Linear(100, 50).double()
        self.batchnorm2 = nn.BatchNorm1d(50).double()
        self.fc3 = nn.Linear(50, 50).double()
        self.batchnorm3 = nn.BatchNorm1d(50).double()
        self.fc4 = nn.Linear(58, 200).double()
        self.fc5 = nn.Linear(200, 100).double()
        self.batchnorm5 = nn.BatchNorm1d(100).double()
        self.fc6 = nn.Linear(100, 50).double()
        self.batchnorm6 = nn.BatchNorm1d(50).double()
        self.fc7 = nn.Linear(50, NUMBER_OF_CLASSES).double()

    def forward(self, input_data_values, metadata_feature):
        x = F.leaky_relu(self.conv1(input_data_values))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.batchnorm1(F.leaky_relu(self.fc1(x)))
        x = self.batchnorm2(F.leaky_relu(self.fc2(x)))
        x = self.batchnorm3(F.leaky_relu(self.fc3(x)))
        x = torch.cat((x, metadata_feature), dim=1)

        x = F.leaky_relu(self.fc4(x))
        x = self.batchnorm5(F.leaky_relu(self.fc5(x)))
        x = self.batchnorm6(F.leaky_relu(self.fc6(x)))
        x = F.softmax(self.fc7(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def show_batch(sample_batched):
    """"Show image with landmarks for a batch of samples"""
    object_id_batch, passbands_batch = sample_batched['object_id'], sample_batched['passbands']
    batch_size = len(object_id_batch)

    for i in range(batch_size):
        plt.figure()
        for key in passbands_dict.keys():
            # print(passbands_batch[0,key].shape)
            x, y = 0, 1
            plt.scatter(passbands_batch[0, key, :, x].numpy(),
                        passbands_batch[0, key, :, y].numpy(),
                        s=4,)
        plt.title("Batch from dataloader")
        plt.show()


def main():
    # define some parameters
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(process)d:%(levelname)s:%(name)s:%(message)s')

    parser = argparse.ArgumentParser(description='Plasticc Training')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--use-learning-rate-decay', action='store_true', default=False,
                        help='make learning rate to be decayed (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--learning-rate-decay', type=float, default=0.9, metavar='LRD',
                        help='the initial learning rate decay rate')
    parser.add_argument('--start-learning-rate-decay', type=int, default=5, help='the epoch to start applying the LRD')
    parser.add_argument('--use-gpu', action='store_true', default=False, help='use the GPU if it is available')

    kwargs = vars(parser.parse_args())
    LOGGER.debug(kwargs)

    if kwargs['use_gpu'] and torch.cuda.is_available():
        LOGGER.info('Using cuda devices: {}'.format(torch.cuda.device_count()))
        kwargs['cuda_device_count'] = torch.cuda.device_count()
        kwargs['using_gpu'] = True
    else:
        LOGGER.info('Using CPU')
        kwargs['cuda_device_count'] = 0
        kwargs['using_gpu'] = False

    if kwargs['using_gpu']:
        model = nn.DataParallel(Net()).cuda()
        train(model, **kwargs)

    else:
        model = Net2()
        model.share_memory()
        train(model, **kwargs)

    '''dataiter = iter(dataloader)
    sample = dataiter.next()
    passbands, target = sample['passbands'], sample['target']
    # create the class list to access class index easily
    class_list = list(CLASSES_DICT.items())
    _, target = torch.max(target, 1)
    print('GroundTruth: ', ' '.join('%5s' % class_list[target[j]][0]
                                    for j in range(4)))

    outputs = net(passbands)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % class_list[predicted[j]][0]
                                  for j in range(4)))'''


if __name__ == '__main__':
    main()