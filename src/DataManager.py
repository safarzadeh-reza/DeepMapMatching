from numpy import testing
import torch
import numpy as np

from src.util import normalization, sequence_data
from torch.utils.data import Dataset, DataLoader


class DataManager(object):
    def __init__(self, input_path, label_path, boundaries, train_ratio, batch_size):
        self.raw_input = np.load(input_path)
        self.boundaries = boundaries
        self.raw_input = normalization(
            self.raw_input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])

        self.raw_label = np.load(label_path)
        self.raw_label[self.raw_label < 0] = 0

        self.train_ratio = train_ratio
        self.test_ratio = 1-train_ratio

        n_data = self.raw_label.shape[0]
        n_train = int(n_data*train_ratio)
        n_test = n_data-n_train

        randidx = np.random.permutation(n_data)
        train_idx = randidx[:n_train]
        test_idx = randidx[n_train:(n_train+n_test)]

        train_input = torch.FloatTensor(self.raw_input[train_idx])
        train_label = torch.LongTensor(self.raw_label[train_idx])
        train_len = train_input[:, :, 0] != -1
        train_len = train_len.sum(axis=1)

        test_input = torch.FloatTensor(self.raw_input[test_idx])
        self.test_input = test_input
        test_target = torch.LongTensor(self.raw_label[test_idx])
        test_len = test_input[:, :, 0] != -1
        test_len = test_len.sum(axis=1)

        train_data = sequence_data(train_input, train_label, train_len)
        test_data = sequence_data(test_input, test_target, test_len)

        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
