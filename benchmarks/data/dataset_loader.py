import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from typing import Tuple
from torch.utils.data.dataset import Dataset


class DatasetLoader:

    @staticmethod
    def get_samplers(train_data: Dataset, valid_size: float = 0.2, shuffle: bool = True) \
            -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
        """
        :param train_data: train data
        :param valid_size: percentage split of the training set used for the validation set. Should be in range [0, 1].
        :param shuffle: whether to shuffle the train/validation indices
        :return: train sampler, validation sampler
        """
        if not 0 <= valid_size <= 1:
            raise ValueError("[!] valid_size should be in the range [0, 1].")

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

        return train_sampler, val_sampler
