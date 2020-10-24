import abc
from pathlib import Path
from typing import Tuple

from filelock import FileLock
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN

from autotune.datasets.dataset_loader import DatasetLoader
from autotune.datasets.MRBI import MRBI


class ImageDatasetLoader(DatasetLoader):

    @abc.abstractmethod
    def _split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        :return: train_data, validation_data, test_data
        """

    def __init__(self, data_dir: str,
                 mean_normalize: Tuple[float, ...] = (), std_normalize: Tuple[float, ...] = (),
                 valid_size: float = 0.2, shuffle: bool = True):

        self.data_dir = data_dir

        # default transform
        if mean_normalize and std_normalize:
            self.normalize = transforms.Normalize(mean=mean_normalize, std=std_normalize)
            self.def_transform = transforms.Compose([transforms.ToTensor(), self.normalize])
        else:
            self.def_transform = transforms.Compose([transforms.ToTensor()])

        # Set the split datasets
        train_data, val_data, test_data = self._split_dataset()
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data

        # Creating datasets samplers
        train_sampler, val_sampler = self.get_samplers(train_data, valid_size, shuffle)
        self._train_sampler, self._val_sampler = train_sampler, val_sampler

        # Note that the loaders are the most important and lucrative parts of this class
        self.val_loader = DataLoader(val_data, batch_size=100, sampler=val_sampler, num_workers=2, pin_memory=False)
        self.test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2, pin_memory=False)
        # The training set loader depends on the batch size -> so it is a function

    def train_loader(self, batch_size: int = 100) -> DataLoader:
        """
        :param batch_size:  batch size can be supplied because it might be used as a hyperparameter
        :return:
        """
        train_data = self.train_data
        train_sampler = self._train_sampler
        return DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=False)


class CIFARLoader(ImageDatasetLoader):

    def __init__(self, data_dir, mean_normalize=(0.485, 0.456, 0.406), std_normalize=(0.229, 0.224, 0.225),
                 augment=True):
        self.augment = augment
        super().__init__(data_dir, mean_normalize, std_normalize)

    def _split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        :return: train_data, validation_data, test_data
        """
        if self.augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            train_transform = self.def_transform

        # load the dataset
        train_data = CIFAR10(root=self.data_dir, train=True, download=True, transform=train_transform)
        val_data = CIFAR10(root=self.data_dir, train=True, download=True, transform=self.def_transform)
        test_data = CIFAR10(root=self.data_dir, train=False, download=True, transform=self.def_transform)

        return train_data, val_data, test_data


class MNISTLoader(ImageDatasetLoader):

    def __init__(self, data_dir, mean_normalize=(0.1307,), std_normalize=(0.3081,)):
        super().__init__(data_dir, mean_normalize, std_normalize)

    def _split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        lock = FileLock(str(Path(self.data_dir) / 'download.lock'))
        with lock:
            train_data = MNIST(root=self.data_dir, train=True, download=True, transform=self.def_transform)
        with lock:
            val_data = MNIST(root=self.data_dir, train=True, download=True, transform=self.def_transform)
        with lock:
            test_data = MNIST(root=self.data_dir, train=False, download=True, transform=self.def_transform)

        return train_data, val_data, test_data


class SVHNLoader(ImageDatasetLoader):

    def __init__(self, data_dir, mean_normalize=(0.4377, 0.4438, 0.4728), std_normalize=(0.1201, 0.1231, 0.1052)):
        super().__init__(data_dir, mean_normalize, std_normalize)

    def _split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        train_data = SVHN(root=self.data_dir, split='train', download=True, transform=self.def_transform)
        val_data = SVHN(root=self.data_dir, split='train', download=True, transform=self.def_transform)
        test_data = SVHN(root=self.data_dir, split='test', download=True, transform=self.def_transform)

        return train_data, val_data, test_data


class MRBILoader(ImageDatasetLoader):

    def __init__(self, data_dir, mean_normalize=(0.5406,), std_normalize=(0.2318,)):
        super().__init__(data_dir, mean_normalize, std_normalize)

    def _split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        train_data = MRBI(root=self.data_dir, split="train", transform=self.def_transform)
        val_data = MRBI(root=self.data_dir, split="train", transform=self.def_transform)
        test_data = MRBI(root=self.data_dir, split="test", transform=self.def_transform)

        return train_data, val_data, test_data
