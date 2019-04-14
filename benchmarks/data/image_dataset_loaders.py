from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN
import abc
from typing import Tuple
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from benchmarks.data.dataset_loader import DatasetLoader
from benchmarks.data.MRBI import MRBI


class ImageDatasetLoader(DatasetLoader):

    @abc.abstractmethod
    def split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        :return: train_data, validation_data, test_data
        """
        pass

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

        # set the split datasets
        train_data, val_data, test_data = self.split_dataset()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # creating data samplers
        train_sampler, val_sampler = self.get_samplers(self.train_data, valid_size, shuffle)
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        # set loaders

        self.val_loader = DataLoader(val_data, batch_size=100, sampler=val_sampler, num_workers=2, pin_memory=False)
        self.test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2, pin_memory=False)
        # the train loader depends on the batch size -> so it is a function

    def train_loader(self, batch_size: int = 100) -> DataLoader:
        """
        :param batch_size:  batch size can be supplied because it might be used as a hyperparameter
        :return:
        """
        train_data = self.train_data
        train_sampler = self.train_sampler
        return DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=False)


class CIFARLoader(ImageDatasetLoader):

    def __init__(self, data_dir, mean_normalize=(0.485, 0.456, 0.406), std_normalize=(0.229, 0.224, 0.225),
                 augment=True):
        self.augment = augment
        super().__init__(data_dir, mean_normalize, std_normalize)

    def split_dataset(self):
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

    def __init__(self, data_dir,  mean_normalize=(0.1307,), std_normalize=(0.3081,)):
        super().__init__(data_dir, mean_normalize, std_normalize)

    def split_dataset(self):
        train_data = MNIST(root=self.data_dir, train=True, download=True, transform=self.def_transform)
        val_data = MNIST(root=self.data_dir, train=True, download=True, transform=self.def_transform)
        test_data = MNIST(root=self.data_dir, train=False, download=True, transform=self.def_transform)

        return train_data, val_data, test_data


class SVHNLoader(ImageDatasetLoader):

    def __init__(self, data_dir, mean_normalize=(0.4377, 0.4438, 0.4728), std_normalize=(0.1201, 0.1231, 0.1052)):
        super().__init__(data_dir, mean_normalize, std_normalize)

    def split_dataset(self):
        train_data = SVHN(root=self.data_dir, split='train', download=True, transform=self.def_transform)
        val_data = SVHN(root=self.data_dir, split='train', download=True, transform=self.def_transform)
        test_data = SVHN(root=self.data_dir, split='test', download=True, transform=self.def_transform)

        return train_data, val_data, test_data


class MRBILoader(ImageDatasetLoader):

    def __init__(self, data_dir, mean_normalize=(0.5406,), std_normalize=(0.2318,)):
        super().__init__(data_dir, mean_normalize, std_normalize)

    def split_dataset(self):
        train_data = MRBI(root=self.data_dir, split="train", transform=self.def_transform)
        val_data = MRBI(root=self.data_dir, split="train", transform=self.def_transform)
        test_data = MRBI(root=self.data_dir, split="test", transform=self.def_transform)

        return train_data, val_data, test_data
