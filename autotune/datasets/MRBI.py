import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing_extensions import Final


class MRBI(Dataset):
    """Please download the new MRBI manually from:

    http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip unzip it and rename the directory
    to "mrbi".
    """

    SPLIT_LIST: Final[Dict[str, List[str]]] = {
        'train': ["url_placeholder", "mrbi/mnist_all_background_images_rotation_normalized_train_valid.amat"],
        'test': ["url_placeholder", "mrbi/mnist_all_background_images_rotation_normalized_test.amat"]
    }

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.SPLIT_LIST:
            raise ValueError('Wrong split entered! Please use split="train" or split="extra" or split="test"')

        self.filename = self.SPLIT_LIST[split][1]
        # reading (loading) amat file as array
        self.data, self.labels = self._get_data(os.path.join(self.root, self.filename))

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.transpose(img, (1, 2, 0))
        img.reshape(28, 28)
        img = Image.fromarray(img.reshape(28, 28), 'L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f'Dataset {self.__class__.__name__} \n' \
               f'    Number of data points: {len(self)}\n' \
               f'    Split: {self.split}\n' \
               f'    Root Location: {self.root}\n' \
               f'    Transforms (if any):\n{self.transform}'

    @staticmethod
    def _parseline(line: str):
        data = np.array([float(i) for i in line.split()])
        x = data[:-1].reshape((28, 28), order='F')
        x = np.array(x*255, dtype=np.uint8)
        x = x[np.newaxis, :, :]
        y = data[-1]
        return x, y

    def _get_data(self, filename: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        print("Processing MRBI dataset")
        with open(filename) as file:
            data, labels = list(zip(*[self._parseline(line) for line in file]))
        return np.array(data), np.array(labels, dtype=int)
