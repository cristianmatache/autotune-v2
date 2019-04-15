import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


class MRBI(Dataset):

    """
    Please download the new MRBI manually from:
    http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip
    unzip it and rename the directory to "mrbi"
    """

    split_list = {
        'train': ["url_placeholder", "mrbi/mnist_all_background_images_rotation_normalized_train_valid.amat"],
        'test': ["url_placeholder", "mrbi/mnist_all_background_images_rotation_normalized_test.amat"]
    }

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" or split="extra" or split="test"')

        self.filename = self.split_list[split][1]
        # reading (loading) amat file as array
        self.data, self.labels = self._get_data(os.path.join(self.root, self.filename))

    def __getitem__(self, index):
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
        img.reshape(28,28)
        img = Image.fromarray(img.reshape(28,28), 'L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

    @staticmethod
    def _parseline(line):
        data = np.array([float(i) for i in line.split()])
        x = data[:-1].reshape((28, 28), order='F')
        x = np.array(x*255, dtype=np.uint8)
        x = x[np.newaxis, :, :]
        y = data[-1]
        return x, y

    def _get_data(self, filename):
        print("Processing MRBI dataset")
        with open(filename) as file:
            data, labels = list(zip(*[self._parseline(line) for line in file]))
        return np.array(data), np.array(labels, dtype=int)
