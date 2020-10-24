import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from autotune.datasets.image_dataset_loaders import CIFARLoader, MRBILoader, SVHNLoader

DATA_DIR = "D:/datasets/"
PROBLEM = 'mrbi'


def get_train_data(problem: str, data_dir: str) -> DataLoader:
    if problem == "cifar":
        return CIFARLoader(data_dir, mean_normalize=(), std_normalize=(), augment=False).train_data
    if problem == "svhn":
        return SVHNLoader(data_dir, mean_normalize=(), std_normalize=()).train_data
    if problem == "mrbi":
        return MRBILoader(data_dir, mean_normalize=(), std_normalize=()).train_data
    raise ValueError('Problem not found')


def main():
    train_data = get_train_data(PROBLEM, DATA_DIR)
    print('Number of samples: ', len(train_data))

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

    def imshow(img):
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

    # Get some random training images
    images, labels = iter(train_loader).__next__()

    # Show images
    imshow(torchvision.utils.make_grid(images))

    # Print labels
    print(' '.join(f'{labels[j]}' for j in range(4)))
    plt.show(block=True)


if __name__ == "__main__":
    main()
